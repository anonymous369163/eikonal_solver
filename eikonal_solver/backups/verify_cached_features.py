"""
Verify that cached samroad_feat_full_*.npy features align with online encoder output.

For each test region:
  1. Load TIF and take a 512x512 crop at (y0, x0)
  2. Run SAM image_encoder on the crop -> feat_online [1, 256, 32, 32]
  3. Load cached .npy and crop at feat[y0//16 : y0//16+32, x0//16 : x0//16+32]
  4. Compare feat_online vs feat_cached (cos similarity, max abs diff)

Findings:
  - Cached features were generated with SAM-Road (cityscale) encoder, not raw SAM.
  - Verification uses cityscale checkpoint by default. Cosine > 0.93 => safe to use --use_cached_features.
  - TIF/Feat are matched by sorted glob; check that crop_* and samroad_feat_full_* share the same location key.

Usage:
  cd MMDataset && python eikonal_solver/verify_cached_features.py
  python eikonal_solver/verify_cached_features.py --region_dir path/to/region
"""
import argparse
import os
import sys
import numpy as np
import torch

# Add parent for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from dataset import _load_rgb_from_tif
from model import SAMRoute


class _MinimalConfig:
    """Minimal config for SAMRoute to load encoder only."""
    def __init__(self):
        self.SAM_VERSION = "vit_b"
        self.PATCH_SIZE = 512
        self.NO_SAM = False
        self.USE_SAM_DECODER = False
        self.ENCODER_LORA = False
        self.FREEZE_ENCODER = True
        self.FOCAL_LOSS = False
        self.TOPONET_VERSION = "default"
        _root = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
        self.SAM_CKPT_PATH = os.path.join(_root, "sam_road_repo/sam_ckpts/sam_vit_b_01ec64.pth")

    def get(self, k, d=None):
        return getattr(self, k, d)


def load_encoder(device, pretrained_ckpt=None):
    cfg = _MinimalConfig()
    model = SAMRoute(cfg).to(device)
    model.eval()
    # 尝试加载 SAM-Road 预训练权重（离线特征可能由此生成）
    _root = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
    for ckpt in [pretrained_ckpt, os.path.join(_root, "checkpoints/cityscale_vitb_512_e10.ckpt")]:
        if ckpt and os.path.isfile(ckpt):
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            clean = {k.replace("model.", ""): v for k, v in sd.items()}
            model.load_state_dict(clean, strict=False)
            return model
    return model


def run_verification(region_dir: str, device: torch.device, y0: int = 256, x0: int = 256, pretrained_ckpt: str = None):
    """
    Verify cached feat vs online encoder for one region.
    y0, x0: top-left of 512x512 crop in full image coordinates.
    """
    region_dir = os.path.abspath(region_dir)
    tifs = sorted(__import__("glob").glob(os.path.join(region_dir, "crop_*.tif")))
    feats = sorted(__import__("glob").glob(os.path.join(region_dir, "samroad_feat_full_*.npy")))

    if not tifs:
        print(f"[ERROR] No crop_*.tif in {region_dir}")
        return False
    if not feats:
        print(f"[ERROR] No samroad_feat_full_*.npy in {region_dir}")
        return False

    # 确保 TIF 与 Feat 一一对应：文件名中的 location key 应一致
    tif_path = tifs[0]
    feat_path = feats[0]
    tif_base = os.path.basename(tif_path).replace("crop_", "").replace("_z16.tif", "").replace(".tif", "")
    feat_base = os.path.basename(feat_path).replace("samroad_feat_full_", "").replace("_z16.npy", "").replace(".npy", "")
    if tif_base != feat_base:
        print(f"[WARN] TIF/Feat filename mismatch: {tif_base} vs {feat_base}")
    print(f"TIF:  {os.path.basename(tif_path)}")
    print(f"Feat: {os.path.basename(feat_path)}")

    # 1. Load full image
    rgb_full = _load_rgb_from_tif(tif_path)  # [H, W, 3]
    h, w = rgb_full.shape[:2]

    # 2. Crop 512x512 (with bounds check)
    ps = 512
    y0 = max(0, min(y0, h - ps))
    x0 = max(0, min(x0, w - ps))
    rgb_crop = rgb_full[y0 : y0 + ps, x0 : x0 + ps]
    if rgb_crop.shape[0] < ps or rgb_crop.shape[1] < ps:
        pad_h, pad_w = ps - rgb_crop.shape[0], ps - rgb_crop.shape[1]
        rgb_crop = np.pad(rgb_crop, ((0, pad_h), (0, pad_w), (0, 0)))
    rgb_t = torch.from_numpy(rgb_crop.astype(np.float32)).unsqueeze(0).to(device)  # [1,H,W,3]

    # 3. Run encoder (optionally with SAM-Road pretrained weights)
    model = load_encoder(device, pretrained_ckpt)
    x = rgb_t.permute(0, 3, 1, 2)  # [1,3,H,W]
    x = (x - model.pixel_mean) / model.pixel_std
    with torch.no_grad():
        feat_online = model.image_encoder(x)  # [1, 256, 32, 32]
    feat_online = feat_online[0].cpu().numpy()  # [256, 32, 32]

    # 4. Load cached feat and crop
    feat_full = np.load(feat_path)  # [C, Hf, Wf]
    stride = 16
    fy0, fx0 = y0 // stride, x0 // stride
    fy1 = min(fy0 + 32, feat_full.shape[1])
    fx1 = min(fx0 + 32, feat_full.shape[2])
    feat_cached = np.ascontiguousarray(feat_full[:, fy0:fy1, fx0:fx1])
    pad_fy = 32 - feat_cached.shape[1]
    pad_fx = 32 - feat_cached.shape[2]
    if pad_fy > 0 or pad_fx > 0:
        feat_cached = np.pad(feat_cached, ((0, 0), (0, pad_fy), (0, pad_fx)))
    feat_cached = feat_cached.astype(np.float32)

    # 5. Compare
    f_o = feat_online.reshape(-1)
    f_c = feat_cached.reshape(-1)
    cos_sim = np.dot(f_o, f_c) / (np.linalg.norm(f_o) * np.linalg.norm(f_c) + 1e-8)
    max_abs_diff = np.abs(feat_online - feat_cached).max()
    mean_abs_diff = np.abs(feat_online - feat_cached).mean()

    print(f"Crop at (y0={y0}, x0={x0}) -> feat crop (fy0={fy0}, fx0={fx0})")
    print(f"feat_online shape: {feat_online.shape}, feat_cached shape: {feat_cached.shape}")
    print(f"Cosine similarity: {cos_sim:.6f} (1.0 = identical)")
    print(f"Max abs diff:       {max_abs_diff:.6f}")
    print(f"Mean abs diff:      {mean_abs_diff:.6f}")

    # Heuristic: cos_sim > 0.93 suggests same encoder + correct spatial correspondence.
    # Drift from: (1) full-image vs crop context (ViT global attn), (2) fp16/fp32, (3) numerical variance.
    ok = cos_sim > 0.93 and max_abs_diff < 1.0
    if ok:
        print("[PASS] Cached features align with online encoder (cos > 0.93). Safe to use --use_cached_features.")
    else:
        print("[FAIL] Cached features do not match. Ensure encoder weights match (e.g. cityscale). Do not use cached.")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Verify cached encoder features vs online")
    ap.add_argument("--region_dir", type=str, default=None,
                    help="Single region dir to test; default: auto-pick first with feat")
    ap.add_argument("--data_root", type=str,
                    default=os.path.join(os.path.dirname(__file__), "../Gen_dataset_V2/Gen_dataset"),
                    help="Dataset root to search regions")
    ap.add_argument("--y0", type=int, default=256, help="Crop top-left y")
    ap.add_argument("--x0", type=int, default=256, help="Crop top-left x")
    ap.add_argument("--n_regions", type=int, default=3, help="Number of regions to test (when auto)")
    ap.add_argument("--pretrained", type=str, default=None,
                    help="SAM-Road ckpt path (default: checkpoints/cityscale_vitb_512_e10.ckpt)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if args.region_dir:
        regions = [os.path.abspath(args.region_dir)]
    else:
        data_root = os.path.abspath(args.data_root)
        regions = []
        for city in sorted(os.listdir(data_root)):
            cp = os.path.join(data_root, city)
            if not os.path.isdir(cp):
                continue
            for reg in sorted(os.listdir(cp)):
                rp = os.path.join(cp, reg)
                if not os.path.isdir(rp):
                    continue
                feats = __import__("glob").glob(os.path.join(rp, "samroad_feat_full_*.npy"))
                tifs = __import__("glob").glob(os.path.join(rp, "crop_*.tif"))
                if feats and tifs:
                    regions.append(rp)
                    if len(regions) >= args.n_regions:
                        break
            if len(regions) >= args.n_regions:
                break

    if not regions:
        print("No regions with both TIF and cached feat found.")
        return 1

    print(f"Testing {len(regions)} region(s)\n")
    passed = 0
    for i, rp in enumerate(regions):
        print(f"--- Region {i+1}/{len(regions)}: {rp} ---")
        try:
            if run_verification(rp, device, args.y0, args.x0, args.pretrained):
                passed += 1
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
        print()
    print(f"Passed {passed}/{len(regions)} regions.")
    return 0 if passed == len(regions) else 1


if __name__ == "__main__":
    sys.exit(main())
