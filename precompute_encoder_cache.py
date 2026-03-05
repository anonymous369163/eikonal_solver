#!/usr/bin/env python3
"""
Pre-compute SAM encoder image embeddings for all satellite images.

Each satellite TIF in Gen_dataset_V2/Gen_dataset/*/*/crop_*.tif gets a
corresponding encoder_cache.pt saved in the same directory.

Unlike road_prob_cache.npz (which caches the decoder output), this caches
the *encoder* output so that the map_decoder can remain trainable during
downstream TSP training — adding ~177K trainable parameters.

Saved fields per file (torch.save dict):
  - features:    float16 tensor [N_patches, 256, 32, 32]
  - patches_info: list of (y0, x0, ph, pw) tuples
  - win2d:       float32 numpy array [patch_size, patch_size]
  - H, W:        original image dimensions
  - patch_size:  encoder input size (512)
  - stride:      sliding-window stride (256)
  - ckpt_path:   checkpoint used for inference
  - ckpt_name:   checkpoint basename
  - timestamp:   ISO 8601 creation time

Usage:
    python precompute_encoder_cache.py [--device cuda] [--ckpt PATH] [--force]
"""
import os, sys, time, argparse, glob, datetime
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

EIKONAL_DIR = os.path.join(SCRIPT_DIR, 'eikonal_solver')
if EIKONAL_DIR not in sys.path:
    sys.path.insert(0, EIKONAL_DIR)


def load_samroute_encoder(ckpt_path, device):
    """Load SAMRoute model for encoder-only inference."""
    from model_multigrid import SAMRoute
    from gradcheck_route_loss_v2_multigrid_fullmap import _load_lightning_ckpt
    from types import SimpleNamespace

    sd = _load_lightning_ckpt(ckpt_path)
    has_lora = any("attn.qkv.linear_a_q" in k for k in sd)
    pe = sd.get('image_encoder.pos_embed')
    patch_size = int(pe.shape[1]) * 16 if pe is not None else 512
    w7 = sd.get('map_decoder.7.weight')
    use_smooth = w7 is not None and w7.shape[1] == 32

    cfg = SimpleNamespace(
        SAM_VERSION='vit_b', PATCH_SIZE=patch_size, NO_SAM=False,
        USE_SAM_DECODER=False, USE_SMOOTH_DECODER=use_smooth,
        ENCODER_LORA=has_lora, LORA_RANK=4, FREEZE_ENCODER=True,
        FOCAL_LOSS=False, TOPONET_VERSION='default',
        SAM_CKPT_PATH=os.path.join(SCRIPT_DIR, 'sam_road_repo', 'sam_ckpts', 'sam_vit_b_01ec64.pth'),
        ROUTE_COST_MODE='add', ROUTE_ADD_ALPHA=20.0, ROUTE_ADD_GAMMA=2.0,
        ROUTE_ADD_BLOCK_ALPHA=0.0, ROUTE_BLOCK_TH=0.0, ROUTE_ROI_MARGIN=64,
        ROUTE_COST_NET=False, ROUTE_COST_NET_CH=8, ROUTE_COST_NET_USE_COORD=False,
    )
    if has_lora:
        cfg.FREEZE_ENCODER = False
        for k, v in sd.items():
            if "linear_a_q.weight" in k:
                cfg.LORA_RANK = v.shape[0]
                break

    model = SAMRoute(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.requires_grad_(False)
    return model, patch_size


def compute_encoder_features(img_np, model, device, patch_size=512, stride=256):
    """Run SAM encoder on all sliding-window patches, return features on CPU."""
    H, W = img_np.shape[:2]
    win1d = np.hanning(patch_size).astype(np.float32)
    win1d = np.maximum(win1d, 1e-3)
    win2d = np.outer(win1d, win1d)

    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    patches_info = []
    features_list = []

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + patch_size, H)
                x1 = min(x0 + patch_size, W)
                patch = img_np[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                rgb_t = torch.tensor(patch, dtype=torch.float32, device=device).unsqueeze(0)
                x_enc = rgb_t.permute(0, 3, 1, 2)
                x_enc = (x_enc - model.pixel_mean) / model.pixel_std
                feat = model.image_encoder(x_enc).cpu().half()  # [1,256,32,32] fp16

                patches_info.append((y0, x0, ph, pw))
                features_list.append(feat.squeeze(0))  # [256,32,32]

    features = torch.stack(features_list, dim=0)  # [N, 256, 32, 32]
    return patches_info, features, win2d, H, W


def main():
    parser = argparse.ArgumentParser(description='Pre-compute SAM encoder embeddings')
    parser.add_argument('--device', default='cuda', help='torch device')
    parser.add_argument('--ckpt', default=os.path.join(
        SCRIPT_DIR, 'training_outputs', 'fulldataset_seg_dist_lora_v2',
        'checkpoints', 'best_seg_dist_lora_v2.ckpt'))
    parser.add_argument('--force', action='store_true', help='overwrite existing caches')
    parser.add_argument('--dataset-root', default=os.path.join(
        SCRIPT_DIR, 'Gen_dataset_V2', 'Gen_dataset'))
    parser.add_argument('--limit', type=int, default=0,
                        help='process only first N images (0=all, for testing)')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(0)

    tif_files = sorted(glob.glob(os.path.join(args.dataset_root, '*', '*', 'crop_*.tif')))
    print(f"Found {len(tif_files)} satellite TIF files")

    if not tif_files:
        print("No TIF files found, exiting.")
        return

    existing = 0
    to_process = []
    for tif in tif_files:
        cache_path = os.path.join(os.path.dirname(tif), 'encoder_cache.pt')
        if os.path.exists(cache_path) and not args.force:
            existing += 1
        else:
            to_process.append(tif)

    if args.limit > 0:
        to_process = to_process[:args.limit]

    print(f"  Already cached: {existing}")
    print(f"  To process: {len(to_process)}")

    if not to_process:
        print("All encoder caches already exist. Use --force to recompute.")
        return

    ckpt_abs = os.path.abspath(args.ckpt)
    ckpt_basename = os.path.basename(ckpt_abs)
    print(f"\nLoading SAMRoute from {ckpt_abs}...")
    model, patch_size = load_samroute_encoder(ckpt_abs, device)
    print(f"SAMRoute loaded (patch_size={patch_size})")

    from PIL import Image

    stride = 256
    total_time = 0.0
    for idx, tif_path in enumerate(to_process):
        cache_path = os.path.join(os.path.dirname(tif_path), 'encoder_cache.pt')
        rel_path = os.path.relpath(tif_path, args.dataset_root)

        t0 = time.time()
        img = Image.open(tif_path).convert('RGB')
        img_np = np.asarray(img, dtype=np.uint8)

        patches_info, features, win2d, H, W = compute_encoder_features(
            img_np, model, device, patch_size=patch_size, stride=stride)

        torch.save({
            'features': features,           # [N, 256, 32, 32] float16
            'patches_info': patches_info,   # list of (y0, x0, ph, pw)
            'win2d': win2d,                 # [patch_size, patch_size] float32
            'H': H,
            'W': W,
            'patch_size': patch_size,
            'stride': stride,
            'ckpt_path': ckpt_abs,
            'ckpt_name': ckpt_basename,
            'timestamp': datetime.datetime.now().isoformat(),
        }, cache_path)

        elapsed = time.time() - t0
        total_time += elapsed
        file_size = os.path.getsize(cache_path) / (1024 * 1024)
        eta = (total_time / (idx + 1)) * (len(to_process) - idx - 1)
        print(f"  [{idx+1}/{len(to_process)}] {rel_path}: "
              f"{H}x{W}, {features.shape[0]} patches -> {file_size:.1f}MB, "
              f"{elapsed:.1f}s (ETA: {eta/60:.0f}min)")

    print(f"\nDone! Processed {len(to_process)} images in {total_time:.0f}s "
          f"({total_time/60:.1f}min)")
    print(f"Total cached: {existing + len(to_process)}")
    print(f"Checkpoint used: {ckpt_abs}")


if __name__ == '__main__':
    main()
