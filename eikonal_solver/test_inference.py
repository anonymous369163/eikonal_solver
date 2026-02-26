"""
SAM-Road inference on a single GeoTIFF.

Modes
-----
center  : crop the central 512×512 patch and run a single forward pass.
          Produces a 3-panel figure: satellite | probability mask | Eikonal field.

sliding : slide a 512×512 window across the full image with configurable stride
          (default 256 px = 50% overlap).  Overlapping predictions are blended
          with a Hanning (cosine) weight window so patch-boundary seams vanish.
          Produces a 2-panel figure: full satellite | full probability map.

Options
-------
--smooth_sigma  apply Gaussian post-processing to the probability map before
                visualisation (sigma=0 = off).  Useful to further reduce the
                16-px ViT token grid without retraining.
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tifffile

from model import SAMRoute

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
# Path to the raw SAM ViT-B image-encoder weights.
# SAMRoute loads this INTERNALLY in __init__ — do NOT pass it as --ckpt.
_SAM_ENCODER_CKPT = os.path.normpath(
    os.path.join(_PROJECT_ROOT, "sam_road_repo", "sam_ckpts", "sam_vit_b_01ec64.pth")
)
# Keep the old alias so InferenceConfig.SAM_CKPT_PATH still works.
_DEFAULT_SAM_CKPT = _SAM_ENCODER_CKPT

# Path to the COMPLETE SAM-Road checkpoint (encoder + map_decoder).
# This is the model to pass as --ckpt for inference.
_SAMROAD_CKPT_DEFAULT = "checkpoints/cityscale_vitb_512_e10.ckpt"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class InferenceConfig:
    def __init__(self):
        self.SAM_VERSION = 'vit_b'
        self.PATCH_SIZE = 512
        self.NO_SAM = False
        self.USE_SAM_DECODER = False
        self.USE_SMOOTH_DECODER = False   # auto-detected from checkpoint shape in load_model()
        self.ENCODER_LORA = False
        self.FREEZE_ENCODER = True
        self.FOCAL_LOSS = False
        self.TOPONET_VERSION = 'default'
        self.SAM_CKPT_PATH = _DEFAULT_SAM_CKPT
        self.ROUTE_COST_MODE = 'add'
        self.ROUTE_ADD_ALPHA = 20.0
        self.ROUTE_ADD_GAMMA = 2.0
        self.ROUTE_ADD_BLOCK_ALPHA = 0.0   # disable hard-block; let wavefront cross gaps
        self.ROUTE_BLOCK_TH = 0.0
        self.ROUTE_EIK_DOWNSAMPLE = 4
        # use_roi=False solves on 512×512; diagonal ≈ 724 px → 512 iters ensures convergence
        self.ROUTE_EIK_ITERS = 512
        self.ROUTE_CKPT_CHUNK = 10
        self.ROUTE_ROI_MARGIN = 64
        self.LORA_RANK = 4

    def get(self, key, default):
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _detect_smooth_decoder(state_dict: dict) -> bool:
    """Return True if the checkpoint was saved with the smooth decoder variant.

    The standard decoder ends with ConvTranspose2d(32→2), so
        map_decoder.7.weight has shape [2, 32, 2, 2]  (out_ch first for ConvTranspose2d).
    The smooth decoder ends with Conv2d(32→2), so
        map_decoder.9.weight has shape [2, 32, 3, 3].
    We detect by checking whether map_decoder.7.weight has 32 output channels
    (i.e., the intermediate ConvTranspose2d(32→32) is present).
    """
    w = state_dict.get("map_decoder.7.weight")
    if w is None:
        return False
    # ConvTranspose2d stores weights as [in_ch, out_ch/groups, kH, kW]
    # Standard: map_decoder.7 = ConvTranspose2d(32→2)  → shape [32, 2, 2, 2]
    # Smooth:   map_decoder.7 = ConvTranspose2d(32→32) → shape [32, 32, 2, 2]
    return w.shape[1] == 32  # out_ch == 32 means it's the smooth variant


def _detect_patch_size(state_dict: dict) -> "int | None":
    """Infer the training patch size from the SAM image encoder's pos_embed.

    SAM ViT encodes images with a stride-16 patch embed, so:
        pos_embed shape = [1, H_tokens, W_tokens, D]
        patch_size = H_tokens * 16

    Returns None if the key is absent (e.g. pure-decoder checkpoint).
    """
    pe = state_dict.get("image_encoder.pos_embed")
    if pe is None:
        return None
    # pe shape: [1, H_tokens, W_tokens, embed_dim]
    return int(pe.shape[1]) * 16


def load_model(ckpt_path: str, config: InferenceConfig, device) -> "SAMRoute":
    """Build SAMRoute, auto-detect decoder variant and patch size, load checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = raw_ckpt.get("state_dict", raw_ckpt)
    clean_sd = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }

    # Guard: raw SAM encoder weights (sam_vit_b_01ec64.pth) contain only
    # image_encoder keys and NO map_decoder keys.  Passing them as the
    # SAM-Road checkpoint leaves the decoder randomly initialised, which
    # produces a uniform ~0.5 probability map (solid pink in magma colormap).
    # SAMRoute already loads these weights internally in __init__; the correct
    # ckpt_path here must be a complete SAM-Road checkpoint (encoder + decoder).
    has_decoder = any(k.startswith("map_decoder.") for k in clean_sd)
    if not has_decoder:
        raise ValueError(
            f"\n\n  ✗ '{os.path.basename(ckpt_path)}' contains only SAM image-encoder "
            f"weights (no map_decoder keys).\n"
            f"  This file is the raw SAM pretrained encoder, NOT a complete SAM-Road "
            f"checkpoint.\n"
            f"  SAMRoute already loads it internally — you do NOT need to pass it as "
            f"--ckpt.\n\n"
            f"  Pass a SAM-Road finetuned checkpoint instead, e.g.:\n"
            f"    --ckpt training_outputs/finetune_demo/checkpoints/"
            f"best_pos8.0_dice0.5_thin4.0.ckpt\n"
        )

    # Auto-detect decoder variant BEFORE constructing the model
    config.USE_SMOOTH_DECODER = _detect_smooth_decoder(clean_sd)
    variant = "smooth (w/ anti-mosaic Conv2d)" if config.USE_SMOOTH_DECODER else "standard"
    print(f"  Decoder variant detected: {variant}")

    # Auto-detect patch size from pos_embed to avoid size-mismatch crashes when
    # loading a checkpoint trained at a different resolution (e.g. 1024 vs 512).
    detected_ps = _detect_patch_size(clean_sd)
    if detected_ps is not None and detected_ps != config.PATCH_SIZE:
        print(f"  Patch size auto-corrected: {config.PATCH_SIZE} → {detected_ps} "
              f"(inferred from image_encoder.pos_embed)")
        config.PATCH_SIZE = detected_ps

    model = SAMRoute(config).to(device)
    model.load_state_dict(clean_sd, strict=False)
    print(f"✓ Loaded weights: {os.path.basename(ckpt_path)}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image loading utilities
# ---------------------------------------------------------------------------

def load_tif(tif_path: str) -> np.ndarray:
    """Load a GeoTIFF and return uint8 RGB array [H, W, 3]."""
    img = tifffile.imread(tif_path)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[0] >= 3:
        img = img[:3, :, :].transpose(1, 2, 0)
    if img.dtype == np.uint16:
        img = (img / 256.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# A1 — Gaussian post-processing
# ---------------------------------------------------------------------------

def smooth_prob(road_prob: np.ndarray, sigma: float) -> np.ndarray:
    """Apply optional Gaussian smoothing to the probability map.

    sigma=0  → no-op (return original array).
    sigma>0  → scipy Gaussian filter (preserves [0,1] range via clip).
    """
    if sigma <= 0.0:
        return road_prob
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(road_prob.astype(np.float32), sigma=sigma).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# A2 — Sliding-window inference
# ---------------------------------------------------------------------------

def sliding_window_inference(
    img_array: np.ndarray,
    model: "SAMRoute",
    device,
    patch_size: int = 512,
    stride: int = 256,
    smooth_sigma: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """Run inference over the full image using a sliding window.

    Overlapping patch predictions are blended with a 2-D Hanning (cosine)
    weight window, which tapers smoothly to zero at the edges.  This
    eliminates visible seams at patch boundaries.

    Args:
        img_array   : uint8 RGB [H, W, 3]
        model       : loaded SAMRoute in eval mode
        device      : torch device
        patch_size  : window side length (must match training patch size)
        stride      : step between windows (stride < patch_size → overlap)
        smooth_sigma: per-patch Gaussian smoothing applied before accumulation
        verbose     : print progress

    Returns:
        full-image probability map [H, W] float32 in [0, 1]
    """
    H, W = img_array.shape[:2]

    # 2-D Hanning window as blend weight (tapers smoothly toward edges).
    # np.hanning() returns exactly 0 at both endpoints, so image border pixels
    # that are only covered by one patch would get weight_sum==0 and stay 0
    # (black border / missed roads).  Clamp endpoints to a small epsilon so
    # every pixel receives a nonzero weight regardless of its position.
    win1d = np.hanning(patch_size).astype(np.float32)
    win1d = np.maximum(win1d, 1e-3)          # prevent zero-weight at endpoints
    win2d = np.outer(win1d, win1d)

    prob_sum   = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    # Build grid of top-left corners, ensuring the last column/row always
    # falls inside the image (pad with edge values if necessary).
    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    # Make sure the bottom / right edge is always covered
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    total = len(ys) * len(xs)
    done  = 0

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + patch_size, H)
            x1 = min(x0 + patch_size, W)
            patch = img_array[y0:y1, x0:x1]

            # Zero-pad if the image boundary is smaller than patch_size
            ph, pw = patch.shape[:2]
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            rgb_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, ms = model._predict_mask_logits_scores(rgb_t)
            prob = ms[0, :, :, 1].cpu().numpy()  # [patch_size, patch_size]

            prob = smooth_prob(prob, smooth_sigma)

            # Accumulate into the full-image arrays (trim back to actual size)
            prob_sum[y0:y0 + ph, x0:x0 + pw]   += prob[:ph, :pw] * win2d[:ph, :pw]
            weight_sum[y0:y0 + ph, x0:x0 + pw] += win2d[:ph, :pw]

            done += 1
            if verbose and (done % 10 == 0 or done == total):
                print(f"  sliding window: {done}/{total} patches", end="\r")

    if verbose:
        print()

    valid  = weight_sum > 0
    result = np.zeros((H, W), dtype=np.float32)
    result[valid] = prob_sum[valid] / weight_sum[valid]
    return result


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------

def run_inference_on_tif(
    tif_path,
    ckpt_path=None,
    save_path=None,
    encoder_lora=False,
    lora_rank=4,
    mode="center",
    stride=256,
    smooth_sigma=0.0,
):
    ckpt_path = ckpt_path or _SAMROAD_CKPT_DEFAULT
    save_path = save_path or "tif_inference_result_fixed.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = InferenceConfig()
    if encoder_lora:
        config.ENCODER_LORA = True
        config.LORA_RANK = lora_rank
        config.FREEZE_ENCODER = False
        print(f"LoRA inference mode: rank={lora_rank}")

    # 1. Load model (auto-detects decoder variant from checkpoint)
    model = load_model(ckpt_path, config, device)

    # 2. Load image
    print(f"Reading image: {tif_path}")
    img_array = load_tif(tif_path)
    h, w = img_array.shape[:2]
    ps = config.PATCH_SIZE

    # ------------------------------------------------------------------
    # Mode: center  — single 512×512 centre crop + Eikonal field
    # ------------------------------------------------------------------
    if mode == "center":
        start_y = max(0, (h - ps) // 2)
        start_x = max(0, (w - ps) // 2)
        img_crop = img_array[start_y:start_y + ps, start_x:start_x + ps]

        if img_crop.shape[0] < ps or img_crop.shape[1] < ps:
            pad = np.zeros((ps, ps, 3), dtype=np.uint8)
            pad[:img_crop.shape[0], :img_crop.shape[1]] = img_crop
            img_crop = pad

        rgb_tensor = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).to(device)

        print("Generating probability mask (center crop)...")
        with torch.no_grad():
            _, mask_scores = model._predict_mask_logits_scores(rgb_tensor)
            road_prob_tensor = mask_scores[..., 1]          # [1, H, W]
            road_prob_raw    = road_prob_tensor[0].cpu().numpy()

        # A1: optional Gaussian post-processing
        road_prob = smooth_prob(road_prob_raw, smooth_sigma)
        if smooth_sigma > 0:
            print(f"  Applied Gaussian smoothing: sigma={smooth_sigma}")

        # Select src / tgt from predicted road pixels for Eikonal demo
        y_coords, x_coords = np.where(road_prob > 0.5)
        if len(y_coords) > 10:
            quarter = len(y_coords) // 4
            src_yx = torch.tensor(
                [[y_coords[quarter], x_coords[quarter]]], dtype=torch.long
            ).to(device)
            tgt_yx = torch.tensor(
                [[y_coords[-quarter - 1], x_coords[-quarter - 1]]], dtype=torch.long
            ).to(device)
        else:
            src_yx = torch.tensor([[64,  64 ]], dtype=torch.long).to(device)
            tgt_yx = torch.tensor([[448, 448]], dtype=torch.long).to(device)

        print(f"  src={src_yx.cpu().tolist()}  tgt={tgt_yx.cpu().tolist()}")

        print("Running Eikonal solver...")
        with torch.no_grad():
            from eikonal import make_source_mask
            from model import _eikonal_soft_sweeping_diff

            B, H_f, W_f = road_prob_tensor.shape
            cost     = model._road_prob_to_cost(road_prob_tensor).to(dtype=torch.float32)
            src_yx_l = src_yx.long().view(B, 1, 2)
            tgt_yx_l = tgt_yx.long().view(B, 2)
            src_mask = make_source_mask(H_f, W_f, src_yx_l)

            T = _eikonal_soft_sweeping_diff(
                cost, src_mask, model.route_eik_cfg, model.route_ckpt_chunk
            )
            yy   = tgt_yx_l[:, 0].clamp(0, H_f - 1)
            xx   = tgt_yx_l[:, 1].clamp(0, W_f - 1)
            dist = T[torch.arange(B, device=device), yy, xx]
            dist_field = T[0].cpu().numpy()

        # --- Visualisation (3-panel) ---
        print("Generating figure...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_crop)
        axes[0].set_title("Cropped Satellite Image (Original Res)")
        axes[0].plot(src_yx[0, 1].cpu(), src_yx[0, 0].cpu(), "g*", markersize=15, label="Start")
        axes[0].plot(tgt_yx[0, 1].cpu(), tgt_yx[0, 0].cpu(), "r*", markersize=15, label="End")
        axes[0].legend()
        axes[0].axis("off")

        sigma_str = f" (σ={smooth_sigma})" if smooth_sigma > 0 else ""
        im1 = axes[1].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title(f"Predicted Probability Mask{sigma_str}")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].axis("off")

        valid_dist = dist_field[dist_field < 1e5]
        vmax = np.percentile(valid_dist, 95) if len(valid_dist) > 0 else 1000
        im2 = axes[2].imshow(dist_field, cmap="viridis", vmax=vmax)
        axes[2].set_title(f"Eikonal Cost Field\n(Target Cost: {dist[0].item():.2f})")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis("off")

    # ------------------------------------------------------------------
    # Mode: sliding — full-image sliding window
    # ------------------------------------------------------------------
    elif mode == "sliding":
        print(f"Running sliding-window inference (stride={stride}, sigma={smooth_sigma})...")
        road_prob = sliding_window_inference(
            img_array, model, device,
            patch_size=ps,
            stride=stride,
            smooth_sigma=smooth_sigma,
            verbose=True,
        )

        # --- Visualisation (2-panel) ---
        print("Generating figure...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].imshow(img_array)
        axes[0].set_title("Full Satellite Image")
        axes[0].axis("off")

        sigma_str = f" (σ={smooth_sigma})" if smooth_sigma > 0 else ""
        im1 = axes[1].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title(f"Predicted Road Probability (Sliding Window){sigma_str}")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].axis("off")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'center' or 'sliding'.")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {os.path.abspath(save_path)}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SAM-Road inference on TIF image")
    p.add_argument(
        "--ckpt", type=str, default=_SAMROAD_CKPT_DEFAULT,
        help="SAM-Road checkpoint path (encoder + decoder). "
             "Default: cityscale_vitb_512_e10.ckpt pretrained model. "
             "Do NOT pass sam_vit_b_01ec64.pth here — that is the raw SAM encoder "
             "and is loaded internally by SAMRoute automatically.",
    )
    p.add_argument(
        "--output", type=str, default="tif_inference_result_fixed.png",
        help="Output figure path",
    )
    p.add_argument(
        "--tif", type=str,
        default="Gen_dataset_V2/Gen_dataset/19.940688_110.276704/00_20.021516_110.190699_3000.0/crop_20.021516_110.190699_3000.0_z16.tif",
        help="Target TIF path",
    )
    p.add_argument(
        "--mode", type=str, default="sliding", choices=["center", "sliding"],
        help="'center': single 512x512 crop + Eikonal. 'sliding': full-image sliding window.",
    )
    p.add_argument(
        "--stride", type=int, default=256,
        help="Sliding-window stride in pixels (default 256 = 50%% overlap)",
    )
    p.add_argument(
        "--smooth_sigma", type=float, default=0.0,
        help="Gaussian post-processing sigma (0 = off). Reduces 16-px ViT token grid.",
    )
    p.add_argument("--encoder_lora", action="store_true", help="Enable LoRA encoder weights")
    p.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (must match training)")
    args = p.parse_args()

    run_inference_on_tif(
        args.tif,
        ckpt_path=args.ckpt,
        save_path=args.output,
        encoder_lora=args.encoder_lora,
        lora_rank=args.lora_rank,
        mode=args.mode,
        stride=args.stride,
        smooth_sigma=args.smooth_sigma,
    )
