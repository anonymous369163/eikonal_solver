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
import torch.nn.functional as F  # NEW: for ROI pooling/padding
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tifffile

from model_multigrid import SAMRoute

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
# Default: fine-tuned LoRA model from finetune_demo.
# _SAMROAD_CKPT_DEFAULT = "checkpoints/cityscale_vitb_512_e10.ckpt"
_SAMROAD_CKPT_DEFAULT = "training_outputs/finetune_demo/checkpoints/best_lora_pos8.0_dice0.5_thin4.0.ckpt"


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
    # Auto-detect LoRA: if checkpoint has LoRA adapter weights (linear_a_q, linear_b_q, etc.)
    # in image_encoder.blocks.*.attn.qkv, we must enable ENCODER_LORA for correct loading.
    has_lora = any("attn.qkv.linear_a_q" in k or "attn.qkv.linear_a_v" in k for k in clean_sd)
    if has_lora:
        config.ENCODER_LORA = True
        config.FREEZE_ENCODER = False
        print(f"  LoRA detected in checkpoint → ENCODER_LORA=True")

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
# NEW — interactive point picking + route visualisation (Eikonal shortest path)
# ---------------------------------------------------------------------------

def _parse_yx(s: str) -> tuple[int, int]:
    """Parse 'y,x' (or 'y x') into integer (y, x)."""
    import re as _re
    parts = [p for p in _re.split(r"[,\s]+", s.strip()) if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate '{s}'. Use 'y,x' (e.g. '120,450').")
    y, x = int(float(parts[0])), int(float(parts[1]))
    return y, x


def _pick_two_points(
    img: np.ndarray,
    road_prob: "np.ndarray | None" = None,
    title: str = "Click START then END (left mouse). Press Enter when done.",
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Interactive picking (matplotlib ginput). Returns (src_yx, tgt_yx)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    if road_prob is not None:
        ax.imshow(road_prob, cmap="magma", vmin=0, vmax=1, alpha=0.35)
    ax.set_title(title)
    ax.axis("off")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise RuntimeError("Point picking cancelled or insufficient points. Need exactly 2 clicks.")
    # ginput returns (x, y) in image coordinates.
    src = (int(round(pts[0][1])), int(round(pts[0][0])))
    tgt = (int(round(pts[1][1])), int(round(pts[1][0])))
    H, W = img.shape[:2]
    src = (max(0, min(H - 1, src[0])), max(0, min(W - 1, src[1])))
    tgt = (max(0, min(H - 1, tgt[0])), max(0, min(W - 1, tgt[1])))
    return src, tgt


def _snap_to_road(
    road_prob: np.ndarray,
    yx: tuple[int, int],
    *,
    radius: int = 0,
    th: float = 0.5,
) -> tuple[int, int]:
    """Snap a clicked point to the nearest high-probability road pixel in a local window."""
    if radius <= 0:
        return yx
    H, W = road_prob.shape[:2]
    y, x = yx
    y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
    x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
    win = road_prob[y0:y1, x0:x1]
    if win.size == 0:
        return yx
    idx = int(np.argmax(win))
    wy, wx = np.unravel_index(idx, win.shape)
    if float(win[wy, wx]) < float(th):
        return yx
    return (y0 + int(wy), x0 + int(wx))


def _trace_path_descend_T(
    T: np.ndarray,
    src_yx: tuple[int, int],
    tgt_yx: tuple[int, int],
    *,
    connectivity8: bool = True,
    max_steps: "int | None" = None,
) -> np.ndarray:
    """Backtrack a shortest path by descending the Eikonal distance field T.

    Note: This is a discrete approximation of gradient-descent on T.
    """
    H, W = T.shape
    sy, sx = src_yx
    ty, tx = tgt_yx
    max_steps = max_steps or int(H * W)

    if connectivity8:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    y, x = ty, tx
    path = [(y, x)]
    eps = 1e-6

    for _ in range(int(max_steps)):
        if (y == sy) and (x == sx):
            break
        cur = float(T[y, x])
        best = None  # (val, ny, nx)
        # Prefer strictly decreasing steps; fall back to smallest neighbour if stuck.
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            val = float(T[ny, nx])
            if val < cur - eps:
                if (best is None) or (val < best[0]):
                    best = (val, ny, nx)
        if best is None:
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                val = float(T[ny, nx])
                if (best is None) or (val < best[0]):
                    best = (val, ny, nx)
        if best is None:
            break
        y, x = best[1], best[2]
        path.append((y, x))

    path = path[::-1]  # src -> tgt
    return np.asarray(path, dtype=np.int32)


def _roi_extract_square(
    prob_map: np.ndarray,
    src_yx: tuple[int, int],
    tgt_yx: tuple[int, int],
    *,
    margin: int = 64,
    min_size: int = 64,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Extract a square ROI patch around src large enough to cover tgt (with padding)."""
    H, W = prob_map.shape[:2]
    sy, sx = src_yx
    ty, tx = tgt_yx
    span = int(max(abs(ty - sy), abs(tx - sx)))
    half = span + int(margin)
    P = max(2 * half + 1, int(min_size))

    y0 = sy - half; x0 = sx - half
    y1 = y0 + P;    x1 = x0 + P

    yy0 = max(y0, 0); xx0 = max(x0, 0)
    yy1 = min(y1, H); xx1 = min(x1, W)

    patch = prob_map[yy0:yy1, xx0:xx1]
    pad_t = yy0 - y0
    pad_l = xx0 - x0
    pad_b = y1 - yy1
    pad_r = x1 - xx1
    if pad_t or pad_l or pad_b or pad_r:
        patch = np.pad(
            patch, ((pad_t, pad_b), (pad_l, pad_r)),
            mode="constant", constant_values=0.0
        )

    src_rel = (int(np.clip(sy - y0, 0, P - 1)), int(np.clip(sx - x0, 0, P - 1)))
    tgt_rel = (int(np.clip(ty - y0, 0, P - 1)), int(np.clip(tx - x0, 0, P - 1)))
    return patch.astype(np.float32), (y0, x0), src_rel, tgt_rel


def _compute_route_roi(
    prob_map: np.ndarray,
    model: "SAMRoute",
    device,
    *,
    src_yx: tuple[int, int],
    tgt_yx: tuple[int, int],
    margin: int,
    downsample: int,
    n_iters: int,
    snap_radius: int = 0,
    snap_th: float = 0.5,
    max_path_steps: "int | None" = None,
) -> dict:
    """Compute ROI Eikonal field + path on a probability map (prob_map in [0,1])."""
    from dataclasses import replace
    from eikonal import eikonal_soft_sweeping

    src0 = _snap_to_road(prob_map, src_yx, radius=snap_radius, th=snap_th)
    tgt0 = _snap_to_road(prob_map, tgt_yx, radius=snap_radius, th=snap_th)

    patch, (y0, x0), src_rel, tgt_rel = _roi_extract_square(
        prob_map, src0, tgt0, margin=margin, min_size=64
    )

    ds = max(1, int(downsample))

    with torch.no_grad():
        patch_t = torch.from_numpy(patch).to(device=device, dtype=torch.float32)

        if ds > 1:
            P = int(patch_t.shape[0])
            P_pad = int(np.ceil(P / ds) * ds)
            if P_pad > P:
                patch_t = F.pad(patch_t, (0, P_pad - P, 0, P_pad - P), value=0.0)

            patch_c = F.max_pool2d(
                patch_t.unsqueeze(0).unsqueeze(0), kernel_size=ds, stride=ds
            ).squeeze(0).squeeze(0)  # [P_c, P_c]

            cost = model._road_prob_to_cost(patch_c.unsqueeze(0)).squeeze(0).to(dtype=torch.float32)
            P_c = int(cost.shape[0])

            src_c = (min(src_rel[0] // ds, P_c - 1), min(src_rel[1] // ds, P_c - 1))
            tgt_c = (min(tgt_rel[0] // ds, P_c - 1), min(tgt_rel[1] // ds, P_c - 1))

            src_mask = torch.zeros((1, P_c, P_c), dtype=torch.bool, device=device)
            src_mask[0, src_c[0], src_c[1]] = True

            # Inference: hard_eval is more faithful than STE.
            cfg = replace(model.route_eik_cfg, n_iters=int(n_iters), h=float(ds), mode="hard_eval")
            T = eikonal_soft_sweeping(cost.unsqueeze(0), src_mask, cfg)  # [1, P_c, P_c]
            T_np = T[0].detach().cpu().numpy()

            dist = float(T[0, tgt_c[0], tgt_c[1]].item())
            path_c = _trace_path_descend_T(T_np, src_c, tgt_c, max_steps=max_path_steps)

            path_full = np.zeros_like(path_c)
            path_full[:, 0] = (path_c[:, 0] * ds + y0).clip(0, prob_map.shape[0] - 1)
            path_full[:, 1] = (path_c[:, 1] * ds + x0).clip(0, prob_map.shape[1] - 1)

            return {
                "src_yx": src0, "tgt_yx": tgt0,
                "dist": dist,
                "T_roi": T_np,
                "path_yx": path_full.astype(np.int32),
                "roi_origin_yx": (y0, x0),
                "ds": ds,
                "roi_size": int(T_np.shape[0]),
            }

        # ds == 1
        cost = model._road_prob_to_cost(patch_t.unsqueeze(0)).squeeze(0).to(dtype=torch.float32)
        P = int(cost.shape[0])

        src_mask = torch.zeros((1, P, P), dtype=torch.bool, device=device)
        src_mask[0, src_rel[0], src_rel[1]] = True

        cfg = replace(model.route_eik_cfg, n_iters=int(n_iters), h=1.0, mode="hard_eval")
        T = eikonal_soft_sweeping(cost.unsqueeze(0), src_mask, cfg)  # [1, P, P]
        T_np = T[0].detach().cpu().numpy()

        dist = float(T[0, tgt_rel[0], tgt_rel[1]].item())
        path_p = _trace_path_descend_T(T_np, src_rel, tgt_rel, max_steps=max_path_steps)

        path_full = np.zeros_like(path_p)
        path_full[:, 0] = (path_p[:, 0] + y0).clip(0, prob_map.shape[0] - 1)
        path_full[:, 1] = (path_p[:, 1] + x0).clip(0, prob_map.shape[1] - 1)

        return {
            "src_yx": src0, "tgt_yx": tgt0,
            "dist": dist,
            "T_roi": T_np,
            "path_yx": path_full.astype(np.int32),
            "roi_origin_yx": (y0, x0),
            "ds": 1,
            "roi_size": int(T_np.shape[0]),
        }


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
    # NEW: routing / path visualisation
    route=False,
    pick_points=False,
    src=None,
    tgt=None,
    roi_margin=None,
    eik_downsample=None,
    eik_iters=None,
    snap_radius=25,
    snap_th=0.5,
    max_path_steps=None,
):
    ckpt_path = ckpt_path or _SAMROAD_CKPT_DEFAULT
    save_path = save_path or "tif_inference_route_result.png"

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

        # NEW: choose src / tgt (crop coordinates)
        if src is not None and tgt is not None:
            src_yx0 = _parse_yx(src) if isinstance(src, str) else tuple(src)
            tgt_yx0 = _parse_yx(tgt) if isinstance(tgt, str) else tuple(tgt)
        elif pick_points:
            src_yx0, tgt_yx0 = _pick_two_points(
                img_crop, road_prob,
                title="CENTER mode — click START then END (coordinates are within this 512×512 crop)"
            )
        else:
            # fallback: pick two points from predicted road pixels
            y_coords, x_coords = np.where(road_prob > 0.5)
            if len(y_coords) > 10:
                quarter = len(y_coords) // 4
                src_yx0 = (int(y_coords[quarter]), int(x_coords[quarter]))
                tgt_yx0 = (int(y_coords[-quarter - 1]), int(x_coords[-quarter - 1]))
            else:
                src_yx0 = (64, 64)
                tgt_yx0 = (448, 448)

        # NEW: optional snapping to nearest road-like pixel (helps with imprecise clicks)
        src_yx0 = _snap_to_road(road_prob, src_yx0, radius=int(snap_radius), th=float(snap_th))
        tgt_yx0 = _snap_to_road(road_prob, tgt_yx0, radius=int(snap_radius), th=float(snap_th))

        src_yx = torch.tensor([list(src_yx0)], dtype=torch.long, device=device)
        tgt_yx = torch.tensor([list(tgt_yx0)], dtype=torch.long, device=device)
        print(f"  src={src_yx.cpu().tolist()}  tgt={tgt_yx.cpu().tolist()}")

        print("Running Eikonal solver...")
        with torch.no_grad():
            from dataclasses import replace
            from eikonal import make_source_mask, eikonal_soft_sweeping

            B, H_f, W_f = road_prob_tensor.shape
            cost = model._road_prob_to_cost(road_prob_tensor).to(dtype=torch.float32)
            src_mask = make_source_mask(H_f, W_f, src_yx.view(B, 1, 2))

            iters = int(eik_iters) if eik_iters is not None else int(model.route_eik_cfg.n_iters)
            cfg = replace(model.route_eik_cfg, n_iters=iters, h=1.0, mode="hard_eval")

            T = eikonal_soft_sweeping(cost, src_mask, cfg)  # [B, H, W]
            yy = tgt_yx[:, 0].clamp(0, H_f - 1)
            xx = tgt_yx[:, 1].clamp(0, W_f - 1)
            dist = T[torch.arange(B, device=device), yy, xx]
            dist_field = T[0].detach().cpu().numpy()

        # NEW: backtrack a discrete path by descending T (for visualisation)
        path_yx = None
        if route:
            path_yx = _trace_path_descend_T(
                dist_field, src_yx0, tgt_yx0, max_steps=max_path_steps
            )

        # --- Visualisation (3-panel) ---
        print("Generating figure...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_crop)
        axes[0].set_title("Cropped Satellite Image (Original Res)")
        axes[0].plot(src_yx[0, 1].cpu(), src_yx[0, 0].cpu(), "g*", markersize=15, label="Start")
        axes[0].plot(tgt_yx[0, 1].cpu(), tgt_yx[0, 0].cpu(), "r*", markersize=15, label="End")
        if path_yx is not None:
            axes[0].plot(path_yx[:, 1], path_yx[:, 0], "-", linewidth=2.0, label="Route")
        axes[0].legend()
        axes[0].axis("off")

        sigma_str = f" (σ={smooth_sigma})" if smooth_sigma > 0 else ""
        im1 = axes[1].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title(f"Predicted Probability Mask{sigma_str}")
        if path_yx is not None:
            axes[1].plot(path_yx[:, 1], path_yx[:, 0], "-", linewidth=2.0)

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

        # NEW: optional route planning (pick 2 points → Eikonal T → backtrack path)
        if route:
            if src is not None and tgt is not None:
                src_yx0 = _parse_yx(src) if isinstance(src, str) else tuple(src)
                tgt_yx0 = _parse_yx(tgt) if isinstance(tgt, str) else tuple(tgt)
            elif pick_points:
                src_yx0, tgt_yx0 = _pick_two_points(
                    img_array, road_prob,
                    title="SLIDING mode — click START then END (full-image coordinates)"
                )
            else:
                # fallback: pick two points from predicted road pixels
                y_coords, x_coords = np.where(road_prob > 0.5)
                if len(y_coords) > 10:
                    quarter = len(y_coords) // 4
                    src_yx0 = (int(y_coords[quarter]), int(x_coords[quarter]))
                    tgt_yx0 = (int(y_coords[-quarter - 1]), int(x_coords[-quarter - 1]))
                else:
                    src_yx0 = (h // 4, w // 4)
                    tgt_yx0 = (3 * h // 4, 3 * w // 4)

            margin = int(roi_margin) if roi_margin is not None else int(getattr(model, "route_roi_margin", 64))
            ds = int(eik_downsample) if eik_downsample is not None else int(getattr(model, "route_eik_downsample", 4))
            iters = int(eik_iters) if eik_iters is not None else int(getattr(model.route_eik_cfg, "n_iters", 200))

            route_out = _compute_route_roi(
                road_prob, model, device,
                src_yx=src_yx0, tgt_yx=tgt_yx0,
                margin=margin,
                downsample=ds,
                n_iters=iters,
                snap_radius=int(snap_radius),
                snap_th=float(snap_th),
                max_path_steps=max_path_steps,
            )
            dist = route_out["dist"]
            path_yx = route_out["path_yx"]
            src_use = route_out["src_yx"]
            tgt_use = route_out["tgt_yx"]
            T_roi = route_out["T_roi"]

            print(f"Route distance (T at target): {dist:.2f}  (ds={route_out['ds']}, ROI={route_out['roi_size']}x{route_out['roi_size']})")

            print("Generating figure...")
            fig, axes = plt.subplots(1, 3, figsize=(22, 8))

            # Panel 1: satellite + route
            axes[0].imshow(img_array)
            axes[0].plot(src_use[1], src_use[0], "g*", markersize=14, label="Start")
            axes[0].plot(tgt_use[1], tgt_use[0], "r*", markersize=14, label="End")
            axes[0].plot(path_yx[:, 1], path_yx[:, 0], "-", linewidth=2.0, label="Route")
            axes[0].set_title("Full Satellite + Planned Route")
            axes[0].legend()
            axes[0].axis("off")

            # Panel 2: probability + route
            sigma_str = f" (σ={smooth_sigma})" if smooth_sigma > 0 else ""
            im1 = axes[1].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
            axes[1].plot(path_yx[:, 1], path_yx[:, 0], "-", linewidth=2.0)
            axes[1].set_title(f"Road Probability + Route{sigma_str}")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].axis("off")

            # Panel 3: ROI T field (coarse)
            valid = T_roi[T_roi < 1e8]
            vmax = np.percentile(valid, 95) if valid.size > 0 else 1000
            im2 = axes[2].imshow(T_roi, cmap="viridis", vmax=vmax)
            axes[2].set_title(f"ROI Eikonal Field (coarse)\nTarget Cost: {dist:.2f}")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            axes[2].axis("off")

        else:
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
             "Default: best_lora_pos8.0_dice0.5_thin4.0.ckpt (finetune_demo). "
             "Do NOT pass sam_vit_b_01ec64.pth here — that is the raw SAM encoder "
             "and is loaded internally by SAMRoute automatically.",
    )
    p.add_argument(
        "--output", type=str, default="tif_inference_route_result.png",
        help="Output figure path (default distinct from test_inference.py)",
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

    # NEW: route planning / distance query
    p.add_argument("--route", action="store_true", default=True,
                   help="Compute road-aware distance T between two points and visualise the route.")
    p.add_argument("--pick_points", action="store_true",
                   help="Interactively click two points on the image (after inference).")
    p.add_argument("--src", type=str, default=None,
                   help="Start point in 'y,x' (pixels). If set, disables interactive picking.")
    p.add_argument("--tgt", type=str, default=None,
                   help="End point in 'y,x' (pixels). If set, disables interactive picking.")
    p.add_argument("--roi_margin", type=int, default=None,
                   help="ROI margin (px) around the src-tgt span for Eikonal solve (default: model ROUTE_ROI_MARGIN).")
    p.add_argument("--eik_downsample", type=int, default=None,
                   help="Downsample factor for route Eikonal solve (default: model ROUTE_EIK_DOWNSAMPLE).")
    p.add_argument("--eik_iters", type=int, default=None,
                   help="Number of Eikonal iterations (default: model ROUTE_EIK_ITERS).")
    p.add_argument("--snap_radius", type=int, default=25,
                   help="Snap clicked points to nearest road-like pixel within this radius (0 disables).")
    p.add_argument("--snap_th", type=float, default=0.5,
                   help="Road probability threshold for snapping.")
    p.add_argument("--max_path_steps", type=int, default=None,
                   help="Max backtracking steps when extracting the path (default: H*W).")
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
        route=args.route,
        pick_points=args.pick_points,
        src=args.src,
        tgt=args.tgt,
        roi_margin=args.roi_margin,
        eik_downsample=args.eik_downsample,
        eik_iters=args.eik_iters,
        snap_radius=args.snap_radius,
        snap_th=args.snap_th,
        max_path_steps=args.max_path_steps,
    )