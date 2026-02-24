"""
validate_sam_route.py
SAMRoute end-to-end validation:
  image → SAMRoute (road_prob) → Eikonal distances → Spearman vs GT vs Euclidean

Usage:
  python validate_sam_route.py --image_path <tif> --config_path <yaml> --ckpt_path <ckpt> --k 5
  python validate_sam_route.py --image_path <tif> --config_path <yaml> --ckpt_path <ckpt> --k 10 --downsample_resolution 1024

conda env: satdino
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, List, Tuple

# ---------------------------------------------------------------------------
# Make the eikonal_solver and sam_road_repo visible on sys.path
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent
_SAM_REPO = _HERE.parent / "sam_road_repo"

# Add SAMRoad sub-directories needed for 'segment_anything' imports.
# We add them AFTER _HERE so that eikonal_solver/utils.py is always found first.
for _p in [
    str(_SAM_REPO / "segment-anything-road"),
    str(_SAM_REPO / "sam"),
    str(_SAM_REPO),
]:
    if _p not in sys.path:
        sys.path.append(_p)

# Guarantee eikonal_solver/ is at the very front so its utils.py wins.
_here_str = str(_HERE)
if _here_str in sys.path:
    sys.path.remove(_here_str)
sys.path.insert(0, _here_str)

from feature_extractor import load_tif_image
from load_nodes_from_npz import load_nodes_from_npz
from utils import (
    _normalize_prob_map,
    _to_uint8_hwc,
    pick_anchor_and_knn,
    mask_finite,
    fmt_rank_table,
    compute_ranking_metrics,
)
from model import SAMRoute


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _find_upwards(start_dir: Path, marker: str, max_depth: int = 10) -> Optional[Path]:
    """Walk up directory tree until a directory containing `marker` is found."""
    cur = start_dir.resolve()
    for _ in range(max_depth):
        if (cur / marker).exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _auto_find_config_and_ckpt() -> Tuple[Optional[str], Optional[str]]:
    """
    Try to locate the SAMRoad config yaml and checkpoint automatically by
    searching upwards from this file's location.
    Returns (config_path, ckpt_path), either may be None if not found.
    """
    root = _find_upwards(_HERE, "sam_road_repo")
    if root is None:
        return None, None

    repo_dir  = root / "sam_road_repo"
    ckpt_dir  = root / "checkpoints"
    cfg_path  = repo_dir / "config" / "toponet_vitb_512_cityscale.yaml"
    ckpt_path = ckpt_dir / "cityscale_vitb_512_e10.ckpt"

    return (
        str(cfg_path)  if cfg_path.exists()  else None,
        str(ckpt_path) if ckpt_path.exists() else None,
    )


def build_sam_route_model(
    config_path: str,
    ckpt_path: str,
    device: torch.device,
    strict: bool = False,
) -> SAMRoute:
    """
    Construct and load SAMRoute from config yaml + checkpoint.

    SAMRoute.__init__ (via SAMRoad.__init__) already loads the SAM ViT-B
    weights from config.SAM_CKPT_PATH.  We then overlay the fine-tuned
    road-segmentation weights from `ckpt_path`.

    Args:
        config_path: path to SAMRoad yaml config  (e.g. toponet_vitb_512_cityscale.yaml)
        ckpt_path:   path to SAMRoad/SAMRoute checkpoint (.ckpt / .pt / .pth)
        device:      target device
        strict:      whether load_state_dict should be strict
    Returns:
        model: SAMRoute in eval mode on `device`
    """
    import yaml
    from addict import Dict as AdDict

    with open(config_path) as f:
        config = AdDict(yaml.safe_load(f))

    # Ensure SAM ViT weights exist relative to config location
    cfg_parent = Path(config_path).parent
    sam_ckpt_candidate = cfg_parent.parent / "sam_ckpts" / "sam_vit_b_01ec64.pth"
    if sam_ckpt_candidate.exists():
        config.SAM_CKPT_PATH = str(sam_ckpt_candidate)

    print(f"[SAMRoute] Building model (SAM_VERSION={config.SAM_VERSION}, PATCH_SIZE={config.PATCH_SIZE})")
    model = SAMRoute(config)

    print(f"[SAMRoute] Loading checkpoint: {ckpt_path}")
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[SAMRoute] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[SAMRoute] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.eval().to(device)
    print(f"[SAMRoute] Ready on {device}")
    return model


# ---------------------------------------------------------------------------
# Road probability extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_road_prob_from_model(
    model: SAMRoute,
    image_path: str,
    rgb_tensor: torch.Tensor,      # [1, 3, H, W] float [0, 1]
    device: torch.device,
    downsample_resolution: Optional[int] = None,
    debug_use_gt: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Extract road probability map.

    For images larger than model.config.PATCH_SIZE (512), this function uses
    SamRoadTiledInferencer (tiled inference at the requested resolution) which
    handles arbitrary-size inputs while using the same SAMRoad backbone weights.

    For images already at PATCH_SIZE, it calls model._predict_mask_logits_scores()
    directly (single-pass, faster).

    Args:
        model:                 SAMRoute model (eval mode), used for direct inference
                               on small images and provides config info
        image_path:            path to the TIF file (for tiled inferencer)
        rgb_tensor:            [1, 3, H, W] float in [0, 1]
        device:                target device
        downsample_resolution: if given, produce road_prob at this resolution
                               (downsamples the longer side)
        debug_use_gt:          use GT road mask instead of model prediction
    Returns:
        road_prob:    [H_out, W_out]  probabilities in [0, 1]
        scale_factor: H_out / H_orig  (1.0 if no downsampling)
    """
    if debug_use_gt:
        from utils import load_ground_truth_road_map_common
        _, _, H_orig, W_orig = rgb_tensor.shape
        road_prob = load_ground_truth_road_map_common(image_path, (H_orig, W_orig), device)
        print("[Debug] Using GT road map")
        return road_prob, 1.0

    _, _, H_orig, W_orig = rgb_tensor.shape
    patch_size = int(model.config.PATCH_SIZE)  # e.g. 512

    # Determine output resolution
    scale_factor = 1.0
    if downsample_resolution and max(H_orig, W_orig) > downsample_resolution:
        scale_factor = downsample_resolution / max(H_orig, W_orig)
    target_H = max(1, int(H_orig * scale_factor))
    target_W = max(1, int(W_orig * scale_factor))

    if max(H_orig, W_orig) <= patch_size:
        # Image fits in one pass — use model directly
        rgb_hwc = rgb_tensor.to(device)[0].permute(1, 2, 0).unsqueeze(0) * 255.0  # [1,H,W,3]
        _, mask_scores = model._predict_mask_logits_scores(rgb_hwc)
        road_prob = mask_scores[0, :, :, 1]  # [H, W]
        if scale_factor != 1.0:
            road_prob = F.interpolate(
                road_prob.unsqueeze(0).unsqueeze(0),
                size=(target_H, target_W), mode="bilinear", align_corners=False,
            ).squeeze()
    else:
        # Large image — use SamRoadTiledInferencer (handles arbitrary sizes)
        from feature_extractor import SamRoadTiledInferencer
        inferencer = SamRoadTiledInferencer(sat_grid=14, sam_road_ckpt_name=_ckpt_name_from_model(model))
        inferencer.ensure_loaded(device)

        results    = inferencer.infer(rgb_tensor.to(device), device, return_mask=True)
        road_np    = results.get("road_mask")
        if road_np is None:
            raise ValueError("SamRoadTiledInferencer inference failed (road_mask is None)")

        road_prob = torch.from_numpy(road_np).to(device, dtype=torch.float32)
        if road_prob.dim() == 3:
            road_prob = road_prob[0] if road_prob.shape[0] == 1 else road_prob.squeeze()

        if scale_factor != 1.0:
            road_prob = F.interpolate(
                road_prob.unsqueeze(0).unsqueeze(0),
                size=(target_H, target_W), mode="bilinear", align_corners=False,
            ).squeeze()

    road_prob = road_prob.clamp(0.0, 1.0)
    if scale_factor != 1.0:
        print(f"[Downsample] road_prob: {H_orig}×{W_orig} → {target_H}×{target_W}  scale={scale_factor:.4f}")
    return road_prob, scale_factor


def _ckpt_name_from_model(model: SAMRoute) -> str:
    """Try to infer checkpoint filename from the already-loaded SAMRoute model config."""
    try:
        ckpt_path = getattr(model.config, "SAM_CKPT_PATH", "")
        # model config doesn't store the fine-tuned ckpt name, fall back to default
        return "cityscale_vitb_512_e10.ckpt"
    except Exception:
        return "cityscale_vitb_512_e10.ckpt"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_knn_paths_sam_route(
    rgb_tensor: torch.Tensor,        # [1, 3, H, W] float [0, 1]
    road_prob: torch.Tensor,         # [H_ds, W_ds]
    T_patch: torch.Tensor,           # [P, P] Eikonal distance field
    top_left_yx: Tuple[int, int],    # (y0, x0) of ROI in road_prob coords
    P: int,
    anchor: torch.Tensor,            # [2] in road_prob coords
    targets: torch.Tensor,           # [K, 2] in road_prob coords
    nn_idx: torch.Tensor,            # [K] node indices
    dist_k: np.ndarray,              # [K] raw Eikonal distances
    pred_norm: np.ndarray,           # [K] normalized pred
    gt: np.ndarray,                  # [K] GT normalized
    scale_factor: float,             # road_prob coords / original image coords
    save_path: str = "images/sam_route_knn_distances.png",
) -> str:
    """
    Visualize SAMRoute distances (Eikonal T values) on the original-resolution image.
    Unlike the old visualizer, no path backtrace is shown — we mark source and targets
    with colored scatter points and annotate each with pred/GT distances.
    """
    # --- Build display image at original resolution ---
    img_np = _to_uint8_hwc(rgb_tensor)     # [H_orig, W_orig, 3]
    H_orig, W_orig = img_np.shape[:2]

    # --- Build road_prob display (upscale to original if needed) ---
    road_np = road_prob.detach().float().cpu()
    if scale_factor != 1.0:
        road_np = F.interpolate(
            road_np.unsqueeze(0).unsqueeze(0),
            size=(H_orig, W_orig), mode="bilinear", align_corners=False,
        ).squeeze().cpu()
    road_np = road_np.numpy()

    # --- Map T_patch onto original-resolution canvas ---
    T_full = np.full((H_orig, W_orig), np.nan, dtype=np.float32)
    inv_sf = 1.0 / scale_factor if scale_factor != 1.0 else 1.0
    y0_ds, x0_ds = top_left_yx
    y0_orig = int(round(y0_ds * inv_sf));  x0_orig = int(round(x0_ds * inv_sf))
    P_orig  = int(round(P * inv_sf))

    # resize T_patch to original-space patch size
    T_orig = F.interpolate(
        T_patch.unsqueeze(0).unsqueeze(0).float(),
        size=(P_orig, P_orig), mode="bilinear", align_corners=False,
    ).squeeze().cpu().numpy()

    yy0 = max(y0_orig, 0);             xx0 = max(x0_orig, 0)
    yy1 = min(y0_orig + P_orig, H_orig); xx1 = min(x0_orig + P_orig, W_orig)
    py0 = yy0 - y0_orig;              px0 = xx0 - x0_orig
    T_full[yy0:yy1, xx0:xx1] = T_orig[py0: py0 + (yy1 - yy0), px0: px0 + (xx1 - xx0)]

    finite = np.isfinite(T_full)
    vmin   = float(np.min(T_full[finite]))      if finite.any() else 0.0
    vmax   = float(np.percentile(T_full[finite], 99)) if finite.any() else 1.0

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(img_np)

    masked_road = np.ma.masked_where(road_np < 0.1, road_np)
    ax.imshow(masked_road, cmap="Reds", vmin=0, vmax=1, alpha=0.45)
    ax.imshow(T_full, cmap="magma", vmin=vmin, vmax=vmax, alpha=0.35)

    # ROI box (in original coords)
    roi_x1 = x0_orig + P_orig;  roi_y1 = y0_orig + P_orig
    ax.plot([x0_orig, roi_x1, roi_x1, x0_orig, x0_orig],
            [y0_orig, y0_orig, roi_y1, roi_y1, y0_orig],
            lw=2.0, color="white", ls="--", alpha=0.8, zorder=9)

    # Targets — colored by rank in pred_norm
    colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))
    for t in range(len(targets)):
        ty_ds, tx_ds = int(targets[t, 0].item()), int(targets[t, 1].item())
        ty = int(round(ty_ds * inv_sf));  tx = int(round(tx_ds * inv_sf))
        c  = colors[t]

        pred_s = f"{pred_norm[t]:.4f}" if np.isfinite(pred_norm[t]) else "inf"
        gt_s   = f"{gt[t]:.4f}"        if np.isfinite(gt[t])        else "N/A"
        label  = f"#{int(nn_idx[t].item())}  pred={pred_s}  gt={gt_s}"

        ax.scatter([tx], [ty], s=260, marker="x", color=c,
                   linewidths=2.5, zorder=14, label=label)

    # Anchor — green circle
    ay_ds, ax_ds = int(anchor[0].item()), int(anchor[1].item())
    ay = int(round(ay_ds * inv_sf));  aax = int(round(ax_ds * inv_sf))
    ax.scatter([aax], [ay], s=350, marker="o", color="lime",
               edgecolors="black", linewidths=3, zorder=15, label="Anchor (src)")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9)
    ax.axis("off")
    ax.set_title("SAMRoute: Eikonal Distance Field  (magma=T, red=road_prob)")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Vis] Saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main demo function
# ---------------------------------------------------------------------------

def demo_sam_route_knn_ranking(
    image_path: str,
    config_path: str,
    ckpt_path: str,
    *,
    k: int = 5,
    anchor_idx: Optional[int] = None,
    road_threshold: float = 0.30,
    downsample_resolution: Optional[int] = 1024,
    eik_iters: int = 200,
    roi_margin: int = 96,
    debug_use_gt: bool = False,
) -> None:
    """
    Single-anchor KNN validation using SAMRoute.

    Pipeline:
      1. Load image
      2. Build SAMRoute model, get road_prob via segmentation head
      3. Load nodes from NPZ file, scale to road_prob coordinates
      4. Pick anchor + K nearest neighbors
      5. model.compute_knn_distances() — single ROI Eikonal solve
      6. Compute Spearman / Kendall / Pairwise vs GT road distances and Euclidean
      7. Print metrics table + save visualization
    """
    if not Path(image_path).exists():
        print(f"[Error] File not found: {image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # ------------------------------------------------------------------ step 1
    print("\n=== STEP 1: Load image ===")
    t0 = time.time()
    rgb_tensor = load_tif_image(image_path).to(device)   # [1, 3, H, W] float [0,1]
    _, _, H_orig, W_orig = rgb_tensor.shape
    print(f"  Image: {H_orig} × {W_orig}  ({time.time()-t0:.2f}s)")

    # ------------------------------------------------------------------ step 2
    print("\n=== STEP 2: Build SAMRoute model + extract road_prob ===")
    t0 = time.time()
    model = build_sam_route_model(config_path, ckpt_path, device)
    t_model = time.time() - t0

    t0 = time.time()
    road_prob, scale_factor = get_road_prob_from_model(
        model, image_path, rgb_tensor, device,
        downsample_resolution=downsample_resolution,
        debug_use_gt=debug_use_gt,
    )
    t_inference = time.time() - t0
    H_ds, W_ds = road_prob.shape
    print(f"  road_prob: {H_ds}×{W_ds}  ({t_inference:.2f}s)")

    # ------------------------------------------------------------------ step 3
    print("\n=== STEP 3: Load nodes from NPZ ===")
    t0 = time.time()
    # load at original resolution first (for GT distance lookup)
    nodes_yx_orig, info = load_nodes_from_npz(
        image_path,
        road_prob if scale_factor == 1.0 else F.interpolate(
            road_prob.unsqueeze(0).unsqueeze(0),
            size=(H_orig, W_orig), mode="bilinear", align_corners=False,
        ).squeeze(),
        p_count=20,
        snap=True,
        snap_threshold=road_threshold,
        snap_win=10,
        verbose=True,
    )
    t_nodes = time.time() - t0

    if nodes_yx_orig.shape[0] < 2:
        print("[Error] Not enough nodes (<2).")
        return

    # scale node coordinates to road_prob resolution
    if scale_factor != 1.0:
        nodes_yx = (nodes_yx_orig.float() * scale_factor).long()
        nodes_yx[:, 0].clamp_(0, H_ds - 1)
        nodes_yx[:, 1].clamp_(0, W_ds - 1)
    else:
        nodes_yx = nodes_yx_orig
    print(f"  Nodes: {nodes_yx.shape[0]}  ({t_nodes:.2f}s)")

    # ------------------------------------------------------------------ step 4
    print("\n=== STEP 4: Pick anchor + KNN ===")
    anchor_idx, nn_idx = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=anchor_idx)
    anchor  = nodes_yx[anchor_idx]    # [2] in road_prob coords
    targets = nodes_yx[nn_idx]        # [K, 2]
    print(f"  Anchor idx={anchor_idx}  yx={anchor.tolist()}")
    print(f"  Neighbors: {nn_idx.tolist()}")

    # ------------------------------------------------------------------ step 5
    print(f"\n=== STEP 5: Eikonal solve (1 anchor → {k} targets) ===")
    from fast_sweeping import EikonalConfig
    from utils import trace_path_from_distance_field_common, calc_path_len_vec
    eik_cfg = EikonalConfig(
        n_iters=eik_iters, tau_min=0.03, tau_branch=0.05, tau_update=0.03,
        use_redblack=True, monotone=True,
    )
    t0 = time.time()
    # Use the full version that also returns T_patch and ROI coords (needed for
    # path backtrace and visualization — avoids running the solver twice)
    dist_k, T_patch_vis, top_left_vis = _compute_knn_distances_with_T(
        model, road_prob, anchor, targets, eik_cfg=eik_cfg, margin=roi_margin,
    )   # [K], [P,P], (y0,x0)
    t_eikonal = time.time() - t0
    P_vis     = T_patch_vis.shape[0]
    y0_roi, x0_roi = top_left_vis
    print(f"  Done in {t_eikonal:.3f}s  patch={P_vis}  dist_k={np.round(dist_k.cpu().numpy(), 2).tolist()}")

    # ------------------------------------------------------------------ step 6
    print("\n=== STEP 6: Path backtrace + Ranking evaluation ===")
    meta_sat_height_px = int(info["meta_sat_height_px"])
    case_idx = info["case_idx"]
    nn_np    = nn_idx.cpu().numpy()
    device   = road_prob.device
    # path scale: T values are in road_prob (downsampled) space; path lengths too
    path_scale_factor = 1.0 / scale_factor if scale_factor != 1.0 else 1.0

    # --- src_rel in patch coords ---
    src_rel = torch.tensor(
        [int(anchor[0].item()) - y0_roi, int(anchor[1].item()) - x0_roi],
        device=device, dtype=torch.long,
    )
    src_rel[0].clamp_(0, P_vis - 1); src_rel[1].clamp_(0, P_vis - 1)

    # --- T-value based pred (raw Eikonal cost) ---
    dist_k_np = dist_k.cpu().numpy()
    tvals_raw = dist_k_np.copy()           # [K] raw T values (downsampled space)

    # T-value normalization: divide by image height in downsampled px
    meta_px_ds = max(meta_sat_height_px * scale_factor, 1)
    pred_norm_T = dist_k_np / meta_px_ds   # comparable in same space

    # --- Path-length based pred (backtrace, same as original script) ---
    pred_pix  = np.full(k, np.nan)
    pred_norm_path = np.full(k, np.nan)
    paths_global   = []
    INF_SENTINEL   = float(1e5)

    for t in range(k):
        tgt_rel = torch.tensor(
            [int(targets[t, 0].item()) - y0_roi, int(targets[t, 1].item()) - x0_roi],
            device=device, dtype=torch.long,
        )
        tval = float(dist_k[t].item())
        if not np.isfinite(tval) or tval >= INF_SENTINEL:
            paths_global.append(None)
            continue
        tgt_rel[0].clamp_(0, P_vis - 1); tgt_rel[1].clamp_(0, P_vis - 1)
        path_patch = trace_path_from_distance_field_common(
            T_patch_vis, src_rel, tgt_rel, device, diag=True, max_steps=200_000,
        )
        if len(path_patch) < 2:
            paths_global.append(None)
            continue
        path_global = [(p[0] + y0_roi, p[1] + x0_roi) for p in path_patch]
        pixel_len   = calc_path_len_vec(path_global) * path_scale_factor
        pred_pix[t]       = pixel_len
        pred_norm_path[t] = pixel_len / max(meta_sat_height_px, 1)
        paths_global.append(path_global)

    gt  = np.asarray(info["undirected_dist_norm"][case_idx][anchor_idx, nn_np], dtype=np.float64)
    euc = np.asarray(info["euclidean_dist_norm"][case_idx][anchor_idx, nn_np],  dtype=np.float64)

    # Metrics for both pred types
    mask_T    = mask_finite(pred_norm_T,    gt, euc)
    mask_path = mask_finite(pred_norm_path, gt, euc)
    metrics_T    = compute_ranking_metrics(nn_idx, pred_norm_T,    gt, euc, mask_T)
    metrics_path = compute_ranking_metrics(nn_idx, pred_norm_path, gt, euc, mask_path)

    # ------------------------------------------------------------------ output
    print("\n" + "=" * 72)
    print("SAMRoute Ranking Evaluation (Anchor + KNN)")
    print("=" * 72)
    print(f"Anchor idx: {anchor_idx}   anchor_yx (road_prob space): {anchor.tolist()}")
    print(f"Neighbors (k={len(nn_idx)}): {nn_idx.tolist()}")
    n_valid_path = metrics_path["n_valid"]
    if n_valid_path < k:
        print(f"[Warn] Valid path backtrace: {n_valid_path}/{k}  (inf = path backtrace failed)")

    hdr = f"{'Metric':<18} {'SAMRoute (path len)':<24} {'SAMRoute (T value)':<24} {'Euclidean':<12}"
    sep = "-" * 78
    print(f"\n{hdr}")
    print(sep)
    print(f"{'Spearman':<18} {metrics_path['s_pred']:>22.4f}   {metrics_T['s_pred']:>22.4f}   {metrics_path['s_euc']:>10.4f}")
    print(f"{'Kendall':<18} {metrics_path['k_pred']:>22.4f}   {metrics_T['k_pred']:>22.4f}   {metrics_path['k_euc']:>10.4f}")
    print(f"{'Pairwise Acc':<18} {metrics_path['pw_pred']:>22.4f}   {metrics_T['pw_pred']:>22.4f}   {metrics_path['pw_euc']:>10.4f}")
    print(f"{'Top-1 (GT)':<18} {'node ' + str(metrics_path['top1_gt']):>22}   {'':>22}   {'node ' + str(metrics_path['top1_euc']):>10}")
    print(f"{'Top-1 (pred)':<18} {'node ' + str(metrics_path['top1_pred']):>22}   {'node ' + str(metrics_T['top1_pred']):>22}")
    print(sep)

    print("\n[Table sorted by GT ascending]  pred=path_len_norm  T(goal)=raw_Eikonal_T")
    valid_arr = np.isfinite(pred_norm_path)
    print(fmt_rank_table(nn_np, pred_norm_path, gt, euc, tvals_raw, valid_arr))

    # timing summary
    print(f"\n{'Timing':<30} {'Time (s)'}")
    print("-" * 42)
    print(f"{'Model build + load':<30} {t_model:.3f}")
    print(f"{'Inference (road_prob)':<30} {t_inference:.3f}")
    print(f"{'Node loading':<30} {t_nodes:.3f}")
    print(f"{'Eikonal solve':<30} {t_eikonal:.3f}")

    # ------------------------------------------------------------------ vis
    print("\n=== STEP 7: Visualization ===")
    save_path = f"images/sam_route_knn_anchor{anchor_idx}.png"
    visualize_knn_paths_sam_route(
        rgb_tensor=rgb_tensor,
        road_prob=road_prob,
        T_patch=T_patch_vis,
        top_left_yx=top_left_vis,
        P=P_vis,
        anchor=anchor,
        targets=targets,
        nn_idx=nn_idx,
        dist_k=tvals_raw,
        pred_norm=pred_norm_path,
        gt=gt,
        scale_factor=scale_factor,
        save_path=save_path,
    )


@torch.no_grad()
def _compute_knn_distances_with_T(
    model: SAMRoute,
    road_prob: torch.Tensor,
    src_yx: torch.Tensor,
    tgt_yx: torch.Tensor,
    eik_cfg=None,
    margin: int = 96,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Same as model.compute_knn_distances() but also returns the T_patch and
    top-left ROI coordinates for visualization.

    Returns:
        dist:       [K] Eikonal distances
        T_patch:    [P, P] full distance field inside ROI
        top_left:   (y0, x0) of ROI in road_prob coordinates
    """
    from fast_sweeping import EikonalConfig, eikonal_soft_sweeping

    cfg    = eik_cfg if eik_cfg is not None else model.route_eik_cfg
    margin = int(margin)

    H, W   = road_prob.shape
    device = road_prob.device
    src    = src_yx.to(device).long()
    tgts   = tgt_yx.to(device).long()
    K      = tgts.shape[0]

    span_max = int(torch.max(torch.abs(tgts.float() - src.float())).item()) if K > 0 else 0
    half     = span_max + margin
    P        = max(2 * half + 1, 512)   # min_patch=512 for sufficient convergence

    y0 = int(src[0].item()) - half;  x0 = int(src[1].item()) - half
    y1 = y0 + P;                     x1 = x0 + P

    yy0 = max(y0, 0); xx0 = max(x0, 0)
    yy1 = min(y1, H); xx1 = min(x1, W)

    patch = torch.zeros(P, P, device=device, dtype=road_prob.dtype)
    patch[yy0 - y0: yy1 - y0, xx0 - x0: xx1 - x0] = road_prob[yy0:yy1, xx0:xx1]

    cost_patch = model._road_prob_to_cost(patch)

    src_rel_y = max(0, min(int(src[0].item()) - y0, P - 1))
    src_rel_x = max(0, min(int(src[1].item()) - x0, P - 1))
    src_mask  = torch.zeros(1, P, P, dtype=torch.bool, device=device)
    src_mask[0, src_rel_y, src_rel_x] = True

    n_iters_eff = max(int(cfg.n_iters), P)
    cfg_eff     = EikonalConfig(
        n_iters=n_iters_eff, h=cfg.h,
        tau_min=cfg.tau_min, tau_branch=cfg.tau_branch, tau_update=cfg.tau_update,
        large_val=cfg.large_val, use_redblack=cfg.use_redblack, monotone=cfg.monotone,
    )
    T_patch = eikonal_soft_sweeping(cost_patch.unsqueeze(0), src_mask, cfg_eff)
    if T_patch.dim() == 3:
        T_patch = T_patch[0]

    if K > 0:
        tgt_rel_y = (tgts[:, 0] - y0).clamp(0, P - 1)
        tgt_rel_x = (tgts[:, 1] - x0).clamp(0, P - 1)
        valid = (
            (tgts[:, 0] >= y0) & (tgts[:, 0] < y1) &
            (tgts[:, 1] >= x0) & (tgts[:, 1] < x1)
        )
        dist = T_patch[tgt_rel_y, tgt_rel_x]
        dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
    else:
        dist = torch.empty(0, device=device)

    return dist, T_patch, (y0, x0)


# ---------------------------------------------------------------------------
# argparse entry point
# ---------------------------------------------------------------------------

def main():
    auto_cfg, auto_ckpt = _auto_find_config_and_ckpt()

    parser = argparse.ArgumentParser(
        description="SAMRoute KNN ranking validation: Eikonal distances vs GT vs Euclidean",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=(
            "Gen_dataset_V2/Gen_dataset/19.940688_110.276704/"
            "00_20.021516_110.190699_3000.0/"
            "crop_20.021516_110.190699_3000.0_z16.tif"
        ),
        help="Path to input satellite image (.tif)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=auto_cfg,
        help="SAMRoad yaml config path (auto-detected if not specified)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=auto_ckpt,
        help="SAMRoad/SAMRoute checkpoint path (auto-detected if not specified)",
    )
    parser.add_argument("--k",                     type=int,   default=5)
    parser.add_argument("--anchor_idx",            type=int,   default=None)
    parser.add_argument("--road_threshold",        type=float, default=0.30)
    parser.add_argument("--downsample_resolution", type=int,   default=1024,
                        help="Downsample longer side to this before SAMRoute inference (0 = disable)")
    parser.add_argument("--eik_iters",             type=int,   default=200)
    parser.add_argument("--roi_margin",            type=int,   default=96)
    parser.add_argument("--debug_use_gt",          action="store_true",
                        help="Use ground-truth road map instead of SAMRoute prediction")

    args = parser.parse_args()

    if args.config_path is None:
        parser.error("--config_path is required (auto-detection failed)")
    if args.ckpt_path is None:
        parser.error("--ckpt_path is required (auto-detection failed)")

    demo_sam_route_knn_ranking(
        image_path=args.image_path,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
        k=args.k,
        anchor_idx=args.anchor_idx,
        road_threshold=args.road_threshold,
        downsample_resolution=args.downsample_resolution if args.downsample_resolution > 0 else None,
        eik_iters=args.eik_iters,
        roi_margin=args.roi_margin,
        debug_use_gt=args.debug_use_gt,
    )


if __name__ == "__main__":
    main()
