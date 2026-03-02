#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradcheck_route_loss_v2_multigrid_fullmap.py

在 gradcheck_route_loss_v2_multigrid.py 的基础上，新增 “整幅大图(≈3k×3k)滑窗分割 + 全图(下采样)路由距离测试” 模式。

核心目标
--------
1) 不再局限于 dataset 采样的 512×512 patch，而是：
   - 对整幅 TIF 做 sliding-window segmentation，得到 full road_prob [H, W]；
   - 在 full road_prob 上选择 src / tgt（可交互点选、也可从 NPZ 节点采样）；
   - 在 full road_prob 的 “全图下采样网格” 上做可微 multigrid(+tube) Eikonal，
     输出 pred_dist，并可选做若干步优化，验证梯度链路/速度。

2) 保留原脚本的 dataset/ROI 模式（默认不变），通过 --tif 启用 fullmap 模式。

注意（非常重要）
----------------
- fullmap 模式下，sliding inference 默认在 torch.no_grad() 里跑（避免巨大计算图）。
  因此 distance loss 的梯度不会回传到 image_encoder/map_decoder（因为 road_prob 被当作常量输入）。
  但梯度仍会回传到 routing 分支参数（cost_log_alpha/cost_log_gamma/eik_gate_logit），以及你若启用
  --optimize_prob，还可以直接把 road_prob 当作可学习变量来验证 “距离监督是否能矫正概率图（在 ROI 子区域）”。

- fullmap “全图求解” 采用下采样网格（ds=args.downsample），避免 ROI span 导致 P>H/W 的巨大 padding。
  这比直接用 _roi_* 在跨越整幅图的两点上更稳健、更可控。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
import re
from dataclasses import replace as dc_replace
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_CKPT_DEFAULT = os.path.join(
    _PROJECT_ROOT,
    "training_outputs", "finetune_demo", "checkpoints",
    "best_lora_pos8.0_dice0.5_thin4.0.ckpt",
)
_DATA_ROOT_DEFAULT = os.path.join(_PROJECT_ROOT, "Gen_dataset_V2", "Gen_dataset")

# model_multigrid exports some helper functions we reuse in fullmap mode
from model_multigrid import (  # noqa: E402
    SAMRoute,
    _mix_eikonal_euclid,
    _eikonal_soft_sweeping_diff,
    _eikonal_soft_sweeping_diff_init,
)
from dataset import build_dataloaders  # noqa: E402
from eikonal import eikonal_soft_sweeping, EikonalConfig  # noqa: E402

# -----------------------------------------------------------------------------
# Minimal config (mirrors finetune_demo.TrainConfig but only keeps what we need)
# -----------------------------------------------------------------------------
class GradcheckConfig:
    def __init__(self):
        # --- SAM / decoder ---
        self.SAM_VERSION = 'vit_b'
        self.PATCH_SIZE = 512
        self.NO_SAM = False
        self.USE_SAM_DECODER = False
        self.USE_SMOOTH_DECODER = False
        self.ENCODER_LORA = False
        self.LORA_RANK = 4
        self.FREEZE_ENCODER = True
        self.FOCAL_LOSS = False
        self.TOPONET_VERSION = 'default'

        # raw SAM encoder weights
        self.SAM_CKPT_PATH = os.path.join(
            _PROJECT_ROOT, "sam_road_repo", "sam_ckpts", "sam_vit_b_01ec64.pth"
        )

        # --- routing branch ---
        self.ROUTE_COST_MODE = 'add'
        self.ROUTE_ADD_ALPHA = 20.0
        self.ROUTE_ADD_GAMMA = 2.0
        self.ROUTE_ADD_BLOCK_ALPHA = 0.0
        self.ROUTE_BLOCK_TH = 0.0
        self.ROUTE_ROI_MARGIN = 64

        # --- neural residual cost net ---
        self.ROUTE_COST_NET = False
        self.ROUTE_COST_NET_CH = 8
        self.ROUTE_COST_NET_USE_COORD = False
        self.ROUTE_COST_NET_DELTA_SCALE = 0.75

        # Eikonal params (CLI override)
        self.ROUTE_EIK_ITERS = 200
        self.ROUTE_EIK_DOWNSAMPLE = 4
        self.ROUTE_CKPT_CHUNK = 10
        self.ROUTE_DIST_NORM_PX = 512.0
        self.ROUTE_GATE_ALPHA = 0.8

        # loss weights (CLI override)
        self.ROUTE_LAMBDA_SEG = 0.0
        self.ROUTE_LAMBDA_DIST = 1.0

        # seg loss hyperparams (keep stable)
        self.ROAD_POS_WEIGHT = 13.9
        self.ROAD_DICE_WEIGHT = 0.0
        self.ROAD_DUAL_TARGET = False
        self.ROAD_THIN_BOOST = 5.0

        # stable defaults for gradcheck
        self.ROUTE_EIK_MODE = "soft_train"
        self.ROUTE_CAP_MODE = "tanh"
        self.ROUTE_CAP_MULT = 10.0

        self.ROUTE_DIST_WARMUP_STEPS = 0
        self.ROUTE_EIK_WARMUP_EPOCHS = 0
        self.ROUTE_EIK_ITERS_MIN = 40

    def get(self, key, default):
        return getattr(self, key, default)


# -----------------------------------------------------------------------------
# IO + full-map sliding inference (borrowed from test_inference_route.py)
# -----------------------------------------------------------------------------
def _load_rgb_from_tif(tif_path: str) -> np.ndarray:
    """Load a GeoTIFF as uint8 RGB array [H, W, 3]."""
    # Prefer rasterio when available (handles multi-band robustly)
    try:
        import rasterio  # type: ignore
        with rasterio.open(tif_path) as src:
            bands = src.read()[:3]  # [3, H, W]
            arr = np.moveaxis(bands, 0, -1)  # [H, W, 3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
    except Exception:
        pass

    # Fallback: tifffile
    try:
        import tifffile  # type: ignore
        arr = tifffile.imread(tif_path)
        # common shapes: [H,W,3] or [3,H,W] or [H,W] (grayscale)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr[:3], 0, -1)
        else:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    except Exception:
        pass

    # Last resort: PIL
    from PIL import Image
    return np.array(Image.open(tif_path).convert("RGB"), dtype=np.uint8)


def smooth_prob(road_prob: np.ndarray, sigma: float) -> np.ndarray:
    """Optional Gaussian smoothing. If scipy unavailable, silently no-op."""
    if sigma <= 0.0:
        return road_prob
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        return gaussian_filter(road_prob.astype(np.float32), sigma=sigma).clip(0.0, 1.0)
    except Exception:
        return road_prob


@torch.no_grad()
def sliding_window_inference(
    img_array: np.ndarray,
    model: "SAMRoute",
    device: torch.device,
    patch_size: int = 512,
    stride: int = 256,
    smooth_sigma: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """Sliding-window inference over full image. Returns road_prob [H,W] float32."""
    H, W = img_array.shape[:2]

    win1d = np.hanning(patch_size).astype(np.float32)
    win1d = np.maximum(win1d, 1e-3)
    win2d = np.outer(win1d, win1d)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    total = len(ys) * len(xs)
    done = 0

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + patch_size, H)
            x1 = min(x0 + patch_size, W)
            patch = img_array[y0:y1, x0:x1]

            ph, pw = patch.shape[:2]
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            rgb_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            _, ms = model._predict_mask_logits_scores(rgb_t)
            prob = ms[0, :, :, 1].float().cpu().numpy()  # [patch_size,patch_size]
            prob = smooth_prob(prob, smooth_sigma)

            prob_sum[y0:y0 + ph, x0:x0 + pw] += prob[:ph, :pw] * win2d[:ph, :pw]
            weight_sum[y0:y0 + ph, x0:x0 + pw] += win2d[:ph, :pw]

            done += 1
            if verbose and (done % 10 == 0 or done == total):
                print(f"  sliding: {done}/{total} patches", end="\r")
    if verbose:
        print()

    out = np.zeros((H, W), dtype=np.float32)
    m = weight_sum > 0
    out[m] = prob_sum[m] / weight_sum[m]
    return out


def _cache_encoder_features(
    img_array: np.ndarray,
    model: "SAMRoute",
    device: torch.device,
    patch_size: int = 512,
    stride: int = 256,
) -> Tuple[List[Tuple[int, int, int, int]], List[torch.Tensor], np.ndarray]:
    """Pre-compute and cache encoder features for all sliding-window patches.

    Returns:
        patches_info: list of (y0, x0, ph, pw) for each patch
        features: list of encoder feature tensors [1, C, Hf, Wf] (on CPU)
        win2d: Hanning window [patch_size, patch_size]
    """
    H, W = img_array.shape[:2]
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
    features = []
    total = len(ys) * len(xs)
    done = 0

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + patch_size, H)
                x1 = min(x0 + patch_size, W)
                patch = img_array[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                rgb_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
                x_enc = rgb_t.permute(0, 3, 1, 2)
                x_enc = (x_enc - model.pixel_mean) / model.pixel_std
                feat = model.image_encoder(x_enc).cpu()

                patches_info.append((y0, x0, ph, pw))
                features.append(feat)

                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  caching encoder features: {done}/{total}", end="\r")
    print()
    return patches_info, features, win2d


def _road_prob_from_cached_features(
    model: "SAMRoute",
    patches_info: List[Tuple[int, int, int, int]],
    features: List[torch.Tensor],
    win2d: np.ndarray,
    H: int, W: int,
    device: torch.device,
) -> torch.Tensor:
    """Recompute road_prob from cached encoder features WITH decoder gradients.

    Returns:
        road_prob: [1, H, W] tensor with gradient through decoder.
    """
    patch_size = win2d.shape[0]
    win_t = torch.from_numpy(win2d).to(device)
    prob_sum = torch.zeros(H, W, device=device)
    weight_sum = torch.zeros(H, W, device=device)

    for (y0, x0, ph, pw), feat in zip(patches_info, features):
        feat_gpu = feat.to(device)
        _, ms = model._predict_mask_logits_scores(None, encoder_feat=feat_gpu)
        prob = ms[0, :, :, 1]  # [patch_size, patch_size]

        prob_sum[y0:y0 + ph, x0:x0 + pw] = prob_sum[y0:y0 + ph, x0:x0 + pw] + \
            prob[:ph, :pw] * win_t[:ph, :pw]
        weight_sum[y0:y0 + ph, x0:x0 + pw] = weight_sum[y0:y0 + ph, x0:x0 + pw] + \
            win_t[:ph, :pw]

    safe_weight = weight_sum.clamp(min=1e-6)
    road_prob = prob_sum / safe_weight
    return road_prob.unsqueeze(0)  # [1, H, W]


# -----------------------------------------------------------------------------
# NPZ node sampling for full-map GT distances
# -----------------------------------------------------------------------------
def _load_npz_nodes(npz_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (coords_norm, undirected_dist_norm, H, W)."""
    d = np.load(npz_path, allow_pickle=True)
    coords = d["matched_node_norm"]            # (B, N, 2) (x_norm, y_norm) bottom-left
    udist  = d["undirected_dist_norm"]         # (B, N, N)
    H = int(d["meta_sat_height_px"][0])
    W = int(d["meta_sat_width_px"][0])
    return coords, udist, H, W


def _normxy_to_yx(xy_norm: np.ndarray, H: int, W: int) -> np.ndarray:
    """(N,2) (x_norm,y_norm bottom-left) -> (N,2) (y,x top-left pixel)."""
    x = np.rint(xy_norm[:, 0] * (W - 1)).astype(np.int64)
    y = np.rint((1.0 - xy_norm[:, 1]) * (H - 1)).astype(np.int64)
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    return np.stack([y, x], axis=1)


def _sample_pairs_from_npz(
    coords: np.ndarray,
    udist: np.ndarray,
    H: int,
    W: int,
    *,
    k_targets: int = 4,
    case_idx: int = 0,
    seed: int = 0,
    min_gt_dist_px: float = 50.0,
    max_pairs_try: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one anchor + K targets on the full image, return (src_yx, tgt_yx_k, gt_dist_k)."""
    rng = np.random.RandomState(seed)
    xy_norm = coords[case_idx]  # (N,2)
    N = xy_norm.shape[0]
    yx = _normxy_to_yx(xy_norm, H, W)  # (N,2) y,x

    # pick an anchor that has enough finite distances
    for _ in range(max_pairs_try):
        a = int(rng.randint(0, N))
        d_row = udist[case_idx, a]  # (N,)
        finite = np.isfinite(d_row) & (d_row > 0)
        finite[a] = False
        idx = np.where(finite)[0]
        if idx.size < k_targets:
            continue
        # choose K nearest by Euclid (good for local supervision; stable)
        ay, ax = yx[a]
        eu = np.sqrt((yx[idx, 0] - ay) ** 2 + (yx[idx, 1] - ax) ** 2)
        order = np.argsort(eu)
        chosen = idx[order[:k_targets]]
        gt_px = (udist[case_idx, a, chosen] * float(H)).astype(np.float32)
        if np.all(gt_px > min_gt_dist_px):
            src = yx[a].astype(np.int64)
            tgt = yx[chosen].astype(np.int64)
            return src, tgt, gt_px

    # fallback: pick arbitrary
    a = int(rng.randint(0, N))
    b = int((a + 1) % N)
    src = yx[a].astype(np.int64)
    tgt = np.tile(yx[b][None, :], (k_targets, 1)).astype(np.int64)
    gt_px = np.full((k_targets,), -1.0, dtype=np.float32)
    return src, tgt, gt_px


def _prepare_all_node_pairs(
    coords: np.ndarray,
    udist: np.ndarray,
    H: int,
    W: int,
    case_idx: int,
    k_neighbors: int = 4,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For one TSP case, build (src_yx, tgt_yx_k, gt_dist_k) for every node as source.

    Returns a list of tuples, one per valid source node:
        src_yx   : (2,)  int64 pixel coords (y, x)
        tgt_yx_k : (K,2) int64 pixel coords
        gt_dist_k: (K,)  float32 GT distance in pixels
    Nodes with fewer than *k_neighbors* valid neighbours are skipped.
    """
    xy_norm = coords[case_idx]           # (N, 2)  x_norm, y_norm (bottom-left)
    N = xy_norm.shape[0]
    yx = _normxy_to_yx(xy_norm, H, W)   # (N, 2)  y, x (top-left pixel)
    dist_mat = udist[case_idx]           # (N, N)

    pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(N):
        d_row = dist_mat[i]
        valid = np.isfinite(d_row) & (d_row > 0)
        valid[i] = False
        idx = np.where(valid)[0]
        if idx.size < k_neighbors:
            continue
        # K nearest by Euclidean distance on pixel grid
        eu = np.sqrt((yx[idx, 0].astype(float) - yx[i, 0]) ** 2
                     + (yx[idx, 1].astype(float) - yx[i, 1]) ** 2)
        order = np.argsort(eu)
        chosen = idx[order[:k_neighbors]]
        gt_px = (d_row[chosen] * float(H)).astype(np.float32)

        pairs.append((
            yx[i].astype(np.int64),
            yx[chosen].astype(np.int64),
            gt_px,
        ))
    return pairs


# -----------------------------------------------------------------------------
# Full-grid multigrid(+tube) differentiable solve (no ROI padding explosion)
# -----------------------------------------------------------------------------
def _pad_to_multiple(x: torch.Tensor, ds: int, value: float = 0.0) -> Tuple[torch.Tensor, int, int]:
    """Pad bottom/right so H,W are multiples of ds."""
    H, W = x.shape[-2], x.shape[-1]
    H_pad = int(math.ceil(H / ds) * ds)
    W_pad = int(math.ceil(W / ds) * ds)
    pad_h = H_pad - H
    pad_w = W_pad - W
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=value)
    return x, H_pad, W_pad


def _maxpool_prob(prob: torch.Tensor, ds: int) -> Tuple[torch.Tensor, int, int]:
    """prob: [B,H,W] -> pooled [B,Hc,Wc], with ceil pooling via padding."""
    if ds <= 1:
        return prob, prob.shape[-2], prob.shape[-1]
    x = prob.unsqueeze(1)  # [B,1,H,W]
    x, H_pad, W_pad = _pad_to_multiple(x, ds, value=0.0)
    y = F.max_pool2d(x, kernel_size=ds, stride=ds).squeeze(1)  # [B,Hc,Wc]
    return y, H_pad, W_pad


def _make_src_mask(B: int, Hc: int, Wc: int, src_yx_c: torch.Tensor, device) -> torch.Tensor:
    m = torch.zeros((B, Hc, Wc), dtype=torch.bool, device=device)
    for b in range(B):
        y = int(src_yx_c[b, 0].item())
        x = int(src_yx_c[b, 1].item())
        y = max(0, min(y, Hc - 1))
        x = max(0, min(x, Wc - 1))
        m[b, y, x] = True
    return m


def _residual_reg(delta_log: torch.Tensor) -> torch.Tensor:
    """L2 + zero-mean constraint: prevents global cost scaling cheats."""
    if delta_log.numel() == 0:
        return delta_log.sum() * 0.0
    return (delta_log ** 2).mean() + (delta_log.mean() ** 2)


def _tv_l1(x: torch.Tensor) -> torch.Tensor:
    """Total variation (L1) on a [B,H,W] or [H,W] tensor."""
    if x.numel() == 0:
        return x.sum() * 0.0
    dx = (x[..., 1:] - x[..., :-1]).abs().mean()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return dx + dy


def _backtrack_path_on_grid(
    T: torch.Tensor,  # [Hc,Wc]
    src_yx: Tuple[int, int],
    tgt_yx: Tuple[int, int],
    max_steps: int = 200000,
) -> List[Tuple[int, int]]:
    """Greedy backtrack along decreasing T (8-neighborhood)."""
    Hc, Wc = T.shape
    sy, sx = src_yx
    ty, tx = tgt_yx
    sy = max(0, min(sy, Hc - 1)); sx = max(0, min(sx, Wc - 1))
    ty = max(0, min(ty, Hc - 1)); tx = max(0, min(tx, Wc - 1))

    path = [(ty, tx)]
    y, x = ty, tx
    for _ in range(max_steps):
        if (y == sy and x == sx):
            break
        y0 = max(0, y - 1); y1 = min(Hc - 1, y + 1)
        x0 = max(0, x - 1); x1 = min(Wc - 1, x + 1)
        neigh = []
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if yy == y and xx == x:
                    continue
                neigh.append((float(T[yy, xx].item()), yy, xx))
        if not neigh:
            break
        neigh.sort(key=lambda t: t[0])
        bestT, by, bx = neigh[0]
        if bestT >= float(T[y, x].item()) + 1e-6:
            break
        y, x = by, bx
        path.append((y, x))
    return path


def fullgrid_multigrid_diff_solve(
    model: SAMRoute,
    road_prob: torch.Tensor,     # [B,H,W]
    src_yx: torch.Tensor,        # [B,2]
    tgt_yx_k: torch.Tensor,      # [B,K,2]
    gt_dist_k: torch.Tensor,     # [B,K] (<=0 padding)
    cfg: EikonalConfig,
    *,
    ds_common: int,
    mg_factor: int,
    mg_iters_coarse: int,
    mg_iters_fine: int,
    mg_detach_coarse: bool,
    mg_interp: str,
    fine_monotone: bool,
    tube_roi: bool,
    tube_min_Pc: int,
    tube_radius_c: int,
    tube_pad_c: int,
    tube_max_area_ratio: float,
    tube_min_side: int,
    tube_iters_floor_full: bool = True,
) -> torch.Tensor:
    """
    Full-image downsampled multigrid(+tube) differentiable solve.
    Domain is the whole image (after max-pool downsample), so no ROI padding explosion.

    Returns:
        pred_dist: [B,K] in pixel units (mix Eikonal with Euclid via gate)
    """
    device = road_prob.device
    B, H, W = road_prob.shape
    K = tgt_yx_k.shape[1]
    large_val = float(cfg.large_val)

    ds_common = max(1, int(ds_common))
    mg_factor = max(1, int(mg_factor))
    ds_coarse = max(1, int(ds_common * mg_factor))

    # -------------------------
    # Build fine grid (downsample ds_common)
    # -------------------------
    prob_f, H_pad_f, W_pad_f = _maxpool_prob(road_prob, ds_common)
    Hf, Wf = prob_f.shape[-2], prob_f.shape[-1]
    cost_f = model._road_prob_to_cost(prob_f).to(dtype=torch.float32)  # [B,Hf,Wf]

    src_c = (src_yx.long() // ds_common).clamp(min=0)
    src_mask_f = _make_src_mask(B, Hf, Wf, src_c, device)

    # targets in fine grid
    tgt_c = (tgt_yx_k.long() // ds_common).clamp(min=0)  # [B,K,2]
    tgt_c[..., 0] = tgt_c[..., 0].clamp(0, Hf - 1)
    tgt_c[..., 1] = tgt_c[..., 1].clamp(0, Wf - 1)

    # -------------------------
    # Build coarse grid (downsample ds_coarse)
    # -------------------------
    prob_c, H_pad_c, W_pad_c = _maxpool_prob(road_prob, ds_coarse)
    Hc, Wc = prob_c.shape[-2], prob_c.shape[-1]
    cost_c = model._road_prob_to_cost(prob_c).to(dtype=torch.float32)

    src_cc = (src_yx.long() // ds_coarse).clamp(min=0)
    src_mask_c = _make_src_mask(B, Hc, Wc, src_cc, device)

    # -------------------------
    # Coarse solve
    # -------------------------
    actual_iters_coarse = int(max(mg_iters_coarse, int(max(Hc, Wc) * 1.5)))
    cfg_c = dc_replace(cfg, h=float(ds_coarse), n_iters=actual_iters_coarse, monotone=True)
    T_c = _eikonal_soft_sweeping_diff(
        cost_c,
        src_mask_c,
        cfg_c,
        checkpoint_chunk=int(getattr(model, "route_ckpt_chunk", 10)),
        gate_alpha=float(getattr(model, "route_gate_alpha", 1.0)),
    )  # [B,Hc,Wc]
    if mg_detach_coarse:
        T_c = T_c.detach()

    # -------------------------
    # Warm-start upsample to fine
    # -------------------------
    if mg_interp == "nearest":
        T_init = F.interpolate(T_c.unsqueeze(1), size=(Hf, Wf), mode="nearest").squeeze(1)
    else:
        T_init = F.interpolate(T_c.unsqueeze(1), size=(Hf, Wf), mode="bilinear", align_corners=False).squeeze(1)

    T_init = T_init.clamp_min(0.0).clamp_max(large_val)
    T_init = torch.where(src_mask_f, torch.zeros_like(T_init), T_init)

    # -------------------------
    # Optional Tube ROI: crop fine refinement domain
    # -------------------------
    use_tube = bool(tube_roi) and (max(Hf, Wf) >= int(tube_min_Pc))
    off_yx = [(0, 0) for _ in range(B)]
    cost_ref = cost_f
    src_ref = src_mask_f
    T_init_ref = T_init

    tube_meta: Dict[str, Any] = {"use_tube": False, "Hf": Hf, "Wf": Wf, "ds": ds_common}

    if use_tube:
        # backtrack on coarse grid T_c for each valid target -> bbox on coarse -> map to fine
        # coarse targets
        tgt_cc = (tgt_yx_k.long() // ds_coarse).clamp(min=0)
        tgt_cc[..., 0] = tgt_cc[..., 0].clamp(0, Hc - 1)
        tgt_cc[..., 1] = tgt_cc[..., 1].clamp(0, Wc - 1)

        radius_c = int(tube_radius_c)
        pad_c = int(tube_pad_c)

        y0_list, y1_list, x0_list, x1_list = [], [], [], []
        for b in range(B):
            # union bbox of all valid target paths
            valid = (gt_dist_k[b] > 0)
            if not valid.any():
                # no valid -> fallback to full
                y0_list.append(0); x0_list.append(0); y1_list.append(Hc); x1_list.append(Wc)
                continue
            sy, sx = int(src_cc[b, 0].item()), int(src_cc[b, 1].item())
            coords = []
            for k in range(K):
                if not bool(valid[k].item()):
                    continue
                ty, tx = int(tgt_cc[b, k, 0].item()), int(tgt_cc[b, k, 1].item())
                path = _backtrack_path_on_grid(T_c[b], (sy, sx), (ty, tx), max_steps=Hc * Wc)
                coords.extend(path)
            if not coords:
                y0_list.append(0); x0_list.append(0); y1_list.append(Hc); x1_list.append(Wc)
                continue
            ys = [p[0] for p in coords]
            xs = [p[1] for p in coords]
            y0 = max(0, min(ys) - radius_c - pad_c)
            y1 = min(Hc, max(ys) + radius_c + pad_c + 1)
            x0 = max(0, min(xs) - radius_c - pad_c)
            x1 = min(Wc, max(xs) + radius_c + pad_c + 1)
            y0_list.append(y0); y1_list.append(y1); x0_list.append(x0); x1_list.append(x1)

        # map coarse bbox -> fine bbox
        # each coarse cell covers mg_factor×mg_factor fine cells
        y0f_list, y1f_list, x0f_list, x1f_list = [], [], [], []
        for b in range(B):
            y0c, y1c, x0c, x1c = y0_list[b], y1_list[b], x0_list[b], x1_list[b]
            y0f = max(0, y0c * mg_factor)
            x0f = max(0, x0c * mg_factor)
            y1f = min(Hf, max(y0f + int(tube_min_side), y1c * mg_factor))
            x1f = min(Wf, max(x0f + int(tube_min_side), x1c * mg_factor))
            y0f_list.append(y0f); y1f_list.append(y1f); x0f_list.append(x0f); x1f_list.append(x1f)

        # compute area ratio, decide enable per-b
        # For simplicity we apply a single bbox = union over B (common in gradcheck B small).
        y0f = int(min(y0f_list)); x0f = int(min(x0f_list))
        y1f = int(max(y1f_list)); x1f = int(max(x1f_list))
        tube_h = y1f - y0f
        tube_w = x1f - x0f
        area_ratio = float(tube_h * tube_w) / float(Hf * Wf + 1e-9)

        if area_ratio >= float(tube_max_area_ratio):
            use_tube = False
        else:
            # crop tensors
            cost_ref = cost_f[:, y0f:y1f, x0f:x1f]
            src_ref = src_mask_f[:, y0f:y1f, x0f:x1f].clone()
            # shift src to cropped coordinates
            for b in range(B):
                off_yx[b] = (y0f, x0f)
            T_init_ref = T_init[:, y0f:y1f, x0f:x1f]
            tube_meta.update({
                "use_tube": True,
                "tube_h": tube_h, "tube_w": tube_w,
                "tube_area_ratio": area_ratio,
                "y0f": y0f, "x0f": x0f,
            })

    # -------------------------
    # Fine refinement
    # -------------------------
    P_for_floor = max(Hf, Wf) if tube_iters_floor_full else (max(cost_ref.shape[-2], cost_ref.shape[-1]))
    actual_iters_fine = int(max(mg_iters_fine, int(P_for_floor * 0.8)))
    cfg_f = dc_replace(cfg, h=float(ds_common), n_iters=actual_iters_fine, monotone=bool(fine_monotone))

    tube_meta["iters_fine"] = actual_iters_fine
    tube_meta["iters_coarse"] = actual_iters_coarse

    T_ref = _eikonal_soft_sweeping_diff_init(
        cost_ref,
        src_ref,
        cfg_f,
        T_init_ref,
        checkpoint_chunk=int(getattr(model, "route_ckpt_chunk", 10)),
        gate_alpha=float(getattr(model, "route_gate_alpha", 1.0)),
    )  # [B, Hr, Wr]

    # -------------------------
    # Read targets (tube-offset aware)
    # -------------------------
    oob_count = 0
    T_eik = []
    for b in range(B):
        yoff, xoff = off_yx[b]
        Tb = T_ref[b]
        Href, Wref = Tb.shape[0], Tb.shape[1]
        d_b = []
        for k in range(K):
            if bool((gt_dist_k[b, k] > 0).item()):
                ty = int(tgt_c[b, k, 0].item()) - yoff
                tx = int(tgt_c[b, k, 1].item()) - xoff
                if ty < 0 or ty >= Href or tx < 0 or tx >= Wref:
                    oob_count += 1
                    d_b.append(torch.tensor(large_val, device=device, dtype=Tb.dtype))
                    continue
                d_b.append(Tb[ty, tx])
            else:
                d_b.append(torch.tensor(large_val, device=device, dtype=Tb.dtype))
        T_eik.append(torch.stack(d_b))
    T_eik = torch.stack(T_eik, dim=0)  # [B,K]

    if use_tube:
        tube_meta["oob_count"] = oob_count
        if oob_count > 0:
            import warnings
            warnings.warn(
                f"[tube] {oob_count} target(s) fell outside tube bbox — "
                f"bbox construction may have an alignment bug"
            )

    # store for debugging
    try:
        model._last_tube_meta = tube_meta
    except Exception:
        pass

    # Euclidean residual blending
    src_f = src_yx.float().unsqueeze(1)   # [B,1,2]
    tgt_f2 = tgt_yx_k.float()             # [B,K,2]
    d_euc = ((src_f - tgt_f2) ** 2).sum(-1).sqrt()  # [B,K]
    pred = _mix_eikonal_euclid(T_eik, d_euc, model.eik_gate_logit)
    return pred


def fullgrid_multigrid_diff_solve_batched(
    model: SAMRoute,
    cost_f: torch.Tensor,        # [1, Hf, Wf] precomputed fine cost
    cost_c: torch.Tensor,        # [1, Hc, Wc] precomputed coarse cost
    src_yx: torch.Tensor,        # [N, 2] pixel coords
    tgt_yx_k: torch.Tensor,      # [N, K, 2]
    gt_dist_k: torch.Tensor,     # [N, K] (<=0 means invalid)
    cfg: EikonalConfig,
    *,
    ds_common: int,
    mg_factor: int,
    mg_iters_coarse: int,
    mg_iters_fine: int,
    mg_detach_coarse: bool,
    mg_interp: str,
    fine_monotone: bool,
    use_compile: bool = False,
    tube_roi: bool = False,
    tube_radius_c: int = 8,
    tube_pad_c: int = 4,
    tube_max_area_ratio: float = 0.90,
    tube_min_side: int = 16,
    tube_min_Pc: int = 256,
) -> torch.Tensor:
    """Batched multigrid solve: N sources sharing the same precomputed cost map.

    Unlike fullgrid_multigrid_diff_solve, this version:
      - Accepts precomputed cost_f / cost_c (avoid recomputing per pair)
      - Processes N source points in a single Eikonal solve (B=N)
      - Optionally uses tube ROI via *union bbox* across all N sources' paths

    Returns:
        pred_dist: [N, K] in pixel units
    """
    device = cost_f.device
    N = src_yx.shape[0]
    K = tgt_yx_k.shape[1]
    Hf, Wf = cost_f.shape[-2], cost_f.shape[-1]
    Hc, Wc = cost_c.shape[-2], cost_c.shape[-1]
    large_val = float(cfg.large_val)
    ds_common = max(1, int(ds_common))
    mg_factor = max(1, int(mg_factor))
    ds_coarse = max(1, ds_common * mg_factor)

    cost_c_n = cost_c.expand(N, -1, -1)

    src_cc = (src_yx.long() // ds_coarse).clamp(min=0)
    src_cc[:, 0] = src_cc[:, 0].clamp(0, Hc - 1)
    src_cc[:, 1] = src_cc[:, 1].clamp(0, Wc - 1)
    src_mask_c = _make_src_mask(N, Hc, Wc, src_cc, device)

    src_fc = (src_yx.long() // ds_common).clamp(min=0)
    src_fc[:, 0] = src_fc[:, 0].clamp(0, Hf - 1)
    src_fc[:, 1] = src_fc[:, 1].clamp(0, Wf - 1)

    tgt_c = (tgt_yx_k.long() // ds_common).clamp(min=0)
    tgt_c[..., 0] = tgt_c[..., 0].clamp(0, Hf - 1)
    tgt_c[..., 1] = tgt_c[..., 1].clamp(0, Wf - 1)

    # --- Coarse solve (always full grid, batched) ---
    actual_iters_coarse = int(max(mg_iters_coarse, int(max(Hc, Wc) * 1.5)))
    cfg_c = dc_replace(cfg, h=float(ds_coarse), n_iters=actual_iters_coarse, monotone=True)
    T_c = _eikonal_soft_sweeping_diff(
        cost_c_n, src_mask_c, cfg_c,
        checkpoint_chunk=int(getattr(model, "route_ckpt_chunk", 10)),
        gate_alpha=float(getattr(model, "route_gate_alpha", 1.0)),
        use_compile=use_compile,
    )
    if mg_detach_coarse:
        T_c = T_c.detach()

    # --- Warm-start upsample ---
    if mg_interp == "nearest":
        T_init = F.interpolate(T_c.unsqueeze(1), size=(Hf, Wf), mode="nearest").squeeze(1)
    else:
        T_init = F.interpolate(
            T_c.unsqueeze(1), size=(Hf, Wf), mode="bilinear", align_corners=False,
        ).squeeze(1)
    T_init = T_init.clamp_min(0.0).clamp_max(large_val)
    src_mask_f = _make_src_mask(N, Hf, Wf, src_fc, device)
    T_init = torch.where(src_mask_f, torch.zeros_like(T_init), T_init)

    # --- Optional tube ROI: union bbox across all N sources ---
    use_tube = bool(tube_roi) and (max(Hf, Wf) >= int(tube_min_Pc))
    y0f, x0f = 0, 0
    cost_ref = cost_f.expand(N, -1, -1)
    src_ref = src_mask_f
    T_init_ref = T_init

    tube_meta: Dict[str, Any] = {
        "use_tube": False, "Hf": Hf, "Wf": Wf, "ds": ds_common,
        "batched_N": N,
    }

    if use_tube:
        tgt_cc = (tgt_yx_k.long() // ds_coarse).clamp(min=0)
        tgt_cc[..., 0] = tgt_cc[..., 0].clamp(0, Hc - 1)
        tgt_cc[..., 1] = tgt_cc[..., 1].clamp(0, Wc - 1)

        T_c_cpu = T_c.detach().cpu()
        all_path_ys: List[int] = []
        all_path_xs: List[int] = []
        for b in range(N):
            sy = int(src_cc[b, 0].item())
            sx = int(src_cc[b, 1].item())
            all_path_ys.append(sy)
            all_path_xs.append(sx)
            for k in range(K):
                if gt_dist_k[b, k].item() <= 0:
                    continue
                ty = int(tgt_cc[b, k, 0].item())
                tx = int(tgt_cc[b, k, 1].item())
                path = _backtrack_path_on_grid(
                    T_c_cpu[b], (sy, sx), (ty, tx), max_steps=Hc * Wc,
                )
                for py, px in path:
                    all_path_ys.append(py)
                    all_path_xs.append(px)

        if all_path_ys:
            rc = int(tube_radius_c)
            pc = int(tube_pad_c)
            y0c = max(0, min(all_path_ys) - rc - pc)
            y1c = min(Hc, max(all_path_ys) + rc + pc + 1)
            x0c = max(0, min(all_path_xs) - rc - pc)
            x1c = min(Wc, max(all_path_xs) + rc + pc + 1)

            y0f = max(0, y0c * mg_factor)
            x0f = max(0, x0c * mg_factor)
            y1f = min(Hf, max(y0f + int(tube_min_side), y1c * mg_factor))
            x1f = min(Wf, max(x0f + int(tube_min_side), x1c * mg_factor))

            tube_h = y1f - y0f
            tube_w = x1f - x0f
            area_ratio = float(tube_h * tube_w) / float(Hf * Wf + 1e-9)

            if area_ratio < float(tube_max_area_ratio):
                cost_ref = cost_f[:, y0f:y1f, x0f:x1f].expand(N, -1, -1)
                src_ref = src_mask_f[:, y0f:y1f, x0f:x1f].clone()
                T_init_ref = T_init[:, y0f:y1f, x0f:x1f]
                use_tube = True
                tube_meta.update({
                    "use_tube": True,
                    "tube_h": tube_h, "tube_w": tube_w,
                    "tube_area_ratio": area_ratio,
                    "y0f": y0f, "x0f": x0f,
                })
            else:
                use_tube = False

    # --- Fine refinement ---
    Hrf, Wrf = cost_ref.shape[-2], cost_ref.shape[-1]
    actual_iters_fine = int(max(mg_iters_fine, int(max(Hrf, Wrf) * 0.8)))
    cfg_f = dc_replace(cfg, h=float(ds_common), n_iters=actual_iters_fine, monotone=bool(fine_monotone))
    T_ref = _eikonal_soft_sweeping_diff_init(
        cost_ref, src_ref, cfg_f, T_init_ref,
        checkpoint_chunk=int(getattr(model, "route_ckpt_chunk", 10)),
        gate_alpha=float(getattr(model, "route_gate_alpha", 1.0)),
        use_compile=use_compile,
    )

    # --- Read target distances (tube-offset aware) ---
    oob_count = 0
    b_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, K)
    if use_tube:
        ty_local = tgt_c[..., 0] - y0f
        tx_local = tgt_c[..., 1] - x0f
        oob = (ty_local < 0) | (ty_local >= Hrf) | (tx_local < 0) | (tx_local >= Wrf)
        oob_count = int(oob.sum().item())
        ty_local = ty_local.clamp(0, Hrf - 1)
        tx_local = tx_local.clamp(0, Wrf - 1)
        T_eik = T_ref[b_idx, ty_local, tx_local]
        T_eik = torch.where(oob, torch.full_like(T_eik, large_val), T_eik)
    else:
        T_eik = T_ref[b_idx, tgt_c[..., 0], tgt_c[..., 1]]

    invalid = gt_dist_k <= 0
    T_eik = torch.where(invalid, torch.full_like(T_eik, large_val), T_eik)

    # --- Euclidean blending ---
    src_f = src_yx.float().unsqueeze(1)
    tgt_f2 = tgt_yx_k.float()
    d_euc = ((src_f - tgt_f2) ** 2).sum(-1).sqrt()
    pred = _mix_eikonal_euclid(T_eik, d_euc, model.eik_gate_logit)

    tube_meta["iters_fine"] = actual_iters_fine
    tube_meta["iters_coarse"] = actual_iters_coarse
    tube_meta["oob_count"] = oob_count
    try:
        model._last_tube_meta = tube_meta
    except Exception:
        pass

    return pred


# -----------------------------------------------------------------------------
# Existing utilities from v2 script (kept minimal)
# -----------------------------------------------------------------------------
def _load_lightning_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    clean = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    return clean


def _detect_smooth_decoder(state_dict: dict) -> bool:
    w = state_dict.get("map_decoder.7.weight")
    if w is None:
        return False
    return w.shape[1] == 32


def _detect_patch_size(state_dict: dict) -> "int | None":
    pe = state_dict.get("image_encoder.pos_embed")
    if pe is None:
        return None
    return int(pe.shape[1]) * 16


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _first_batch_with_dist(loader) -> Dict[str, Any]:
    for batch in loader:
        if "gt_dist" in batch:
            return batch
    raise RuntimeError("No batch with gt_dist found. Check include_dist=True and dataset.")


def _choose_first_valid_pair(gt_dist: torch.Tensor) -> Tuple[int, int]:
    # gt_dist: [B,K]
    idx = (gt_dist > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return -1, -1
    b, k = int(idx[0, 0].item()), int(idx[0, 1].item())
    return b, k


def _grad_norm(x: Optional[torch.Tensor]) -> float:
    if x is None:
        return 0.0
    if not torch.isfinite(x).all():
        return float("nan")
    return float(x.norm().item())


def _apply_cap(
    pred_dist: torch.Tensor,
    gt_dist: torch.Tensor,
    norm: float,
    cap_mult: float,
    cap_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cap = norm * float(cap_mult)
    cap_mode = str(cap_mode).lower()

    if cap_mode == "log":
        pred_in = torch.log1p(pred_dist / norm)
        gt_in = torch.log1p(gt_dist / norm)
        sat = torch.zeros_like(pred_dist, dtype=torch.bool)
        return pred_in, gt_in, sat

    if cap_mode == "none":
        pred_c = pred_dist
        sat = torch.zeros_like(pred_dist, dtype=torch.bool)
    elif cap_mode == "tanh":
        pred_c = cap * torch.tanh(pred_dist / cap)
        sat = pred_dist > cap
    else:
        pred_c = pred_dist.clamp(max=cap)
        sat = pred_dist > cap

    return pred_c / norm, gt_dist / norm, sat


# -----------------------------------------------------------------------------
# Visualization (optional): overlay path on full image
# -----------------------------------------------------------------------------
def _save_route_overlay(
    out_path: str,
    rgb: np.ndarray,            # [H,W,3] uint8
    road_prob: np.ndarray,      # [H,W] float
    src_yx: Tuple[int, int],
    tgt_yx: Tuple[int, int],
    path_xy: List[Tuple[int, int]],  # list of (y,x) on full-res pixels
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    H, W = rgb.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb)
    ax.imshow(road_prob, cmap="magma", vmin=0, vmax=1, alpha=0.35)
    if path_xy:
        ys = [p[0] for p in path_xy]
        xs = [p[1] for p in path_xy]
        ax.plot(xs, ys, linewidth=2)
    ax.scatter([src_yx[1]], [src_yx[0]], s=60, marker="o")
    ax.scatter([tgt_yx[1]], [tgt_yx[0]], s=60, marker="x")
    ax.set_title(title)
    ax.set_xlim([0, W - 1]); ax.set_ylim([H - 1, 0])
    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _hard_eval_route_fullgrid(
    model: SAMRoute,
    road_prob: torch.Tensor,  # [1,H,W]
    src_yx: Tuple[int, int],
    tgt_yx: Tuple[int, int],
    *,
    ds: int,
    n_iters: int,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Hard-eval Eikonal on full downsampled grid; return (dist_px, path_fullres)."""
    device = road_prob.device
    B, H, W = road_prob.shape
    assert B == 1
    ds = max(1, int(ds))
    # downsample
    prob_f, _, _ = _maxpool_prob(road_prob, ds)  # [1,Hf,Wf]
    Hf, Wf = prob_f.shape[-2], prob_f.shape[-1]
    cost = model._road_prob_to_cost(prob_f).to(dtype=torch.float32)

    sy, sx = src_yx
    ty, tx = tgt_yx
    src_c = (sy // ds, sx // ds)
    tgt_c = (ty // ds, tx // ds)
    src_mask = torch.zeros((1, Hf, Wf), dtype=torch.bool, device=device)
    src_mask[0, max(0, min(src_c[0], Hf - 1)), max(0, min(src_c[1], Wf - 1))] = True

    cfg = dc_replace(model.route_eik_cfg, h=float(ds), n_iters=int(n_iters), mode="hard_eval", monotone=True)
    T = eikonal_soft_sweeping(cost, src_mask, cfg)  # [1,Hf,Wf]
    dist = float(T[0, max(0, min(tgt_c[0], Hf - 1)), max(0, min(tgt_c[1], Wf - 1))].item())

    path_c = _backtrack_path_on_grid(T[0], src_c, tgt_c, max_steps=Hf * Wf)
    # map to full-res pixels (center of cells)
    path_full = [(int(y * ds), int(x * ds)) for (y, x) in path_c]
    return dist, path_full


# -----------------------------------------------------------------------------
# TSP multi-anchor training & evaluation
# -----------------------------------------------------------------------------

def _tsp_evaluate(
    model: SAMRoute,
    road_prob_t: torch.Tensor,
    coords: np.ndarray,
    udist: np.ndarray,
    H: int,
    W: int,
    case_indices: List[int],
    args,
    cfg,
    device: torch.device,
    tag: str = "eval",
) -> Dict[str, float]:
    """Evaluate model distance predictions over multiple TSP cases (no_grad).

    Returns dict with aggregate metrics.
    """
    model.eval()
    K = int(args.tsp_k_neighbors)
    norm = float(getattr(cfg, "ROUTE_DIST_NORM_PX", 512.0))
    eps = float(args.prob_eps)

    mg_itc = int(args.mg_iters_coarse) if int(args.mg_iters_coarse) > 0 else max(20, int(args.eik_iters * 0.25))
    mg_itf = int(args.mg_iters_fine)   if int(args.mg_iters_fine)   > 0 else max(20, int(args.eik_iters * 0.80))

    all_mae: List[float] = []
    all_rel: List[float] = []
    all_gt:  List[float] = []
    n_oob_total = 0

    rp_in = road_prob_t.clamp(eps, 1.0 - eps)

    ds = int(args.downsample)
    if args.multigrid:
        mg_f = int(args.mg_factor)
        ds_coarse = ds * mg_f
        prob_f, _, _ = _maxpool_prob(rp_in, ds)
        Hf, Wf = prob_f.shape[-2], prob_f.shape[-1]
        cost_f = model._road_prob_to_cost(prob_f).to(dtype=torch.float32)
        prob_c, _, _ = _maxpool_prob(rp_in, ds_coarse)
        cost_c = model._road_prob_to_cost(prob_c).to(dtype=torch.float32)
    else:
        prob_f, _, _ = _maxpool_prob(rp_in, ds)
        Hf, Wf = prob_f.shape[-2], prob_f.shape[-1]
        cost_f = model._road_prob_to_cost(prob_f).to(dtype=torch.float32)

    for ci_pos, ci in enumerate(case_indices):
        pairs = _prepare_all_node_pairs(coords, udist, H, W, ci, K)
        if not pairs:
            if (ci_pos + 1) % max(1, args.tsp_log_interval) == 0:
                print(f"  [{tag}] {ci_pos+1}/{len(case_indices)} cases done ...")
            continue

        N_p = len(pairs)
        src_all = torch.from_numpy(np.stack([p[0] for p in pairs])).to(device, torch.long)
        tgt_all = torch.from_numpy(np.stack([p[1] for p in pairs])).to(device, torch.long)
        gt_all_np = np.stack([p[2] for p in pairs])
        gt_t = torch.from_numpy(gt_all_np).to(device, torch.float32)

        if args.multigrid:
            pred = fullgrid_multigrid_diff_solve_batched(
                model, cost_f, cost_c, src_all, tgt_all, gt_t, model.route_eik_cfg,
                ds_common=ds, mg_factor=mg_f,
                mg_iters_coarse=mg_itc, mg_iters_fine=mg_itf,
                mg_detach_coarse=True, mg_interp=str(args.mg_interp),
                fine_monotone=bool(args.mg_fine_monotone),
                use_compile=bool(getattr(args, "compile_eikonal", False)),
                tube_roi=bool(args.tube_roi),
                tube_radius_c=int(args.tube_radius_c),
                tube_pad_c=int(args.tube_pad_c),
                tube_max_area_ratio=float(args.tube_max_area_ratio),
                tube_min_side=int(args.tube_min_side),
                tube_min_Pc=int(args.tube_min_pc),
            )
        else:
            cost_n = cost_f.expand(N_p, -1, -1)
            src_c = (src_all // ds).clamp(0, max(Hf - 1, 0))
            src_mask = _make_src_mask(N_p, Hf, Wf, src_c, device)
            cfg_f = dc_replace(model.route_eik_cfg, h=float(ds), n_iters=int(args.eik_iters))
            T = _eikonal_soft_sweeping_diff(
                cost_n, src_mask, cfg_f,
                checkpoint_chunk=int(getattr(model, "route_ckpt_chunk", 10)),
                gate_alpha=float(getattr(model, "route_gate_alpha", 1.0)),
            )
            tgt_c = (tgt_all // ds).clamp(min=0)
            tgt_c[..., 0] = tgt_c[..., 0].clamp(0, Hf - 1)
            tgt_c[..., 1] = tgt_c[..., 1].clamp(0, Wf - 1)
            b_idx = torch.arange(N_p, device=device).unsqueeze(1).expand(-1, K)
            Te = T[b_idx, tgt_c[..., 0], tgt_c[..., 1]]
            d_euc = ((src_all.float().unsqueeze(1) - tgt_all.float()) ** 2).sum(-1).sqrt()
            pred = _mix_eikonal_euclid(Te, d_euc, model.eik_gate_logit)

        pred_np = pred.detach().cpu().numpy()
        valid = gt_all_np > 0
        if valid.any():
            err = np.abs(pred_np[valid] - gt_all_np[valid])
            rel = err / np.maximum(gt_all_np[valid], 1.0)
            all_mae.extend(err.tolist())
            all_rel.extend(rel.tolist())
            all_gt.extend(gt_all_np[valid].tolist())

        meta = getattr(model, "_last_tube_meta", {})
        n_oob_total += int(meta.get("oob_count", 0))

        if (ci_pos + 1) % max(1, args.tsp_log_interval) == 0:
            print(f"  [{tag}] {ci_pos+1}/{len(case_indices)} cases done ...")

    if not all_mae:
        print(f"  [{tag}] WARNING: no valid predictions!")
        return {}

    mae_arr = np.array(all_mae)
    rel_arr = np.array(all_rel)
    gt_arr  = np.array(all_gt)

    # per-distance-bin breakdown
    bins = {"short(<500px)": gt_arr < 500,
            "medium(500-2000px)": (gt_arr >= 500) & (gt_arr < 2000),
            "long(>2000px)": gt_arr >= 2000}

    results: Dict[str, float] = {
        "n_pairs": len(mae_arr),
        "mae_mean": float(np.mean(mae_arr)),
        "mae_median": float(np.median(mae_arr)),
        "mae_p90": float(np.percentile(mae_arr, 90)),
        "rel_mean": float(np.mean(rel_arr)),
        "rel_median": float(np.median(rel_arr)),
        "oob_total": n_oob_total,
    }
    for bname, bmask in bins.items():
        if bmask.any():
            results[f"mae_{bname}"] = float(np.mean(mae_arr[bmask]))
            results[f"rel_{bname}"] = float(np.mean(rel_arr[bmask]))
            results[f"n_{bname}"] = int(bmask.sum())

    print(f"\n  [{tag}] ====== Evaluation Summary ({len(case_indices)} cases, {len(mae_arr)} pairs) ======")
    print(f"    MAE  : mean={results['mae_mean']:.1f}px  median={results['mae_median']:.1f}px  p90={results['mae_p90']:.1f}px")
    print(f"    RelErr: mean={results['rel_mean']:.4f}  median={results['rel_median']:.4f}")
    for bname in bins:
        if f"mae_{bname}" in results:
            print(f"    {bname:>20s}: n={results[f'n_{bname}']:5d}  MAE={results[f'mae_{bname}']:.1f}px  RelErr={results[f'rel_{bname}']:.4f}")
    if n_oob_total > 0:
        print(f"    OOB targets: {n_oob_total}")
    print()

    return results


def _run_tsp_train(
    model: SAMRoute,
    road_prob_np: np.ndarray,
    rgb: np.ndarray,
    cfg,
    args,
    device: torch.device,
):
    """TSP multi-anchor training: iterate over NPZ cases, train on distance loss."""
    print("\n[mode] TSP multi-anchor training")

    H, W = road_prob_np.shape[:2]
    eps = float(args.prob_eps)

    # --- load NPZ ---
    npz_path = args.npz
    if not npz_path:
        tif_name = os.path.basename(args.tif)
        location_key = tif_name.replace("crop_", "").replace("_z16.tif", "").replace(".tif", "")
        npz_name = f"distance_dataset_all_{location_key}_p{int(args.p_count)}.npz"
        npz_path = os.path.join(os.path.dirname(args.tif), npz_name)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    coords, udist, H_npz, W_npz = _load_npz_nodes(npz_path)
    N_cases = coords.shape[0]
    print(f"[npz] {npz_path}  cases={N_cases}  nodes_per_case={coords.shape[1]}  H={H_npz} W={W_npz}")
    if H_npz != H or W_npz != W:
        print(f"  [warn] NPZ meta H/W={H_npz}/{W_npz} != tif H/W={H}/{W}")

    # --- split train / eval ---
    rng = np.random.RandomState(args.seed)
    all_indices = rng.permutation(N_cases).tolist()

    n_train = min(int(args.tsp_n_train), N_cases)
    n_eval  = int(args.tsp_n_eval) if int(args.tsp_n_eval) > 0 else N_cases
    n_eval  = min(n_eval, N_cases)

    train_cases = all_indices[:n_train]
    # eval from the end to avoid overlap when possible
    eval_cases  = all_indices[n_train:n_train + n_eval] if (n_train + n_eval <= N_cases) else all_indices[:n_eval]
    print(f"[split] train={len(train_cases)} cases  eval={len(eval_cases)} cases  "
          f"overlap={len(set(train_cases) & set(eval_cases))}")

    # --- build road_prob tensor ---
    road_prob_t = torch.from_numpy(road_prob_np).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
    road_prob_t = road_prob_t.clamp(eps, 1.0 - eps)

    # --- optimizer ---
    model.train()
    if args.freeze_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    # --- load routing checkpoint if provided ---
    if getattr(args, "tsp_load_ckpt", "") and os.path.isfile(args.tsp_load_ckpt):
        rckpt = torch.load(args.tsp_load_ckpt, map_location=device, weights_only=False)
        with torch.no_grad():
            if "cost_log_alpha" in rckpt:
                model.cost_log_alpha.copy_(rckpt["cost_log_alpha"].to(device))
            if "cost_log_gamma" in rckpt:
                model.cost_log_gamma.copy_(rckpt["cost_log_gamma"].to(device))
            if "eik_gate_logit" in rckpt:
                model.eik_gate_logit.copy_(rckpt["eik_gate_logit"].to(device))
        if "cost_net_state_dict" in rckpt and getattr(model, "cost_net", None) is not None:
            model.cost_net.load_state_dict(rckpt["cost_net_state_dict"])
        if "decoder_state_dict" in rckpt and hasattr(model, "map_decoder"):
            model.map_decoder.load_state_dict(rckpt["decoder_state_dict"])
        print(f"[resumed] routing params from {args.tsp_load_ckpt}")

    # --- online decoder: cache encoder features ---
    online_decoder = getattr(args, "online_decoder", False)
    enc_patches_info = None
    enc_features = None
    enc_win2d = None
    if online_decoder:
        print("[online_decoder] Caching encoder features for all patches ...")
        enc_patches_info, enc_features, enc_win2d = _cache_encoder_features(
            rgb, model, device,
            patch_size=int(cfg.PATCH_SIZE),
            stride=int(args.stride),
        )
        print(f"  Cached {len(enc_features)} patches of encoder features")
        if hasattr(model, "map_decoder"):
            for p in model.map_decoder.parameters():
                p.requires_grad = True
            print(f"  Decoder unfrozen for distance-driven fine-tuning")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    print(f"Trainable params: {sum(p.numel() for p in params)}")

    with torch.no_grad():
        _a = model.cost_log_alpha.clamp(math.log(5.0), math.log(100.0)).exp()
        _g = model.cost_log_gamma.clamp(math.log(0.5), math.log(4.0)).exp()
        _gate = torch.sigmoid(model.eik_gate_logit).clamp(0.3, 0.95)
        print(f"[init] cost_alpha={_a.item():.3f}  cost_gamma={_g.item():.3f}  "
              f"eik_gate={_gate.item():.4f}")

    K = int(args.tsp_k_neighbors)
    norm = float(getattr(cfg, "ROUTE_DIST_NORM_PX", 512.0))
    mg_itc = int(args.mg_iters_coarse) if int(args.mg_iters_coarse) > 0 else max(20, int(args.eik_iters * 0.25))
    mg_itf = int(args.mg_iters_fine)   if int(args.mg_iters_fine)   > 0 else max(20, int(args.eik_iters * 0.80))
    out_dir = args.save_debug or "/tmp/route_tsp_train"
    os.makedirs(out_dir, exist_ok=True)

    # --- optional baseline eval ---
    baseline_metrics: Dict[str, float] = {}
    if args.tsp_eval_before:
        print("\n===== Baseline Evaluation (before training) =====")
        with torch.no_grad():
            baseline_metrics = _tsp_evaluate(
                model, road_prob_t, coords, udist, H, W,
                eval_cases, args, cfg, device, tag="baseline",
            )

    # --- training loop ---
    print(f"\n===== TSP Training: {len(train_cases)} cases x {args.tsp_epochs} epochs =====")
    if online_decoder:
        print(f"  [online_decoder] recompute road_prob every {args.decoder_grad_every} cases")

    rp_in = road_prob_t.clamp(eps, 1.0 - eps)

    ds = int(args.downsample)
    mg_f = int(args.mg_factor)
    ds_coarse_t = ds * mg_f
    with torch.no_grad():
        prob_f_cached, _, _ = _maxpool_prob(rp_in, ds)
        Hf_cached, Wf_cached = prob_f_cached.shape[-2], prob_f_cached.shape[-1]
        if args.multigrid:
            prob_c_cached, _, _ = _maxpool_prob(rp_in, ds_coarse_t)

    for epoch in range(int(args.tsp_epochs)):
        epoch_losses: List[float] = []
        epoch_maes:   List[float] = []
        t_epoch_start = time.perf_counter()

        case_order = train_cases.copy()
        rng.shuffle(case_order)

        _need_rp_refresh = True  # force first recompute at epoch start

        for ci_pos, ci in enumerate(case_order):
            opt.zero_grad(set_to_none=True)

            # online_decoder: periodically recompute road_prob through decoder
            if online_decoder and _need_rp_refresh:
                rp_online = _road_prob_from_cached_features(
                    model, enc_patches_info, enc_features, enc_win2d,
                    H, W, device,
                )
                rp_online = rp_online.clamp(eps, 1.0 - eps)
                prob_f_cached, _, _ = _maxpool_prob(rp_online, ds)
                Hf_cached, Wf_cached = prob_f_cached.shape[-2], prob_f_cached.shape[-1]
                if args.multigrid:
                    prob_c_cached, _, _ = _maxpool_prob(rp_online, ds_coarse_t)
                _need_rp_refresh = False

            pairs = _prepare_all_node_pairs(coords, udist, H, W, ci, K)
            if not pairs:
                continue

            n_pairs = len(pairs)
            model.train()

            src_all = torch.from_numpy(np.stack([p[0] for p in pairs])).to(device, torch.long)
            tgt_all = torch.from_numpy(np.stack([p[1] for p in pairs])).to(device, torch.long)
            gt_all_np = np.stack([p[2] for p in pairs])
            gt_t = torch.from_numpy(gt_all_np).to(device, torch.float32)

            if args.multigrid:
                cost_f = model._road_prob_to_cost(prob_f_cached).to(dtype=torch.float32)
                cost_c = model._road_prob_to_cost(prob_c_cached).to(dtype=torch.float32)
                pred = fullgrid_multigrid_diff_solve_batched(
                    model, cost_f, cost_c, src_all, tgt_all, gt_t, model.route_eik_cfg,
                    ds_common=ds, mg_factor=mg_f,
                    mg_iters_coarse=mg_itc, mg_iters_fine=mg_itf,
                    mg_detach_coarse=bool(args.mg_detach_coarse),
                    mg_interp=str(args.mg_interp),
                    fine_monotone=bool(args.mg_fine_monotone),
                    use_compile=bool(getattr(args, "compile_eikonal", False)),
                    tube_roi=bool(args.tube_roi),
                    tube_radius_c=int(args.tube_radius_c),
                    tube_pad_c=int(args.tube_pad_c),
                    tube_max_area_ratio=float(args.tube_max_area_ratio),
                    tube_min_side=int(args.tube_min_side),
                    tube_min_Pc=int(args.tube_min_pc),
                )
            else:
                cost_f = model._road_prob_to_cost(prob_f_cached).to(dtype=torch.float32)
                cost_n = cost_f.expand(n_pairs, -1, -1)
                src_c = (src_all // ds).clamp(0, max(Hf_cached - 1, 0))
                src_mask = _make_src_mask(n_pairs, Hf_cached, Wf_cached, src_c, device)
                cfg_f = dc_replace(model.route_eik_cfg, h=float(ds), n_iters=int(args.eik_iters))
                T = _eikonal_soft_sweeping_diff(
                    cost_n, src_mask, cfg_f,
                    checkpoint_chunk=int(getattr(model, "route_ckpt_chunk", 10)),
                    gate_alpha=float(getattr(model, "route_gate_alpha", 1.0)),
                )
                tgt_c = (tgt_all // ds).clamp(min=0)
                tgt_c[..., 0] = tgt_c[..., 0].clamp(0, Hf_cached - 1)
                tgt_c[..., 1] = tgt_c[..., 1].clamp(0, Wf_cached - 1)
                b_idx = torch.arange(n_pairs, device=device).unsqueeze(1).expand(-1, K)
                Te = T[b_idx, tgt_c[..., 0], tgt_c[..., 1]]
                d_euc = ((src_all.float().unsqueeze(1) - tgt_all.float()) ** 2).sum(-1).sqrt()
                pred = _mix_eikonal_euclid(Te, d_euc, model.eik_gate_logit)

            valid = gt_t > 0
            n_valid_nodes = 0
            if valid.any():
                pred_in, gt_in, _sat = _apply_cap(pred, gt_t, norm, args.cap_mult, args.cap_mode)
                loss_case = model.dist_criterion(pred_in[valid], gt_in[valid])

                # cost_net regularization
                if getattr(model, "cost_net", None) is not None:
                    delta_f = model._cost_net_delta_log(prob_f_cached)
                    if delta_f is not None:
                        loss_case = loss_case + float(args.lambda_cost_reg) * _residual_reg(delta_f)
                        if float(args.lambda_cost_tv) > 0:
                            loss_case = loss_case + float(args.lambda_cost_tv) * _tv_l1(delta_f)
                    if bool(args.reg_on_coarse) and args.multigrid:
                        delta_c = model._cost_net_delta_log(prob_c_cached)
                        if delta_c is not None:
                            loss_case = loss_case + float(args.lambda_cost_reg) * _residual_reg(delta_c)

                loss_case.backward()
                with torch.no_grad():
                    case_mae = float((pred[valid] - gt_t[valid]).abs().mean().item())
                epoch_losses.append(float(loss_case.item()))
                epoch_maes.append(case_mae)
                n_valid_nodes = 1

            # after backward, detach cached probs (graph is freed)
            if online_decoder:
                prob_f_cached = prob_f_cached.detach()
                if args.multigrid:
                    prob_c_cached = prob_c_cached.detach()
                if (ci_pos + 1) % max(1, int(args.decoder_grad_every)) == 0:
                    _need_rp_refresh = True

            # gradient guard
            has_bad = False
            for p in params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_bad = True
                    break
            if has_bad:
                print(f"  [epoch {epoch}] case {ci}: NaN/Inf gradient — skipping step")
            else:
                opt.step()

            if (ci_pos + 1) % max(1, args.tsp_log_interval) == 0:
                recent_loss = np.mean(epoch_losses[-args.tsp_log_interval:]) if epoch_losses else 0
                recent_mae  = np.mean(epoch_maes[-args.tsp_log_interval:]) if epoch_maes else 0
                print(f"  [epoch {epoch}] {ci_pos+1}/{len(case_order)} cases  "
                      f"recent_loss={recent_loss:.6f}  recent_MAE={recent_mae:.1f}px")

        t_epoch = time.perf_counter() - t_epoch_start
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0
        mean_mae  = float(np.mean(epoch_maes)) if epoch_maes else 0
        with torch.no_grad():
            _a = model.cost_log_alpha.clamp(math.log(5.0), math.log(100.0)).exp()
            _g = model.cost_log_gamma.clamp(math.log(0.5), math.log(4.0)).exp()
            _gate = torch.sigmoid(model.eik_gate_logit).clamp(0.3, 0.95)
        print(f"\n  [epoch {epoch}] DONE  time={t_epoch:.1f}s  avg_loss={mean_loss:.6f}  "
              f"avg_MAE={mean_mae:.1f}px")
        print(f"  [params] cost_alpha={_a.item():.3f}  cost_gamma={_g.item():.3f}  "
              f"eik_gate={_gate.item():.4f}")

    # --- final evaluation ---
    print(f"\n===== Final Evaluation ({len(eval_cases)} cases) =====")
    eval_rp = road_prob_t
    if online_decoder:
        print("  [online_decoder] Recomputing road_prob through updated decoder for final eval...")
        with torch.no_grad():
            eval_rp = _road_prob_from_cached_features(
                model, enc_patches_info, enc_features, enc_win2d,
                H, W, device,
            )
    with torch.no_grad():
        final_metrics = _tsp_evaluate(
            model, eval_rp, coords, udist, H, W,
            eval_cases, args, cfg, device, tag="final",
        )

    # --- save checkpoint ---
    if args.tsp_save_ckpt:
        ckpt_dir = os.path.dirname(args.tsp_save_ckpt)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        save_dict = {
            "cost_log_alpha": model.cost_log_alpha.detach().cpu(),
            "cost_log_gamma": model.cost_log_gamma.detach().cpu(),
            "eik_gate_logit": model.eik_gate_logit.detach().cpu(),
            "args": vars(args),
            "final_metrics": final_metrics,
        }
        if getattr(model, "cost_net", None) is not None:
            save_dict["cost_net_state_dict"] = {
                k: v.detach().cpu() for k, v in model.cost_net.state_dict().items()
            }
        if getattr(args, "online_decoder", False) and hasattr(model, "map_decoder"):
            save_dict["decoder_state_dict"] = {
                k: v.detach().cpu() for k, v in model.map_decoder.state_dict().items()
            }
        torch.save(save_dict, args.tsp_save_ckpt)
        print(f"[saved] routing checkpoint: {args.tsp_save_ckpt}")

    # --- save metrics summary ---
    summary_path = os.path.join(out_dir, "tsp_metrics.txt")
    with open(summary_path, "w") as f:
        if baseline_metrics:
            f.write("=== BASELINE ===\n")
            for k, v in baseline_metrics.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write("=== AFTER TRAINING ===\n")
        for k, v in final_metrics.items():
            f.write(f"  {k}: {v}\n")
    print(f"[saved] {summary_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # --- common ---
    ap.add_argument("--ckpt", type=str, default=_CKPT_DEFAULT, help="SAMRoute lightning ckpt (.ckpt)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_debug", type=str, default="", help="output dir to save debug images")

    # solver / loss knobs
    ap.add_argument("--eik_iters", type=int, default=200)
    ap.add_argument("--downsample", type=int, default=48)
    ap.add_argument("--eik_mode", type=str, default="soft_train",
                    choices=["ste_train", "soft_train", "hard_eval"])
    ap.add_argument("--cap_mode", type=str, default="tanh", choices=["clamp", "tanh", "log", "none"])
    ap.add_argument("--cap_mult", type=float, default=10.0)
    ap.add_argument("--lambda_seg", type=float, default=0.0)
    ap.add_argument("--lambda_dist", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--profile_time", action="store_true")
    ap.add_argument("--gate_alpha", type=float, default=0.8,
                    help="GPPN residual gate alpha (0.8 recommended for multigrid)")

    # cost net
    ap.add_argument("--cost_net", action=argparse.BooleanOptionalAction, default=False,
                    help="Enable ResidualCostNet in prob->cost mapping")
    ap.add_argument("--cost_net_ch", type=int, default=8,
                    help="ResidualCostNet hidden channels")
    ap.add_argument("--cost_net_use_coord", action="store_true",
                    help="Append (y,x) coord channels to cost net input")
    ap.add_argument("--cost_net_delta_scale", type=float, default=0.75,
                    help="Scale of bounded log-residual (tanh * scale)")
    ap.add_argument("--lambda_cost_reg", type=float, default=1e-3,
                    help="L2/mean regularization on cost residual (prevents global cheating)")
    ap.add_argument("--lambda_cost_tv", type=float, default=0.0,
                    help="TV regularization on cost residual (optional smoothing)")
    ap.add_argument("--reg_on_coarse", action="store_true",
                    help="Also regularize coarse-grid cost residual")

    # multigrid / tube
    ap.add_argument("--multigrid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mg_factor", type=int, default=4)
    ap.add_argument("--mg_iters_coarse", type=int, default=0)
    ap.add_argument("--mg_iters_fine", type=int, default=0)
    ap.add_argument("--mg_detach_coarse", action="store_true")
    ap.add_argument("--mg_interp", type=str, default="bilinear", choices=["nearest", "bilinear"])
    ap.add_argument("--mg_fine_monotone", action="store_true")

    ap.add_argument("--tube_roi", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tube_min_pc", type=int, default=256)
    ap.add_argument("--tube_radius_c", type=int, default=8)
    ap.add_argument("--tube_pad_c", type=int, default=4)
    ap.add_argument("--tube_max_area_ratio", type=float, default=0.90)
    ap.add_argument("--tube_min_side", type=int, default=16)

    ap.add_argument("--tube_iters_floor_full", action="store_true",
                    help="Fine iters floor uses full grid size (recommended for training); "
                         "if off, uses tube bbox size (faster but may slow convergence).")

    # --- dataset/ROI mode (original) ---
    ap.add_argument("--data_root", type=str, default=_DATA_ROOT_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--npz_variant", type=str, default="p20")
    ap.add_argument("--k_targets", type=int, default=4)
    ap.add_argument("--samples_per_region", type=int, default=10)
    ap.add_argument("--preload_to_ram", action="store_true")

    # --- fullmap mode ---
    ap.add_argument("--tif", type=str, default="", help="If set, run full-map mode on this GeoTIFF")
    ap.add_argument("--npz", type=str, default="", help="Optional NPZ for GT distances; if empty, try auto-detect near tif")
    ap.add_argument("--p_count", type=int, default=20, help="NPZ problem size (20/50) for auto-detect")
    ap.add_argument("--case_idx", type=int, default=0, help="NPZ case idx when matched_node_norm is 3D")
    ap.add_argument("--stride", type=int, default=256, help="Sliding stride for full-image segmentation")
    ap.add_argument("--smooth_sigma", type=float, default=0.0)
    ap.add_argument("--cache_prob", type=str, default="", help="Path to .npy for caching full road_prob")
    ap.add_argument("--src", type=str, default="", help="src 'y,x' in full image coordinates")
    ap.add_argument("--tgt", type=str, default="", help="tgt 'y,x' in full image coordinates (single target)")
    ap.add_argument("--sample_from_npz", action=argparse.BooleanOptionalAction, default=True,
                    help="Sample src + K targets from NPZ nodes instead of --src/--tgt")
    ap.add_argument("--min_gt_dist_px", type=float, default=50.0)

    ap.add_argument("--teacher_hard_eval", action="store_true",
                    help="If no GT provided, use hard_eval distance (high iters) as pseudo-GT")
    ap.add_argument("--teacher_iters", type=int, default=600)
    ap.add_argument("--vis_full_route", action="store_true",
                    help="Save full-image route overlay before/after (hard_eval on downsampled grid)")
    ap.add_argument("--vis_ds", type=int, default=4)
    ap.add_argument("--vis_iters", type=int, default=600)

    ap.add_argument("--optimize_prob", action="store_true",
                    help="Treat full road_prob as learnable variable (not the segmentation network) and optimize it.")
    ap.add_argument("--prob_eps", type=float, default=1e-6)

    # --- TSP multi-anchor training ---
    ap.add_argument("--tsp_train", action=argparse.BooleanOptionalAction, default=True,
                    help="TSP multi-anchor training: iterate over NPZ cases, "
                         "use all 20 nodes as src with K nearest neighbors as targets")
    ap.add_argument("--tsp_n_train", type=int, default=50,
                    help="Number of NPZ cases to train on")
    ap.add_argument("--tsp_n_eval", type=int, default=100,
                    help="Number of NPZ cases to evaluate on (0=all)")
    ap.add_argument("--tsp_epochs", type=int, default=3,
                    help="Number of training epochs over the selected cases")
    ap.add_argument("--tsp_k_neighbors", type=int, default=4,
                    help="Number of nearest neighbor targets per source node")
    ap.add_argument("--tsp_eval_before", action=argparse.BooleanOptionalAction, default=True,
                    help="Run evaluation before training to establish baseline metrics")
    ap.add_argument("--tsp_log_interval", type=int, default=10,
                    help="Print training progress every N cases")
    ap.add_argument("--tsp_save_ckpt", type=str, default="",
                    help="Path to save routing-branch checkpoint after TSP training")
    ap.add_argument("--tsp_load_ckpt", type=str, default="",
                    help="Path to load routing-branch checkpoint (resumes cost params + CostNet)")
    ap.add_argument("--online_decoder", action="store_true",
                    help="Recompute road_prob through decoder each step (enables decoder gradient)")
    ap.add_argument("--decoder_grad_every", type=int, default=5,
                    help="Recompute road_prob through decoder every N cases (amortizes cost)")

    # --- performance ---
    ap.add_argument("--compile_eikonal", action="store_true",
                    help="Use torch.compile on Eikonal iteration kernel (PyTorch 2.0+)")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # ---- load ckpt ----
    sd = _load_lightning_ckpt(args.ckpt)
    cfg = GradcheckConfig()

    has_lora = any("attn.qkv.linear_a_q" in k or "attn.qkv.linear_a_v" in k for k in sd)
    if has_lora:
        cfg.ENCODER_LORA = True
        cfg.FREEZE_ENCODER = False
        print("  LoRA detected in checkpoint → ENCODER_LORA=True")

    has_decoder = any(k.startswith("map_decoder.") for k in sd)
    if not has_decoder:
        raise ValueError("Checkpoint contains no map_decoder.* keys (not a finetuned SAM-Road ckpt).")

    cfg.USE_SMOOTH_DECODER = _detect_smooth_decoder(sd)
    ps = _detect_patch_size(sd)
    if ps is not None:
        cfg.PATCH_SIZE = ps

    # override knobs
    cfg.ROUTE_EIK_ITERS = int(args.eik_iters)
    cfg.ROUTE_EIK_DOWNSAMPLE = int(args.downsample)
    cfg.ROUTE_LAMBDA_SEG = float(args.lambda_seg)
    cfg.ROUTE_LAMBDA_DIST = float(args.lambda_dist)

    # cost net config
    cfg.ROUTE_COST_NET = bool(args.cost_net)
    cfg.ROUTE_COST_NET_CH = int(args.cost_net_ch)
    cfg.ROUTE_COST_NET_USE_COORD = bool(args.cost_net_use_coord)
    cfg.ROUTE_COST_NET_DELTA_SCALE = float(args.cost_net_delta_scale)

    print(f"Model patch_size={cfg.PATCH_SIZE}, smooth_decoder={cfg.USE_SMOOTH_DECODER}")
    print(f"Eikonal iters(train)={cfg.ROUTE_EIK_ITERS}, downsample(train)={cfg.ROUTE_EIK_DOWNSAMPLE}")
    print(f"Loss weights: lambda_seg={cfg.ROUTE_LAMBDA_SEG}, lambda_dist={cfg.ROUTE_LAMBDA_DIST}")
    print(f"cap_mode={args.cap_mode}, cap_mult={args.cap_mult}")
    if cfg.ROUTE_COST_NET:
        print(f"[cost_net] enabled: ch={cfg.ROUTE_COST_NET_CH}, "
              f"coord={cfg.ROUTE_COST_NET_USE_COORD}, "
              f"delta_scale={cfg.ROUTE_COST_NET_DELTA_SCALE}")
    if args.multigrid:
        print(f"[multigrid] enabled: mg_factor={args.mg_factor}")

    model = SAMRoute(cfg).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded ckpt: missing={len(missing)}, unexpected={len(unexpected)}")

    if args.freeze_encoder or (cfg.FREEZE_ENCODER and not cfg.ENCODER_LORA):
        if hasattr(model, "image_encoder"):
            for p in model.image_encoder.parameters():
                p.requires_grad_(False)
        if hasattr(model, "topo_net"):
            for p in model.topo_net.parameters():
                p.requires_grad_(False)

    # switch mode for gradient stability
    old_mode = model.route_eik_cfg.mode
    if old_mode == "ste_train":
        model.route_eik_cfg = dc_replace(model.route_eik_cfg, mode=args.eik_mode)
        if args.eik_mode != old_mode:
            print(f"[mode] Eikonal mode: {old_mode} -> {args.eik_mode}")

    # override gate_alpha if requested
    if abs(args.gate_alpha - model.route_gate_alpha) > 1e-9:
        print(f"[gate] route_gate_alpha: {model.route_gate_alpha} -> {args.gate_alpha}")
        model.route_gate_alpha = float(args.gate_alpha)
    else:
        print(f"[gate] route_gate_alpha={model.route_gate_alpha} (default)")

    # ---- choose mode ----
    fullmap_mode = bool(args.tif)

    if not fullmap_mode:
        # -------------------------
        # Original dataset/ROI mode
        # -------------------------
        model.train()

        use_cached = not has_lora
        if has_lora:
            print("  LoRA model → disabling cached encoder features (must run encoder online)")

        train_loader, _ = build_dataloaders(
            root_dir=args.data_root,
            patch_size=cfg.PATCH_SIZE,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_dist=True,
            npz_variant=args.npz_variant,
            samples_per_region=args.samples_per_region,
            preload_to_ram=args.preload_to_ram,
            k_targets=args.k_targets,
            min_in_patch=2,
            use_cached_features=use_cached,
        )
        batch = _to_device(_first_batch_with_dist(train_loader), device)

        # optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr)
        print(f"Trainable params: {sum(p.numel() for p in params)}")

        mg_itc = int(args.mg_iters_coarse) if int(args.mg_iters_coarse) > 0 else max(20, int(args.eik_iters * 0.25))
        mg_itf = int(args.mg_iters_fine)   if int(args.mg_iters_fine)   > 0 else max(20, int(args.eik_iters * 0.80))

        # run steps (same as original, kept minimal)
        for step in range(int(args.steps)):
            opt.zero_grad(set_to_none=True)
            loss_seg, road_prob_raw = model._seg_forward(batch)
            road_prob = road_prob_raw.clone()
            road_prob.retain_grad()

            src_yx = batch["src_yx"]
            tgt_yx = batch["tgt_yx"]
            gt_dist = batch["gt_dist"].to(torch.float32)

            if args.multigrid:
                pred_dist = model._roi_multi_target_multigrid_diff_solve(
                    road_prob, src_yx, tgt_yx, gt_dist, model.route_eik_cfg,
                    mg_factor=int(args.mg_factor),
                    mg_iters_coarse=mg_itc,
                    mg_iters_fine=mg_itf,
                    mg_detach_coarse=bool(args.mg_detach_coarse),
                    mg_interp=str(args.mg_interp),
                    fine_monotone=bool(args.mg_fine_monotone),
                    tube_roi=bool(args.tube_roi),
                    tube_min_Pc=int(args.tube_min_pc),
                    tube_radius_c=int(args.tube_radius_c),
                    tube_pad_c=int(args.tube_pad_c),
                    tube_max_area_ratio=float(args.tube_max_area_ratio),
                    tube_min_side=int(args.tube_min_side),
                )
            else:
                pred_dist = model._roi_multi_target_diff_solve(road_prob, src_yx, tgt_yx, gt_dist, model.route_eik_cfg)

            norm = float(getattr(cfg, "ROUTE_DIST_NORM_PX", 512.0))
            valid = gt_dist > 0
            if valid.any():
                pred_in, gt_in, sat = _apply_cap(pred_dist, gt_dist, norm, args.cap_mult, args.cap_mode)
                loss_dist = model.dist_criterion(pred_in[valid], gt_in[valid])
                sat_ratio = float(sat[valid].float().mean().item())
                max_pred = float(pred_dist[valid].max().item())
            else:
                loss_dist = torch.tensor(0.0, device=device)
                sat_ratio = 0.0
                max_pred = 0.0

            loss = cfg.ROUTE_LAMBDA_SEG * loss_seg + cfg.ROUTE_LAMBDA_DIST * loss_dist
            loss.backward()
            opt.step()

            with torch.no_grad():
                print(
                    f"step={step:03d} loss={float(loss.item()):.6f} "
                    f"seg={float(loss_seg.item()):.6f} dist={float(loss_dist.item()):.6f} "
                    f"road_prob_grad_norm={_grad_norm(road_prob.grad):.6g} "
                    f"sat_ratio={sat_ratio:.3f} max_pred_dist={max_pred:.1f}"
                )

        return

    # -------------------------
    # Full-map mode
    # -------------------------
    print("[mode] fullmap (GeoTIFF + sliding segmentation + full-grid downsampled routing)")

    # 1) load rgb
    rgb = _load_rgb_from_tif(args.tif)
    H, W = rgb.shape[:2]
    print(f"[tif] {args.tif}  shape={H}x{W}")

    # 2) get road_prob full image (cacheable)
    if args.cache_prob and os.path.isfile(args.cache_prob):
        road_prob_np = np.load(args.cache_prob).astype(np.float32)
        print(f"[cache] loaded road_prob: {args.cache_prob}")
    else:
        model.eval()
        road_prob_np = sliding_window_inference(
            rgb, model, device,
            patch_size=int(cfg.PATCH_SIZE),
            stride=int(args.stride),
            smooth_sigma=float(args.smooth_sigma),
            verbose=True,
        )
        if args.cache_prob:
            _d = os.path.dirname(args.cache_prob)
            if _d:
                os.makedirs(_d, exist_ok=True)
            np.save(args.cache_prob, road_prob_np)
            print(f"[cache] saved road_prob: {args.cache_prob}")

    # =====================================================================
    # TSP multi-anchor training mode
    # =====================================================================
    if args.tsp_train:
        _run_tsp_train(model, road_prob_np, rgb, cfg, args, device)
        return

    # 3) choose src/tgt + gt_dist
    npz_path = args.npz
    if (not npz_path) and args.sample_from_npz:
        # auto-detect by tif name
        tif_name = os.path.basename(args.tif)
        location_key = tif_name.replace("crop_", "").replace("_z16.tif", "").replace(".tif", "")
        npz_name = f"distance_dataset_all_{location_key}_p{int(args.p_count)}.npz"
        npz_path = os.path.join(os.path.dirname(args.tif), npz_name)

    src_yx_np: np.ndarray
    tgt_yx_np: np.ndarray
    gt_dist_np: np.ndarray

    if args.sample_from_npz:
        if not npz_path or (not os.path.isfile(npz_path)):
            raise FileNotFoundError(f"NPZ not found for sampling: {npz_path}")
        coords, udist, H_npz, W_npz = _load_npz_nodes(npz_path)
        if H_npz != H or W_npz != W:
            print(f"[warn] NPZ meta H/W={H_npz}/{W_npz} != tif H/W={H}/{W} (still proceeding)")
        src_yx_np, tgt_yx_np, gt_dist_np = _sample_pairs_from_npz(
            coords, udist, H, W,
            k_targets=int(args.k_targets),
            case_idx=int(args.case_idx),
            seed=int(args.seed),
            min_gt_dist_px=float(args.min_gt_dist_px),
        )
        print(f"[pair@npz] src={tuple(src_yx_np.tolist())}  tgt0={tuple(tgt_yx_np[0].tolist())}  "
              f"gt0={float(gt_dist_np[0]):.1f}px  npz={npz_path}")
    else:
        # from CLI coords
        if not args.src or not args.tgt:
            raise ValueError("fullmap mode requires either --sample_from_npz or both --src/--tgt")
        def _parse_yx(s: str) -> Tuple[int, int]:
            parts = [p for p in re.split(r"[,\s]+", s.strip()) if p]
            if len(parts) != 2:
                raise ValueError(f"Invalid yx: '{s}' (use 'y,x')")
            return int(float(parts[0])), int(float(parts[1]))
        sy, sx = _parse_yx(args.src)
        ty, tx = _parse_yx(args.tgt)
        src_yx_np = np.array([sy, sx], dtype=np.int64)
        tgt_yx_np = np.array([[ty, tx]], dtype=np.int64)
        gt_dist_np = np.array([-1.0], dtype=np.float32)
        # pad targets to K
        if int(args.k_targets) > 1:
            pad = int(args.k_targets) - 1
            tgt_yx_np = np.concatenate([tgt_yx_np, np.full((pad, 2), -1, np.int64)], axis=0)
            gt_dist_np = np.concatenate([gt_dist_np, np.full((pad,), -1.0, np.float32)], axis=0)

    # 4) build tensors
    road_prob_t = torch.from_numpy(road_prob_np).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
    # numeric safety: avoid exact 0/1 saturation in cost mapping
    eps = float(args.prob_eps)
    road_prob_t = road_prob_t.clamp(eps, 1.0 - eps)

    src_yx_t = torch.from_numpy(src_yx_np[None, :]).to(device=device, dtype=torch.long)  # [1,2]
    tgt_yx_t = torch.from_numpy(tgt_yx_np[None, :, :]).to(device=device, dtype=torch.long)  # [1,K,2]
    gt_dist_t = torch.from_numpy(gt_dist_np[None, :]).to(device=device, dtype=torch.float32)  # [1,K]

    # 5) pseudo-GT if requested
    if (not (gt_dist_t > 0).any()) and args.teacher_hard_eval:
        # use hard-eval distance on downsample grid as pseudo-GT for each valid target (here K targets)
        print("[teacher] computing hard_eval pseudo-GT distances...")
        gt_list = []
        for k in range(int(args.k_targets)):
            yx = tgt_yx_np[k]
            if yx[0] < 0:
                gt_list.append(-1.0)
                continue
            d, _ = _hard_eval_route_fullgrid(
                model, road_prob_t, (int(src_yx_np[0]), int(src_yx_np[1])),
                (int(yx[0]), int(yx[1])),
                ds=int(args.vis_ds),
                n_iters=int(args.teacher_iters),
            )
            gt_list.append(float(d))
        gt_dist_t = torch.tensor(gt_list, device=device, dtype=torch.float32).view(1, -1)
        print(f"[teacher] gt_dist[0]={gt_list[0]:.1f}px")

    # 6) optimization setup
    # road_prob optimization (optional)
    if args.optimize_prob:
        road_prob_var = torch.nn.Parameter(road_prob_t.detach().clone())
        prob_params = [road_prob_var]
        print("[opt] optimize_prob enabled: updating road_prob as a variable (NOT the segmentation network).")
    else:
        road_prob_var = None
        prob_params = []

    # model params (routing branch)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    # if we only want to tune routing params (recommended here), optionally filter:
    # keep everything trainable by default; user can freeze via ckpt or flags.
    opt = torch.optim.AdamW(params + prob_params, lr=float(args.lr))
    print(f"Trainable params: {sum(p.numel() for p in params)}  (+prob={sum(p.numel() for p in prob_params)})")

    mg_itc = int(args.mg_iters_coarse) if int(args.mg_iters_coarse) > 0 else max(20, int(args.eik_iters * 0.25))
    mg_itf = int(args.mg_iters_fine)   if int(args.mg_iters_fine)   > 0 else max(20, int(args.eik_iters * 0.80))

    # optional visualization before
    out_dir = args.save_debug or "/tmp/route_gradcheck_fullmap"
    os.makedirs(out_dir, exist_ok=True)

    if args.vis_full_route:
        with torch.no_grad():
            d0, path0 = _hard_eval_route_fullgrid(
                model, road_prob_t,
                (int(src_yx_np[0]), int(src_yx_np[1])),
                (int(tgt_yx_np[0, 0]), int(tgt_yx_np[0, 1])),
                ds=int(args.vis_ds),
                n_iters=int(args.vis_iters),
            )
        _save_route_overlay(
            os.path.join(out_dir, "route_before.png"),
            rgb, road_prob_np,
            (int(src_yx_np[0]), int(src_yx_np[1])),
            (int(tgt_yx_np[0, 0]), int(tgt_yx_np[0, 1])),
            path0,
            title=f"BEFORE  hard_eval dist={d0:.1f}px (ds={args.vis_ds})",
        )

    # 7) run steps on fixed full-map pair(s)
    _use_cuda_sync = bool(args.profile_time) and device.type == "cuda"

    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)

        rp_in = road_prob_var if road_prob_var is not None else road_prob_t
        rp_in = rp_in.clamp(eps, 1.0 - eps)

        # --- forward (distance prediction) ---
        if _use_cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if args.multigrid:
            pred_dist = fullgrid_multigrid_diff_solve(
                model,
                rp_in,
                src_yx_t,
                tgt_yx_t,
                gt_dist_t,
                model.route_eik_cfg,
                ds_common=int(args.downsample),
                mg_factor=int(args.mg_factor),
                mg_iters_coarse=mg_itc,
                mg_iters_fine=mg_itf,
                mg_detach_coarse=bool(args.mg_detach_coarse),
                mg_interp=str(args.mg_interp),
                fine_monotone=bool(args.mg_fine_monotone),
                tube_roi=bool(args.tube_roi),
                tube_min_Pc=int(args.tube_min_pc),
                tube_radius_c=int(args.tube_radius_c),
                tube_pad_c=int(args.tube_pad_c),
                tube_max_area_ratio=float(args.tube_max_area_ratio),
                tube_min_side=int(args.tube_min_side),
                tube_iters_floor_full=bool(args.tube_iters_floor_full),
            )
        else:
            # non-multigrid fullgrid: single-stage diff solve without warm-start
            # (we still run on full downsample grid)
            ds = int(args.downsample)
            prob_f, _, _ = _maxpool_prob(rp_in, ds)
            Hf, Wf = prob_f.shape[-2], prob_f.shape[-1]
            cost = model._road_prob_to_cost(prob_f).to(dtype=torch.float32)
            src_c = (src_yx_t // ds).clamp(0, max(Hf - 1, 0))
            src_mask = _make_src_mask(1, Hf, Wf, src_c, device)
            cfg_f = dc_replace(model.route_eik_cfg, h=float(ds), n_iters=int(args.eik_iters))
            T = _eikonal_soft_sweeping_diff(
                cost, src_mask, cfg_f,
                checkpoint_chunk=int(getattr(model, "route_ckpt_chunk", 10)),
                gate_alpha=float(getattr(model, "route_gate_alpha", 1.0)),
            )
            # read targets
            tgt_c = (tgt_yx_t // ds).clamp(min=0)
            tgt_c[..., 0] = tgt_c[..., 0].clamp(0, Hf - 1)
            tgt_c[..., 1] = tgt_c[..., 1].clamp(0, Wf - 1)
            Te = []
            for k in range(tgt_c.shape[1]):
                Te.append(T[0, tgt_c[0, k, 0], tgt_c[0, k, 1]])
            Te = torch.stack(Te).view(1, -1)
            d_euc = ((src_yx_t.float().unsqueeze(1) - tgt_yx_t.float()) ** 2).sum(-1).sqrt()
            pred_dist = _mix_eikonal_euclid(Te, d_euc, model.eik_gate_logit)

        if _use_cuda_sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # --- loss ---
        norm = float(getattr(cfg, "ROUTE_DIST_NORM_PX", 512.0))
        valid = gt_dist_t > 0
        if valid.any():
            pred_in, gt_in, sat = _apply_cap(pred_dist, gt_dist_t, norm, args.cap_mult, args.cap_mode)
            loss_dist = model.dist_criterion(pred_in[valid], gt_in[valid])
            sat_ratio = float(sat[valid].float().mean().item())
            max_pred = float(pred_dist[valid].max().item())
        else:
            # no GT -> just print pred
            loss_dist = pred_dist.mean() * 0.0
            sat_ratio = 0.0
            max_pred = float(pred_dist.max().item())

        loss = float(args.lambda_dist) * loss_dist

        if _use_cuda_sync:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        if _use_cuda_sync:
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        # gradient guard
        has_bad = False
        for p in (params + prob_params):
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                has_bad = True
                break
        if has_bad:
            print("  *** NaN/Inf gradient detected — skipping opt.step() ***")
        else:
            opt.step()
            if road_prob_var is not None:
                with torch.no_grad():
                    road_prob_var.clamp_(eps, 1.0 - eps)

        with torch.no_grad():
            rp_gn = _grad_norm(road_prob_var.grad) if road_prob_var is not None else 0.0
            print(
                f"step={step:03d} loss={float(loss.item()):.6f} "
                f"dist={float(loss_dist.item()):.6f} "
                f"sat_ratio={sat_ratio:.3f} max_pred={max_pred:.1f} "
                f"prob_grad_norm={rp_gn:.6g}"
            )
            if args.profile_time:
                print(f"  [time] pred={t1-t0:.3f}s  bwd={t3-t2:.3f}s  total={t3-t0:.3f}s")
            if args.multigrid and hasattr(model, "_last_tube_meta"):
                meta = getattr(model, "_last_tube_meta", {})
                if meta:
                    oob = meta.get('oob_count', 0)
                    oob_str = f"  oob={oob}" if oob > 0 else ""
                    print(f"  [tube_meta] use={meta.get('use_tube', False)}  "
                          f"Hf={meta.get('Hf','?')}  Wf={meta.get('Wf','?')}  "
                          f"tube={meta.get('tube_h','?')}x{meta.get('tube_w','?')}  "
                          f"area_ratio={meta.get('tube_area_ratio', 1.0):.3f}  "
                          f"iters_fine={meta.get('iters_fine', '?')}  "
                          f"iters_coarse={meta.get('iters_coarse', '?')}{oob_str}")

    # 8) visualization after
    if args.vis_full_route:
        rp_vis = (road_prob_var.detach() if road_prob_var is not None else road_prob_t).clamp(eps, 1.0 - eps)
        road_prob_after = rp_vis[0].float().cpu().numpy()
        with torch.no_grad():
            d1, path1 = _hard_eval_route_fullgrid(
                model, rp_vis,
                (int(src_yx_np[0]), int(src_yx_np[1])),
                (int(tgt_yx_np[0, 0]), int(tgt_yx_np[0, 1])),
                ds=int(args.vis_ds),
                n_iters=int(args.vis_iters),
            )
        _save_route_overlay(
            os.path.join(out_dir, "route_after.png"),
            rgb, road_prob_after,
            (int(src_yx_np[0]), int(src_yx_np[1])),
            (int(tgt_yx_np[0, 0]), int(tgt_yx_np[0, 1])),
            path1,
            title=f"AFTER   hard_eval dist={d1:.1f}px (ds={args.vis_ds})",
        )
        print(f"[saved] {os.path.join(out_dir, 'route_before.png')}")
        print(f"[saved] {os.path.join(out_dir, 'route_after.png')}")

    # also dump probability maps for debugging
    np.save(os.path.join(out_dir, "road_prob_full.npy"), road_prob_np)
    print(f"[saved] {os.path.join(out_dir, 'road_prob_full.npy')}")


if __name__ == "__main__":
    main()
