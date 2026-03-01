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

        # --- neural residual cost net (adds spatial context to prob→cost) ---
        self.ROUTE_COST_NET = True
        self.ROUTE_COST_NET_CH = 8
        self.ROUTE_COST_NET_USE_COORD = False
        self.ROUTE_COST_NET_DELTA_SCALE = 0.75

        # Eikonal params (CLI override)
        self.ROUTE_EIK_ITERS = 40
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
                neigh.append((float(T[yy, xx].item()), yy, xx))
        neigh.sort(key=lambda t: t[0])
        # take the smallest neighbor; if stuck (no decrease), break
        bestT, by, bx = neigh[0]
        if bestT > float(T[y, x].item()) + 1e-6:
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
    cfg_c = dc_replace(cfg, h=float(ds_coarse), n_iters=int(mg_iters_coarse), monotone=False)
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

    # store for debugging
    try:
        model._last_tube_meta = tube_meta
    except Exception:
        pass

    # -------------------------
    # Fine refinement
    # -------------------------
    P_for_floor = max(Hf, Wf) if tube_iters_floor_full else (max(cost_ref.shape[-2], cost_ref.shape[-1]))
    actual_iters_fine = int(max(mg_iters_fine, int(P_for_floor * 0.8)))
    cfg_f = dc_replace(cfg, h=float(ds_common), n_iters=actual_iters_fine, monotone=bool(fine_monotone))

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
    T_eik = []
    for b in range(B):
        yoff, xoff = off_yx[b]
        Tb = T_ref[b]
        d_b = []
        for k in range(K):
            if bool((gt_dist_k[b, k] > 0).item()):
                ty = int(tgt_c[b, k, 0].item()) - yoff
                tx = int(tgt_c[b, k, 1].item()) - xoff
                # Do NOT silently clamp (debug-friendly). If out-of-bounds, return large_val.
                if ty < 0 or ty >= Tb.shape[0] or tx < 0 or tx >= Tb.shape[1]:
                    d_b.append(torch.tensor(large_val, device=device, dtype=Tb.dtype))
                else:
                    d_b.append(Tb[ty, tx])
            else:
                d_b.append(torch.tensor(large_val, device=device, dtype=Tb.dtype))
        T_eik.append(torch.stack(d_b))
    T_eik = torch.stack(T_eik, dim=0)  # [B,K]

    # Euclidean residual blending
    # d_euc in pixel units (full-res)
    src_f = src_yx.float().unsqueeze(1)   # [B,1,2]
    tgt_f2 = tgt_yx_k.float()             # [B,K,2]
    d_euc = ((src_f - tgt_f2) ** 2).sum(-1).sqrt()  # [B,K]
    pred = _mix_eikonal_euclid(T_eik, d_euc, model.eik_gate_logit)
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
    if cap_mode == "tanh":
        pred_c = cap * torch.tanh(pred_dist / cap)
        sat = pred_dist > cap
    elif cap_mode == "clamp":
        pred_c = pred_dist.clamp(max=cap)
        sat = pred_dist > cap
    elif cap_mode == "log":
        pred_c = torch.log1p(pred_dist / max(1e-6, norm)) * norm
        sat = torch.zeros_like(pred_dist, dtype=torch.bool)
    else:
        pred_c = pred_dist
        sat = torch.zeros_like(pred_dist, dtype=torch.bool)
    return pred_c / norm, gt_dist / norm, sat


# -----------------------------------------------------------------------------
# Visualization (optional): overlay path on full image
# -----------------------------------------------------------------------------

def _tv_l1(x: torch.Tensor) -> torch.Tensor:
    """Total variation (L1) on a [B,H,W] tensor."""
    if x.numel() == 0:
        return x.sum() * 0.0
    dx = (x[..., 1:] - x[..., :-1]).abs().mean()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return dx + dy


def _residual_reg(delta_log: torch.Tensor) -> torch.Tensor:
    """Prevent 'global cheating': encourage small, zero-mean residual."""
    if delta_log.numel() == 0:
        return delta_log.sum() * 0.0
    return (delta_log ** 2).mean() + (delta_log.mean() ** 2)

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
    ap.add_argument("--eik_iters", type=int, default=80)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--eik_mode", type=str, default="soft_train",
                    choices=["ste_train", "soft_train", "hard_eval"])
    ap.add_argument("--cap_mode", type=str, default="tanh", choices=["clamp", "tanh", "log", "none"])
    ap.add_argument("--cap_mult", type=float, default=10.0)

    # --- neural cost net ---
    ap.add_argument("--cost_net", dest="cost_net", action="store_true", help="Enable ResidualCostNet in prob→cost (default: on)")
    ap.add_argument("--no_cost_net", dest="cost_net", action="store_false", help="Disable ResidualCostNet")
    ap.set_defaults(cost_net=None)
    ap.add_argument("--cost_net_ch", type=int, default=8, help="ResidualCostNet hidden channels")
    ap.add_argument("--cost_net_use_coord", action="store_true", help="Append (y,x) coord channels to cost net input")
    ap.add_argument("--cost_net_delta_scale", type=float, default=0.75, help="Scale of bounded log-residual (tanh*scale)")
    ap.add_argument("--lambda_cost_reg", type=float, default=1e-3, help="L2/mean regularization weight on cost residual (prevents global cheating)")
    ap.add_argument("--lambda_cost_tv", type=float, default=0.0, help="TV regularization weight on cost residual (optional smoothing)")
    ap.add_argument("--reg_on_coarse", action="store_true", help="Also regularize coarse-grid residual (default: off)")
    ap.add_argument("--lambda_seg", type=float, default=0.0)
    ap.add_argument("--lambda_dist", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--profile_time", action="store_true")

    # multigrid / tube
    ap.add_argument("--multigrid", action="store_true")
    ap.add_argument("--mg_factor", type=int, default=4)
    ap.add_argument("--mg_iters_coarse", type=int, default=0)
    ap.add_argument("--mg_iters_fine", type=int, default=0)
    ap.add_argument("--mg_detach_coarse", action="store_true")
    ap.add_argument("--mg_interp", type=str, default="bilinear", choices=["nearest", "bilinear"])
    ap.add_argument("--mg_fine_monotone", action="store_true")

    ap.add_argument("--tube_roi", action="store_true")
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
    ap.add_argument("--sample_from_npz", action="store_true",
                    help="If set, sample src + K targets from NPZ nodes instead of --src/--tgt")
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
    # neural cost net overrides
    if args.cost_net is not None:
        cfg.ROUTE_COST_NET = bool(args.cost_net)
    cfg.ROUTE_COST_NET_CH = int(args.cost_net_ch)
    cfg.ROUTE_COST_NET_USE_COORD = bool(args.cost_net_use_coord)
    cfg.ROUTE_COST_NET_DELTA_SCALE = float(args.cost_net_delta_scale)

    cfg.ROUTE_EIK_ITERS = int(args.eik_iters)
    cfg.ROUTE_EIK_DOWNSAMPLE = int(args.downsample)
    cfg.ROUTE_LAMBDA_SEG = float(args.lambda_seg)
    cfg.ROUTE_LAMBDA_DIST = float(args.lambda_dist)

    print(f"Model patch_size={cfg.PATCH_SIZE}, smooth_decoder={cfg.USE_SMOOTH_DECODER}")
    print(f"Eikonal iters(train)={cfg.ROUTE_EIK_ITERS}, downsample(train)={cfg.ROUTE_EIK_DOWNSAMPLE}")
    print(f"Loss weights: lambda_seg={cfg.ROUTE_LAMBDA_SEG}, lambda_dist={cfg.ROUTE_LAMBDA_DIST}")
    print(f"cap_mode={args.cap_mode}, cap_mult={args.cap_mult}")
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
            # --- optional regularization on neural cost residual (prevents global scaling cheats) ---
            loss_reg = torch.tensor(0.0, device=device)
            if getattr(cfg, "ROUTE_COST_NET", False) and getattr(model, "cost_net", None) is not None:
                ds = int(args.downsample)
                prob_f, _, _ = _maxpool_prob(road_prob, ds)
                delta_f = model._cost_net_delta_log(prob_f)
                if delta_f is not None:
                    loss_reg = loss_reg + float(args.lambda_cost_reg) * _residual_reg(delta_f)
                    if float(args.lambda_cost_tv) > 0:
                        loss_reg = loss_reg + float(args.lambda_cost_tv) * _tv_l1(delta_f)

            loss = loss + loss_reg
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
            os.makedirs(os.path.dirname(args.cache_prob), exist_ok=True)
            np.save(args.cache_prob, road_prob_np)
            print(f"[cache] saved road_prob: {args.cache_prob}")

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
# --- optional regularization on neural cost residual (prevents global scaling cheats) ---
loss_reg = torch.tensor(0.0, device=device)
if getattr(cfg, "ROUTE_COST_NET", False) and getattr(model, "cost_net", None) is not None:
    ds = int(args.downsample)
    prob_f, _, _ = _maxpool_prob(rp_in, ds)
    delta_f = model._cost_net_delta_log(prob_f)
    if delta_f is not None:
        loss_reg = loss_reg + float(args.lambda_cost_reg) * _residual_reg(delta_f)
        if float(args.lambda_cost_tv) > 0:
            loss_reg = loss_reg + float(args.lambda_cost_tv) * _tv_l1(delta_f)

    if bool(args.reg_on_coarse) and bool(args.multigrid):
        ds_c = int(args.downsample) * int(args.mg_factor)
        prob_c, _, _ = _maxpool_prob(rp_in, ds_c)
        delta_c = model._cost_net_delta_log(prob_c)
        if delta_c is not None:
            loss_reg = loss_reg + float(args.lambda_cost_reg) * _residual_reg(delta_c)

        loss = loss + loss_reg

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
                    print(f"  [tube_meta] use={meta.get('use_tube', False)}  "
                          f"tube_area_ratio={meta.get('tube_area_ratio', 1.0):.3f}  "
                          f"tube={meta.get('tube_h','?')}x{meta.get('tube_w','?')}")

    # 8) visualization after
    if args.vis_full_route:
        rp_vis = (road_prob_var.detach() if road_prob_var is not None else road_prob_t).clamp(eps, 1.0 - eps)
        road_prob_after = rp_vis[0].float().cpu().numpy()
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
