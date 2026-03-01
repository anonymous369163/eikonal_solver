#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gradcheck_route_loss_v2.py

在训练阶段验证“可微 Eikonal 路由距离监督”的最小闭环，并可视化优化前后路径差异。

相较 v1（gradcheck_route_loss.py），本版本做了 3 类增强：
1) 工程稳健性：显式冻结 encoder（可选），并从 optimizer 参数里排除；避免未来配置变动导致意外反传/显存飙升。
2) 梯度不消失：提供 cap_mode（clamp / tanh / log / none），默认 tanh（软上限），避免 pred_dist 饱和导致梯度为 0。
3) 可视化：保存“优化前/后规划路径”对比图（route_compare.png），直观看出路线是否更沿路。

运行示例
--------
# 先做一次 backward，确认梯度链路通
python gradcheck_route_loss_v2.py \
  --data_root Gen_dataset_V2/Gen_dataset \
  --ckpt training_outputs/finetune_demo/checkpoints/best.ckpt \
  --downsample 8 --eik_iters 100 --eik_mode soft_train \
  --lambda_seg 0.0 --lambda_dist 1.0 \
  --steps 1 --lr 1e-4 \
  --save_debug /tmp/route_gradcheck

# 固定一个 batch 过拟合 20 步 + 可视化路径变化
python gradcheck_route_loss_v2.py \
  --data_root Gen_dataset_V2/Gen_dataset \
  --ckpt training_outputs/finetune_demo/checkpoints/best.ckpt \
  --downsample 8 --eik_iters 100 --eik_mode soft_train \
  --lambda_seg 0.0 --lambda_dist 1.0 \
  --steps 20 --lr 1e-4 \
  --save_debug /tmp/route_gradcheck \
  --vis_route --vis_eik_iters 200 --vis_downsample 4

# 对比 max_pool2d 在不同 downsample 下的道路概率图
python gradcheck_route_loss_v2.py \
  --data_root Gen_dataset_V2/Gen_dataset \
  --vis_ds_compare --save_debug /tmp/route_gradcheck --steps 0

NOTE on eik_mode
----------------
ste_train (the model default) causes gradient explosion/NaN when n_iters > ~20
because the STE introduces a forward-backward mismatch: hard min in forward
leaks gradient to the non-selected branch via soft min in backward, and this
amplification compounds exponentially over many iterations.
soft_train uses smooth softmin in both passes and produces stable gradients.

说明
----
- 训练监督用 dataset.MMRouteDataset(include_dist=True) 提供的 gt_dist（像素单位）。
- 训练距离预测用 SAMRoute 内置可微 solver：
    _roi_multi_target_diff_solve -> _eikonal_soft_sweeping_diff（checkpointed, ste_train）
- 路径可视化用非可微 hard_eval solver（eikonal_soft_sweeping），只用于画图，不影响训练梯度。

"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Path setup: make sure we can import model.py / dataset.py when run from repo root
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

from model_multigrid import SAMRoute  # noqa: E402
from dataset import build_dataloaders  # noqa: E402
from eikonal import eikonal_soft_sweeping, EikonalConfig  # noqa: E402
from dataclasses import replace as dc_replace  # noqa: E402


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

        # IMPORTANT: this path is only used inside SAMRoad.__init__ to load raw SAM encoder weights.
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

        # Eikonal params (will be overridden by CLI)
        self.ROUTE_EIK_ITERS = 40
        self.ROUTE_EIK_DOWNSAMPLE = 4
        self.ROUTE_CKPT_CHUNK = 10
        self.ROUTE_DIST_NORM_PX = 512.0
        self.ROUTE_GATE_ALPHA = 0.8

        # loss weights (overridden by CLI)
        self.ROUTE_LAMBDA_SEG = 0.0
        self.ROUTE_LAMBDA_DIST = 1.0

        # seg loss hyperparams (keep stable)
        self.ROAD_POS_WEIGHT = 13.9
        self.ROAD_DICE_WEIGHT = 0.0
        self.ROAD_DUAL_TARGET = False
        self.ROAD_THIN_BOOST = 5.0

        # Eikonal solver mode and distance cap (must match model.py defaults)
        self.ROUTE_EIK_MODE = "soft_train"
        self.ROUTE_CAP_MODE = "tanh"
        self.ROUTE_CAP_MULT = 10.0

        # warmups off for gradcheck simplicity
        self.ROUTE_DIST_WARMUP_STEPS = 0
        self.ROUTE_EIK_WARMUP_EPOCHS = 0
        self.ROUTE_EIK_ITERS_MIN = 40

    def get(self, key, default):
        return getattr(self, key, default)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _load_lightning_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    clean = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    return clean


def _detect_smooth_decoder(state_dict: Dict[str, torch.Tensor]) -> bool:
    w = state_dict.get("map_decoder.7.weight")
    if w is None:
        return False
    return w.shape[1] == 32


def _detect_patch_size(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    pe = state_dict.get("image_encoder.pos_embed")
    if pe is None:
        return None
    return int(pe.shape[1]) * 16


def _first_batch_with_dist(loader) -> Dict[str, Any]:
    for batch in loader:
        if "src_yx" in batch and "tgt_yx" in batch and "gt_dist" in batch:
            return batch
    raise RuntimeError("No batch with distance supervision found. Try include_dist=True or check NPZ availability.")


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _grad_norm(t: Optional[torch.Tensor]) -> float:
    if t is None:
        return 0.0
    return float(t.detach().norm().item())


def _print_param_grads(model: torch.nn.Module, keys: Tuple[str, ...]) -> None:
    print("\n[Grad norms: selected parameters]")
    for name, p in model.named_parameters():
        if any(name.endswith(k) or name == k for k in keys):
            g = p.grad
            print(f"  {name:50s}  grad_norm={_grad_norm(g):.6g}  req_grad={p.requires_grad}")


def _choose_first_valid_pair(gt_dist: torch.Tensor) -> Tuple[int, int]:
    """
    gt_dist: [B, K] float, padding <=0
    return (b, k) for first valid, else (-1, -1)
    """
    B, K = gt_dist.shape
    for b in range(B):
        for k in range(K):
            if float(gt_dist[b, k].item()) > 0:
                return b, k
    return -1, -1


def _roi_extract_single(
    prob_b: torch.Tensor,    # [H,W]
    src_yx: torch.Tensor,    # [2] long
    tgt_yx: torch.Tensor,    # [2] long
    margin: int,
    ds: int,
) -> Tuple[torch.Tensor, Tuple[int,int], Tuple[int,int], int, int, int]:
    """
    复刻 model._roi_eikonal_solve 的 ROI crop + padding + downsample 逻辑（单样本单目标），用于可视化。
    返回：
      patch_coarse: [P_c, P_c] prob after maxpool
      (y0,x0): ROI top-left in original patch coordinate
      (src_c_y,src_c_x), (tgt_c_y,tgt_c_x): coarse coords
      ds, P (original ROI size), P_c
    """
    H, W = prob_b.shape
    src = src_yx.long()
    tgt = tgt_yx.long()

    span = int(torch.max(torch.abs(tgt.float() - src.float())).item())
    half = span + int(margin)
    P = max(2 * half + 1, 64)

    y0 = int(src[0].item()) - half
    x0 = int(src[1].item()) - half
    y1 = y0 + P
    x1 = x0 + P

    yy0 = max(y0, 0); xx0 = max(x0, 0)
    yy1 = min(y1, H); xx1 = min(x1, W)

    patch = F.pad(
        prob_b[yy0:yy1, xx0:xx1],
        (xx0 - x0, x1 - xx1, yy0 - y0, y1 - yy1),
        value=0.0,
    )  # [P,P]

    src_rel_y = max(0, min(int(src[0].item()) - y0, P - 1))
    src_rel_x = max(0, min(int(src[1].item()) - x0, P - 1))
    tgt_rel_y = max(0, min(int(tgt[0].item()) - y0, P - 1))
    tgt_rel_x = max(0, min(int(tgt[1].item()) - x0, P - 1))

    ds = max(1, int(ds))
    if ds > 1:
        P_pad = int(np.ceil(P / ds) * ds)
        if P_pad > P:
            patch = F.pad(patch, (0, P_pad - P, 0, P_pad - P), value=0.0)
        patch_coarse = F.max_pool2d(
            patch.unsqueeze(0).unsqueeze(0), kernel_size=ds, stride=ds
        ).squeeze(0).squeeze(0)  # [P_c,P_c]
    else:
        patch_coarse = patch

    P_c = int(patch_coarse.shape[0])
    src_c_y = max(0, min(src_rel_y // ds, P_c - 1))
    src_c_x = max(0, min(src_rel_x // ds, P_c - 1))
    tgt_c_y = max(0, min(tgt_rel_y // ds, P_c - 1))
    tgt_c_x = max(0, min(tgt_rel_x // ds, P_c - 1))

    return patch_coarse, (y0, x0), (src_c_y, src_c_x), (tgt_c_y, tgt_c_x), ds, P, P_c


def _backtrack_path(
    T: torch.Tensor,                 # [H,W]
    src: Tuple[int,int],
    tgt: Tuple[int,int],
    large_val: float,
    max_steps: int = 100000,
) -> List[Tuple[int,int]]:
    """
    在距离场 T 上从 tgt 回溯到 src：每步走向 T 更小的邻居（8邻域）。
    返回 coarse-grid 的 (y,x) 列表（从 tgt 到 src）。
    """
    H, W = T.shape
    sy, sx = src
    ty, tx = tgt

    def inb(y, x): return (0 <= y < H) and (0 <= x < W)

    path = [(ty, tx)]
    cy, cx = ty, tx

    # 如果目标本身不可达（T ~ large），直接返回
    if float(T[cy, cx].item()) > large_val * 0.9:
        return path

    for _ in range(max_steps):
        if cy == sy and cx == sx:
            break

        cur = float(T[cy, cx].item())
        best = (cy, cx)
        best_val = cur

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if not inb(ny, nx):
                    continue
                v = float(T[ny, nx].item())
                if v < best_val:
                    best_val = v
                    best = (ny, nx)

        # stuck：无法继续下降（可能未收敛/平坦区）
        if best == (cy, cx):
            break

        cy, cx = best
        path.append((cy, cx))

        # 额外保险：防止走入不可达大值区
        if best_val > large_val * 0.9:
            break

    return path


def _coarse_path_to_patch_xy(
    path_yx: List[Tuple[int,int]],
    y0x0: Tuple[int,int],
    ds: int,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    coarse 的 (y,x) 序列映射回原 patch 坐标系（用于画线）。
    使用 cell center: y = y0 + y*ds + ds/2
    返回 (xs, ys) for matplotlib plot
    """
    y0, x0 = y0x0
    xs, ys = [], []
    for (cy, cx) in path_yx:
        yy = y0 + cy * ds + ds * 0.5
        xx = x0 + cx * ds + ds * 0.5
        yy = float(np.clip(yy, 0, H - 1))
        xx = float(np.clip(xx, 0, W - 1))
        xs.append(xx)
        ys.append(yy)
    return np.asarray(xs), np.asarray(ys)


def _compute_route_path_for_vis(
    model: SAMRoute,
    prob_b: torch.Tensor,            # [H,W] road_prob (0..1)
    src_yx: torch.Tensor,            # [2]
    tgt_yx: torch.Tensor,            # [2]
    vis_iters: int,
    vis_downsample: int,
    margin: int,
) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """
    用 hard_eval 求 T 场并回溯路径，只用于可视化：
      返回 (dist_at_tgt, xs, ys, reached_src)
    """
    device = prob_b.device
    H, W = prob_b.shape

    patch_c, y0x0, src_c, tgt_c, ds, _P, _P_c = _roi_extract_single(
        prob_b=prob_b,
        src_yx=src_yx,
        tgt_yx=tgt_yx,
        margin=margin,
        ds=vis_downsample,
    )

    # prob -> cost（使用模型当前可学习 alpha/gamma）
    with torch.no_grad():
        cost_c = model._road_prob_to_cost(patch_c.to(dtype=torch.float32))

        cfg_vis = dc_replace(
            model.route_eik_cfg,
            n_iters=int(vis_iters),
            h=float(ds),
            mode="hard_eval",
        )

        src_mask = torch.zeros(1, cost_c.shape[0], cost_c.shape[1], dtype=torch.bool, device=device)
        src_mask[0, src_c[0], src_c[1]] = True

        T = eikonal_soft_sweeping(cost_c.unsqueeze(0), src_mask, cfg_vis)  # [1,Hc,Wc]
        T0 = T[0]
        large_val = float(cfg_vis.large_val)
        dist = float(T0[tgt_c[0], tgt_c[1]].item())

        path_c = _backtrack_path(T0, src=src_c, tgt=tgt_c, large_val=large_val)
        xs, ys = _coarse_path_to_patch_xy(path_c, y0x0=y0x0, ds=ds, H=H, W=W)
        reached = (len(path_c) > 0 and path_c[-1][0] == src_c[0] and path_c[-1][1] == src_c[1])

    return dist, xs, ys, reached


def _save_debug_figs(
    out_dir: str,
    rgb: torch.Tensor,               # [H, W, 3] float 0..255
    road_mask: torch.Tensor,         # [H, W] 0/1
    prob_before: torch.Tensor,       # [H, W]
    prob_after: torch.Tensor,        # [H, W]
    src_yx: torch.Tensor,            # [2]
    tgt_yx_k: torch.Tensor,          # [K,2]
    gt_dist_k: torch.Tensor,         # [K]
    pred_before_k: torch.Tensor,     # [K]
    pred_after_k: torch.Tensor,      # [K]
    route_pair: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (tgt_one, gt_dist_one)
    route_before: Optional[Tuple[float, np.ndarray, np.ndarray, bool]] = None,
    route_after: Optional[Tuple[float, np.ndarray, np.ndarray, bool]] = None,
    vis_meta: str = "",
) -> None:
    """
    保存两张图：
      - gradcheck_debug.png：prob/mask/数值对比
      - route_compare.png   ：优化前后路径对比（若提供 route_*）
    """
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    rgb_np = rgb.detach().cpu().numpy().astype(np.uint8)
    mask_np = road_mask.detach().cpu().numpy()
    pb = prob_before.detach().cpu().numpy()
    pa = prob_after.detach().cpu().numpy()
    df = (pa - pb)

    src = src_yx.detach().cpu().numpy().tolist()
    tgts = tgt_yx_k.detach().cpu().numpy()
    gt = gt_dist_k.detach().cpu().numpy()
    pr0 = pred_before_k.detach().cpu().numpy()
    pr1 = pred_after_k.detach().cpu().numpy()

    # --- 1) 基础 debug 图 ---
    fig = plt.figure(figsize=(18, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(rgb_np)
    ax1.set_title("RGB")
    ax1.scatter([src[1]], [src[0]], c="lime", s=60, marker="x")
    valid = gt > 0
    if valid.any():
        ax1.scatter(tgts[valid, 1], tgts[valid, 0], c="yellow", s=40, marker="o")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(mask_np, cmap="gray")
    ax2.set_title("GT road_mask")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(pb, cmap="magma", vmin=0, vmax=1)
    ax3.set_title("road_prob BEFORE")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(pa, cmap="magma", vmin=0, vmax=1)
    ax4.set_title("road_prob AFTER")

    ax5 = fig.add_subplot(2, 3, 5)
    im = ax5.imshow(df, cmap="coolwarm")
    ax5.set_title("Δ prob (after - before)")
    fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    lines = ["K-target distance supervision (pixels):"]
    for i in range(tgts.shape[0]):
        if gt[i] <= 0:
            continue
        lines.append(
            f"  k={i:02d}  gt={gt[i]:8.1f}  pred0={pr0[i]:8.1f}  pred1={pr1[i]:8.1f}"
        )
    if vis_meta:
        lines.append("")
        lines.append(vis_meta)
    ax6.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "gradcheck_debug.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")

    # --- 2) 路径对比图 ---
    if route_pair is not None and route_before is not None and route_after is not None:
        tgt_one, gt_one = route_pair
        tgt = tgt_one.detach().cpu().numpy().tolist()
        gt_dist_one = float(gt_one.item())

        dist0, xs0, ys0, ok0 = route_before
        dist1, xs1, ys1, ok1 = route_after

        fig2 = plt.figure(figsize=(18, 8))

        axA = fig2.add_subplot(1, 2, 1)
        axA.imshow(rgb_np)
        axA.set_title(f"Route BEFORE  (gt={gt_dist_one:.1f}, pred={dist0:.1f})  ok={ok0}")
        axA.scatter([src[1]], [src[0]], c="lime", s=60, marker="x")
        axA.scatter([tgt[1]], [tgt[0]], c="red", s=60, marker="x")
        if xs0.size > 1:
            axA.plot(xs0, ys0, linewidth=2.0)

        axB = fig2.add_subplot(1, 2, 2)
        axB.imshow(rgb_np)
        axB.set_title(f"Route AFTER   (gt={gt_dist_one:.1f}, pred={dist1:.1f})  ok={ok1}")
        axB.scatter([src[1]], [src[0]], c="lime", s=60, marker="x")
        axB.scatter([tgt[1]], [tgt[0]], c="red", s=60, marker="x")
        if xs1.size > 1:
            axB.plot(xs1, ys1, linewidth=2.0)

        fig2.tight_layout()
        path2 = os.path.join(out_dir, "route_compare.png")
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        print(f"[saved] {path2}")


def _apply_cap(
    pred_dist: torch.Tensor,
    gt_dist: torch.Tensor,
    norm: float,
    cap_mult: float,
    cap_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 pred_loss_input, gt_loss_input, sat_mask（用于统计饱和比例）
    - clamp: pred 被截断 -> 饱和区梯度=0
    - tanh : 软上限 -> 处处可导（推荐用于 gradcheck）
    - log  : log1p 压缩 -> 处处可导
    - none : 不做上限
    """
    cap = norm * float(cap_mult)
    cap_mode = str(cap_mode).lower()

    if cap_mode == "log":
        # log supervision: both pred and gt in log space, no cap
        pred_in = torch.log1p(pred_dist / norm)
        gt_in = torch.log1p(gt_dist / norm)
        sat = torch.zeros_like(pred_dist, dtype=torch.bool)
        return pred_in, gt_in, sat

    if cap_mode == "none":
        pred_c = pred_dist
        sat = torch.zeros_like(pred_dist, dtype=torch.bool)
    elif cap_mode == "tanh":
        # smooth cap
        pred_c = cap * torch.tanh(pred_dist / cap)
        sat = pred_dist > cap
    else:
        # default clamp
        pred_c = pred_dist.clamp(max=cap)
        sat = pred_dist > cap

    pred_in = pred_c / norm
    gt_in = gt_dist / norm
    return pred_in, gt_in, sat


# -----------------------------------------------------------------------------
# Downsample comparison visualization
# -----------------------------------------------------------------------------

def _vis_downsample_compare(
    road_prob: torch.Tensor,
    src_yx: torch.Tensor,
    tgt_yx_k: torch.Tensor,
    gt_dist_k: torch.Tensor,
    margin: int,
    out_dir: str,
):
    """Save a side-by-side comparison of road_prob at ds=1,4,8,16."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob0 = road_prob[0].detach().cpu().float()  # [H, W]
    H, W = prob0.shape

    src = src_yx[0].long()
    valid = gt_dist_k[0] > 0
    tgts = tgt_yx_k[0][valid].long() if valid.any() else torch.zeros(0, 2, dtype=torch.long)

    span_max = 0
    if tgts.numel() > 0:
        span_max = int(torch.max(torch.abs(tgts.float() - src.float())).item())

    ds_list = [1, 4, 8, 16]
    pooled = {}
    for ds in ds_list:
        if ds == 1:
            pooled[ds] = prob0
        else:
            p = prob0.unsqueeze(0).unsqueeze(0)
            pad_to = int(np.ceil(H / ds) * ds)
            if pad_to > H:
                p = F.pad(p, (0, pad_to - W, 0, pad_to - H), value=0.0)
            pooled[ds] = F.max_pool2d(p, kernel_size=ds, stride=ds).squeeze()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    for ax, ds in zip(axes, ds_list):
        img = pooled[ds].numpy()
        ph, pw = img.shape
        half = span_max + margin
        P = max(2 * half + 1, 64)
        P_c = int(np.ceil(P / ds))

        ax.imshow(img, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(
            f"ds={ds}  ({ph}x{pw})\n"
            f"P_c={P_c}  (span={span_max}, margin={margin})",
            fontsize=10,
        )

        s_y = max(0, min(int(src[0].item()) // ds, ph - 1))
        s_x = max(0, min(int(src[1].item()) // ds, pw - 1))
        ax.plot(s_x, s_y, "c*", markersize=10, label="src")
        for ti in range(tgts.shape[0]):
            t_y = max(0, min(int(tgts[ti, 0].item()) // ds, ph - 1))
            t_x = max(0, min(int(tgts[ti, 1].item()) // ds, pw - 1))
            ax.plot(t_x, t_y, "gx", markersize=8, markeredgewidth=2)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"max_pool2d downsample comparison  |  span_max={span_max}px  margin={margin}px",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ds_compare.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


# -----------------------------------------------------------------------------
# ROI geometry diagnostics
# -----------------------------------------------------------------------------

def _print_roi_geometry(
    src_yx: torch.Tensor,
    tgt_yx_k: torch.Tensor,
    gt_dist_k: torch.Tensor,
    margin: int,
    ds: int,
    eik_iters: int,
):
    """Print ROI geometry for each batch element and warn if iterations are insufficient."""
    B = src_yx.shape[0]
    P_c_max = 0
    print("\n[ROI geometry diagnostics]")
    for b in range(B):
        src = src_yx[b].long()
        valid = gt_dist_k[b] > 0
        if valid.any():
            tgts = tgt_yx_k[b][valid]
            span_max = int(torch.max(torch.abs(tgts.float() - src.float())).item())
        else:
            span_max = 0
        half = span_max + margin
        P = max(2 * half + 1, 64)
        P_c = int(np.ceil(P / ds))
        P_c_max = max(P_c_max, P_c)
        sufficient = "OK" if eik_iters >= P_c else "INSUFFICIENT"
        print(
            f"  b={b}: span_max={span_max:4d}px  P={P:4d}  "
            f"P_c={P_c:4d} (ds={ds})  eik_iters={eik_iters}  [{sufficient}]"
        )
    if eik_iters < P_c_max:
        print(
            f"  WARNING: eik_iters={eik_iters} < P_c_max={P_c_max}. "
            f"Wavefront cannot reach all targets — gradients will be zero for far targets."
        )
    print()
    return P_c_max


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=_DATA_ROOT_DEFAULT, help="Gen_dataset_V2/Gen_dataset")
    ap.add_argument("--ckpt", type=str, default=_CKPT_DEFAULT, help="SAMRoute/SAMRoad lightning ckpt (.ckpt)")

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--npz_variant", type=str, default="p20")
    ap.add_argument("--k_targets", type=int, default=4)
    ap.add_argument("--samples_per_region", type=int, default=10)
    ap.add_argument("--preload_to_ram", action="store_true")

    ap.add_argument("--eik_iters", type=int, default=40)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--lambda_seg", type=float, default=0.0)
    ap.add_argument("--lambda_dist", type=float, default=1.0)

    ap.add_argument("--steps", type=int, default=1, help="optimizer steps on a fixed batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_debug", type=str, default="", help="output dir to save debug png")
    ap.add_argument("--freeze_encoder", action="store_true", help="explicitly set image_encoder.requires_grad=False")

    # gradient stability knobs
    ap.add_argument("--cap_mode", type=str, default="tanh", choices=["clamp", "tanh", "log", "none"])
    ap.add_argument("--cap_mult", type=float, default=10.0, help="cap = norm * cap_mult (for clamp/tanh)")
    ap.add_argument("--gate_alpha", type=float, default=0.8,
                    help="GPPN residual gate (1.0=no gating, <1.0=blend with previous T)")

    # route visualization
    ap.add_argument("--vis_route", action="store_true", help="save route_compare.png")
    ap.add_argument("--vis_eik_iters", type=int, default=0, help="hard_eval iters for visualization (0 -> use eik_iters)")
    ap.add_argument("--vis_downsample", type=int, default=0, help="downsample for visualization (0 -> use downsample)")

    # downsample comparison visualization
    ap.add_argument("--vis_ds_compare", action="store_true", help="save ds_compare.png (ds=1/4/8/16 side-by-side)")

    # eikonal solver mode (ste_train causes gradient explosion at >20 iters)
    ap.add_argument("--eik_mode", type=str, default="soft_train",
                    choices=["ste_train", "soft_train", "hard_eval"],
                    help="Eikonal solver mode for backward pass (default: soft_train)")

    # [MOD] multigrid warm-start options (coarse-to-fine Eikonal)
    ap.add_argument("--multigrid", action="store_true",
                    help="Enable multigrid warm-start: coarse solve -> upsample -> fine refine")
    ap.add_argument("--mg_factor", type=int, default=4,
                    help="coarse_ds = downsample * mg_factor")
    ap.add_argument("--mg_iters_coarse", type=int, default=0,
                    help="coarse-stage iters (0 -> auto)")
    ap.add_argument("--mg_iters_fine", type=int, default=0,
                    help="fine-stage iters (0 -> auto)")
    ap.add_argument("--mg_detach_coarse", action="store_true",
                    help="Detach coarse T warm-start to reduce backward cost")
    ap.add_argument("--mg_interp", type=str, default="bilinear", choices=["nearest", "bilinear"],
                    help="Upsample mode for warm-start T")
    ap.add_argument("--mg_fine_monotone", action="store_true",
                    help="Use monotone=True in fine stage (requires warm-start upper bound)")

    # [MOD] Tube ROI options (further acceleration for large ROIs)
    ap.add_argument("--tube_roi", action="store_true",
                    help="Enable Tube ROI: crop fine stage to corridor bbox (multigrid only)")
    ap.add_argument("--tube_min_pc", type=int, default=256,
                    help="Only enable tube when fine grid side P_c >= this")
    ap.add_argument("--tube_radius_c", type=int, default=8,
                    help="Coarse-grid radius (cells) around backtracked path")
    ap.add_argument("--tube_pad_c", type=int, default=4,
                    help="Extra coarse-grid padding on tube bbox")
    ap.add_argument("--tube_max_area_ratio", type=float, default=0.90,
                    help="Skip tube if bbox area / full area >= this")
    ap.add_argument("--tube_min_side", type=int, default=16,
                    help="Minimum tube bbox side length (fine grid)")
    ap.add_argument("--profile_time", action="store_true",
                    help="Print per-step timing (pred/backward/step); CUDA sync")
    ap.add_argument("--tube_compare_baseline", action="store_true",
                    help="At step 0, also compute non-tube multigrid and report distance diff")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # ---- load checkpoint to detect decoder variant / patch size ----
    sd = _load_lightning_ckpt(args.ckpt)
    cfg = GradcheckConfig()

    # Auto-detect LoRA (mirrors test_inference_route.load_model)
    has_lora = any("attn.qkv.linear_a_q" in k or "attn.qkv.linear_a_v" in k for k in sd)
    if has_lora:
        cfg.ENCODER_LORA = True
        cfg.FREEZE_ENCODER = False
        print("  LoRA detected in checkpoint → ENCODER_LORA=True")

    # Validate that this is a full SAM-Road checkpoint (not raw SAM encoder weights)
    has_decoder = any(k.startswith("map_decoder.") for k in sd)
    if not has_decoder:
        raise ValueError(
            f"\n\n  '{os.path.basename(args.ckpt)}' contains only SAM image-encoder weights "
            f"(no map_decoder keys).\n"
            f"  Pass a full SAM-Road finetuned checkpoint instead.\n"
        )

    cfg.USE_SMOOTH_DECODER = _detect_smooth_decoder(sd)
    ps = _detect_patch_size(sd)
    if ps is not None:
        cfg.PATCH_SIZE = ps

    # override routing / loss knobs
    cfg.ROUTE_EIK_ITERS = int(args.eik_iters)
    cfg.ROUTE_EIK_DOWNSAMPLE = int(args.downsample)
    cfg.ROUTE_LAMBDA_SEG = float(args.lambda_seg)
    cfg.ROUTE_LAMBDA_DIST = float(args.lambda_dist)

    print(f"Model patch_size={cfg.PATCH_SIZE}, smooth_decoder={cfg.USE_SMOOTH_DECODER}")
    print(f"Eikonal iters(train)={cfg.ROUTE_EIK_ITERS}, downsample(train)={cfg.ROUTE_EIK_DOWNSAMPLE}")
    print(f"Loss weights: lambda_seg={cfg.ROUTE_LAMBDA_SEG}, lambda_dist={cfg.ROUTE_LAMBDA_DIST}")
    print(f"cap_mode={args.cap_mode}, cap_mult={args.cap_mult}")
    if args.multigrid:
        print(
            f"[multigrid] enabled: mg_factor={args.mg_factor}, "
            f"mg_iters_coarse={args.mg_iters_coarse or 'auto'}, "
            f"mg_iters_fine={args.mg_iters_fine or 'auto'}, "
            f"detach_coarse={args.mg_detach_coarse}, interp={args.mg_interp}, "
            f"fine_monotone={args.mg_fine_monotone}"
        )

    model = SAMRoute(cfg).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded ckpt: missing={len(missing)}, unexpected={len(unexpected)}")

    # ---- explicit freezing (recommended for stability) ----
    if args.freeze_encoder or (cfg.FREEZE_ENCODER and not cfg.ENCODER_LORA):
        if hasattr(model, "image_encoder"):
            for p in model.image_encoder.parameters():
                p.requires_grad_(False)
        # topo_net 在该脚本不会产生梯度，冻结可省遍历开销
        if hasattr(model, "topo_net"):
            for p in model.topo_net.parameters():
                p.requires_grad_(False)

    # We want gradients, so training mode.
    model.train()

    # ---- data ----
    # LoRA models were trained with online encoder forward; cached features
    # (samroad_feat_full_*.npy) were generated by the base SAM encoder and
    # are INCOMPATIBLE with LoRA-modified encoder outputs.
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

    batch = _first_batch_with_dist(train_loader)
    batch = _to_device(batch, device)

    # ---- ROI geometry diagnostics ----
    with torch.no_grad():
        _, road_prob_diag = model._seg_forward(batch)

    P_c_max = _print_roi_geometry(
        batch["src_yx"], batch["tgt_yx"], batch["gt_dist"],
        margin=int(cfg.ROUTE_ROI_MARGIN),
        ds=int(cfg.ROUTE_EIK_DOWNSAMPLE),
        eik_iters=int(cfg.ROUTE_EIK_ITERS),
    )

    # ---- ds comparison visualization (optional) ----
    if args.vis_ds_compare:
        out_dir = args.save_debug or "/tmp/route_gradcheck"
        _vis_downsample_compare(
            road_prob_diag,
            batch["src_yx"], batch["tgt_yx"], batch["gt_dist"],
            margin=int(cfg.ROUTE_ROI_MARGIN),
            out_dir=out_dir,
        )

    # ---- auto-adjust eik_iters if insufficient ----
    required_iters = int(P_c_max * 1.5)
    if cfg.ROUTE_EIK_ITERS < required_iters:
        print(
            f"[auto-adjust] eik_iters: {cfg.ROUTE_EIK_ITERS} -> {required_iters}  "
            f"(P_c_max={P_c_max}, need ~P_c*1.5 for convergence)"
        )
        cfg.ROUTE_EIK_ITERS = required_iters
        model.route_eik_cfg = dc_replace(model.route_eik_cfg, n_iters=required_iters)

    # ---- switch eikonal mode to avoid gradient explosion ----
    # ste_train causes gradient explosion (inf at ~20 iters, NaN at ~50) due to
    # mismatch between hard-min forward and soft-min backward in the STE.
    # soft_train uses smooth operations in both forward and backward, yielding
    # stable gradients even at 200+ iterations.
    old_mode = model.route_eik_cfg.mode
    if old_mode == "ste_train":
        model.route_eik_cfg = dc_replace(model.route_eik_cfg, mode=args.eik_mode)
        if args.eik_mode != old_mode:
            print(f"[mode] Eikonal mode: {old_mode} -> {args.eik_mode}  (ste_train causes grad explosion at >20 iters)")

    del road_prob_diag

    # ---- override gate_alpha if requested ----
    if abs(args.gate_alpha - model.route_gate_alpha) > 1e-9:
        print(f"[gate] route_gate_alpha: {model.route_gate_alpha} -> {args.gate_alpha}")
        model.route_gate_alpha = float(args.gate_alpha)
    else:
        print(f"[gate] route_gate_alpha={model.route_gate_alpha} (default)")

    # ---- optimizer ----
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"Trainable params: {sum(p.numel() for p in params)}")

    # ---- pre-compute multigrid iteration counts ----
    if args.multigrid:
        mg_itc = int(args.mg_iters_coarse) if int(args.mg_iters_coarse) > 0 else max(20, int(args.eik_iters * 0.25))
        mg_itf = int(args.mg_iters_fine)   if int(args.mg_iters_fine)   > 0 else max(20, int(args.eik_iters * 0.80))
    else:
        mg_itc = mg_itf = 0

    # ---- run steps ----
    prob_before = None
    pred_before = None

    # choose one pair for route visualization (b,k)
    b0, k0 = _choose_first_valid_pair(batch["gt_dist"])
    if b0 < 0:
        print("WARNING: no valid (b,k) found in this batch for route visualization.")
    else:
        print(f"[vis] using pair: b={b0}, k={k0}")

    _use_cuda_sync = args.profile_time and device.type == "cuda"

    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)

        loss_seg, road_prob_raw = model._seg_forward(batch)

        # retain grad on a safe tensor (avoid view edge cases across versions)
        road_prob = road_prob_raw.clone()
        road_prob.retain_grad()

        loss = cfg.ROUTE_LAMBDA_SEG * loss_seg

        src_yx = batch["src_yx"]
        tgt_yx = batch["tgt_yx"]
        gt_dist = batch["gt_dist"].to(torch.float32)

        # --- profiling start ---
        if _use_cuda_sync:
            torch.cuda.synchronize()
        t_pred_start = time.perf_counter()

        # distance prediction (differentiable): baseline vs multigrid warm-start
        if args.multigrid:
            pred_dist = model._roi_multi_target_multigrid_diff_solve(
                road_prob,
                src_yx,
                tgt_yx,
                gt_dist,
                model.route_eik_cfg,
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

        if _use_cuda_sync:
            torch.cuda.synchronize()
        t_pred_end = time.perf_counter()

        #归一化 + 距离损失（支持 cap_mode）
        norm = float(getattr(cfg, "ROUTE_DIST_NORM_PX", 512.0))
        valid = gt_dist > 0
        if valid.any():
            pred_in, gt_in, sat = _apply_cap(
                pred_dist=pred_dist,
                gt_dist=gt_dist,
                norm=norm,
                cap_mult=args.cap_mult,
                cap_mode=args.cap_mode,
            )
            loss_dist = model.dist_criterion(pred_in[valid], gt_in[valid])
            sat_ratio = float(sat[valid].float().mean().item())
            max_pred = float(pred_dist[valid].max().item())
        else:
            loss_dist = torch.tensor(0.0, device=device)
            sat_ratio = 0.0
            max_pred = 0.0

        loss = loss + cfg.ROUTE_LAMBDA_DIST * loss_dist

        if _use_cuda_sync:
            torch.cuda.synchronize()
        t_bwd_start = time.perf_counter()
        loss.backward()
        if _use_cuda_sync:
            torch.cuda.synchronize()
        t_bwd_end = time.perf_counter()

        # --- tube_compare_baseline: at step 0, compare tube vs non-tube ---
        if step == 0 and args.tube_compare_baseline and args.multigrid and args.tube_roi:
            with torch.no_grad():
                pred_base = model._roi_multi_target_multigrid_diff_solve(
                    road_prob.detach(),
                    src_yx, tgt_yx, gt_dist, model.route_eik_cfg,
                    mg_factor=int(args.mg_factor),
                    mg_iters_coarse=mg_itc,
                    mg_iters_fine=mg_itf,
                    mg_detach_coarse=bool(args.mg_detach_coarse),
                    mg_interp=str(args.mg_interp),
                    fine_monotone=bool(args.mg_fine_monotone),
                    tube_roi=False,
                )
                v = gt_dist > 0
                if v.any():
                    diff = (pred_dist.detach()[v] - pred_base[v]).abs()
                    rel = diff / (pred_base[v].abs() + 1e-6)
                    print(f"  [tube_vs_base] abs_diff_mean={diff.mean():.3f}  "
                          f"abs_diff_max={diff.max():.3f}  "
                          f"rel_mean={rel.mean():.4f}  rel_max={rel.max():.4f}")

        # ---- print sanity stats ----
        with torch.no_grad():
            rp_grad = road_prob.grad

            prob_sat_hi = float((road_prob > 1 - 1e-5).float().mean().item())
            prob_sat_lo = float((road_prob < 1e-5).float().mean().item())

            print(
                f"step={step:03d}  loss={float(loss.item()):.6f}  "
                f"seg={float(loss_seg.item()):.6f}  dist={float(loss_dist.item()):.6f}  "
                f"road_prob_grad_norm={_grad_norm(rp_grad):.6g}  "
                f"sat_ratio={sat_ratio:.3f}  max_pred_dist={max_pred:.1f}  "
                f"prob_sat(hi/lo)={prob_sat_hi:.4f}/{prob_sat_lo:.4f}"
            )

        _print_param_grads(
            model,
            keys=(
                "map_decoder.0.weight",
                "map_decoder.0.bias",
                "map_decoder.3.weight",
                "map_decoder.6.weight",
                "map_decoder.7.weight",
                "map_decoder.9.weight",
                "cost_log_alpha",
                "cost_log_gamma",
                "eik_gate_logit",
            ),
        )

        # --- profiling: print per-step timing ---
        if args.profile_time:
            print(f"  [time] pred={t_pred_end - t_pred_start:.3f}s  "
                  f"bwd={t_bwd_end - t_bwd_start:.3f}s  "
                  f"total_fwd_bwd={t_bwd_end - t_pred_start:.3f}s")

        # --- tube_meta: print tube usage info ---
        if args.multigrid and hasattr(model, '_last_tube_meta'):
            meta = model._last_tube_meta
            oob = meta.get('oob_count', 0)
            oob_str = f"  oob={oob}" if oob > 0 else ""
            print(f"  [tube_meta] use={meta.get('use_tube', False)}  "
                  f"P_c={meta.get('P_c')}  tube={meta.get('tube_h')}x{meta.get('tube_w')}  "
                  f"area_ratio={meta.get('tube_area_ratio', 1.0):.3f}  "
                  f"iters_fine={meta.get('iters_fine', '?')}  "
                  f"iters_coarse={meta.get('iters_coarse', '?')}{oob_str}")

        if step == 0:
            prob_before = road_prob_raw.detach().clone()   # 用原始 road_prob（未 clone）保存“优化前”
            pred_before = pred_dist.detach().clone()

        # NaN gradient guard: skip update to prevent parameter corruption
        has_nan = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in params
        )
        if has_nan:
            print("  *** NaN/Inf gradient detected — skipping opt.step() ***")
        else:
            opt.step()

    # ---- after ----
    with torch.no_grad():
        loss_seg2, road_prob2 = model._seg_forward(batch)
        if args.multigrid:
            pred_dist2 = model._roi_multi_target_multigrid_diff_solve(
                road_prob2,
                batch["src_yx"],
                batch["tgt_yx"],
                batch["gt_dist"].to(torch.float32),
                model.route_eik_cfg,
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
            pred_dist2 = model._roi_multi_target_diff_solve(
                road_prob2, batch["src_yx"], batch["tgt_yx"], batch["gt_dist"].to(torch.float32), model.route_eik_cfg
            )

    # steps==0 时循环体不执行，prob_before/pred_before 仍为 None
    if prob_before is None:
        prob_before = road_prob2.detach().clone()
        pred_before = pred_dist2.detach().clone()

    if prob_before is not None:
        m = batch["road_mask"].to(torch.float32)
        pb = prob_before
        pa = road_prob2
        eps = 1e-6
        with torch.no_grad():
            road_mean_b = float((pb * (m > 0.5)).sum().item() / ((m > 0.5).sum().item() + eps))
            road_mean_a = float((pa * (m > 0.5)).sum().item() / ((m > 0.5).sum().item() + eps))
            bg_mean_b = float((pb * (m <= 0.5)).sum().item() / ((m <= 0.5).sum().item() + eps))
            bg_mean_a = float((pa * (m <= 0.5)).sum().item() / ((m <= 0.5).sum().item() + eps))
            print("\n[road_prob mean on GT mask]")
            print(f"  road: before={road_mean_b:.4f}  after={road_mean_a:.4f}")
            print(f"  bg  : before={bg_mean_b:.4f}  after={bg_mean_a:.4f}")

            if pred_before is not None:
                gt0 = batch["gt_dist"][0].detach().cpu().numpy()
                pr0 = pred_before[0].detach().cpu().numpy()
                pr1 = pred_dist2[0].detach().cpu().numpy()
                valid0 = gt0 > 0
                if valid0.any():
                    err0 = np.abs(pr0[valid0] - gt0[valid0]).mean()
                    err1 = np.abs(pr1[valid0] - gt0[valid0]).mean()
                    print(f"\n[dist MAE on sample0]  before={err0:.3f}  after={err1:.3f}")

    # ---- visualization ----
    if args.save_debug:
        vis_b = max(0, b0)

        rgb0 = batch.get("rgb", None)
        if rgb0 is None:
            rgb0 = torch.zeros(cfg.PATCH_SIZE, cfg.PATCH_SIZE, 3, device=device)
        else:
            rgb0 = rgb0[vis_b]

        route_pair = None
        route_before = None
        route_after = None
        vis_meta = ""

        if args.vis_route and b0 >= 0:
            vis_iters = int(args.vis_eik_iters) if int(args.vis_eik_iters) > 0 else int(args.eik_iters)
            vis_ds = int(args.vis_downsample) if int(args.vis_downsample) > 0 else int(args.downsample)

            src_one = batch["src_yx"][b0]
            tgt_one = batch["tgt_yx"][b0, k0]
            gt_one = batch["gt_dist"][b0, k0].to(torch.float32)

            # hard-eval route on BEFORE/AFTER prob maps
            dist0, xs0, ys0, ok0 = _compute_route_path_for_vis(
                model=model,
                prob_b=prob_before[b0],
                src_yx=src_one,
                tgt_yx=tgt_one,
                vis_iters=vis_iters,
                vis_downsample=vis_ds,
                margin=int(cfg.ROUTE_ROI_MARGIN),
            )
            dist1, xs1, ys1, ok1 = _compute_route_path_for_vis(
                model=model,
                prob_b=road_prob2[b0].detach(),
                src_yx=src_one,
                tgt_yx=tgt_one,
                vis_iters=vis_iters,
                vis_downsample=vis_ds,
                margin=int(cfg.ROUTE_ROI_MARGIN),
            )

            route_pair = (tgt_one, gt_one)
            route_before = (dist0, xs0, ys0, ok0)
            route_after = (dist1, xs1, ys1, ok1)
            vis_meta = f"[route vis] hard_eval iters={vis_iters}, ds={vis_ds}  |  pair(b={b0},k={k0})"

        _save_debug_figs(
            out_dir=args.save_debug,
            rgb=rgb0,
            road_mask=batch["road_mask"][vis_b],
            prob_before=prob_before[vis_b],
            prob_after=road_prob2[vis_b].detach(),
            src_yx=batch["src_yx"][vis_b],
            tgt_yx_k=batch["tgt_yx"][vis_b],
            gt_dist_k=batch["gt_dist"][vis_b],
            pred_before_k=pred_before[vis_b],
            pred_after_k=pred_dist2[vis_b],
            route_pair=route_pair,
            route_before=route_before,
            route_after=route_after,
            vis_meta=vis_meta,
        )


if __name__ == "__main__":
    main()
