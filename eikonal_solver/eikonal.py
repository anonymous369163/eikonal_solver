# -*- coding: utf-8 -*-
"""
eikonal.py — Eikonal soft-sweeping solver (pure PyTorch, GPU-optimized).

Core components:
  - EikonalConfig:           solver hyperparameters
  - eikonal_soft_sweeping:   non-differentiable fast solver (red-black Gauss-Seidel)
  - softmin_algebraic:       smooth min approximation
  - prob_to_cost:            road probability -> travel cost
  - make_source_mask:        build boolean source mask from (y, x) coordinates
  - fill_dense_with_scaled_euclid: approximate dense distance matrix from KNN samples
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def softmin_algebraic(x: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Smooth min via sqrt: 0.5*(x+y - sqrt((x-y)^2 + tau^2))."""
    dist = x - y
    smooth_factor = float(tau * tau)
    return 0.5 * (x + y - torch.sqrt(dist * dist + smooth_factor + 1e-8))


def _ste(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator: forward=hard, backward=soft."""
    return hard + (soft - soft.detach())


def _min2(x: torch.Tensor, y: torch.Tensor, tau: float, mode: str) -> torch.Tensor:
    """Binary min: hard / soft / ste by mode."""
    if mode == "soft_train":
        return softmin_algebraic(x, y, tau)
    hard = torch.minimum(x, y)
    if mode == "ste_train":
        soft = softmin_algebraic(x, y, tau)
        return _ste(hard, soft)
    return hard


def prob_to_cost(
    prob: torch.Tensor,
    gamma: float = 3.0,
    eps: float = 1e-6,
    road_block_th: float = 0.05,
    offroad_penalty: float = 100.0,
    **kwargs,
) -> torch.Tensor:
    """Convert road probability [0,1] to travel cost."""
    prob = prob.clamp(0.0, 1.0)
    k = np.log(float(offroad_penalty))
    cost = torch.exp(k * torch.pow(1.0 - prob, gamma))
    if road_block_th > 0:
        cost = torch.where(prob < road_block_th, cost * 10.0, cost)
    return cost.clamp_min(float(eps))


def make_source_mask(H: int, W: int, sources_yx: torch.Tensor) -> torch.Tensor:
    """Build (B, H, W) boolean source mask from source coordinates."""
    if sources_yx.dim() == 2:
        sources_yx = sources_yx.unsqueeze(0)
    B, S, _ = sources_yx.shape
    mask = torch.zeros((B, H, W), dtype=torch.bool, device=sources_yx.device)
    for b in range(B):
        y = sources_yx[b, :, 0].clamp(0, H - 1)
        x = sources_yx[b, :, 1].clamp(0, W - 1)
        mask[b, y, x] = True
    return mask


def fill_dense_with_scaled_euclid(
    nodes_yx: torch.Tensor,
    knn_idx: torch.Tensor,
    dist_knn: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-6,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Approximate dense distance matrix from sparse KNN road distances."""
    device = nodes_yx.device
    nodes = nodes_yx.float()
    N = nodes.size(0)
    if N <= 1:
        return torch.zeros((N, N), device=device)

    d_e = torch.cdist(nodes, nodes)
    d_e.fill_diagonal_(0.0)
    if knn_idx.numel() == 0:
        return d_e

    eu_knn = d_e[torch.arange(N, device=device)[:, None], knn_idx].clamp_min(eps)
    ratio = (dist_knn / eu_knn).detach()
    scale = torch.median(ratio, dim=1).values.clamp_min(0.5).clamp_max(5.0)
    dist = (scale[:, None] * d_e) * float(alpha) + d_e * float(1.0 - alpha)
    dist[torch.arange(N, device=device)[:, None], knn_idx] = dist_knn

    if symmetrize:
        dist = 0.5 * (dist + dist.T)
        dist.fill_diagonal_(0.0)
    return dist


# ---------------------------------------------------------------------------
# Eikonal solver
# ---------------------------------------------------------------------------

@dataclass
class EikonalConfig:
    n_iters: int = 100
    h: float = 1.0
    tau_min: float = 0.1
    tau_branch: float = 0.1
    tau_update: float = 0.01  # soft monotone 时更接近 hard min，误差更小
    large_val: float = 1e6
    eps: float = 1e-6
    use_redblack: bool = True
    monotone: bool = True
    mode: Optional[str] = None  # "hard_eval" | "ste_train" | "soft_train"
    monotone_hard: bool = False  # deprecated: use mode; compat: monotone_hard=True -> hard_eval


def _local_update_fast(
    T: torch.Tensor, cost: torch.Tensor,
    h: float, large_val: float,
    tau_min: float, tau_branch: float,
    mode: str,
) -> torch.Tensor:
    """One Eikonal update step using pad+slice (2-4x faster than torch.roll)."""
    Tpad = F.pad(T, (1, 1, 1, 1), value=large_val)
    T_l = Tpad[:, 1:-1, 0:-2]
    T_r = Tpad[:, 1:-1, 2:]
    T_u = Tpad[:, 0:-2, 1:-1]
    T_d = Tpad[:, 2:, 1:-1]

    a = _min2(T_l, T_r, tau_min, mode)
    b = _min2(T_u, T_d, tau_min, mode)

    hc = cost * h
    d = torch.abs(a - b)
    s = torch.sigmoid((d - hc) / max(float(tau_branch), 1e-6))
    m = _min2(a, b, tau_min, mode)

    t1 = m + hc
    rad = torch.clamp(2.0 * hc * hc - d * d, min=0.0)
    t2 = 0.5 * (a + b + torch.sqrt(rad + 1e-8))

    if mode == "soft_train":
        return s * t1 + (1.0 - s) * t2
    hard_sel = torch.where(d >= hc, t1, t2)
    if mode == "ste_train":
        soft_sel = s * t1 + (1.0 - s) * t2
        return _ste(hard_sel, soft_sel)
    return hard_sel


def _monotone_update(
    T: torch.Tensor, T_new: torch.Tensor,
    tau_update: float, mode: str, monotone: bool,
) -> torch.Tensor:
    if not monotone:
        return T_new
    if mode == "soft_train":
        return softmin_algebraic(T, T_new, tau_update)
    hard = torch.minimum(T, T_new)
    if mode == "ste_train":
        soft = softmin_algebraic(T, T_new, tau_update)
        return _ste(hard, soft)
    return hard


def eikonal_soft_sweeping(
    cost: torch.Tensor,
    source_mask: Optional[torch.Tensor],
    cfg: EikonalConfig,
    source_yx: Optional[Tuple[int, int]] = None,
    return_convergence: bool = False,
):
    """
    Eikonal soft-sweeping solver (red-black Gauss-Seidel).

    Args:
        cost:               (B, H, W) or (H, W) travel-cost map.
        source_mask:        (B, H, W) bool mask, or None if source_yx is given.
        cfg:                solver configuration.
        source_yx:          optional (y, x) for single-point source (faster path).
        return_convergence: if True, also return a list of per-iteration max|ΔT| values
                            for convergence diagnostics. Signature becomes
                            (T, conv_curve) instead of just T.

    Returns:
        T (distance field) if return_convergence=False, else (T, conv_curve).
        conv_curve is a list of floats with length n_iters.
    """
    squeeze_back = False
    if cost.dim() == 2:
        cost = cost.unsqueeze(0)
        if source_mask is not None and source_mask.dim() == 2:
            source_mask = source_mask.unsqueeze(0)
        squeeze_back = True
    elif source_mask is not None and source_mask.dim() == 2:
        source_mask = source_mask.unsqueeze(0)

    B, H, W = cost.shape
    device = cost.device
    T = torch.full((B, H, W), cfg.large_val, device=device, dtype=cost.dtype)

    if source_yx is not None:
        sy, sx = int(source_yx[0]), int(source_yx[1])
        T[:, sy, sx] = 0.0
    else:
        T = torch.where(source_mask, torch.zeros_like(T), T)

    h_val = float(cfg.h)
    large_val = float(cfg.large_val)
    tau_min = float(cfg.tau_min)
    tau_branch = float(cfg.tau_branch)
    tau_update = float(cfg.tau_update)
    mode = getattr(cfg, 'mode', None)
    if mode is None:
        mode = "hard_eval" if getattr(cfg, 'monotone_hard', False) else "ste_train"

    if cfg.use_redblack:
        yy = torch.arange(H, device=device)[:, None]
        xx = torch.arange(W, device=device)[None, :]
        even = ((yy + xx) % 2 == 0)[None, :, :]
        odd = ~even

    conv_curve: list[float] = []

    for _ in range(int(cfg.n_iters)):
        if cfg.use_redblack:
            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch, mode)
            T_e = _monotone_update(T, T_new, tau_update, mode, cfg.monotone)
            if return_convergence:
                # Even half-step: delta on even cells (T only decreases → T - T_e >= 0)
                delta_e = (T - T_e).clamp_min(0.0)
                delta_e_finite = delta_e[even & (T < large_val * 0.9)]
            T = torch.where(even, T_e, T)
            if source_yx is not None:
                T[:, sy, sx] = 0.0
            else:
                T = torch.where(source_mask, torch.zeros_like(T), T)

            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch, mode)
            T_o = _monotone_update(T, T_new, tau_update, mode, cfg.monotone)
            if return_convergence:
                # Odd half-step delta
                delta_o = (T - T_o).clamp_min(0.0)
                delta_o_finite = delta_o[odd & (T < large_val * 0.9)]
                all_deltas = torch.cat([
                    delta_e_finite.reshape(-1),
                    delta_o_finite.reshape(-1)
                ])
                max_delta = float(all_deltas.max().item()) if all_deltas.numel() > 0 else 0.0
                conv_curve.append(max_delta)
            T = torch.where(odd, T_o, T)
            if source_yx is not None:
                T[:, sy, sx] = 0.0
            else:
                T = torch.where(source_mask, torch.zeros_like(T), T)
        else:
            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch, mode)
            T_upd = _monotone_update(T, T_new, tau_update, mode, cfg.monotone)
            if return_convergence:
                delta = (T - T_upd).clamp_min(0.0)
                delta_finite = delta[T < large_val * 0.9]
                max_delta = float(delta_finite.max().item()) if delta_finite.numel() > 0 else 0.0
                conv_curve.append(max_delta)
            T = T_upd
            if source_yx is not None:
                T[:, sy, sx] = 0.0
            else:
                T = torch.where(source_mask, torch.zeros_like(T), T)

    T_out = T.squeeze(0) if squeeze_back else T
    if return_convergence:
        return T_out, conv_curve
    return T_out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eikonal self-test] device={device}")

    H, W = 128, 128
    prob = torch.ones(H, W, device=device) * 0.8
    prob[40:60, 40:80] = 0.01
    cost = prob_to_cost(prob)
    src = torch.zeros(1, H, W, dtype=torch.bool, device=device)
    src[0, 64, 10] = True
    cfg = EikonalConfig(n_iters=200)
    T = eikonal_soft_sweeping(cost, src, cfg)
    print(f"  T shape={T.shape}  min={T.min():.2f}  max={T.max():.2f}")
    print("  OK")
