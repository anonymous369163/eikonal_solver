#!/usr/bin/env python3
"""
Clean ablation: decouple ds / n_iters, test hard vs soft GT prob.

Answers two questions:
  1. What Spearman ceiling can the Eikonal backend reach with ideal prob maps?
  2. Which knob matters: n_iters (convergence), ds (grid resolution), or alpha/gamma?

Usage:
    python eikonal_solver/test_gt_ablation.py --n_cases 5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eikonal import EikonalConfig, eikonal_soft_sweeping
from backups.evaluator import load_tif_image, pick_anchor_and_knn, spearmanr
from dataset import load_nodes_from_npz

# ── Model loading utility (aligned with test_inference) ────────────────────────

@torch.no_grad()
def load_model_road_prob(
    config_path: str, ckpt_path: str,
    rgb_ds: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    """Load a trained SAMRoute model and predict road probability on rgb_ds.

    Uses test_inference.load_model for checkpoint compatibility (decoder variant,
    patch size auto-detection). Sliding-window + Hann blend for full-image prob.

    Args:
        config_path: path to YAML config (optional overrides; InferenceConfig used as base)
        ckpt_path:   path to the checkpoint (.ckpt)
        rgb_ds:      downsampled RGB tensor [1, 3, H, W] in [0, 1]
        device:      torch device

    Returns:
        road_prob: [H, W] road probability tensor in [0, 1]
    """
    from test_inference import InferenceConfig, load_model, sliding_window_inference

    config = InferenceConfig()
    if config_path and os.path.isfile(config_path):
        import yaml
        from addict import Dict as AdDict
        with open(config_path) as f:
            cfg = AdDict(yaml.safe_load(f))
        config_dir = os.path.dirname(os.path.abspath(config_path))
        # Resolve paths: config may live in backups/; '../' means from eikonal_solver upward
        eikonal_dir = os.path.dirname(config_dir)  # eikonal_solver when config in backups/
        for key in ("SAM_CKPT_PATH", "PRETRAINED_CKPT"):
            val = str(cfg.get(key, "") or "")
            if val and not os.path.isabs(val):
                base = eikonal_dir if val.startswith("../") else config_dir
                setattr(config, key, os.path.normpath(os.path.join(base, val)))
        if "PATCH_SIZE" in cfg:
            config.PATCH_SIZE = int(cfg.PATCH_SIZE)

    model = load_model(ckpt_path, config, device)
    _, _, H, W = rgb_ds.shape
    img_array = (rgb_ds[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    prob_np = sliding_window_inference(
        img_array, model, device,
        patch_size=config.PATCH_SIZE,
        stride=config.PATCH_SIZE // 2,
        smooth_sigma=0.0,
        verbose=False,
    )
    road_prob = torch.from_numpy(prob_np).float().to(device)
    v = road_prob.cpu().numpy().ravel()
    in_01 = ((v > 0.01) & (v < 0.99)).sum() / max(len(v), 1)
    print(f"  model road_prob: mean={road_prob.mean():.4f}, "
          f"p_in_(0,1)={in_01:.4f} ({in_01*100:.1f}%)")
    return road_prob


# ── GT mask / node loading utilities ─────────────────────────────────────────

def load_gt_mask(region_dir: str, variant: str) -> np.ndarray:
    """Load a GT road mask as float32 [H, W] in [0, 1].

    variant: 'thin' for original roadnet PNG, 'r5' for dilated r=5, etc.
    """
    if variant == "thin":
        pngs = sorted(glob.glob(os.path.join(region_dir, "roadnet_*.png")))
        pngs = [p for p in pngs if "normalized" not in os.path.basename(p)]
        if not pngs:
            raise FileNotFoundError(f"No thin GT mask found in {region_dir}")
        path = pngs[0]
    else:
        path = os.path.join(region_dir, f"roadnet_normalized_{variant}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"GT mask not found: {path}")
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        m = arr
    else:
        m = arr[..., 0]
    return m.astype(np.float32) / 255.0


def load_nodes_for_case(tif_path: str, H: int, W: int, case_idx: int,
                        device: torch.device, scale: float,
                        H_ds: int, W_ds: int):
    """Load nodes for a specific case_idx (deterministic)."""
    rp_dummy = torch.zeros(H, W)
    nodes_orig, info = load_nodes_from_npz(
        tif_path, rp_dummy, p_count=20, snap=False, verbose=False,
        case_idx=case_idx)
    meta_h, meta_w = int(info["meta_sat_height_px"]), int(info["meta_sat_width_px"])
    if (meta_h, meta_w) != (H, W):
        raise ValueError(
            f"NPZ meta (H={meta_h}, W={meta_w}) does not match TIF size ({H}, {W}). "
            "Node coordinates would be misaligned."
        )
    if scale != 1.0:
        nodes_yx = (nodes_orig.float() * scale).long()
        nodes_yx[:, 0].clamp_(0, H_ds - 1)
        nodes_yx[:, 1].clamp_(0, W_ds - 1)
    else:
        nodes_yx = nodes_orig
    return nodes_yx.to(device), info


# ── Soft GT probability generation ───────────────────────────────────────────

def make_soft_gt(mask: torch.Tensor, mode: str = "blur",
                 sigma: float = 5.0) -> torch.Tensor:
    """Convert binary GT mask to soft probability map.

    mode='blur':  Gaussian blur with reflect padding (no edge artifacts)
    mode='hard':  Keep binary (baseline)
    """
    if mode == "hard":
        return mask
    if mode == "blur":
        k = int(6 * sigma + 1) | 1
        x = torch.arange(k, device=mask.device, dtype=torch.float32) - k // 2
        gauss = (-x ** 2 / (2 * sigma ** 2)).exp()
        gauss = gauss / gauss.sum()
        kernel = gauss[None, None, :, None] * gauss[None, None, None, :]
        m4d = mask[None, None]
        pad = k // 2
        m4d_padded = F.pad(m4d, (pad, pad, pad, pad), mode="reflect")
        soft = F.conv2d(m4d_padded, kernel, padding=0).squeeze()
        return soft.clamp(0.0, 1.0)
    raise ValueError(f"Unknown mode: {mode}")


# ── Cost function ────────────────────────────────────────────────────────────

def prob_to_cost(p: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """Pure additive cost without block term — cleanest for ablation."""
    p = p.clamp(0.0, 1.0)
    return (1.0 + alpha * (1.0 - p).pow(gamma)).clamp_min(1e-6)

# ── Patch Cropping Helper ────────────────────────────────────────────────────

def crop_square_patch(img_2d: torch.Tensor, cy: int, cx: int, patch: int) -> tuple[torch.Tensor, int, int]:
    """
    Safely crop a square patch from a full 2D image. Handles out-of-bounds by zero-padding.
    Returns: (cropped_patch [patch, patch], global_y0, global_x0)
    """
    H, W = img_2d.shape
    half = patch // 2
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + patch, x0 + patch

    yy0, xx0 = max(y0, 0), max(x0, 0)
    yy1, xx1 = min(y1, H), min(x1, W)

    sub = img_2d[yy0:yy1, xx0:xx1]
    pad_left = xx0 - x0
    pad_right = x1 - xx1
    pad_top = yy0 - y0
    pad_bottom = y1 - yy1
    
    out = F.pad(sub, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return out, y0, x0


# ── Eikonal solve (with FIXED ds, decoupled from n_iters) ───────────────────

def eikonal_solve_fixed_ds(
    rp: torch.Tensor,       # [H, W] local road prob
    anc_yx: torch.Tensor,   # [2] local yx
    tgt_yx: torch.Tensor,   # [K, 2] local yx
    device: torch.device,
    margin: int,
    n_iters: int,
    ds: int,
    alpha: float,
    gamma: float,
    gate: float,
    pool_mode: str = "avg",
    return_T: bool = False,
    n_iters_adaptive: bool = False,
    return_convergence: bool = False,
) -> dict:
    """Fixed-scale Eikonal solver within a given local patch ROI.

    n_iters_adaptive: if True, use max(n_iters, 2 * Pc) so that the wave can
        propagate across the full grid (one traversal ~ Pc iterations for RB-GS).
    return_convergence: if True, include a 'conv_curve' list of per-iteration
        max|ΔT| values in the returned dict (for convergence diagnostics).
    """
    K = tgt_yx.shape[0]
    if K == 0:
        return {"pred": np.empty(0), "diag": {}}

    span_max = int(torch.max(torch.abs(tgt_yx.float() - anc_yx.float())).item())
    half = span_max + margin
    P = max(2 * half + 1, 64)

    H_r, W_r = rp.shape
    y0 = int(anc_yx[0]) - half
    x0 = int(anc_yx[1]) - half
    y1, x1 = y0 + P, x0 + P
    yy0, xx0 = max(y0, 0), max(x0, 0)
    yy1, xx1 = min(y1, H_r), min(x1, W_r)
    roi = F.pad(rp[yy0:yy1, xx0:xx1],
                (xx0 - x0, x1 - xx1, yy0 - y0, y1 - yy1), value=0.0)

    sr_y = max(0, min(int(anc_yx[0]) - y0, P - 1))
    sr_x = max(0, min(int(anc_yx[1]) - x0, P - 1))

    if ds > 1:
        P_pad = math.ceil(P / ds) * ds
        if P_pad > P:
            roi = F.pad(roi, (0, P_pad - P, 0, P_pad - P), value=0.0)
        if pool_mode == "max":
            roi_c = F.max_pool2d(roi[None, None], kernel_size=ds, stride=ds).squeeze()
        else:
            # Avg pool + threshold: avoids max_pool's "road fattening" and corner-cutting
            # while preventing avg_pool's over-dilution (thin roads -> near-zero -> wall)
            avg_pooled = F.avg_pool2d(roi[None, None], kernel_size=ds, stride=ds).squeeze()
            roi_c = torch.where(avg_pooled > 0.02, torch.clamp(avg_pooled * 5.0, 0.0, 1.0), torch.zeros_like(avg_pooled))
    else:
        roi_c = roi

    cost = prob_to_cost(roi_c, alpha, gamma)
    Pc = cost.shape[0]
    sc_y = max(0, min(sr_y // ds, Pc - 1))
    sc_x = max(0, min(sr_x // ds, Pc - 1))
    smask = torch.zeros(1, Pc, Pc, dtype=torch.bool, device=device)
    smask[0, sc_y, sc_x] = True

    actual_n_iters = max(n_iters, 2 * Pc) if n_iters_adaptive else n_iters
    cfg = EikonalConfig(n_iters=actual_n_iters, h=float(ds), tau_min=0.03,
                        tau_branch=0.05, tau_update=0.03,
                        use_redblack=True, monotone=True)
    eik_out = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg,
                                    return_convergence=return_convergence)
    if return_convergence:
        T, conv_curve = eik_out[0][0], eik_out[1]
    else:
        T = eik_out[0]
        conv_curve = None

    T_vals = []
    for ki in range(K):
        tr_y = max(0, min((int(tgt_yx[ki, 0]) - y0) // ds, Pc - 1))
        tr_x = max(0, min((int(tgt_yx[ki, 1]) - x0) // ds, Pc - 1))
        T_vals.append(float(T[tr_y, tr_x].item()))
    T_arr = np.array(T_vals, dtype=np.float64)

    if gate < 1.0:
        d_euc = np.sqrt(((anc_yx.cpu().numpy().astype(np.float64)
                          - tgt_yx.cpu().numpy().astype(np.float64)) ** 2).sum(-1))
        T_arr = d_euc + gate * (T_arr - d_euc)

    p_vals = roi_c.detach().cpu().numpy().ravel()
    in_01 = ((p_vals > 0.01) & (p_vals < 0.99)).sum() / max(len(p_vals), 1)
    roi_truncated = (yy0 > y0) or (yy1 < y1) or (xx0 > x0) or (xx1 < x1)
    valid_h, valid_w = yy1 - yy0, xx1 - xx0
    pad_frac = 1.0 - (valid_h * valid_w) / max(P * P, 1)

    diag = {
        "P": P, "ds": ds, "Pc": Pc,
        "actual_n_iters": actual_n_iters,
        "cost_min": float(cost.min()), "cost_max": float(cost.max()),
        "p_in_01_frac": float(in_01),
        "roi_truncated": bool(roi_truncated),
        "pad_frac": float(pad_frac),
    }
    result = {"pred": T_arr, "diag": diag}
    if conv_curve is not None:
        result["conv_curve"] = conv_curve
    
    if return_T:
        T_up = F.interpolate(T[None, None].float(), size=(P, P),
                             mode="bilinear", align_corners=False).squeeze()
        tgt_coarse = []
        for ki in range(K):
            tc_y = max(0, min((int(tgt_yx[ki, 0]) - y0) // ds, Pc - 1))
            tc_x = max(0, min((int(tgt_yx[ki, 1]) - x0) // ds, Pc - 1))
            tgt_coarse.append((tc_y, tc_x))
        result.update({
            "T_patch": T_up, "T_coarse": T, "roi_y0": y0, "roi_x0": x0,
            "roi_P": P, "ds": ds, "Pc": Pc, "src_coarse": (sc_y, sc_x),
            "tgt_coarse": tgt_coarse
        })
    return result


# ── True Multiscale Eikonal (Dynamic Global Cropping & Target Fusion) ────────

def eikonal_solve_true_multiscale(
    prob_full: torch.Tensor,
    anc_global: torch.Tensor,
    tgt_global: torch.Tensor,
    device: torch.device,
    fine_patch: int,
    coarse_max_patch: int,
    margin_coarse: int,
    margin_fine: int,
    ds_coarse: int,
    ds_fine: int,
    n_iters_coarse: int,
    n_iters_fine: int,
    alpha: float,
    gamma: float,
    gate: float,
    pool_mode: str = "avg",
    return_T: bool = False,
    n_iters_adaptive: bool = False,
) -> dict:
    """True Multiscale: Fine solver for close targets, Dynamic Coarse solver for distant targets."""
    K = tgt_global.shape[0]
    if K == 0:
        return {"pred": np.empty(0), "diag": {}}

    cy, cx = int(anc_global[0]), int(anc_global[1])

    # --- 1. Fine Branch ---
    fine_rp, fy0, fx0 = crop_square_patch(prob_full, cy, cx, fine_patch)
    anc_f = anc_global - torch.tensor([fy0, fx0], device=device)
    tgt_f = tgt_global - torch.tensor([fy0, fx0], device=device)

    # Which targets are naturally inside the 512x512 patch?
    in_fine = ((tgt_f[:, 0] >= 0) & (tgt_f[:, 0] < fine_patch) &
               (tgt_f[:, 1] >= 0) & (tgt_f[:, 1] < fine_patch))
    idx_fine = torch.where(in_fine)[0]

    res_fine = None
    if idx_fine.numel() > 0:
        res_fine = eikonal_solve_fixed_ds(
            fine_rp, anc_f, tgt_f[idx_fine], device, margin_fine,
            n_iters_fine, ds_fine, alpha, gamma, gate, pool_mode=pool_mode, return_T=return_T,
            n_iters_adaptive=n_iters_adaptive)

    # --- 2. Dynamic Coarse Branch (Lazy Evaluation) ---
    # Only use coarse for targets OUTSIDE the 512 patch. Never discard fine results due to
    # roi_truncated: ds=8 fine is always better than ds=16 coarse for in-patch targets
    # (quantization error from ds=16 collapses nearby points to same grid cell).
    need_coarse = ~in_fine
    idx_coarse = torch.where(need_coarse)[0]

    res_coarse = None
    if idx_coarse.numel() > 0:
        # Dynamically size the coarse patch based on target span
        tgt_c_global = tgt_global[idx_coarse]
        span_max = int(torch.max(torch.abs(tgt_c_global.float() - anc_global.float())).item())
        needed = max(2 * (span_max + margin_coarse) + 1, fine_patch * 2)
        coarse_patch = min(needed, coarse_max_patch)

        coarse_rp, cy0, cx0 = crop_square_patch(prob_full, cy, cx, coarse_patch)
        anc_c = anc_global - torch.tensor([cy0, cx0], device=device)
        tgt_c = tgt_c_global - torch.tensor([cy0, cx0], device=device)

        res_coarse = eikonal_solve_fixed_ds(
            coarse_rp, anc_c, tgt_c, device, margin_coarse,
            n_iters_coarse, ds_coarse, alpha, gamma, gate, pool_mode=pool_mode, return_T=return_T,
            n_iters_adaptive=n_iters_adaptive)

    # --- 3. Fusion ---
    pred = np.zeros(K, dtype=np.float64)
    if res_fine is not None:
        pred[idx_fine.cpu().numpy()] = res_fine["pred"]
    if res_coarse is not None:
        pred[idx_coarse.cpu().numpy()] = res_coarse["pred"]

    diag = {
        "used_coarse": idx_coarse.numel() > 0,
        "used_coarse_ratio": idx_coarse.numel() / max(K, 1),
        "P": res_fine["diag"]["P"] if res_fine else 0,
        "Pc": res_fine["diag"]["Pc"] if res_fine else 0,
        "ds": ds_fine,
        "actual_n_iters": res_fine["diag"].get("actual_n_iters", n_iters_fine) if res_fine else n_iters_fine,
        "cost_min": res_coarse["diag"]["cost_min"] if res_coarse else (res_fine["diag"]["cost_min"] if res_fine else 0),
        "cost_max": res_coarse["diag"]["cost_max"] if res_coarse else (res_fine["diag"]["cost_max"] if res_fine else 0),
        "p_in_01_frac": res_coarse["diag"]["p_in_01_frac"] if res_coarse else (res_fine["diag"]["p_in_01_frac"] if res_fine else 0),
    }
    
    if res_coarse is not None:
        diag["P_coarse"] = res_coarse["diag"]["P"]
        diag["Pc_coarse"] = res_coarse["diag"]["Pc"]

    result = {"pred": pred, "diag": diag}
    
    # Pass out visualization variables matching whichever solver was primarily used
    if return_T:
        src = res_fine if res_fine else res_coarse
        if src:
            for k in ("T_patch", "T_coarse", "roi_y0", "roi_x0", "roi_P", "ds", "Pc", "src_coarse", "tgt_coarse"):
                if k in src:
                    result[k] = src[k]
            # Offset tracking for rendering paths in global space (use "is" for identity, not "==")
            result["global_y0"] = cy0 if (src is res_coarse) else fy0
            result["global_x0"] = cx0 if (src is res_coarse) else fx0
            # Indices of targets that have T values (for viz alignment with tgts_global)
            result["viz_tgt_indices"] = idx_coarse.cpu().numpy() if (src is res_coarse) else idx_fine.cpu().numpy()
            
    return result


# ── Evaluate one configuration over all cases ────────────────────────────────

def eval_config(
    mask: torch.Tensor,
    all_cases: list,
    *,
    k: int, patch_size: int, margin: int, pscale: float,
    n_iters: int, ds: int, alpha: float, gamma: float, gate: float,
    device: torch.device,
    use_multiscale: bool = False,
    coarse_max_patch: int = 2048,
    margin_coarse: int = 128,
    margin_fine: int = 64,
    pool_mode: str = "avg",
    n_iters_adaptive: bool = False,
) -> dict:
    sp_eik, sp_euc = [], []
    all_P, all_Pc = [], []
    all_P_coarse, all_Pc_coarse = [], []
    all_actual_n_iters = []
    cost_mins, cost_maxs, p01_fracs = [], [], []
    used_coarse_cnt = 0
    roi_truncated_cnt = 0

    for nodes_yx, info in all_cases:
        case = info["case_idx"]
        meta_px = int(info["meta_sat_height_px"])
        H_ds, W_ds = mask.shape[-2:]
        N = nodes_yx.shape[0]

        for anc_idx in range(N):
            _, nn_global = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=anc_idx)
            anc = nodes_yx[anc_idx]
            tgts_g = nodes_yx[nn_global]

            if use_multiscale:
                # ── TRUE MULTISCALE: Use all K targets, pass full mask ──
                result = eikonal_solve_true_multiscale(
                    mask, anc, tgts_g, device,
                    fine_patch=patch_size, coarse_max_patch=coarse_max_patch,
                    margin_coarse=margin_coarse, margin_fine=margin_fine,
                    ds_coarse=2 * ds, ds_fine=ds,
                    n_iters_coarse=max(20, n_iters // 2), n_iters_fine=n_iters,
                    alpha=alpha, gamma=gamma, gate=gate, pool_mode=pool_mode,
                    n_iters_adaptive=n_iters_adaptive)
                
                pred_px = result["pred"]
                nn_kept = nn_global  # Keep ALL targets!
            else:
                # ── LEGACY SINGLE-SCALE: Strict 512 patch drop ──
                ay, ax = int(anc[0]), int(anc[1])
                py0 = max(0, min(ay - patch_size // 2, H_ds - patch_size))
                px0 = max(0, min(ax - patch_size // 2, W_ds - patch_size))
                py1, px1 = py0 + patch_size, px0 + patch_size
                
                in_p = ((tgts_g[:, 0] >= py0) & (tgts_g[:, 0] < py1) &
                        (tgts_g[:, 1] >= px0) & (tgts_g[:, 1] < px1))
                keep = torch.where(in_p)[0]
                
                if keep.numel() < 3:
                    continue
                
                nn_kept = nn_global[keep]
                tgt_patch = tgts_g[keep] - torch.tensor([py0, px0], device=device)
                anc_patch = torch.tensor([ay - py0, ax - px0], device=device).long()

                rp = mask[py0:py1, px0:px1]
                if rp.shape[0] < patch_size or rp.shape[1] < patch_size:
                    rp = F.pad(rp, (0, patch_size - rp.shape[1], 0, patch_size - rp.shape[0]))

                result = eikonal_solve_fixed_ds(
                    rp, anc_patch, tgt_patch, device, margin,
                    n_iters, ds, alpha, gamma, gate, pool_mode=pool_mode,
                    n_iters_adaptive=n_iters_adaptive)
                pred_px = result["pred"]

            # Shared Stats Calculation
            # Eikonal T ≈ path_length in grid units; physical pixels ≈ T * ds
            T_norm = pred_px * ds * pscale / max(meta_px, 1)
            dd = result["diag"]
            
            all_P.append(dd.get("P", 0))
            all_Pc.append(dd.get("Pc", 0))
            all_actual_n_iters.append(dd.get("actual_n_iters", n_iters))
            if use_multiscale and "P_coarse" in dd:
                all_P_coarse.append(dd["P_coarse"])
                all_Pc_coarse.append(dd["Pc_coarse"])
                used_coarse_cnt += int(dd.get("used_coarse", False))
                
            cost_mins.append(dd.get("cost_min", 0.0))
            cost_maxs.append(dd.get("cost_max", 0.0))
            p01_fracs.append(dd.get("p_in_01_frac", 0.0))
            if dd.get("roi_truncated", False):
                roi_truncated_cnt += 1

            # GT & Metrics
            nn_np = nn_kept.cpu().numpy().astype(int)
            gt_arr = np.asarray(info["undirected_dist_norm"][case][anc_idx, nn_np], dtype=np.float64)
            
            valid = np.isfinite(T_norm) & (T_norm < 1e3) & np.isfinite(gt_arr)
            if valid.sum() < 3:
                continue
            sp_eik.append(spearmanr(T_norm[valid], gt_arr[valid]))

            # Euc distance always measured in global coordinates for consistency
            anc_global_np = anc.cpu().numpy().astype(np.float64)
            tgt_global_np = nodes_yx[nn_kept].cpu().numpy().astype(np.float64)
            d_euc = np.sqrt(((anc_global_np - tgt_global_np)**2).sum(-1)) * pscale / max(meta_px, 1)
            
            v2 = valid & np.isfinite(d_euc)
            if v2.sum() >= 3:
                sp_euc.append(spearmanr(d_euc[v2], gt_arr[v2]))

    n_anchors_total = len(all_P)
    diag_agg = {
        "P": int(np.mean(all_P)) if all_P else 0,
        "ds": ds,
        "Pc": int(np.mean(all_Pc)) if all_Pc else 0,
        "actual_n_iters": int(np.mean(all_actual_n_iters)) if all_actual_n_iters else n_iters,
        "cost_min": float(np.mean(cost_mins)) if cost_mins else 0,
        "cost_max": float(np.mean(cost_maxs)) if cost_maxs else 0,
        "p_in_01_frac": float(np.mean(p01_fracs)) if p01_fracs else 0,
        "roi_truncated_ratio": roi_truncated_cnt / max(n_anchors_total, 1),
    }
    if use_multiscale and all_P_coarse:
        diag_agg["P_coarse"] = int(np.mean(all_P_coarse))
        diag_agg["Pc_coarse"] = int(np.mean(all_Pc_coarse))
        diag_agg["used_coarse_ratio"] = used_coarse_cnt / max(len(all_P), 1)
        
    return {
        "sp_mean": float(np.mean(sp_eik)) if sp_eik else float("nan"),
        "sp_std": float(np.std(sp_eik)) if sp_eik else 0.0,
        "sp_euc": float(np.mean(sp_euc)) if sp_euc else float("nan"),
        "n_anchors": len(sp_eik),
        "diag": diag_agg,
    }


# ── Visualization ────────────────────────────────────────────────────────────

def visualize_cases(
    mask: torch.Tensor, gt_mask_hard: torch.Tensor, rgb_ds: torch.Tensor,
    all_cases: list, *,
    k: int, patch_size: int, margin: int, pscale: float,
    n_iters: int, ds: int, alpha: float, gamma: float, gate: float,
    device: torch.device, save_dir: str,
    max_cases: int = 3, max_anchors_per_case: int = 3,
    use_multiscale: bool = False,
    margin_coarse: int = 128,
    margin_fine: int = 64,
    pool_mode: str = "avg",
):
    try:
        from backups.evaluator import trace_path_from_T
        from backups.trainer import _plot_paths_standalone
    except ImportError as e:
        print(f"  [WARN] Visualization skipped: could not import backups ({e})")
        return
    os.makedirs(save_dir, exist_ok=True)
    H_ds, W_ds = mask.shape[-2:]
    count = 0

    for ci, (nodes_yx, info) in enumerate(all_cases[:max_cases]):
        case = info["case_idx"]
        N = nodes_yx.shape[0]

        for anc_idx in range(min(N, max_anchors_per_case)):
            _, nn_global = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=anc_idx)
            anc = nodes_yx[anc_idx]
            tgts_g = nodes_yx[nn_global]
            ay, ax = int(anc[0]), int(anc[1])

            if use_multiscale:
                result = eikonal_solve_true_multiscale(
                    mask, anc, tgts_g, device,
                    fine_patch=patch_size, coarse_max_patch=2048,
                    margin_coarse=margin_coarse, margin_fine=margin_fine,
                    ds_coarse=2 * ds, ds_fine=ds,
                    n_iters_coarse=max(20, n_iters // 2), n_iters_fine=n_iters,
                    alpha=alpha, gamma=gamma, gate=gate, pool_mode=pool_mode, return_T=True)
                
                nn_kept = nn_global
                global_y0, global_x0 = result.get("global_y0", ay - patch_size//2), result.get("global_x0", ax - patch_size//2)
                py0, px0 = global_y0, global_x0 # Treat dynamic offset as the patch offset for Viz
            else:
                py0 = max(0, min(ay - patch_size // 2, H_ds - patch_size))
                px0 = max(0, min(ax - patch_size // 2, W_ds - patch_size))
                py1, px1 = py0 + patch_size, px0 + patch_size
                
                in_p = ((tgts_g[:, 0] >= py0) & (tgts_g[:, 0] < py1) &
                        (tgts_g[:, 1] >= px0) & (tgts_g[:, 1] < px1))
                keep = torch.where(in_p)[0]
                if keep.numel() < 3: continue
                
                nn_kept = nn_global[keep]
                tgt_patch = tgts_g[keep] - torch.tensor([py0, px0], device=device)
                anc_patch = torch.tensor([ay - py0, ax - px0], device=device).long()
                rp = mask[py0:py1, px0:px1]
                if rp.shape[0] < patch_size or rp.shape[1] < patch_size:
                    rp = F.pad(rp, (0, patch_size - rp.shape[1], 0, patch_size - rp.shape[0]))

                result = eikonal_solve_fixed_ds(
                    rp, anc_patch, tgt_patch, device, margin,
                    n_iters, ds, alpha, gamma, gate, pool_mode=pool_mode, return_T=True)

            if "T_coarse" not in result: continue # Edge case failure

            T_patch, T_coarse = result["T_patch"], result["T_coarse"]
            roi_y0, roi_x0, P = result["roi_y0"], result["roi_x0"], result["roi_P"]
            cur_ds, sc_y, sc_x = result["ds"], result["src_coarse"][0], result["src_coarse"][1]
            tgt_coarse = result["tgt_coarse"]

            src_c = torch.tensor([sc_y, sc_x], device=device, dtype=torch.long)
            K = len(tgt_coarse)
            paths_global, t_vals = [], []
            for ki in range(K):
                tc_y, tc_x = tgt_coarse[ki]
                t_vals.append(float(T_coarse[tc_y, tc_x].item()))
                tgt_c = torch.tensor([tc_y, tc_x], device=device, dtype=torch.long)
                pp = trace_path_from_T(T_coarse, src_c, tgt_c, device, diag=True)
                
                if len(pp) >= 2:
                    # Map from Coarse grid -> ROI local -> Global Patch
                    paths_global.append(
                        [(p[0] * cur_ds + roi_y0 + py0, p[1] * cur_ds + roi_x0 + px0) for p in pp]
                    )
                else:
                    paths_global.append(None)

            anc_global = torch.tensor([ay, ax], device=device, dtype=torch.long)
            # When multiscale, t_vals/paths correspond to a subset (viz_tgt_indices)
            viz_idx = result.get("viz_tgt_indices")
            if viz_idx is not None:
                tgts_global = nodes_yx[nn_kept][viz_idx]
                nn_np = nn_kept.cpu().numpy().astype(int)[viz_idx]
            else:
                tgts_global = nodes_yx[nn_kept]
                nn_np = nn_kept.cpu().numpy().astype(int)

            save_path = os.path.join(save_dir, f"case{case}_anc{anc_idx}.png")
            _plot_paths_standalone(
                rgb=rgb_ds, road_prob=mask, gt_mask=gt_mask_hard, src=anc_global,
                tgts=tgts_global, T_patch=T_patch, t_y0=roi_y0 + py0, t_x0=roi_x0 + px0,
                P=P, patch_y0=py0, patch_x0=px0, patch_h=patch_size, patch_w=patch_size,
                paths_global=paths_global, t_vals=t_vals, nn_np=nn_np, save_path=save_path
            )
            count += 1
            print(f"  Saved: {save_path}")

    print(f"Total {count} visualizations saved to {save_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif_path", type=str,
                        default="Gen_dataset_V2/Gen_dataset/34.269888_108.947180/"
                                "01_34.350697_108.914569_3000.0/"
                                "crop_34.350697_108.914569_3000.0_z16.tif")
    parser.add_argument("--downsample", type=int, default=0)
    parser.add_argument("--k", type=int, default=5,
                    help="KNN neighborhood size for evaluation")
    parser.add_argument("--n_cases", type=int, default=5)
    parser.add_argument("--vis_dir", type=str, default="images/gt_ablation_vis",
                        help="Directory to save path visualizations")
    parser.add_argument("--max_vis_cases", type=int, default=3)
    parser.add_argument("--max_vis_anchors", type=int, default=3)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config for SAMRoute model (enables GT vs model comparison)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Checkpoint path for SAMRoute model")
    parser.add_argument("--multiscale", action="store_true", default=True,
                        help="Use multiscale Eikonal (default: True, reduces ROI truncation)")
    parser.add_argument("--no-multiscale", action="store_false", dest="multiscale",
                        help="Disable multiscale, use legacy 512-patch mode")
    parser.add_argument("--pool_mode", type=str, default="avg", choices=["max", "avg"],
                        help="Pooling mode for downsampling the mask (max or avg)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tif_path = args.tif_path
    if not os.path.isabs(tif_path):
        tif_path = os.path.join(str(_HERE.parent), tif_path)
    region_dir = os.path.dirname(tif_path)

    rgb = load_tif_image(tif_path).to(device)
    _, _, H_orig, W_orig = rgb.shape
    scale = 1.0
    if args.downsample > 0 and max(H_orig, W_orig) > args.downsample:
        scale = args.downsample / max(H_orig, W_orig)
    H_ds, W_ds = max(1, int(H_orig * scale)), max(1, int(W_orig * scale))
    pscale = 1.0 / scale if scale != 1.0 else 1.0
    if scale != 1.0:
        rgb_ds = F.interpolate(rgb, size=(H_ds, W_ds),
                               mode="bilinear", align_corners=False)
    else:
        rgb_ds = rgb
    print(f"Device: {device}")
    print(f"Image: {H_orig}x{W_orig} -> {H_ds}x{W_ds}")

    m = load_gt_mask(region_dir, "r5")
    if m.shape != (H_orig, W_orig):
        raise ValueError(
            f"GT mask shape {m.shape} does not match TIF shape ({H_orig}, {W_orig}). "
            "Mask and TIF must share the same pixel dimensions for correct alignment."
        )
    mt = torch.from_numpy(m).to(device)
    if scale != 1.0:
        mt = F.interpolate(mt[None, None], size=(H_ds, W_ds),
                           mode="bilinear", align_corners=False).squeeze()

    masks = {
        "hard": make_soft_gt(mt, "hard"),
        "blur_s3": make_soft_gt(mt, "blur", sigma=3.0),
        "blur_s8": make_soft_gt(mt, "blur", sigma=8.0),
    }
    for name, mk in masks.items():
        v = mk.cpu().numpy().ravel()
        in_01 = ((v > 0.01) & (v < 0.99)).sum() / len(v)
        print(f"  mask '{name}': mean={mk.mean():.4f}, "
              f"p_in_(0,1)={in_01:.4f} ({in_01*100:.1f}%)")
    print()

    all_cases = []
    for ci in range(args.n_cases):
        ny, nf = load_nodes_for_case(tif_path, H_orig, W_orig, case_idx=ci,
                                     device=device, scale=scale,
                                     H_ds=H_ds, W_ds=W_ds)
        all_cases.append((ny, nf))
    print(f"Loaded {args.n_cases} cases\n")

    common = dict(k=args.k, patch_size=512, margin=64, pscale=pscale, device=device, pool_mode=args.pool_mode)

    # ── Ablation matrix ──────────────────────────────────────────────────────
    # (label, mask_key, ds, n_iters, alpha, gamma, gate)
    configs = []

    # Group 1: Decouple ds vs n_iters (hard mask, default alpha/gamma)
    # Fix ds=16, vary n_iters → isolate convergence effect
    for n in [20, 40, 80, 160, 320]:
        configs.append((f"hard ds=16 n={n:<3d} g=0.8", "hard", 16, n, 20., 2., 0.8))
    # Fix ds=8, vary n_iters
    for n in [20, 40, 80, 160, 320]:
        configs.append((f"hard ds=8  n={n:<3d} g=0.8", "hard", 8, n, 20., 2., 0.8))
    # Fix n_iters=80, vary ds → isolate grid resolution
    for d in [32, 16, 8, 4]:
        configs.append((f"hard ds={d:<2d} n=80  g=0.8", "hard", d, 80, 20., 2., 0.8))

    # Group 2: gate sweep at best solver (ds=8, n=80)
    for g in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        configs.append((f"hard ds=8  n=80  g={g:.1f}", "hard", 8, 80, 20., 2., g))

    # Group 3: alpha/gamma on SOFT mask (blur_s8) to test if they matter
    for a in [5, 20, 80]:
        for gm in [1.0, 2.0, 4.0]:
            configs.append((f"soft8 a={a:<3d} g2={gm:.0f} ds=8 n=80",
                            "blur_s8", 8, 80, float(a), gm, 0.8))

    # Group 4: hard vs soft mask comparison at same solver
    for mk in ["hard", "blur_s3", "blur_s8"]:
        configs.append((f"{mk:7s} ds=8  n=80  g=0.8", mk, 8, 80, 20., 2., 0.8))

    # Group 5: multiscale (when --multiscale)
    if args.multiscale:
        configs.append((f"multiscale ds16+8 g=0.8", "hard", 8, 80, 20., 2., 0.8, True))

    # ── Run ──────────────────────────────────────────────────────────────────
    results = []
    for tup in configs:
        use_multiscale = len(tup) == 8 and tup[7] is True
        if use_multiscale:
            label, mk, ds, n_it, alpha, gamma, gate, _ = tup
        else:
            label, mk, ds, n_it, alpha, gamma, gate = tup
        r = eval_config(masks[mk], all_cases, n_iters=n_it, ds=ds,
                        alpha=alpha, gamma=gamma, gate=gate,
                        use_multiscale=use_multiscale, **common)
        results.append((label, r))
        d = r["diag"]
        p_str = f"{d.get('P',0):>4d}" if "P_coarse" not in d else f"{d.get('P',0)}/{d.get('P_coarse',0)}"
        pc_str = f"{d.get('Pc',0):>3d}" if "Pc_coarse" not in d else f"{d.get('Pc',0)}/{d.get('Pc_coarse',0)}"
        fallback_str = f" fallback={d.get('used_coarse_ratio',0):.1%}" if "used_coarse_ratio" in d else ""
        trunc_str = f" trunc={d.get('roi_truncated_ratio',0):.1%}" if d.get('roi_truncated_ratio', 0) > 0 else ""
        print(f"{label:40s}  Sp={r['sp_mean']:+.4f}±{r['sp_std']:.3f}  "
              f"Euc={r['sp_euc']:+.4f}  "
              f"P~{p_str:>9s}  ds={d.get('ds',0):>2d}  Pc~{pc_str:>7s}  "
              f"cost~[{d.get('cost_min',0):.1f},{d.get('cost_max',0):.1f}]  "
              f"p01={d.get('p_in_01_frac',0):.3f}  "
              f"anc={r['n_anchors']}{fallback_str}{trunc_str}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*115}")
    print(f"{'Config':40s}  {'Sp':>8s} {'±std':>6s}  {'Euc':>8s}  "
          f"{'P~':>5s} {'ds':>3s} {'Pc~':>4s}  {'cost~[min,max]':>16s}  {'p01%':>6s}")
    print(f"{'-'*115}")
    for label, r in results:
        d = r["diag"]
        cr = f"[{d.get('cost_min',0):.1f},{d.get('cost_max',0):.1f}]"
        p_str = str(d.get('P',0)) if "P_coarse" not in d else f"{d.get('P',0)}/{d.get('P_coarse',0)}"
        pc_str = str(d.get('Pc',0)) if "Pc_coarse" not in d else f"{d.get('Pc',0)}/{d.get('Pc_coarse',0)}"
        fallback_str = f"  fb={d.get('used_coarse_ratio',0):.0%}" if "used_coarse_ratio" in d else ""
        trunc_str = f"  trunc={d.get('roi_truncated_ratio',0):.0%}" if d.get('roi_truncated_ratio', 0) > 0 else ""
        print(f"{label:40s}  {r['sp_mean']:>+8.4f} {r['sp_std']:>5.3f}  "
              f"{r['sp_euc']:>+8.4f}  "
              f"{p_str:>9s}  {d.get('ds',0):>3}  {pc_str:>7s}  "
              f"{cr:>16s}  {d.get('p_in_01_frac',0)*100:>5.1f}%{fallback_str}{trunc_str}")
    print(f"{'='*115}")

    # ── Path visualization (representative config: hard, ds=8, n_iters=80) ──
    vis_ds, vis_n, vis_alpha, vis_gamma, vis_gate = 8, 80, 20.0, 2.0, 0.8
    print(f"\n{'='*60}")
    print("Generating GT path visualizations ...")
    visualize_cases(
        mask=masks["hard"],
        gt_mask_hard=masks["hard"],
        rgb_ds=rgb_ds,
        all_cases=all_cases,
        n_iters=vis_n, ds=vis_ds,
        alpha=vis_alpha, gamma=vis_gamma, gate=vis_gate,
        save_dir=os.path.join(args.vis_dir, "gt"),
        max_cases=args.max_vis_cases,
        max_anchors_per_case=args.max_vis_anchors,
        use_multiscale=args.multiscale,
        **common,
    )

    # ── Model comparison (optional) ───────────────────────────────────────
    if args.config and args.ckpt_path:
        print(f"\n{'='*60}")
        print("Loading model for GT vs Model comparison ...")
        model_rp = load_model_road_prob(args.config, args.ckpt_path, rgb_ds, device)
        masks["model"] = model_rp

        # Run a focused comparison: GT variants vs Model, same solver params
        cmp_configs = []
        for mk in ["hard", "blur_s3", "blur_s8", "model"]:
            for g in [0.8]:
                cmp_configs.append(
                    (f"{mk:7s} ds=8  n=80  g={g:.1f}", mk, 8, 80, 20., 2., g))

        cmp_results = []
        for label, mk, ds, n_it, alpha, gamma, gate in cmp_configs:
            r = eval_config(masks[mk], all_cases, n_iters=n_it, ds=ds,
                            alpha=alpha, gamma=gamma, gate=gate,
                            use_multiscale=args.multiscale, **common)
            cmp_results.append((label, r))

        print(f"\n{'='*80}")
        print(f"  GT vs Model Comparison  (same Eikonal solver: ds=8, n=80, g=0.8)")
        print(f"{'='*80}")
        print(f"{'Source':40s}  {'Sp':>8s} {'±std':>6s}  {'Euc':>8s}  {'n_anc':>6s}")
        print(f"{'-'*80}")
        for label, r in cmp_results:
            print(f"{label:40s}  {r['sp_mean']:>+8.4f} {r['sp_std']:>5.3f}  "
                  f"{r['sp_euc']:>+8.4f}  {r['n_anchors']:>6d}")
        print(f"{'='*80}")

        # Visualize model paths
        print("\nGenerating Model path visualizations ...")
        visualize_cases(
            mask=masks["model"],
            gt_mask_hard=masks["hard"],
            rgb_ds=rgb_ds,
            all_cases=all_cases,
            n_iters=vis_n, ds=vis_ds,
            alpha=vis_alpha, gamma=vis_gamma, gate=vis_gate,
            save_dir=os.path.join(args.vis_dir, "model"),
            max_cases=args.max_vis_cases,
            max_anchors_per_case=args.max_vis_anchors,
            use_multiscale=args.multiscale,
            **common,
        )


if __name__ == "__main__":
    main()
