#!/usr/bin/env python3
"""
Diagnostic: compute Spearman coefficient using GT road masks instead of model
predictions, at multiple downsample factors.

This answers: "If we had perfect segmentation, would the Eikonal solver give
correct distance ranking?"  If not, the bottleneck is in ds / cost function,
not in the model.

Usage:
    cd eikonal_solver
    python test_gt_spearman.py [--tif_path PATH] [--downsample 1024] [--k 10]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eikonal import EikonalConfig, eikonal_soft_sweeping
from evaluator import (
    load_tif_image,
    pick_anchor_and_knn,
    spearmanr,
    kendall_tau,
    pairwise_order_acc,
)
from dataset import load_nodes_from_npz


def prob_to_cost(prob: torch.Tensor,
                 alpha: float = 20.0, gamma: float = 2.0,
                 block_th: float = 0.05, block_alpha: float = 50.0,
                 block_smooth: float = 50.0,
                 cost_mode: str = "additive") -> torch.Tensor:
    """Standalone cost function with multiple modes.

    cost_mode:
        "additive"  — original: 1 + alpha*(1-p)^gamma + block  (default)
        "binary"    — hard threshold: road=1, off-road=high_cost
    """
    prob = prob.clamp(0.0, 1.0)
    if cost_mode == "binary":
        road = (prob > 0.3).float()
        cost = 1.0 + (1.0 - road) * 100.0
        return cost
    # default: additive
    cost = 1.0 + alpha * (1.0 - prob).pow(gamma)
    if block_th > 0.0 and block_alpha > 0.0:
        block = block_alpha * torch.sigmoid(-block_smooth * (prob - block_th))
        cost = cost + block
    return cost.clamp_min(1e-6)


def eikonal_one_src_k_tgts(
    rp: torch.Tensor,
    anc_patch: torch.Tensor,
    tgt_patch: torch.Tensor,
    device: torch.device,
    margin: int,
    n_iters: int,
    min_ds: int,
    pool_mode: str = "avg",
    cost_mode: str = "additive",
) -> np.ndarray:
    """One Eikonal solve covering all K targets; return K T-values.

    pool_mode: "avg" (default) or "max"
    cost_mode: "additive" (default) or "binary"
    """
    K = tgt_patch.shape[0]
    if K == 0:
        return np.empty(0, dtype=np.float64)

    H_r, W_r = rp.shape
    span_max = int(torch.max(torch.abs(tgt_patch.float() - anc_patch.float())).item())
    half = span_max + margin
    P = max(2 * half + 1, 64)

    y0 = int(anc_patch[0]) - half
    x0 = int(anc_patch[1]) - half
    y1, x1 = y0 + P, x0 + P
    yy0, xx0 = max(y0, 0), max(x0, 0)
    yy1, xx1 = min(y1, H_r), min(x1, W_r)

    roi = F.pad(rp[yy0:yy1, xx0:xx1],
                (xx0 - x0, x1 - xx1, yy0 - y0, y1 - yy1), value=0.0)

    sr_y = max(0, min(int(anc_patch[0]) - y0, P - 1))
    sr_x = max(0, min(int(anc_patch[1]) - x0, P - 1))

    ds = max(min_ds, math.ceil(P / max(n_iters, 1)))
    if ds > 1:
        P_pad = math.ceil(P / ds) * ds
        if P_pad > P:
            roi = F.pad(roi, (0, P_pad - P, 0, P_pad - P), value=0.0)
        roi_4d = roi[None, None]
        if pool_mode == "max":
            roi_c = F.max_pool2d(roi_4d, kernel_size=ds, stride=ds).squeeze()
        else:
            roi_c = F.avg_pool2d(roi_4d, kernel_size=ds, stride=ds).squeeze()
        cost = prob_to_cost(roi_c, cost_mode=cost_mode)
        Pc = cost.shape[0]
        sc_y = max(0, min(sr_y // ds, Pc - 1))
        sc_x = max(0, min(sr_x // ds, Pc - 1))
        smask = torch.zeros(1, Pc, Pc, dtype=torch.bool, device=device)
        smask[0, sc_y, sc_x] = True
        cfg = EikonalConfig(n_iters=n_iters, h=float(ds), tau_min=0.03,
                            tau_branch=0.05, tau_update=0.03,
                            use_redblack=True, monotone=True)
        T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)[0]
        vals = []
        for ki in range(K):
            tr_y = max(0, min((int(tgt_patch[ki, 0]) - y0) // ds, Pc - 1))
            tr_x = max(0, min((int(tgt_patch[ki, 1]) - x0) // ds, Pc - 1))
            vals.append(float(T[tr_y, tr_x].item()))
    else:
        cost = prob_to_cost(roi, cost_mode=cost_mode)
        smask = torch.zeros(1, P, P, dtype=torch.bool, device=device)
        smask[0, sr_y, sr_x] = True
        cfg = EikonalConfig(n_iters=max(n_iters, P), h=1.0, tau_min=0.03,
                            tau_branch=0.05, tau_update=0.03,
                            use_redblack=True, monotone=True)
        T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)[0]
        vals = []
        for ki in range(K):
            tr_y = max(0, min(int(tgt_patch[ki, 0]) - y0, P - 1))
            tr_x = max(0, min(int(tgt_patch[ki, 1]) - x0, P - 1))
            vals.append(float(T[tr_y, tr_x].item()))

    return np.array(vals, dtype=np.float64)


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
    return arr[:, :, 0].astype(np.float32) / 255.0


def evaluate_with_mask(
    mask_ds: torch.Tensor,
    nodes_yx: torch.Tensor,
    info: dict,
    *,
    k: int = 10,
    min_in_patch: int = 3,
    patch_size: int = 512,
    margin: int = 64,
    n_iters: int = 20,
    min_ds: int = 8,
    pscale: float = 1.0,
    device: torch.device = torch.device("cpu"),
    pool_mode: str = "avg",
    cost_mode: str = "additive",
) -> Dict[str, float]:
    """Run Spearman evaluation using a given prob map (GT mask or model pred)."""
    H_ds, W_ds = mask_ds.shape
    case = info["case_idx"]
    meta_px = int(info["meta_sat_height_px"])
    N = nodes_yx.shape[0]

    sp_acc, kd_acc, pw_acc, sp_euc_acc = [], [], [], []

    for anc_idx in range(N):
        _, nn_global = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=anc_idx)
        anc = nodes_yx[anc_idx]
        ay, ax = int(anc[0]), int(anc[1])
        py0 = max(0, min(ay - patch_size // 2, H_ds - patch_size))
        px0 = max(0, min(ax - patch_size // 2, W_ds - patch_size))
        py1, px1 = py0 + patch_size, px0 + patch_size
        tgts_g = nodes_yx[nn_global]
        in_p = ((tgts_g[:, 0] >= py0) & (tgts_g[:, 0] < py1) &
                (tgts_g[:, 1] >= px0) & (tgts_g[:, 1] < px1))
        keep = torch.where(in_p)[0]
        if keep.numel() < min_in_patch:
            continue
        nn_kept = nn_global[keep]
        tgt_patch = tgts_g[keep] - torch.tensor([py0, px0], device=device)
        anc_patch = torch.tensor([ay - py0, ax - px0], device=device).long()

        rp = mask_ds[py0:py1, px0:px1]
        if rp.shape[0] < patch_size or rp.shape[1] < patch_size:
            rp = F.pad(rp, (0, patch_size - rp.shape[1], 0, patch_size - rp.shape[0]))

        T_raw = eikonal_one_src_k_tgts(
            rp, anc_patch, tgt_patch, device, margin, n_iters, min_ds,
            pool_mode=pool_mode, cost_mode=cost_mode)
        T_norm = T_raw * pscale / max(meta_px, 1)
        nn_np = nn_kept.cpu().numpy().astype(int)
        gt_arr = np.asarray(info["undirected_dist_norm"][case][anc_idx, nn_np],
                            dtype=np.float64)
        euc_arr = np.asarray(info["euclidean_dist_norm"][case][anc_idx, nn_np],
                             dtype=np.float64)

        valid = np.isfinite(T_norm) & (T_norm < 1e3) & np.isfinite(gt_arr)
        if valid.sum() < 3:
            continue

        pv, gv = T_norm[valid], gt_arr[valid]
        sp_acc.append(spearmanr(pv, gv))
        kd_acc.append(kendall_tau(pv, gv))
        pw_acc.append(pairwise_order_acc(pv, gv))

        valid_euc = valid & np.isfinite(euc_arr)
        if valid_euc.sum() >= 3:
            sp_euc_acc.append(spearmanr(euc_arr[valid_euc], gt_arr[valid_euc]))

    if not sp_acc:
        return {"spearman": float("nan"), "kendall": float("nan"),
                "pw_acc": float("nan"), "spearman_euc": float("nan"),
                "n_anchors": 0}
    return {
        "spearman": float(np.mean(sp_acc)),
        "kendall": float(np.mean(kd_acc)),
        "pw_acc": float(np.nanmean(pw_acc)),
        "spearman_euc": float(np.mean(sp_euc_acc)) if sp_euc_acc else float("nan"),
        "n_anchors": len(sp_acc),
    }


def load_nodes_for_case(tif_path: str, H: int, W: int, case_idx: int,
                        device: torch.device, scale: float,
                        H_ds: int, W_ds: int):
    """Load nodes for a specific case_idx (deterministic)."""
    rp_dummy = torch.zeros(H, W, device=device)
    nodes_orig, info = load_nodes_from_npz(
        tif_path, rp_dummy, p_count=20, snap=False, verbose=False,
        case_idx=case_idx)
    if scale != 1.0:
        nodes_yx = (nodes_orig.float() * scale).long()
        nodes_yx[:, 0].clamp_(0, H_ds - 1)
        nodes_yx[:, 1].clamp_(0, W_ds - 1)
    else:
        nodes_yx = nodes_orig
    return nodes_yx.to(device), info


def main():
    parser = argparse.ArgumentParser(description="GT mask Spearman diagnostic")
    parser.add_argument("--tif_path", type=str,
                        default="Gen_dataset_V2/Gen_dataset/34.269888_108.947180/"
                                "01_34.350697_108.914569_3000.0/"
                                "crop_34.350697_108.914569_3000.0_z16.tif")
    parser.add_argument("--downsample", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--margin", type=int, default=64)
    parser.add_argument("--n_cases", type=int, default=10,
                        help="Number of deterministic cases to average over")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tif_path = args.tif_path
    if not os.path.isabs(tif_path):
        tif_path = os.path.join(str(_HERE.parent), tif_path)

    region_dir = os.path.dirname(tif_path)
    print(f"Region: {region_dir}")
    print(f"Device: {device}")
    print(f"Averaging over {args.n_cases} deterministic cases (case_idx = 0..{args.n_cases-1})")
    print()

    # --- Load TIF for coordinate mapping ---
    rgb = load_tif_image(tif_path).to(device)
    _, _, H_orig, W_orig = rgb.shape
    scale = 1.0
    if args.downsample > 0 and max(H_orig, W_orig) > args.downsample:
        scale = args.downsample / max(H_orig, W_orig)
    H_ds = max(1, int(H_orig * scale))
    W_ds = max(1, int(W_orig * scale))
    pscale = 1.0 / scale if scale != 1.0 else 1.0
    print(f"Image: {H_orig}x{W_orig} -> {H_ds}x{W_ds}  (scale={scale:.4f})")

    # --- Load GT masks ---
    masks = {}
    for variant in ["thin", "r5"]:
        try:
            m = load_gt_mask(region_dir, variant)
            mt = torch.from_numpy(m).to(device)
            if scale != 1.0:
                mt = F.interpolate(mt[None, None], size=(H_ds, W_ds),
                                   mode="bilinear", align_corners=False).squeeze()
            masks[variant] = mt
            road_frac = float((mt > 0.5).sum()) / mt.numel()
            print(f"  Loaded GT mask '{variant}': {mt.shape}, road fraction={road_frac:.4f}")
        except FileNotFoundError as e:
            print(f"  Skip '{variant}': {e}")
    print()

    # --- Define test configurations ---
    # (label, mask_key, min_ds, n_iters, pool_mode, cost_mode)
    configs = [
        # Old baseline: avg_pool + additive cost
        ("avg+add  r5 ds=8 n=20",  "r5", 8,  20, "avg", "additive"),
        # New default: max_pool + additive cost
        ("max+add  r5 ds=8 n=20",  "r5", 8,  20, "max", "additive"),
        ("max+add  r5 ds=8 n=30",  "r5", 8,  30, "max", "additive"),
        ("max+add  r5 ds=4 n=20",  "r5", 4,  20, "max", "additive"),
        ("max+add  r5 ds=4 n=40",  "r5", 4,  40, "max", "additive"),
    ]

    # --- Run evaluations over multiple cases ---
    config_results: Dict[str, List[Dict[str, float]]] = {c[0]: [] for c in configs}

    for ci in range(args.n_cases):
        nodes_yx, info = load_nodes_for_case(
            tif_path, H_orig, W_orig, case_idx=ci,
            device=device, scale=scale, H_ds=H_ds, W_ds=W_ds)
        print(f"--- Case {ci}: N={nodes_yx.shape[0]} nodes ---")

        for label, mask_key, min_ds, n_iters, pool_mode, cost_mode in configs:
            if mask_key not in masks:
                continue
            res = evaluate_with_mask(
                masks[mask_key], nodes_yx, info,
                k=args.k, min_in_patch=3, patch_size=args.patch_size,
                margin=args.margin, n_iters=n_iters, min_ds=min_ds,
                pscale=pscale, device=device,
                pool_mode=pool_mode, cost_mode=cost_mode)
            config_results[label].append(res)
            print(f"  {label}  Sp={res['spearman']:+.4f}  Euc={res['spearman_euc']:+.4f}")

    # --- Aggregate and print summary ---
    print()
    print("=" * 105)
    print(f"{'Configuration':<25s}  {'Sp mean':>8s} {'±std':>6s}  "
          f"{'Kd mean':>8s}  {'PW mean':>8s}  "
          f"{'Euc mean':>8s} {'±std':>6s}  {'Anchors':>7s}")
    print("-" * 105)
    agg_rows = []
    for label, *_ in configs:
        case_list = config_results[label]
        if not case_list:
            continue
        sp_vals = [r["spearman"] for r in case_list if np.isfinite(r["spearman"])]
        kd_vals = [r["kendall"] for r in case_list if np.isfinite(r["kendall"])]
        pw_vals = [r["pw_acc"] for r in case_list if np.isfinite(r["pw_acc"])]
        eu_vals = [r["spearman_euc"] for r in case_list if np.isfinite(r["spearman_euc"])]
        anc_vals = [r["n_anchors"] for r in case_list]

        sp_m, sp_s = np.mean(sp_vals), np.std(sp_vals)
        kd_m = np.mean(kd_vals) if kd_vals else float("nan")
        pw_m = np.mean(pw_vals) if pw_vals else float("nan")
        eu_m, eu_s = (np.mean(eu_vals), np.std(eu_vals)) if eu_vals else (float("nan"), 0)
        anc_m = np.mean(anc_vals)

        print(f"{label:<25s}  {sp_m:>+8.4f} {sp_s:>5.4f}  "
              f"{kd_m:>+8.4f}  {pw_m:>8.4f}  "
              f"{eu_m:>+8.4f} {eu_s:>5.4f}  {anc_m:>7.1f}")
        agg_rows.append({
            "label": label, "sp_mean": sp_m, "sp_std": sp_s,
            "kd_mean": kd_m, "pw_mean": pw_m,
            "eu_mean": eu_m, "eu_std": eu_s, "anc_mean": anc_m,
        })
    print("=" * 105)
    print()

    # --- Save CSV ---
    csv_path = os.path.join(str(_HERE.parent), "gt_spearman_diagnostic.csv")
    with open(csv_path, "w") as f:
        f.write("config,sp_mean,sp_std,kd_mean,pw_mean,eu_mean,eu_std,n_cases\n")
        for r in agg_rows:
            f.write(f"{r['label']},{r['sp_mean']:.6f},{r['sp_std']:.6f},"
                    f"{r['kd_mean']:.6f},{r['pw_mean']:.6f},"
                    f"{r['eu_mean']:.6f},{r['eu_std']:.6f},{args.n_cases}\n")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
