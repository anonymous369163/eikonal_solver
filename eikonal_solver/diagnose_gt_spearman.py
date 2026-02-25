#!/usr/bin/env python3
"""
Diagnostic script for GT Spearman ceiling.

Per the GT Spearman diagnosis plan, runs:
  1. Data alignment: mask.shape vs rgb, NPZ meta vs TIF
  2. ROI stats: roi_truncated, pad_frac per anchor
  3. Unit comparison: (T_norm, undirected_dist_norm) sample pairs
  4. Iteration impact: Spearman vs n_iters

Usage:
    python eikonal_solver/diagnose_gt_spearman.py
    python eikonal_solver/diagnose_gt_spearman.py --tif_path path/to/crop.tif --n_cases 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluator import load_tif_image, pick_anchor_and_knn, spearmanr
from dataset import load_nodes_from_npz

# Import helpers from test script
from test_gt_ablation2 import (
    load_gt_mask,
    load_nodes_for_case,
    make_soft_gt,
    eval_config,
    eikonal_solve_fixed_ds,
    prob_to_cost,
    crop_square_patch,
)
from eikonal import EikonalConfig, eikonal_soft_sweeping


def run_data_alignment_check(
    region_dir: str, H_orig: int, W_orig: int, meta_h: int, meta_w: int
):
    """1. Data alignment: mask vs rgb, NPZ meta vs TIF. Uses pre-loaded dims to avoid redundant I/O."""
    print("\n" + "=" * 60)
    print("1. DATA ALIGNMENT CHECK")
    print("=" * 60)

    print(f"  TIF shape: ({H_orig}, {W_orig})")

    m = load_gt_mask(region_dir, "r5")
    print(f"  Mask shape: {m.shape}")
    if m.shape != (H_orig, W_orig):
        print("  [FAIL] Mask and TIF dimensions differ!")
        return False
    print("  [PASS] Mask shape matches TIF.")

    print(f"  NPZ meta: (H={meta_h}, W={meta_w})")
    if (meta_h, meta_w) != (H_orig, W_orig):
        print("  [FAIL] NPZ meta does not match TIF!")
        return False
    print("  [PASS] NPZ meta matches TIF.")
    return True


def visualize_nodes_on_mask(mask: np.ndarray, nodes_yx: np.ndarray, save_path: str):
    """Overlay nodes on mask for visual verification."""
    vis = np.stack([mask, mask, mask], axis=-1)
    vis = (vis * 255).astype(np.uint8)
    for i, (y, x) in enumerate(nodes_yx):
        y, x = int(y), int(x)
        r0, r1 = max(0, y - 3), min(mask.shape[0], y + 4)
        c0, c1 = max(0, x - 3), min(mask.shape[1], x + 4)
        vis[r0:r1, c0:c1, 0] = 255
        vis[r0:r1, c0:c1, 1] = 0
        vis[r0:r1, c0:c1, 2] = 0
    Image.fromarray(vis).save(save_path)
    print(f"  Saved node overlay: {save_path}")


def run_roi_stats(mask: torch.Tensor, all_cases: list, k: int, patch_size: int,
                  margin: int, n_iters: int, ds: int, device: torch.device, pool_mode: str = "avg"):
    """2. ROI stats: roi_truncated, pad_frac per anchor (legacy mode)."""
    print("\n" + "=" * 60)
    print("2. ROI TRUNCATION STATS (legacy 512-patch)")
    print("=" * 60)

    trunc_list, pad_frac_list = [], []
    for nodes_yx, info in all_cases[:2]:  # First 2 cases
        for anc_idx in range(min(5, nodes_yx.shape[0])):
            _, nn_global = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=anc_idx)
            anc = nodes_yx[anc_idx]
            ay, ax = int(anc[0]), int(anc[1])
            py0 = max(0, min(ay - patch_size // 2, mask.shape[0] - patch_size))
            px0 = max(0, min(ax - patch_size // 2, mask.shape[1] - patch_size))
            py1, px1 = py0 + patch_size, px0 + patch_size
            tgts_g = nodes_yx[nn_global]
            in_p = ((tgts_g[:, 0] >= py0) & (tgts_g[:, 0] < py1) &
                    (tgts_g[:, 1] >= px0) & (tgts_g[:, 1] < px1))
            keep = torch.where(in_p)[0]
            if keep.numel() < 3:
                continue
            tgt_patch = tgts_g[keep] - torch.tensor([py0, px0], device=device)
            anc_patch = torch.tensor([ay - py0, ax - px0], device=device).long()
            rp = mask[py0:py1, px0:px1]
            if rp.shape[0] < patch_size or rp.shape[1] < patch_size:
                rp = F.pad(rp, (0, patch_size - rp.shape[1], 0, patch_size - rp.shape[0]))
            result = eikonal_solve_fixed_ds(
                rp, anc_patch, tgt_patch, device, margin,
                n_iters, ds, 20.0, 2.0, 0.8, pool_mode=pool_mode)
            dd = result["diag"]
            trunc_list.append(dd.get("roi_truncated", False))
            pad_frac_list.append(dd.get("pad_frac", 0.0))
            print(f"  case0 anc{anc_idx}: roi_truncated={dd.get('roi_truncated')} "
                  f"pad_frac={dd.get('pad_frac', 0):.3f} P={dd.get('P', 0)}")

    n = len(trunc_list)
    trunc_ratio = sum(trunc_list) / max(n, 1)
    avg_pad = np.mean(pad_frac_list) if pad_frac_list else 0
    print(f"  Summary (sample): roi_truncated_ratio={trunc_ratio:.1%} avg_pad_frac={avg_pad:.3f}")


def run_unit_comparison(mask: torch.Tensor, all_cases: list, common: dict,
                        n_iters: int, ds: int, meta_px: int):
    """3. Unit comparison: sample (T_norm, undirected_dist_norm) pairs."""
    print("\n" + "=" * 60)
    print("3. UNIT COMPARISON (T_norm vs undirected_dist_norm)")
    print("=" * 60)

    pscale = common["pscale"]
    device = common["device"]
    k = common["k"]
    patch_size = common["patch_size"]
    margin = common["margin"]

    pairs = []
    for nodes_yx, info in all_cases[:1]:
        case = info["case_idx"]
        for anc_idx in range(min(3, nodes_yx.shape[0])):
            _, nn_global = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=anc_idx)
            anc = nodes_yx[anc_idx]
            tgts_g = nodes_yx[nn_global]
            ay, ax = int(anc[0]), int(anc[1])
            py0 = max(0, min(ay - patch_size // 2, mask.shape[0] - patch_size))
            px0 = max(0, min(ax - patch_size // 2, mask.shape[1] - patch_size))
            in_p = ((tgts_g[:, 0] >= py0) & (tgts_g[:, 0] < py0 + patch_size) &
                    (tgts_g[:, 1] >= px0) & (tgts_g[:, 1] < px0 + patch_size))
            keep = torch.where(in_p)[0]
            if keep.numel() < 2:
                continue
            tgt_patch = tgts_g[keep] - torch.tensor([py0, px0], device=device)
            anc_patch = torch.tensor([ay - py0, ax - px0], device=device).long()
            rp = mask[py0:py0 + patch_size, px0:px0 + patch_size]
            pool_mode = common.get("pool_mode", "avg")
            result = eikonal_solve_fixed_ds(
                rp, anc_patch, tgt_patch, device, margin,
                n_iters, ds, 20.0, 2.0, 0.8, pool_mode=pool_mode)
            pred = result["pred"]
            # Eikonal h=ds already yields T in physical pixel scale; do NOT multiply by ds again
            T_norm = pred * pscale / max(meta_px, 1)
            nn_np = nn_global[keep].cpu().numpy().astype(int)
            gt_arr = np.asarray(info["undirected_dist_norm"][case][anc_idx, nn_np], dtype=np.float64)
            for i in range(min(5, len(T_norm))):
                pairs.append((float(T_norm[i]), float(gt_arr[i])))

    print("  Sample (T_norm, undirected_dist_norm) pairs:")
    for t, g in pairs[:10]:
        print(f"    ({t:.6f}, {g:.6f})  ratio={t/max(g,1e-9):.2f}")
    if pairs:
        ratios = [t / max(g, 1e-9) for t, g in pairs]
        print(f"  Ratio T_norm/gt range: [{min(ratios):.2f}, {max(ratios):.2f}] mean={np.mean(ratios):.2f}")


def run_iteration_ablation(mask: torch.Tensor, all_cases: list, common: dict, alpha: float = 500.0):
    """4. Iteration impact: Spearman vs n_iters.
    Uses gate=1.0 (pure Eikonal, no Euclidean blend).
    """
    patch = common.get("patch_size", 1024)
    print("\n" + "=" * 60)
    print("4. ITERATION IMPACT (Spearman vs n_iters)")
    print(f"  alpha={alpha}, gate=1.0, fine_patch={patch}")
    print("=" * 60)

    for n in [20, 40, 80, 160, 320]:
        r = eval_config(mask, all_cases, n_iters=n, ds=8,
                        alpha=alpha, gamma=2.0, gate=1.0,
                        use_multiscale=True, **common)
        d = r["diag"]
        fallback_ratio = d.get("used_coarse_ratio", 0)
        print(f"  n_iters={n:3d}  Sp={r['sp_mean']:+.4f}  fallback={fallback_ratio:.1%}")


def run_mask_comparison(masks: dict, all_cases: list, common: dict, alpha: float = 500.0):
    """5. Mask comparison: hard (binary) vs blur_s8 (gradient dilation) at aggressive ds=16.
    Validates that Gaussian-blur 'soft dilation' creates a cost funnel that prevents
    thin-road断裂 under downsampling while avoiding corner-cutting.
    """
    print("\n" + "=" * 60)
    print("5. MASK COMPARISON: hard vs blur_s8 (gradient dilation) at ds=16")
    print(f"  alpha={alpha}, gate=1.0, n_iters=80")
    print("=" * 60)

    for mask_name in ["hard", "blur_s8"]:
        mask = masks[mask_name]
        r = eval_config(mask, all_cases, n_iters=80, ds=16,
                        alpha=alpha, gamma=2.0, gate=1.0,
                        use_multiscale=True, **common)
        d = r["diag"]
        fallback_ratio = d.get("used_coarse_ratio", 0)
        print(f"  {mask_name:10s}  Sp={r['sp_mean']:+.4f}  fallback={fallback_ratio:.1%}")


def run_convergence_check(mask: torch.Tensor, all_cases: list, common: dict,
                          alpha: float = 500.0, n_iters: int = 2000):
    """6a. Convergence monitor: run a single anchor with return_convergence=True.
    Prints max|ΔT| at each iteration to reveal when the Eikonal field stabilises.
    Reports: final max|ΔT|, iteration at which max|ΔT| < 0.1 (converged threshold),
    and a down-sampled curve for readability.

    Uses actual k-NN (not full-image patch) for realistic but manageable grid size.
    """
    device = common["device"]
    margin = common.get("margin", 64)
    k = common.get("k", 5)
    alpha_v = alpha
    gamma_v = 2.0

    nodes_yx, info = all_cases[0]
    anc_idx, nn_idx = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=0)
    anc = nodes_yx[anc_idx]
    tgts = nodes_yx[nn_idx]

    # Pass full mask; eikonal_solve_fixed_ds crops a P×P ROI around anchor
    result = eikonal_solve_fixed_ds(
        mask, anc, tgts, device, margin,
        n_iters, 1, alpha_v, gamma_v, 1.0,
        return_convergence=True, n_iters_adaptive=False)

    conv_curve = result.get("conv_curve", [])
    Pc = result["diag"].get("Pc", 0)
    actual_n = result["diag"].get("actual_n_iters", n_iters)

    print("\n" + "=" * 60)
    print("6a. CONVERGENCE MONITOR (one anchor, ds=1)")
    print(f"  Pc={Pc}, actual_n_iters={actual_n}, alpha={alpha_v}")
    print("=" * 60)

    if not conv_curve:
        print("  [No convergence curve returned]")
        return

    # Find first iteration (after the initial ramp-up) where max|ΔT| < 0.1
    # Skip iter 0 (always 0) and look for sustained convergence after wave front passes
    final_delta = conv_curve[-1]
    peak_idx = max(range(len(conv_curve)), key=lambda i: conv_curve[i])
    peak_val = conv_curve[peak_idx]
    # Converged = first iter AFTER peak where delta < 0.1
    conv_iter = next((i for i in range(peak_idx, len(conv_curve)) if conv_curve[i] < 0.1), None)
    print(f"  Final max|ΔT|={final_delta:.4f}  Peak at iter {peak_idx+1} (val={peak_val:.2f})")
    print(f"  Converged (max|ΔT|<0.1 after peak) at iter: "
          f"{conv_iter+1 if conv_iter is not None else 'NOT YET (need more iters)'}")

    # Print down-sampled curve (every stride steps)
    stride = max(1, len(conv_curve) // 20)
    max_v = max(conv_curve) if conv_curve else 1.0
    print(f"  Convergence curve (max|ΔT| every {stride} iters, peak={max_v:.2f}):")
    for i in range(0, len(conv_curve), stride):
        bar_len = int(min(conv_curve[i] / max(max_v, 1e-6) * 30, 30))
        bar = "#" * bar_len
        mark = " ← peak" if i == peak_idx else ""
        print(f"    iter {i+1:5d}: {conv_curve[i]:10.3f}  |{bar}{mark}")
    # Always print last
    last = len(conv_curve) - 1
    if last % stride != 0:
        print(f"    iter {last+1:5d}: {conv_curve[last]:10.3f}  (final)")


def run_fullres_high_iters(mask: torch.Tensor, all_cases: list, common: dict, alpha: float = 500.0,
                           n_iters_list: list | None = None, k: int = 10):
    """6. Full resolution (ds=1, no downsampling) with very high n_iters.
    Tests whether Spearman can reach ~1.0 when grid is pixel-perfect and convergence
    is sufficient. Uses full-image fine_patch so targets use ds=1 (not coarse).
    Uses n_iters_adaptive=True so each solve scales to max(user_n, 2*Pc).
    WARNING: ds=1 is slow; grid Pc can be 500-2000+.
    """
    if n_iters_list is None:
        n_iters_list = [500, 1000, 2000, 5000]
    H, W = mask.shape[-2:]
    fullres_patch = min(4096, H, W)  # Cover full image so all targets use fine (ds=1)
    common_fullres = {**common, "patch_size": fullres_patch, "k": k}
    print("\n" + "=" * 60)
    print("6. FULL RESOLUTION (ds=1) + HIGH ITERATIONS → Spearman ceiling?")
    print(f"  alpha={alpha}, gate=1.0, fine_patch={fullres_patch} (full-image), ds=1, k={k}")
    print(f"  n_iters to test: {n_iters_list}")
    print("  [Grid Pc = full ROI.  RB-GS converges when n_iters >= 2*Pc.]")
    print("=" * 60)

    first_Pc = None
    for n in n_iters_list:
        r = eval_config(mask, all_cases, n_iters=n, ds=1,
                        alpha=alpha, gamma=2.0, gate=1.0,
                        use_multiscale=True, n_iters_adaptive=False,
                        **common_fullres)
        d = r["diag"]
        fallback_ratio = d.get("used_coarse_ratio", 0)
        Pc = d.get("Pc", 0)
        if first_Pc is None:
            first_Pc = Pc
        converged = "CONV" if (Pc > 0 and n >= 2 * Pc) else f"need~{2*Pc}"
        print(f"  n_iters={n:5d}  Sp={r['sp_mean']:+.4f}"
              f"  Pc~{Pc}  fallback={fallback_ratio:.1%}  [{converged}]")

    if first_Pc is not None:
        print(f"\n  NOTE: Convergence likely needs n_iters >= 2*Pc = {2*first_Pc}."
              f" Consider running with that many iterations.")


def run_thin_vs_r5_comparison(region_dir: str, H_orig: int, W_orig: int,
                              all_cases: list, common: dict, device: torch.device,
                              alpha: float = 500.0, n_iters: int = 3000):
    """7. Thin mask vs r=5 dilated mask Spearman comparison at ds=1 + adaptive iters.

    Purpose: isolate whether the Spearman ceiling is due to:
      (a) Eikonal convergence issue  → thin ≈ r5 at same n_iters
      (b) r=5 dilation shortcuts     → thin >> r5
      (c) GT graph vs continuous Eikonal mismatch → both plateau below 1.0

    Loads:
      - 'thin' : original 1-pixel-wide road mask
      - 'r5'   : r=5 dilated mask (11-pixel roads)
    """
    H, W = H_orig, W_orig
    fullres_patch = min(4096, H, W)

    # Load thin mask
    try:
        m_thin = load_gt_mask(region_dir, "thin")
    except FileNotFoundError as e:
        print(f"\n  [Step 7 SKIPPED] Could not load thin mask: {e}")
        return

    mt_thin = torch.from_numpy(m_thin).to(device)
    mask_thin = make_soft_gt(mt_thin, "hard")

    # Load r5 mask
    m_r5 = load_gt_mask(region_dir, "r5")
    mt_r5 = torch.from_numpy(m_r5).to(device)
    mask_r5 = make_soft_gt(mt_r5, "hard")

    k = common.get("k", 10)
    common_fullres = {**common, "patch_size": fullres_patch, "k": k}

    print("\n" + "=" * 60)
    print("7. THIN vs R5 MASK COMPARISON @ ds=1 (structural ceiling test)")
    print(f"  alpha={alpha}, gate=1.0, n_iters={n_iters}, k={k}")
    print(f"  fine_patch={fullres_patch}")
    print("  Hypothesis: if thin Sp >> r5 Sp → r=5 dilation causes structural ceiling")
    print("=" * 60)

    for mask_name, mask_t in [("r5_dilated", mask_r5), ("thin_1px", mask_thin)]:
        r = eval_config(mask_t, all_cases, n_iters=n_iters, ds=1,
                        alpha=alpha, gamma=2.0, gate=1.0,
                        use_multiscale=True, n_iters_adaptive=False,
                        **common_fullres)
        d = r["diag"]
        fallback_ratio = d.get("used_coarse_ratio", 0)
        Pc = d.get("Pc", 0)
        converged = "CONV?" if n_iters >= 2 * max(Pc, 1) else f"need~{2*Pc}"
        print(f"  {mask_name:12s}  Sp={r['sp_mean']:+.4f}"
              f"  Pc~{Pc}  fallback={fallback_ratio:.1%}  [{converged}]")

    print()
    print("  Interpretation:")
    print("    thin_1px Sp >> r5_dilated Sp  →  dilation shortcuts are the ceiling")
    print("    thin_1px Sp ≈  r5_dilated Sp  →  convergence or GT-graph mismatch is the ceiling")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif_path", type=str,
                        default="Gen_dataset_V2/Gen_dataset/34.269888_108.947180/"
                                "01_34.350697_108.914569_3000.0/"
                                "crop_34.350697_108.914569_3000.0_z16.tif")
    parser.add_argument("--n_cases", type=int, default=5,
                        help="Number of cases for ablation; 5+ recommended to reduce Spearman variance")
    parser.add_argument("--save_dir", type=str, default="images/gt_spearman_diagnostic")
    parser.add_argument("--pool_mode", type=str, default="avg", choices=["max", "avg"],
                        help="Pooling mode for downsampling (max or avg)")
    parser.add_argument("--fine_patch", type=int, default=1024,
                        help="Fine patch size for multiscale (default 1024 to reduce fallback)")
    parser.add_argument("--alpha", type=float, default=500.0,
                        help="Off-road cost penalty (try 5000 to prevent wavefront bleeding; 5000 may cause NaN at low n_iters)")
    parser.add_argument("--only_fullres", action="store_true",
                        help="Only run steps 6/6a/7: ds=1 + high n_iters + thin-vs-r5 (skip steps 1-5)")
    parser.add_argument("--fullres_n_iters", type=str, default="500,1000,2000,5000",
                        help="Comma-separated minimum n_iters for step 6 (adaptive will scale up)")
    parser.add_argument("--fullres_k", type=int, default=10,
                        help="Number of NN targets per anchor in step 6 (default 10 for robust Spearman)")
    parser.add_argument("--conv_check_iters", type=int, default=2000,
                        help="n_iters for step 6a convergence monitor (default 2000)")
    parser.add_argument("--thin_vs_r5_iters", type=int, default=3000,
                        help="n_iters for step 7 thin-vs-r5 comparison (default 3000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tif_path = args.tif_path
    if not os.path.isabs(tif_path):
        tif_path = os.path.join(str(_HERE.parent), tif_path)
    region_dir = os.path.dirname(tif_path)

    print("GT Spearman Diagnostic")
    print(f"  TIF: {tif_path}")

    # Load data
    rgb = load_tif_image(tif_path).to(device)
    _, _, H_orig, W_orig = rgb.shape
    m = load_gt_mask(region_dir, "r5")
    if m.shape != (H_orig, W_orig):
        print(f"[FAIL] Mask shape {m.shape} != TIF ({H_orig}, {W_orig})")
        sys.exit(1)
    mt = torch.from_numpy(m).to(device)
    masks = {
        "hard": make_soft_gt(mt, "hard"),
        "blur_s3": make_soft_gt(mt, "blur", sigma=3.0),
        "blur_s8": make_soft_gt(mt, "blur", sigma=8.0),
    }

    all_cases = []
    for ci in range(args.n_cases):
        ny, nf = load_nodes_for_case(tif_path, H_orig, W_orig, case_idx=ci,
                                     device=device, scale=1.0, H_ds=H_orig, W_ds=W_orig)
        all_cases.append((ny, nf))

    common = dict(k=5, patch_size=args.fine_patch, margin=64, pscale=1.0, device=device,
                  pool_mode=args.pool_mode)
    meta_px = int(all_cases[0][1]["meta_sat_height_px"])

    if not args.only_fullres:
        # 1. Data alignment (pass pre-loaded dims to avoid redundant I/O)
        meta_h = int(all_cases[0][1]["meta_sat_height_px"])
        meta_w = int(all_cases[0][1]["meta_sat_width_px"])
        ok = run_data_alignment_check(region_dir, H_orig, W_orig, meta_h, meta_w)
        if not ok:
            sys.exit(1)

        # Visualize nodes on mask
        os.makedirs(args.save_dir, exist_ok=True)
        nodes_yx = all_cases[0][0].cpu().numpy()
        visualize_nodes_on_mask(m, nodes_yx, os.path.join(args.save_dir, "nodes_on_mask.png"))

        # 2. ROI stats
        run_roi_stats(masks["hard"], all_cases, k=5, patch_size=512, margin=64,
                      n_iters=80, ds=8, device=device, pool_mode=common["pool_mode"])

        # 3. Unit comparison
        meta_px = int(all_cases[0][1]["meta_sat_height_px"])
        run_unit_comparison(masks["hard"], all_cases, common, n_iters=80, ds=8, meta_px=meta_px)

        # 4. Iteration ablation
        run_iteration_ablation(masks["hard"], all_cases, common, alpha=args.alpha)

        # 5. Mask comparison: hard vs blur_s8 (gradient dilation) at ds=16
        run_mask_comparison(masks, all_cases, common, alpha=args.alpha)

    # 6a. Convergence monitor: single anchor, return per-iteration max|ΔT|
    run_convergence_check(masks["hard"], all_cases, common, alpha=args.alpha,
                          n_iters=args.conv_check_iters)

    # 6. Full resolution (ds=1) + very high n_iters, n_iters_adaptive=True
    n_iters_list = [int(x.strip()) for x in args.fullres_n_iters.split(",") if x.strip()]
    run_fullres_high_iters(masks["hard"], all_cases, common, alpha=args.alpha,
                           n_iters_list=n_iters_list if n_iters_list else None,
                           k=args.fullres_k)

    # 7. Thin vs r=5 dilated mask comparison: structural ceiling test
    run_thin_vs_r5_comparison(region_dir, H_orig, W_orig, all_cases, common, device,
                               alpha=args.alpha, n_iters=args.thin_vs_r5_iters)

    print("\n" + "=" * 60)
    print("Diagnostic complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
