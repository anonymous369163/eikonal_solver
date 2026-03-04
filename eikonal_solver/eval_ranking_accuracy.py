#!/usr/bin/env python3
"""Compare edge ranking accuracy: Eikonal predictions vs Euclidean distances vs GT.

For NCO/TSP, what matters is not absolute distance accuracy but whether the
**relative ordering of edges** is correct. This script computes:

1. Pairwise ordering accuracy: for all pairs of edges from the same source,
   what fraction have the correct relative ordering?
2. Kendall's tau rank correlation
3. Top-K recall: among the K nearest neighbors in GT, how many are also in
   the K nearest predicted?

Usage:
  python eval_ranking_accuracy.py \
    --tif .../crop_...tif \
    --tsp_load_ckpt /tmp/distance_training/phase1/.../routing.pt \
    --n_cases 200
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model_multigrid import (
    SAMRoute, EikonalConfig,
    _eikonal_soft_sweeping_diff, _eikonal_soft_sweeping_diff_init,
)
from gradcheck_route_loss_v2_multigrid_fullmap import (
    sliding_window_inference,
    _load_npz_nodes,
    _normxy_to_yx,
    _load_lightning_ckpt,
    _detect_smooth_decoder,
    _detect_patch_size,
    _load_rgb_from_tif,
    _maxpool_prob,
    _make_src_mask,
    fullgrid_multigrid_diff_solve_batched,
    _mix_eikonal_euclid,
    GradcheckConfig,
    smooth_prob,
)
from dataclasses import replace as dc_replace


def _avgpool_prob(prob: torch.Tensor, ds: int):
    """prob: [B,H,W] -> pooled [B,Hc,Wc] using average pooling."""
    if ds <= 1:
        return prob, prob.shape[-2], prob.shape[-1]
    x = prob.unsqueeze(1)  # [B,1,H,W]
    pad_h = (ds - prob.shape[-2] % ds) % ds
    pad_w = (ds - prob.shape[-1] % ds) % ds
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    H_pad, W_pad = x.shape[-2], x.shape[-1]
    y = F.avg_pool2d(x, kernel_size=ds, stride=ds).squeeze(1)
    return y, H_pad, W_pad


_CKPT_DEFAULT = os.path.join(
    _PROJECT_ROOT,
    "training_outputs", "finetune_demo", "checkpoints",
    "best_lora_pos8.0_dice0.5_thin4.0.ckpt",
)


def _kendall_tau(x, y):
    """Kendall's tau-b rank correlation between two arrays."""
    n = len(x)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _pairwise_ordering_accuracy(pred, gt):
    """Fraction of pairs (i,j) where pred and gt agree on ordering."""
    n = len(pred)
    if n < 2:
        return 0.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if gt[i] == gt[j]:
                continue
            total += 1
            if (pred[i] - pred[j]) * (gt[i] - gt[j]) > 0:
                correct += 1
    return correct / total if total > 0 else 0.0


def _topk_recall(pred, gt, k):
    """Among the K nearest in GT, how many appear in K nearest predicted?"""
    if len(pred) <= k:
        return 1.0
    gt_topk = set(np.argsort(gt)[:k])
    pred_topk = set(np.argsort(pred)[:k])
    return len(gt_topk & pred_topk) / k


def _solve_all_sources_once(
    model, cost_f, cost_c, all_src_yx, cfg,
    ds, mg_f, mg_itc, mg_itf, device,
    iter_floor_c: float = 1.5,
    iter_floor_f: float = 0.8,
):
    """Solve Eikonal for ALL source points in a single batched call.

    Returns the full fine-resolution distance map for each source.

    Args:
        all_src_yx: [B, 2] pixel coords (each row a DIFFERENT source)
        iter_floor_c: coarse iteration floor = max(Hc,Wc) * this
        iter_floor_f: fine iteration floor = max(Hf,Wf) * this
    Returns:
        T_fine: [B, Hf, Wf] distance map per source
    """
    B = all_src_yx.shape[0]
    Hf, Wf = cost_f.shape[-2], cost_f.shape[-1]
    Hc, Wc = cost_c.shape[-2], cost_c.shape[-1]
    large_val = float(cfg.large_val)
    ds_coarse = ds * mg_f
    ckpt_chunk = int(getattr(model, "route_ckpt_chunk", 10))
    gate_alpha = float(getattr(model, "route_gate_alpha", 1.0))

    # --- Coarse solve ---
    cost_c_b = cost_c.expand(B, -1, -1)
    src_cc = (all_src_yx.long() // ds_coarse).clamp(min=0)
    src_cc[:, 0] = src_cc[:, 0].clamp(0, Hc - 1)
    src_cc[:, 1] = src_cc[:, 1].clamp(0, Wc - 1)
    src_mask_c = _make_src_mask(B, Hc, Wc, src_cc, device)

    actual_iters_c = int(max(mg_itc, int(max(Hc, Wc) * iter_floor_c)))
    cfg_c = dc_replace(cfg, h=float(ds_coarse), n_iters=actual_iters_c, monotone=True)
    T_c = _eikonal_soft_sweeping_diff(
        cost_c_b, src_mask_c, cfg_c,
        checkpoint_chunk=ckpt_chunk, gate_alpha=gate_alpha,
    ).detach()

    # --- Warm-start upsample ---
    T_init = F.interpolate(
        T_c.unsqueeze(1), size=(Hf, Wf), mode="bilinear", align_corners=False,
    ).squeeze(1).clamp_min(0.0).clamp_max(large_val)
    src_fc = (all_src_yx.long() // ds).clamp(min=0)
    src_fc[:, 0] = src_fc[:, 0].clamp(0, Hf - 1)
    src_fc[:, 1] = src_fc[:, 1].clamp(0, Wf - 1)
    src_mask_f = _make_src_mask(B, Hf, Wf, src_fc, device)
    T_init = torch.where(src_mask_f, torch.zeros_like(T_init), T_init)

    # --- Fine solve (full grid, no tube needed — single solve per source) ---
    cost_f_b = cost_f.expand(B, -1, -1)
    actual_iters_f = int(max(mg_itf, int(max(Hf, Wf) * iter_floor_f)))
    cfg_f = dc_replace(cfg, h=float(ds), n_iters=actual_iters_f, monotone=False)
    T_fine = _eikonal_soft_sweeping_diff_init(
        cost_f_b, src_mask_f, cfg_f, T_init,
        checkpoint_chunk=ckpt_chunk, gate_alpha=gate_alpha,
    )
    return T_fine


def _record(m, eik, euc, gt, detour):
    """Append ranking metrics for one source into metrics dict *m*."""
    m["eik_pairwise"].append(_pairwise_ordering_accuracy(eik, gt))
    m["euc_pairwise"].append(_pairwise_ordering_accuracy(euc, gt))
    m["eik_tau"].append(_kendall_tau(eik, gt))
    m["euc_tau"].append(_kendall_tau(euc, gt))

    n = len(gt)
    if n >= 1:
        m["eik_top1"].append(_topk_recall(eik, gt, 1))
        m["euc_top1"].append(_topk_recall(euc, gt, 1))
    if n >= 3:
        m["eik_top3"].append(_topk_recall(eik, gt, 3))
        m["euc_top3"].append(_topk_recall(euc, gt, 3))
    if n >= 5:
        m["eik_top5"].append(_topk_recall(eik, gt, 5))
        m["euc_top5"].append(_topk_recall(euc, gt, 5))

    eik_rel = np.abs(eik - gt) / np.maximum(gt, 1.0)
    euc_rel = np.abs(euc - gt) / np.maximum(gt, 1.0)
    m["eik_rel"].extend(eik_rel.tolist())
    m["euc_rel"].extend(euc_rel.tolist())
    m["eik_dist"].extend(eik.tolist())
    m["euc_dist"].extend(euc.tolist())
    m["gt_dist"].extend(gt.tolist())
    m["detour_ratio"].extend(detour.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=_CKPT_DEFAULT)
    ap.add_argument("--tsp_load_ckpt", type=str, default="")
    ap.add_argument("--n_cases", type=int, default=200)
    ap.add_argument("--downsample", type=int, default=16,
                     help="Eikonal grid downsample factor. 16 is optimal (SWEEP 3).")
    ap.add_argument("--eik_iters", type=int, default=40,
                     help="Eikonal iteration hint (actual count controlled by iter_floor).")
    ap.add_argument("--cost_net", action="store_true")
    ap.add_argument("--cost_net_ch", type=int, default=16)
    ap.add_argument("--p_count", type=int, default=20,
                     help="Node count variant: 20 or 50 (selects matching NPZ)")
    ap.add_argument("--cache_prob", type=str, default="")
    ap.add_argument("--gt_mask", type=str, default="",
                     help="Path to GT road mask image (e.g. roadnet_normalized_r3.png). "
                          "If given, use this as road_prob instead of model prediction.")
    ap.add_argument("--gate_override", type=float, default=1.0,
                     help="Eikonal/Euclidean gate value (0-1). "
                          "1.0 = pure Eikonal (optimal, SWEEP 1). -1 = use checkpoint value.")
    ap.add_argument("--alpha_override", type=float, default=50,
                     help="Cost alpha. 50 is optimal (SWEEP 2). -1 = use checkpoint value.")
    ap.add_argument("--gamma_override", type=float, default=2.0,
                     help="Cost gamma. 2.0 is optimal (SWEEP 2). -1 = use checkpoint value.")
    ap.add_argument("--pool_mode", type=str, default="max", choices=["max", "avg"],
                     help="Pooling strategy for road_prob downsampling.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", default=True,
                     help="Optimized mode: solve once per source, read all targets (default ON).")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                     help="Disable fast mode, use original per-target-batch solve.")
    ap.add_argument("--iter_floor_c", type=float, default=1.2,
                     help="Coarse iteration floor multiplier. 1.2 is optimal (SWEEP 7b).")
    ap.add_argument("--iter_floor_f", type=float, default=0.65,
                     help="Fine iteration floor multiplier. 0.65 is optimal (SWEEP 7b).")
    args = ap.parse_args()

    device = torch.device(args.device)
    np.random.seed(args.seed)

    # --- load model ---
    sd = _load_lightning_ckpt(args.ckpt)
    cfg = GradcheckConfig()
    has_lora = any("attn.qkv.linear_a_q" in k or "attn.qkv.linear_a_v" in k for k in sd)
    if has_lora:
        cfg.ENCODER_LORA = True
        cfg.FREEZE_ENCODER = False
        for k, v in sd.items():
            if "linear_a_q.weight" in k:
                cfg.LORA_RANK = v.shape[0]
                break
    cfg.USE_SMOOTH_DECODER = _detect_smooth_decoder(sd)
    ps = _detect_patch_size(sd)
    if ps is not None:
        cfg.PATCH_SIZE = ps
    cfg.ROUTE_EIK_ITERS = args.eik_iters
    cfg.ROUTE_EIK_DOWNSAMPLE = args.downsample
    if args.cost_net:
        cfg.ROUTE_COST_NET = True
        cfg.ROUTE_COST_NET_CH = args.cost_net_ch
    if "block_log_alpha" in sd or "block_th_logit" in sd:
        cfg.LEARNABLE_BLOCK = True

    model = SAMRoute(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    for p in model.image_encoder.parameters():
        p.requires_grad_(False)

    # load routing checkpoint
    if args.tsp_load_ckpt and os.path.isfile(args.tsp_load_ckpt):
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
        print(f"[loaded] routing ckpt: {args.tsp_load_ckpt}")

    # --- apply parameter overrides ---
    with torch.no_grad():
        if args.gate_override >= 0:
            logit = 10.0 if args.gate_override >= 0.9999 else math.log(
                args.gate_override / max(1.0 - args.gate_override, 1e-9))
            model.eik_gate_logit.fill_(logit)
            print(f"[override] gate={args.gate_override:.4f} (logit={logit:.2f})")
        if args.alpha_override > 0:
            model.cost_log_alpha.fill_(math.log(args.alpha_override))
            print(f"[override] alpha={args.alpha_override:.1f}")
        if args.gamma_override > 0:
            model.cost_log_gamma.fill_(math.log(args.gamma_override))
            print(f"[override] gamma={args.gamma_override:.2f}")

    model.eval()
    model.route_gate_alpha = 0.8

    # --- load image + road_prob ---
    rgb = _load_rgb_from_tif(args.tif)
    H, W = rgb.shape[:2]
    print(f"[tif] {args.tif}  shape={H}x{W}")

    if args.gt_mask:
        from PIL import Image as _PILImage
        _gt_img = np.array(_PILImage.open(args.gt_mask))
        if _gt_img.ndim == 3:
            _gt_img = _gt_img[:, :, 0]
        road_prob_np = (_gt_img > 0).astype(np.float32)
        if road_prob_np.shape != (H, W):
            road_prob_np = np.array(
                _PILImage.fromarray(road_prob_np).resize((W, H), _PILImage.BILINEAR)
            )
        print(f"[GT mask] {args.gt_mask}  road_ratio={road_prob_np.mean():.4f}")
    else:
        cache = args.cache_prob or os.path.join("/tmp/ranking_eval", "road_prob.npy")
        if os.path.isfile(cache):
            road_prob_np = np.load(cache).astype(np.float32)
            print(f"[cache] loaded: {cache}")
        else:
            road_prob_np = sliding_window_inference(rgb, model, device, patch_size=int(cfg.PATCH_SIZE))
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, road_prob_np)
            print(f"[cache] saved: {cache}")

    # --- load NPZ GT ---
    tif_name = os.path.basename(args.tif)
    location_key = tif_name.replace("crop_", "").replace("_z16.tif", "").replace(".tif", "")
    npz_name = f"distance_dataset_all_{location_key}_p{args.p_count}.npz"
    npz_path = os.path.join(os.path.dirname(args.tif), npz_name)
    coords, udist, H_npz, W_npz = _load_npz_nodes(npz_path)
    N_total = coords.shape[0]
    N_nodes = coords.shape[1]
    print(f"[npz] {npz_path}  cases={N_total}  nodes={N_nodes}")

    # --- prepare Eikonal solver ---
    eps = 1e-6
    ds = args.downsample
    mg_f = 4
    road_prob_t = torch.from_numpy(road_prob_np).to(device, torch.float32).unsqueeze(0)
    road_prob_t = road_prob_t.clamp(eps, 1.0 - eps)

    _pool_fn = _avgpool_prob if args.pool_mode == "avg" else _maxpool_prob

    with torch.no_grad():
        prob_f, _, _ = _pool_fn(road_prob_t, ds)
        cost_f = model._road_prob_to_cost(prob_f).to(torch.float32)
        prob_c, _, _ = _pool_fn(road_prob_t, ds * mg_f)
        cost_c = model._road_prob_to_cost(prob_c).to(torch.float32)

    mg_itc = max(20, int(args.eik_iters * 0.25))
    mg_itf = max(20, int(args.eik_iters * 0.80))

    # --- evaluate ranking over cases ---
    n_cases = min(args.n_cases, N_total)
    case_indices = np.random.permutation(N_total)[:n_cases]

    _empty_metrics = lambda: {
        "eik_pairwise": [], "euc_pairwise": [],
        "eik_tau": [], "euc_tau": [],
        "eik_top1": [], "euc_top1": [],
        "eik_top3": [], "euc_top3": [],
        "eik_top5": [], "euc_top5": [],
        "eik_rel": [], "euc_rel": [],
        "eik_dist": [], "euc_dist": [], "gt_dist": [],
        "detour_ratio": [],
    }
    metrics = _empty_metrics()

    Hf, Wf = cost_f.shape[-2], cost_f.shape[-1]
    gate_logit = model.eik_gate_logit

    t0 = time.time()
    for ci_idx, ci in enumerate(case_indices):
        xy_norm = coords[ci]
        dist_mat = udist[ci]  # (N, N) normalized
        yx = _normxy_to_yx(xy_norm, H, W)

        if args.fast:
            # --- Optimized: solve all sources in one batched call ---
            all_src_yx = torch.from_numpy(yx).to(device, torch.long)  # [N_nodes, 2]

            with torch.no_grad():
                try:
                    T_maps = _solve_all_sources_once(
                        model, cost_f, cost_c, all_src_yx,
                        model.route_eik_cfg, ds, mg_f, mg_itc, mg_itf, device,
                        iter_floor_c=args.iter_floor_c,
                        iter_floor_f=args.iter_floor_f,
                    )  # [N_nodes, Hf, Wf]
                except Exception:
                    continue

            for src_i in range(N_nodes):
                gt_row = dist_mat[src_i]
                valid = np.isfinite(gt_row) & (gt_row > 0)
                valid[src_i] = False
                tgt_indices = np.where(valid)[0]
                if len(tgt_indices) < 3:
                    continue

                gt_px_all = (gt_row[tgt_indices] * float(H)).astype(np.float32)
                src_yx_i = yx[src_i]
                tgt_yx = yx[tgt_indices]
                euc_dist_all = np.sqrt(
                    (tgt_yx[:, 0].astype(float) - src_yx_i[0]) ** 2 +
                    (tgt_yx[:, 1].astype(float) - src_yx_i[1]) ** 2
                ).astype(np.float32)

                tgt_fc = (torch.from_numpy(tgt_yx).to(device, torch.long) // ds).clamp(min=0)
                tgt_fc[:, 0] = tgt_fc[:, 0].clamp(0, Hf - 1)
                tgt_fc[:, 1] = tgt_fc[:, 1].clamp(0, Wf - 1)
                T_eik = T_maps[src_i, tgt_fc[:, 0], tgt_fc[:, 1]]

                d_euc_t = torch.from_numpy(euc_dist_all).to(device, torch.float32)
                eik_dist_all = _mix_eikonal_euclid(
                    T_eik.unsqueeze(0), d_euc_t.unsqueeze(0), gate_logit,
                ).squeeze(0).detach().cpu().numpy()

                detour_all = gt_px_all / np.maximum(euc_dist_all, 1.0)
                _record(metrics, eik_dist_all, euc_dist_all, gt_px_all, detour_all)
        else:
            # --- Original: per-source batched solve (redundant) ---
            for src_i in range(N_nodes):
                gt_row = dist_mat[src_i]
                valid = np.isfinite(gt_row) & (gt_row > 0)
                valid[src_i] = False
                tgt_indices = np.where(valid)[0]
                if len(tgt_indices) < 3:
                    continue

                gt_px_all = (gt_row[tgt_indices] * float(H)).astype(np.float32)

                src_yx_i = yx[src_i]
                tgt_yx = yx[tgt_indices]
                euc_dist_all = np.sqrt(
                    (tgt_yx[:, 0].astype(float) - src_yx_i[0]) ** 2 +
                    (tgt_yx[:, 1].astype(float) - src_yx_i[1]) ** 2
                ).astype(np.float32)

                K_batch = len(tgt_indices)
                src_expanded = torch.from_numpy(
                    np.tile(src_yx_i[None, :], (K_batch, 1))
                ).to(device, torch.long)
                tgt_expanded = torch.from_numpy(tgt_yx[:, None, :]).to(device, torch.long)
                gt_expanded = torch.from_numpy(gt_px_all[:, None]).to(device, torch.float32)

                with torch.no_grad():
                    try:
                        pred = fullgrid_multigrid_diff_solve_batched(
                            model, cost_f, cost_c,
                            src_expanded, tgt_expanded, gt_expanded,
                            model.route_eik_cfg,
                            ds_common=ds, mg_factor=mg_f,
                            mg_iters_coarse=mg_itc, mg_iters_fine=mg_itf,
                            mg_detach_coarse=True, mg_interp="bilinear",
                            fine_monotone=False, use_compile=False,
                            tube_roi=True,
                            tube_radius_c=8, tube_pad_c=4,
                            tube_max_area_ratio=0.90,
                            tube_min_side=16, tube_min_Pc=256,
                        )
                        eik_dist_all = pred[:, 0].cpu().numpy()
                    except Exception:
                        continue

                detour_all = gt_px_all / np.maximum(euc_dist_all, 1.0)
                _record(metrics, eik_dist_all, euc_dist_all, gt_px_all, detour_all)

        if (ci_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            m = metrics
            print(f"  [{ci_idx+1}/{n_cases}] {elapsed:.1f}s  "
                  f"eik_pw={np.mean(m['eik_pairwise']):.4f}  "
                  f"euc_pw={np.mean(m['euc_pairwise']):.4f}  "
                  f"eik_tau={np.mean(m['eik_tau']):.4f}  "
                  f"euc_tau={np.mean(m['euc_tau']):.4f}  "
                  f"detour={np.mean(m['detour_ratio']):.3f}")

    elapsed = time.time() - t0

    # --- final report ---
    m = metrics
    n_src = len(m["eik_pairwise"])
    print(f"\n{'='*70}")
    print(f"EDGE RANKING: Eikonal vs Euclidean")
    print(f"  cases={n_cases}, nodes_per_case={N_nodes}, "
          f"n_sources_evaluated={n_src}")
    using_gt = "GT road mask" if args.gt_mask else "predicted road_prob"
    mode_str = "FAST (solve-once)" if args.fast else "ORIGINAL (per-target batch)"
    gate_val = torch.sigmoid(model.eik_gate_logit).item()
    alpha_val = model.cost_log_alpha.clamp(math.log(5.0), math.log(100.0)).exp().item()
    gamma_val = model.cost_log_gamma.clamp(math.log(0.5), math.log(4.0)).exp().item()
    print(f"  mode: {mode_str}")
    print(f"  road_prob source: {using_gt}")
    if args.fast:
        _Hf, _Wf = cost_f.shape[-2], cost_f.shape[-1]
        _Hc, _Wc = cost_c.shape[-2], cost_c.shape[-1]
        _aic = int(max(mg_itc, int(max(_Hc, _Wc) * args.iter_floor_c)))
        _aif = int(max(mg_itf, int(max(_Hf, _Wf) * args.iter_floor_f)))
        print(f"  actual_iters: coarse={_aic}, fine={_aif}  "
              f"(floor_c={args.iter_floor_c}, floor_f={args.iter_floor_f})")
    print(f"  downsample={ds}, eik_iters={args.eik_iters}, pool={args.pool_mode}")
    print(f"  gate={gate_val:.4f}, alpha={alpha_val:.2f}, gamma={gamma_val:.4f}")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s\n")

    pw_eik = np.mean(m["eik_pairwise"])
    pw_euc = np.mean(m["euc_pairwise"])
    print(f"  Pairwise Ordering Accuracy (higher=better):")
    print(f"    Eikonal:   {pw_eik:.4f}  (median={np.median(m['eik_pairwise']):.4f})")
    print(f"    Euclidean: {pw_euc:.4f}  (median={np.median(m['euc_pairwise']):.4f})")
    print(f"    >>> Eikonal advantage: {pw_eik - pw_euc:+.4f}")

    tau_eik = np.mean(m["eik_tau"])
    tau_euc = np.mean(m["euc_tau"])
    print(f"\n  Kendall's Tau (higher=better):")
    print(f"    Eikonal:   {tau_eik:.4f}")
    print(f"    Euclidean: {tau_euc:.4f}")
    print(f"    >>> Eikonal advantage: {tau_eik - tau_euc:+.4f}")

    for k_label, eik_key, euc_key in [
        ("Top-1", "eik_top1", "euc_top1"),
        ("Top-3", "eik_top3", "euc_top3"),
        ("Top-5", "eik_top5", "euc_top5"),
    ]:
        if m[eik_key]:
            v_eik = np.mean(m[eik_key])
            v_euc = np.mean(m[euc_key])
            print(f"\n  {k_label} Recall:")
            print(f"    Eikonal:   {v_eik:.4f}  ({v_eik*100:.1f}%)")
            print(f"    Euclidean: {v_euc:.4f}  ({v_euc*100:.1f}%)")
            print(f"    >>> Eikonal advantage: {v_eik - v_euc:+.4f}")

    eik_rel = np.array(m["eik_rel"])
    euc_rel = np.array(m["euc_rel"])
    print(f"\n  Distance Relative Error (lower=better):")
    print(f"    Eikonal:   mean={eik_rel.mean():.4f}  median={np.median(eik_rel):.4f}")
    print(f"    Euclidean: mean={euc_rel.mean():.4f}  median={np.median(euc_rel):.4f}")

    det = np.array(m["detour_ratio"])
    print(f"\n  Detour Ratio (GT_road / Euclidean, 1.0=straight line):")
    print(f"    mean={det.mean():.3f}  median={np.median(det):.3f}  "
          f"p90={np.percentile(det,90):.3f}  max={det.max():.3f}")

    # --- detour-ratio breakdown ---
    print(f"\n{'='*70}")
    print("BREAKDOWN BY DETOUR RATIO (GT_road / Euclidean)")
    print(f"{'='*70}")

    all_det = np.array(m["detour_ratio"])
    all_eik_r = np.array(m["eik_rel"])
    all_euc_r = np.array(m["euc_rel"])

    bins = [
        ("detour<1.2 (nearly straight)", all_det < 1.2),
        ("1.2<=detour<1.5 (moderate)", (all_det >= 1.2) & (all_det < 1.5)),
        ("1.5<=detour<2.0 (significant)", (all_det >= 1.5) & (all_det < 2.0)),
        ("detour>=2.0 (heavy detour)", all_det >= 2.0),
    ]
    for bname, bmask in bins:
        if bmask.sum() == 0:
            continue
        n_edges = int(bmask.sum())
        eik_err = all_eik_r[bmask].mean()
        euc_err = all_euc_r[bmask].mean()
        eik_wins = int((all_eik_r[bmask] < all_euc_r[bmask]).sum())

        print(f"\n  {bname}  (n={n_edges} edges)")
        print(f"    Eikonal  RelErr: {eik_err:.4f}  ({eik_err*100:.1f}%)")
        print(f"    Euclidean RelErr: {euc_err:.4f}  ({euc_err*100:.1f}%)")
        print(f"    Eikonal closer to GT: {eik_wins}/{n_edges} ({eik_wins/n_edges*100:.1f}%)")

    print()


if __name__ == "__main__":
    main()
