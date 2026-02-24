"""
eval_multi_anchor.py  —  Multi-anchor averaged evaluation of distance prediction.

Strict training-consistency:
  1. Crop PATCH_SIZE×PATCH_SIZE centered on anchor (same as training).
  2. Only evaluate in-patch targets (same as _sample_anchor_targets).
  3. The "training-path" solver calls _model._roi_multi_target_diff_solve
     under torch.no_grad() — same differentiable solver, same gate_alpha,
     same adaptive-ds, same ONE-ROI-for-all-K-targets as training_step.
  4. Two normalization views:
       Train-norm: pred/NORM vs gt_px/NORM  (NORM=ROUTE_DIST_NORM_PX=512)
                   where gt_px = undirected_dist_norm * meta_px (original px)
                   and pred comes from the Eikonal solver (Eikonal cost units)
       Task-norm : ordering only — Spearman/Kendall are norm-invariant

Configs:
  [Ref]   Infer-A 700/ds1  : tiled road_prob + non-diff Eikonal (reference)
  [TRAIN] Model-B train    : model-forward road_prob + _roi_multi_target_diff_solve
                             (true training path under no_grad)
  [Cmp]   Non-diff 20/ds8  : model-forward road_prob + non-diff 20-iter (ablation)
  [Base]  Euclidean        : pure Euclidean distance

Usage:
    python eikonal_solver/eval_multi_anchor.py
    python eikonal_solver/eval_multi_anchor.py --min_in_patch 3
"""

import sys
import math
import time
from pathlib import Path
from collections import defaultdict

_here = Path(__file__).resolve().parent
_repo = _here.parent / "sam_road_repo"
sys.path.insert(0, str(_here))
for _p in [str(_repo / "segment-anything-road"),
           str(_repo / "sam"),
           str(_repo)]:
    if _p not in sys.path:
        sys.path.append(_p)

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from addict import Dict as AdDict
from scipy.stats import spearmanr, kendalltau

from model import SAMRoute, EikonalConfig, eikonal_soft_sweeping
from feature_extractor import load_tif_image, SamRoadTiledInferencer
from load_nodes_from_npz import load_nodes_from_npz

# ─────────────────────────────────────────────────────────────────────────────
_root = _here.parent
_CKPT_PATH   = str(_root / "checkpoints/cityscale_vitb_512_e10.ckpt")
_DEFAULT_IMG = str(
    _root / "Gen_dataset_V2/Gen_dataset/34.269888_108.947180"
    / "01_34.350697_108.914569_3000.0"
    / "crop_34.350697_108.914569_3000.0_z16.tif"
)

import argparse
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--image_path",   default=_DEFAULT_IMG)
ap.add_argument("--ckpt_path",    default=_CKPT_PATH,
                help="Checkpoint for SAMRoute model (default: pretrained cityscale)")
ap.add_argument("--config_path",  default=str(_here / "config_sam_route_phase2.yaml"),
                help="SAMRoute yaml config for model construction")
ap.add_argument("--k",            type=int, default=19)
ap.add_argument("--min_in_patch", type=int, default=3)
ap.add_argument("--downsample",   type=int, default=1024)
ap.add_argument("--save_vis",     default=None,
                help="If set, also save a visualize_paths figure for the first valid anchor")
args = ap.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build model with Phase-2 config
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== Build SAMRoute ===")
print(f"  Config: {args.config_path}")
print(f"  Ckpt:   {args.ckpt_path}")
_cfg2 = AdDict(yaml.safe_load(open(args.config_path)))
_cfg_dir = Path(args.config_path).resolve().parent
for _key in ("SAM_CKPT_PATH", "PRETRAINED_CKPT"):
    _v = str(_cfg2.get(_key, "") or "")
    if _v and not Path(_v).is_absolute():
        _cfg2[_key] = str((_cfg_dir / _v).resolve())
_sam_ckpt = _root / "sam_road_repo" / "sam_ckpts" / "sam_vit_b_01ec64.pth"
if _sam_ckpt.exists():
    _cfg2.SAM_CKPT_PATH = str(_sam_ckpt)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model  = SAMRoute(_cfg2)
_ckpt   = torch.load(args.ckpt_path, map_location="cpu")
_model.load_state_dict(_ckpt.get("state_dict", _ckpt), strict=False)
_model.eval().to(_device)

_PATCH    = int(_cfg2.get("PATCH_SIZE", 512))
_margin   = int(_cfg2.get("ROUTE_ROI_MARGIN", 64))
_NORM_PX  = float(_cfg2.get("ROUTE_DIST_NORM_PX", 512.0))   # training loss normalisation

# Training solver config (same as model.__init__ builds route_eik_cfg)
_n_iters_train = int(_cfg2.get("ROUTE_EIK_ITERS", 20))
_min_ds_train  = int(_cfg2.get("ROUTE_EIK_DOWNSAMPLE", 8))
print(f"  Device={_device}  PATCH={_PATCH}  NORM_PX={_NORM_PX}")
print(f"  Training Eikonal: n_iters={_n_iters_train}  min_ds={_min_ds_train}  margin={_margin}")
print(f"  gate_alpha={_model.route_gate_alpha}  ckpt_chunk={_model.route_ckpt_chunk}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load image + tiled road_prob [Source A]
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Load image + tiled road_prob ===")
_rgb_orig = load_tif_image(args.image_path).to(_device)
_, _, _H_orig, _W_orig = _rgb_orig.shape

_inf = SamRoadTiledInferencer(sat_grid=14,
                               sam_road_ckpt_name=Path(_CKPT_PATH).name)
_inf.ensure_loaded(_device)
_rp_inf_full = torch.from_numpy(
    _inf.infer(_rgb_orig, _device, return_mask=True)["road_mask"]
).to(_device).float().squeeze()

_scale = 1.0
if args.downsample > 0 and max(_rp_inf_full.shape) > args.downsample:
    _scale = args.downsample / max(_rp_inf_full.shape)
    _nH = max(1, int(_rp_inf_full.shape[0] * _scale))
    _nW = max(1, int(_rp_inf_full.shape[1] * _scale))
    _rp_inf_full = F.interpolate(
        _rp_inf_full.unsqueeze(0).unsqueeze(0), size=(_nH, _nW),
        mode="bilinear", align_corners=False).squeeze()
_H_ds, _W_ds = _rp_inf_full.shape
_rgb_ds = F.interpolate(_rgb_orig, size=(_H_ds, _W_ds),
                        mode="bilinear", align_corners=False)
print(f"  Working res: {_H_ds}×{_W_ds}  scale={_scale:.4f}")
print(f"  NOTE: eval patch is 512px at scale={_scale:.4f}  "
      f"≙ {int(512/_scale)}px original = {512/_scale/_H_orig*100:.1f}% of image height")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load nodes
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Load nodes ===")
_rp_snap = _rp_inf_full if _scale == 1.0 else F.interpolate(
    _rp_inf_full.unsqueeze(0).unsqueeze(0), size=(_H_orig, _W_orig),
    mode="bilinear", align_corners=False).squeeze()
_nodes_orig, _info = load_nodes_from_npz(
    args.image_path, _rp_snap, p_count=20, snap=True,
    snap_threshold=0.30, snap_win=10, verbose=False)
_nodes_yx = _nodes_orig
if _scale != 1.0:
    _nodes_yx = (_nodes_orig.float() * _scale).long()
    _nodes_yx[:, 0].clamp_(0, _H_ds - 1)
    _nodes_yx[:, 1].clamp_(0, _W_ds - 1)
N = _nodes_yx.shape[0]
_case    = _info["case_idx"]
_meta_px = int(_info["meta_sat_height_px"])   # original image height in pixels
_pscale  = 1.0 / _scale if _scale != 1.0 else 1.0
print(f"  Nodes: {N}   meta_px={_meta_px}")

# ─────────────────────────────────────────────────────────────────────────────
# Cache: model-forward road_prob per patch
# ─────────────────────────────────────────────────────────────────────────────
_rp_model_cache: dict = {}

def get_model_rp(py0, py1, px0, px1):
    key = (py0, py1, px0, px1)
    if key not in _rp_model_cache:
        rgb_p = _rgb_ds[:, :, py0:py1, px0:px1]
        if rgb_p.shape[2] < _PATCH or rgb_p.shape[3] < _PATCH:
            rgb_p = F.pad(rgb_p, (0, _PATCH - rgb_p.shape[3],
                                   0, _PATCH - rgb_p.shape[2]), value=0.0)
        rgb_hwc = (rgb_p * 255.0).squeeze(0).permute(1, 2, 0).unsqueeze(0)
        with torch.no_grad():
            _, ms = _model._predict_mask_logits_scores(rgb_hwc)
            _rp_model_cache[key] = ms[0, :, :, 1]
    return _rp_model_cache[key]

# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

def solve_training_path(rp_patch, anc_patch, tgt_patch):
    """
    TRUE training solver: _roi_multi_target_diff_solve under no_grad.
    - Differentiable Eikonal (_eikonal_soft_sweeping_diff)
    - gate_alpha from model config
    - ONE ROI covering all K targets (same as training_step)
    - Adaptive ds (same as training)
    Returns: np.ndarray [K] of raw T values
    """
    K = tgt_patch.shape[0]
    rp_b   = rp_patch.unsqueeze(0)              # [1, H, W]
    src_b  = anc_patch.unsqueeze(0)             # [1, 2]
    tgt_b  = tgt_patch.unsqueeze(0)             # [1, K, 2]
    # gt_dist_k: need > 0 for all valid targets (validity check in solver)
    gt_b   = torch.ones(1, K, device=_device)  # [1, K]
    with torch.no_grad():
        T = _model._roi_multi_target_diff_solve(rp_b, src_b, tgt_b, gt_b)
    return T[0].cpu().numpy()   # [K]


@torch.no_grad()
def solve_nondiff(rp_patch, anc_patch, tgt_patch, n_iters, min_ds):
    """
    Non-diff eikonal_soft_sweeping, K separate ROIs (ablation).
    """
    H, W   = rp_patch.shape
    src    = anc_patch.long()
    K      = tgt_patch.shape[0]
    T_vals = []
    for k in range(K):
        tgt  = tgt_patch[k].long()
        span = int(torch.max(torch.abs(tgt.float() - src.float())).item())
        half = span + _margin
        P    = max(2 * half + 1, 64)
        y0   = int(src[0].item()) - half;  x0 = int(src[1].item()) - half
        y1   = y0 + P;                     x1 = x0 + P
        yy0  = max(y0, 0); xx0 = max(x0, 0)
        yy1  = min(y1, H); xx1 = min(x1, W)
        roi  = F.pad(rp_patch[yy0:yy1, xx0:xx1],
                     (xx0-x0, x1-xx1, yy0-y0, y1-yy1), value=0.0)
        sr_y = max(0, min(int(src[0].item())-y0, P-1))
        sr_x = max(0, min(int(src[1].item())-x0, P-1))
        tr_y = max(0, min(int(tgt[0].item())-y0, P-1))
        tr_x = max(0, min(int(tgt[1].item())-x0, P-1))
        ds   = max(min_ds, math.ceil(P / max(n_iters, 1)))
        if ds > 1:
            P_pad = math.ceil(P / ds) * ds
            if P_pad > P:
                roi = F.pad(roi, (0, P_pad-P, 0, P_pad-P), value=0.0)
            roi_c  = F.avg_pool2d(roi.unsqueeze(0).unsqueeze(0),
                                  kernel_size=ds, stride=ds).squeeze()
            cost   = _model._road_prob_to_cost(roi_c)
            Pc     = cost.shape[0]
            sc_y   = max(0, min(sr_y//ds, Pc-1)); sc_x = max(0, min(sr_x//ds, Pc-1))
            tc_y   = max(0, min(tr_y//ds, Pc-1)); tc_x = max(0, min(tr_x//ds, Pc-1))
            smask  = torch.zeros(1, Pc, Pc, dtype=torch.bool, device=_device)
            smask[0, sc_y, sc_x] = True
            cfg    = EikonalConfig(n_iters=n_iters, h=float(ds),
                                   tau_min=0.03, tau_branch=0.05,
                                   tau_update=0.03, use_redblack=True, monotone=True)
            T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)
            if T.dim() == 3: T = T[0]
            T_vals.append(float(T[tc_y, tc_x].item()))
        else:
            cost   = _model._road_prob_to_cost(roi)
            smask  = torch.zeros(1, P, P, dtype=torch.bool, device=_device)
            smask[0, sr_y, sr_x] = True
            cfg    = EikonalConfig(n_iters=n_iters, h=1.0,
                                   tau_min=0.03, tau_branch=0.05,
                                   tau_update=0.03, use_redblack=True, monotone=True)
            T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)
            if T.dim() == 3: T = T[0]
            T_vals.append(float(T[tr_y, tr_x].item()))
    return np.array(T_vals)

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def _rank_metrics(pred, ref):
    """Spearman, Kendall, pairwise-acc between pred and ref (both 1-D arrays)."""
    valid = np.isfinite(pred) & (pred < 1e5) & np.isfinite(ref)
    if valid.sum() < 2:
        return dict(spearman=np.nan, kendall=np.nan, pw=np.nan)
    p, r = pred[valid], ref[valid]
    s, _ = spearmanr(p, r)
    k, _ = kendalltau(p, r)
    n     = valid.sum()
    pairs, correct = 0, 0
    idx = np.where(valid)[0]
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            ii, jj = idx[i], idx[j]
            if ref[ii] == ref[jj]: continue
            pairs += 1
            if (pred[ii] < pred[jj]) == (ref[ii] < ref[jj]):
                correct += 1
    return dict(spearman=float(s), kendall=float(k),
                pw=(correct/pairs if pairs else np.nan))

def _mae(pred, ref):
    valid = np.isfinite(pred) & (pred < 1e5) & np.isfinite(ref)
    if valid.sum() == 0: return np.nan
    return float(np.mean(np.abs(pred[valid] - ref[valid])))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Loop over all anchors
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== Multi-anchor eval (N={N}, min_in_patch={args.min_in_patch}) ===\n")

# Accumulators keyed by config label
acc = defaultdict(lambda: defaultdict(list))

n_valid = 0
t0_total = time.time()

for anc_idx in range(N):
    anc = _nodes_yx[anc_idx]
    # KNN by Euclidean distance
    diffs  = (_nodes_yx.float() - anc.float())
    euclid = torch.sqrt((diffs**2).sum(1))
    euclid[anc_idx] = 1e9
    nn_all = torch.argsort(euclid)[:args.k]

    # Patch bounds centered on anchor
    ay, ax = int(anc[0].item()), int(anc[1].item())
    py0 = max(0, ay - _PATCH//2); px0 = max(0, ax - _PATCH//2)
    py1 = py0 + _PATCH;           px1 = px0 + _PATCH
    if py1 > _H_ds: py0, py1 = _H_ds - _PATCH, _H_ds
    if px1 > _W_ds: px0, px1 = _W_ds - _PATCH, _W_ds
    py0 = max(0, py0); px0 = max(0, px0)

    # Filter to in-patch targets
    tgts_all = _nodes_yx[nn_all]
    in_mask  = ((tgts_all[:,0] >= py0) & (tgts_all[:,0] < py1) &
                (tgts_all[:,1] >= px0) & (tgts_all[:,1] < px1))
    keep = torch.where(in_mask)[0]
    if len(keep) < args.min_in_patch:
        continue

    nn_idx  = nn_all[keep]
    tgts    = tgts_all[keep]
    K       = len(nn_idx)
    nn_np   = nn_idx.cpu().numpy()

    # Patch-relative coords
    anc_p = torch.tensor([ay-py0, ax-px0], dtype=torch.long, device=_device)
    tgt_p = (tgts - torch.tensor([py0, px0], device=_device)).long()   # [K,2]

    # GT distances (undirected_dist_norm is normed by meta_px at original resolution)
    gt_norm = np.asarray(_info["undirected_dist_norm"][_case][anc_idx, nn_np], dtype=np.float64)
    euc_norm = np.asarray(_info["euclidean_dist_norm"][_case][anc_idx, nn_np], dtype=np.float64)

    # GT in original pixels and in training-norm units
    gt_orig_px = gt_norm * _meta_px                     # original pixel distance
    gt_train   = gt_orig_px / _NORM_PX                  # training-loss normalisation

    # road_prob patches
    rp_A = _rp_inf_full[py0:py1, px0:px1]              # [PATCH,PATCH] tiled inferencer
    rp_B = get_model_rp(py0, py1, px0, px1)            # [PATCH,PATCH] model forward

    def record(label, T_raw):
        # --- ranking metrics (norm-invariant) ---
        m = _rank_metrics(T_raw, gt_norm)
        for k, v in m.items():
            if not np.isnan(v):
                acc[label]["rank_" + k].append(v)
        # --- MAE under training normalisation ---
        # pred in Eikonal cost units ≈ pixel distance at downsampled res
        # gt_train in original-pixel/NORM_PX units
        # To make them comparable: convert pred to same scale
        # pred_ds_px ≈ T_raw (Eikonal cost, roughly proportional to ds-pixel path)
        # pred_train = pred_ds_px * _pscale / _NORM_PX  (upscale to orig, then norm)
        valid = np.isfinite(T_raw) & (T_raw < 1e5)
        pred_train = T_raw * _pscale / _NORM_PX
        mae_train = _mae(pred_train, gt_train)
        if not np.isnan(mae_train):
            acc[label]["mae_train"].append(mae_train)

    # Euclidean baseline
    record("Euclidean", euc_norm * _meta_px)   # raw units matching T ordering

    # [Ref] Infer-A + non-diff 700-iter
    T_ref = solve_nondiff(rp_A, anc_p, tgt_p, n_iters=700, min_ds=1)
    record("Ref Infer-A 700/ds1", T_ref)

    # [TRAIN] Model-B + TRUE training solver (diff, multi-target ROI)
    T_train = solve_training_path(rp_B, anc_p, tgt_p)
    record("TRAIN Model-B diff", T_train)

    # [Cmp] Model-B + non-diff 20/ds8
    T_nd = solve_nondiff(rp_B, anc_p, tgt_p, n_iters=_n_iters_train, min_ds=_min_ds_train)
    record("Cmp  Model-B 20/ds8 non-diff", T_nd)

    # [Cmp] Model-B + non-diff 50/ds1 (more iterations, full resolution)
    T_nd50 = solve_nondiff(rp_B, anc_p, tgt_p, n_iters=50, min_ds=1)
    record("Cmp  Model-B 50/ds1 non-diff", T_nd50)

    # [Cmp] Model-B + non-diff 700/ds1 (reference quality on model road_prob)
    T_nd700 = solve_nondiff(rp_B, anc_p, tgt_p, n_iters=700, min_ds=1)
    record("Ref  Model-B 700/ds1", T_nd700)

    # [Cmp] Infer-A + non-diff 20/ds8
    T_ia = solve_nondiff(rp_A, anc_p, tgt_p, n_iters=_n_iters_train, min_ds=_min_ds_train)
    record("Cmp  Infer-A 20/ds8 non-diff", T_ia)

    n_valid += 1
    print(f"  anchor {anc_idx:>2}  K={K}  ✓  "
          f"T_train[0]={T_train[0]:.1f}  T_nd[0]={T_nd[0]:.1f}")

    if args.save_vis and n_valid == 1:
        _eik_vis = EikonalConfig(n_iters=700, tau_min=0.03, tau_branch=0.05,
                                 tau_update=0.03, use_redblack=True, monotone=True)
        _rgb_patch = _rgb_ds[:, :, py0:py1, px0:px1]
        _model.visualize_paths(
            rgb=_rgb_patch,
            road_prob=rp_B,
            src_yx=anc_p,
            tgt_yx=tgt_p,
            labels=[f"node {int(i)}" for i in nn_np],
            save_path=args.save_vis,
            eik_cfg=_eik_vis,
            margin=_margin,
            show_T_field=True,
            show_road_prob=True,
            figsize=(14, 14),
            dpi=150,
        )

elapsed = time.time() - t0_total

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*82}")
print(f"  Multi-anchor summary  |  valid={n_valid}/{N}  time={elapsed:.1f}s")
print(f"  PATCH={_PATCH}  NORM_PX={_NORM_PX}  scale={_scale:.4f}")
print(f"\n  NOTE on normalization:")
print(f"    Training gt_dist = d_norm * meta_px (in original pixels, from NPZ)")
print(f"    Eikonal T_raw   ≈ path cost in downsampled-pixel units")
print(f"    pred_train = T_raw * pscale / NORM_PX  (upscale → original → /512)")
print(f"    gt_train   = d_norm * meta_px / NORM_PX  (= d_norm * {_meta_px}/{int(_NORM_PX)})")

_col = 34
print(f"\n  {'Config':<{_col}} {'Spearman':>9} {'±':>6} "
      f"{'Kendall':>9} {'±':>6} {'PW-Acc':>8} {'MAE-tr':>8} {'N':>3}")
print("  " + "-" * (_col + 55))

ORDER = [
    "Euclidean",
    "Ref Infer-A 700/ds1",
    "Ref  Model-B 700/ds1",
    "TRAIN Model-B diff",
    "Cmp  Model-B 50/ds1 non-diff",
    "Cmp  Model-B 20/ds8 non-diff",
    "Cmp  Infer-A 20/ds8 non-diff",
]
for label in ORDER:
    v = acc[label]
    if not v["rank_spearman"]:
        print(f"  {label:<{_col}}  (no data)")
        continue
    s  = np.array(v["rank_spearman"])
    k  = np.array(v["rank_kendall"])
    pw = np.array(v["rank_pw"])
    mae = np.array(v["mae_train"]) if v["mae_train"] else np.array([np.nan])
    tag = "◀ TRAIN" if "TRAIN" in label else ("◀ REF" if "Ref" in label else "")
    print(f"  {label:<{_col}} {np.nanmean(s):>+9.4f} {np.nanstd(s):>6.4f} "
          f"{np.nanmean(k):>+9.4f} {np.nanstd(k):>6.4f} "
          f"{np.nanmean(pw):>8.4f} {np.nanmean(mae):>8.4f} "
          f"{len(s):>3}  {tag}")

print("  " + "-" * (_col + 55))
print("\n  Spearman: higher=better ranking (1.0=perfect, 0=random)")
print("  MAE-tr: mean |pred_train - gt_train| under training normalisation")
print("  ◀ TRAIN = actual training forward path (diff solver, model road_prob)")
print("\nDone.")
