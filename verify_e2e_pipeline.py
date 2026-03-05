#!/usr/bin/env python3
"""
Phase 1 & 2 verification for the end-to-end TSP pipeline (encoder_mode='e2e_eikonal').

Usage:
    python verify_e2e_pipeline.py

This script checks:
  Phase 1: Computation graph connectivity
    - distance_matrix retains requires_grad after _compute_distance_matrix_e2e
    - SAMRoute cost params (cost_log_alpha, cost_log_gamma, eik_gate_logit) receive non-zero grad
    - distance_matrix has no NaN/Inf
  Phase 2: Online distance vs NPZ distance comparison
"""

import os
import sys
import time
import math
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EIKONAL_DIR = os.path.join(SCRIPT_DIR, 'eikonal_solver')
if EIKONAL_DIR not in sys.path:
    sys.path.insert(0, EIKONAL_DIR)


def find_project_root(start_path, marker='MMDataset'):
    current = os.path.abspath(start_path)
    for _ in range(10):
        if os.path.isdir(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.abspath(start_path)


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'MMDataset', 'Gen_dataset_V2', 'Gen_dataset')

from utils.utils import create_logger, copy_all_src


def run_phase1(device):
    """Phase 1: Computation graph connectivity verification."""
    print("\n" + "=" * 70)
    print("Phase 1: Computation Graph Connectivity")
    print("=" * 70)

    from model_multigrid import SAMRoute
    from gradcheck_route_loss_v2_multigrid_fullmap import (
        sliding_window_inference as swi,
        _load_lightning_ckpt,
        _detect_smooth_decoder,
        _detect_patch_size,
    )

    ckpt_path = os.path.join(
        SCRIPT_DIR, 'training_outputs', 'fulldataset_seg_dist_lora_v2',
        'checkpoints', 'best_seg_dist_lora_v2.ckpt')
    if not os.path.isfile(ckpt_path):
        print(f"[SKIP] Checkpoint not found: {ckpt_path}")
        return False

    sd = _load_lightning_ckpt(ckpt_path)
    from types import SimpleNamespace

    has_lora = any("attn.qkv.linear_a_q" in k or "attn.qkv.linear_a_v" in k for k in sd)
    pe = sd.get('image_encoder.pos_embed')
    patch_size = int(pe.shape[1]) * 16 if pe is not None else 512
    w7 = sd.get('map_decoder.7.weight')
    use_smooth = w7 is not None and w7.shape[1] == 32

    cfg = SimpleNamespace(
        SAM_VERSION='vit_b', PATCH_SIZE=patch_size, NO_SAM=False,
        USE_SAM_DECODER=False, USE_SMOOTH_DECODER=use_smooth,
        ENCODER_LORA=has_lora, LORA_RANK=4, FREEZE_ENCODER=True,
        FOCAL_LOSS=False, TOPONET_VERSION='default',
        SAM_CKPT_PATH=os.path.join(SCRIPT_DIR, 'sam_road_repo', 'sam_ckpts', 'sam_vit_b_01ec64.pth'),
        ROUTE_COST_MODE='add', ROUTE_ADD_ALPHA=20.0, ROUTE_ADD_GAMMA=2.0,
        ROUTE_ADD_BLOCK_ALPHA=0.0, ROUTE_BLOCK_TH=0.0,
        ROUTE_ROI_MARGIN=64, ROUTE_COST_NET=False, ROUTE_COST_NET_CH=8,
        ROUTE_COST_NET_USE_COORD=False,
    )
    if has_lora:
        cfg.FREEZE_ENCODER = False
        for k, v in sd.items():
            if "linear_a_q.weight" in k:
                cfg.LORA_RANK = v.shape[0]
                break

    model = SAMRoute(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.requires_grad_(False)
    model.cost_log_alpha.requires_grad_(True)
    model.cost_log_gamma.requires_grad_(True)
    model.eik_gate_logit.requires_grad_(True)
    model.eval()

    print(f"[OK] SAMRoute loaded. cost_log_alpha={model.cost_log_alpha.item():.4f}, "
          f"cost_log_gamma={model.cost_log_gamma.item():.4f}, "
          f"eik_gate_logit={model.eik_gate_logit.item():.4f}")

    # --- find a test TIF + NPZ ---
    test_tif = None
    test_npz = None
    if os.path.isdir(DATASET_ROOT):
        for city in os.listdir(DATASET_ROOT):
            city_dir = os.path.join(DATASET_ROOT, city)
            if not os.path.isdir(city_dir):
                continue
            for sg in os.listdir(city_dir):
                sg_dir = os.path.join(city_dir, sg)
                if not os.path.isdir(sg_dir):
                    continue
                tifs = [f for f in os.listdir(sg_dir) if f.startswith('crop_') and f.endswith('.tif')]
                npzs = [f for f in os.listdir(sg_dir) if f.endswith('_p20.npz')]
                if tifs and npzs:
                    test_tif = os.path.join(sg_dir, tifs[0])
                    test_npz = os.path.join(sg_dir, npzs[0])
                    break
            if test_tif:
                break

    if test_tif is None:
        print("[SKIP] No test TIF/NPZ found under", DATASET_ROOT)
        return False

    print(f"[INFO] Using TIF: {test_tif}")
    print(f"[INFO] Using NPZ: {test_npz}")

    # --- load image & compute road_prob ---
    from PIL import Image
    img = Image.open(test_tif).convert('RGB')
    img_np = np.asarray(img, dtype=np.uint8)
    H_img, W_img = img_np.shape[:2]
    print(f"[INFO] Image size: {H_img}x{W_img}")

    t0 = time.time()
    road_prob_np = swi(img_np, model, device, patch_size=patch_size, verbose=False).astype(np.float32)
    t_swi = time.time() - t0
    print(f"[OK] road_prob computed in {t_swi:.2f}s, shape={road_prob_np.shape}, "
          f"min={road_prob_np.min():.4f}, max={road_prob_np.max():.4f}")

    road_prob = torch.from_numpy(road_prob_np).to(device, torch.float32).unsqueeze(0).clamp(1e-6, 1.0 - 1e-6)

    # --- load nodes from NPZ ---
    data = np.load(test_npz)
    node_coords_xy = data['matched_node_norm']
    dist_norm = data['undirected_dist_norm']
    case_idx = 0
    xy = node_coords_xy[case_idx]
    N = xy.shape[0]
    print(f"[INFO] Nodes: N={N}")

    x_norm = torch.from_numpy(xy[:, 0]).to(device, torch.float32)
    y_norm = torch.from_numpy(xy[:, 1]).to(device, torch.float32)
    y_pix = torch.round((1.0 - y_norm) * (H_img - 1)).long()
    x_pix = torch.round(x_norm * (W_img - 1)).long()
    nodes_yx = torch.stack([y_pix, x_pix], dim=-1)

    # --- compute distance matrix (differentiable) ---
    t0 = time.time()
    with torch.amp.autocast("cuda", enabled=False):
        D = model.forward_distance_matrix(
            nodes_yx,
            road_prob=road_prob,
            ds=16, eik_iters=40, pool_mode='max',
            iter_floor_c=1.2, iter_floor_f=0.65,
        )
    t_fdm = time.time() - t0
    D_norm = D.float() / float(H_img)

    # --- Phase 1 assertions ---
    passed = True

    # 1. requires_grad
    print(f"\n[CHECK 1] D.requires_grad = {D.requires_grad}")
    if D.requires_grad:
        print("  PASS: distance_matrix retains computation graph")
    else:
        print("  FAIL: distance_matrix has no gradient!")
        passed = False

    # 2. NaN / Inf
    has_nan = torch.isnan(D_norm).any().item()
    has_inf = torch.isinf(D_norm).any().item()
    print(f"[CHECK 2] NaN={has_nan}, Inf={has_inf}")
    if not has_nan and not has_inf:
        print("  PASS: no NaN/Inf")
    else:
        print("  FAIL: numerical instability detected!")
        passed = False

    print(f"[INFO] D_norm: min={D_norm.min().item():.6f}, max={D_norm.max().item():.6f}, "
          f"mean={D_norm.mean().item():.6f}")

    # 3. backward + grad check
    dummy_loss = D_norm.sum()
    model.zero_grad()
    dummy_loss.backward()

    for name, param in [('cost_log_alpha', model.cost_log_alpha),
                        ('cost_log_gamma', model.cost_log_gamma),
                        ('eik_gate_logit', model.eik_gate_logit)]:
        if param.grad is None:
            print(f"[CHECK 3] {name}.grad = None  FAIL")
            passed = False
        else:
            gn = param.grad.abs().mean().item()
            print(f"[CHECK 3] {name}: value={param.item():.6f}, grad_norm={gn:.8f}", end="")
            if gn > 0:
                print("  PASS")
            else:
                print("  FAIL (grad is zero!)")
                passed = False

    print(f"\n[TIMING] sliding_window: {t_swi:.2f}s, forward_distance_matrix: {t_fdm:.2f}s")

    # --- Phase 2: compare with NPZ distance ---
    print("\n" + "=" * 70)
    print("Phase 2: Online Distance vs NPZ Distance Comparison")
    print("=" * 70)

    D_norm_np = D_norm.detach().cpu().numpy()
    npz_dist = dist_norm[case_idx]

    mask = ~np.eye(N, dtype=bool) & np.isfinite(npz_dist) & (npz_dist > 0)
    if mask.any():
        rel_err = np.abs(D_norm_np[mask] - npz_dist[mask]) / np.maximum(npz_dist[mask], 1e-6)
        mean_err = rel_err.mean()
        median_err = np.median(rel_err)
        print(f"[CHECK] Relative error: mean={mean_err:.4f}, median={median_err:.4f}")
        if mean_err < 0.30:
            print(f"  PASS: mean relative error < 30%")
        else:
            print(f"  WARNING: mean relative error >= 30% (may indicate coord/normalization bug)")

        print(f"[INFO] Online D_norm: range=[{D_norm_np[mask].min():.4f}, {D_norm_np[mask].max():.4f}]")
        print(f"[INFO] NPZ dist_norm: range=[{npz_dist[mask].min():.4f}, {npz_dist[mask].max():.4f}]")
    else:
        print("[SKIP] No valid off-diagonal entries in NPZ distance matrix")

    print("\n" + "=" * 70)
    if passed:
        print("ALL Phase 1 CHECKS PASSED")
    else:
        print("SOME Phase 1 CHECKS FAILED - review output above")
    print("=" * 70)
    return passed


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run_phase1(device)
