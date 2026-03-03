"""
Predict distance using the trained model.
""" 



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal distance prediction (call-only, no duplicated defs).

Usage:
  python predict_distance_minimal_callonly.py \
    --tif  /path/to/crop_xxx.tif \
    --ckpt /path/to/best.ckpt \
    --nodes "100,120;200,300;350,400" \
    --out /tmp/D.npy \
    --ds 16 \
    --device cuda
"""

import argparse
import math
import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model_multigrid import SAMRoute

from gradcheck_route_loss_v2_multigrid_fullmap import (
    sliding_window_inference,
    _load_lightning_ckpt,
    _detect_smooth_decoder,
    _detect_patch_size,
    _load_rgb_from_tif,
    _maxpool_prob,
    _mix_eikonal_euclid,
    GradcheckConfig,
)

# 直接复用你已有的函数：不要在本文件重复定义
# 方案 A：如果你保留在 eval_ranking_accuracy.py 里，就这样 import
from eval_ranking_accuracy import _avgpool_prob, _solve_all_sources_once


def _parse_nodes(nodes_str: str) -> np.ndarray:
    """Parse 'y,x;y,x;...' -> (N,2) int64, yx pixel coords."""
    parts = [p.strip() for p in nodes_str.split(";") if p.strip()]
    yx = []
    for p in parts:
        y_s, x_s = [t.strip() for t in p.split(",")]
        yx.append([int(float(y_s)), int(float(x_s))])
    return np.asarray(yx, dtype=np.int64)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--nodes", type=str, required=True,
                    help='Nodes: "y,x;y,x;..." in pixel coords.')
    ap.add_argument("--out", type=str, default="/tmp/D.npy")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ds", type=int, default=16)
    ap.add_argument("--eik_iters", type=int, default=40)
    ap.add_argument("--pool_mode", type=str, default="max", choices=["max", "avg"])
    ap.add_argument("--gate", type=float, default=1.0, help="0..1, 1=pure Eikonal")
    ap.add_argument("--alpha", type=float, default=50.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--iter_floor_c", type=float, default=1.2)
    ap.add_argument("--iter_floor_f", type=float, default=0.65)
    args = ap.parse_args()

    device = torch.device(args.device)
    nodes_yx = _parse_nodes(args.nodes)

    # ---- load model ----
    sd = _load_lightning_ckpt(args.ckpt)
    cfg = GradcheckConfig()
    cfg.USE_SMOOTH_DECODER = _detect_smooth_decoder(sd)
    ps = _detect_patch_size(sd)
    if ps is not None:
        cfg.PATCH_SIZE = ps
    cfg.ROUTE_EIK_ITERS = int(args.eik_iters)
    cfg.ROUTE_EIK_DOWNSAMPLE = int(args.ds)

    model = SAMRoute(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    for p in model.image_encoder.parameters():
        p.requires_grad_(False)

    # ---- override routing params (optional) ----
    with torch.no_grad():
        g = float(np.clip(args.gate, 0.0, 1.0))
        if g >= 0.9999:
            logit = 10.0
        elif g <= 1e-6:
            logit = -10.0
        else:
            logit = math.log(g / max(1.0 - g, 1e-9))
        model.eik_gate_logit.fill_(logit)
        model.cost_log_alpha.fill_(math.log(float(args.alpha)))
        model.cost_log_gamma.fill_(math.log(float(args.gamma)))

    model.route_gate_alpha = 0.8

    # ---- load image ----
    rgb = _load_rgb_from_tif(args.tif)
    H, W = rgb.shape[:2]

    # ---- infer road_prob ----
    road_prob_np = sliding_window_inference(rgb, model, device, patch_size=int(cfg.PATCH_SIZE)).astype(np.float32)
    road_prob_t = torch.from_numpy(road_prob_np).to(device, torch.float32).unsqueeze(0).clamp(1e-6, 1.0 - 1e-6)

    # ---- build cost maps ----
    ds = int(args.ds)
    mg_f = 4
    pool_fn = _avgpool_prob if args.pool_mode == "avg" else _maxpool_prob

    prob_f, _, _ = pool_fn(road_prob_t, ds)
    cost_f = model._road_prob_to_cost(prob_f)[0].to(torch.float32)  # [Hf,Wf]

    prob_c, _, _ = pool_fn(road_prob_t, ds * mg_f)
    cost_c = model._road_prob_to_cost(prob_c)[0].to(torch.float32)  # [Hc,Wc]

    # ---- solve all sources once ----
    nodes_yx[:, 0] = np.clip(nodes_yx[:, 0], 0, H - 1)
    nodes_yx[:, 1] = np.clip(nodes_yx[:, 1], 0, W - 1)
    all_src_yx = torch.from_numpy(nodes_yx).to(device, torch.long)  # [N,2]

    mg_itc = max(20, int(args.eik_iters * 0.25))
    mg_itf = max(20, int(args.eik_iters * 0.80))

    T_maps = _solve_all_sources_once(
        model, cost_f, cost_c, all_src_yx,
        model.route_eik_cfg, ds, mg_f, mg_itc, mg_itf, device,
        iter_floor_c=args.iter_floor_c,
        iter_floor_f=args.iter_floor_f,
    )  # [N,Hf,Wf]

    # ---- lookup to build NxN distance matrix ----
    N = nodes_yx.shape[0]
    Hf, Wf = cost_f.shape[-2], cost_f.shape[-1]
    tgt_fc = (all_src_yx // ds).clamp(min=0)
    tgt_fc[:, 0] = tgt_fc[:, 0].clamp(0, Hf - 1)
    tgt_fc[:, 1] = tgt_fc[:, 1].clamp(0, Wf - 1)

    D_eik = torch.empty((N, N), device=device, dtype=torch.float32)
    for i in range(N):
        D_eik[i] = T_maps[i, tgt_fc[:, 0], tgt_fc[:, 1]]

    # Euclidean for optional mix
    y = nodes_yx[:, 0].astype(np.float32)
    x = nodes_yx[:, 1].astype(np.float32)
    dy = y[:, None] - y[None, :]
    dx = x[:, None] - x[None, :]
    D_euc = np.sqrt(dy * dy + dx * dx).astype(np.float32)

    D = _mix_eikonal_euclid(D_eik, torch.from_numpy(D_euc).to(device), model.eik_gate_logit)
    D = D.detach().cpu().numpy().astype(np.float32)
    np.fill_diagonal(D, 0.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, D)
    print(f"[saved] {args.out}  shape={D.shape}  min/mean/max={D.min():.3f}/{D.mean():.3f}/{D.max():.3f}")


if __name__ == "__main__":
    main()