#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict N×N distance matrix using trained SAMRoute model.

Usage:
  python predict_dist_demo.py \
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
    GradcheckConfig,
)


# ---------------------------------------------------------------------------
# Thin wrapper — delegates to model.forward_distance_matrix
# ---------------------------------------------------------------------------

def predict_distance_matrix(
    model: SAMRoute,
    road_prob_t: torch.Tensor,
    nodes_yx: torch.Tensor,
    *,
    ds: int = 16,
    mg_f: int = 4,
    eik_iters: int = 40,
    pool_mode: str = "max",
    iter_floor_c: float = 1.2,
    iter_floor_f: float = 0.65,
) -> torch.Tensor:
    """Differentiable N×N distance matrix prediction.

    Thin wrapper around ``model.forward_distance_matrix`` for backward
    compatibility.  See :meth:`SAMRoute.forward_distance_matrix` for details.
    """
    return model.forward_distance_matrix(
        nodes_yx,
        road_prob=road_prob_t,
        ds=ds,
        mg_factor=mg_f,
        eik_iters=eik_iters,
        pool_mode=pool_mode,
        iter_floor_c=iter_floor_c,
        iter_floor_f=iter_floor_f,
    )


# ---------------------------------------------------------------------------
# Node parser
# ---------------------------------------------------------------------------

def _parse_nodes(nodes_str: str) -> np.ndarray:
    """Parse 'y,x;y,x;...' -> (N,2) int64, yx pixel coords."""
    parts = [p.strip() for p in nodes_str.split(";") if p.strip()]
    yx = []
    for p in parts:
        y_s, x_s = [t.strip() for t in p.split(",")]
        yx.append([int(float(y_s)), int(float(x_s))])
    return np.asarray(yx, dtype=np.int64)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tsp_load_ckpt", type=str, default="",
                    help="Routing checkpoint (alpha/gamma/gate). Overridden by --gate/--alpha/--gamma if >= 0.")
    ap.add_argument("--nodes", type=str, required=True,
                    help='Nodes: "y,x;y,x;..." in pixel coords.')
    ap.add_argument("--out", type=str, default="/tmp/D.npy")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ds", type=int, default=16)
    ap.add_argument("--eik_iters", type=int, default=40)
    ap.add_argument("--pool_mode", type=str, default="max", choices=["max", "avg"])
    ap.add_argument("--gate", type=float, default=1.0,
                    help="0..1, 1=pure Eikonal. -1=use checkpoint value.")
    ap.add_argument("--alpha", type=float, default=50.0,
                    help="Cost alpha. -1=use checkpoint value.")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="Cost gamma. -1=use checkpoint value.")
    ap.add_argument("--cost_net", action="store_true")
    ap.add_argument("--cost_net_ch", type=int, default=16)
    ap.add_argument("--iter_floor_c", type=float, default=1.2)
    ap.add_argument("--iter_floor_f", type=float, default=0.65)
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---- load checkpoint & auto-detect config ----
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
    cfg.ROUTE_EIK_ITERS = int(args.eik_iters)
    cfg.ROUTE_EIK_DOWNSAMPLE = int(args.ds)

    if args.cost_net:
        cfg.ROUTE_COST_NET = True
        cfg.ROUTE_COST_NET_CH = args.cost_net_ch

    if "block_log_alpha" in sd or "block_th_logit" in sd:
        cfg.LEARNABLE_BLOCK = True

    model = SAMRoute(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    for p in model.image_encoder.parameters():
        p.requires_grad_(False)

    # ---- load routing checkpoint (optional) ----
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

    # ---- apply parameter overrides (eval-aligned: -1 = keep checkpoint value) ----
    with torch.no_grad():
        if args.gate >= 0:
            logit = 10.0 if args.gate >= 0.9999 else math.log(
                args.gate / max(1.0 - args.gate, 1e-9))
            model.eik_gate_logit.fill_(logit)
            print(f"[override] gate={args.gate:.4f} (logit={logit:.2f})")
        if args.alpha > 0:
            model.cost_log_alpha.fill_(math.log(args.alpha))
            print(f"[override] alpha={args.alpha:.1f}")
        if args.gamma > 0:
            model.cost_log_gamma.fill_(math.log(args.gamma))
            print(f"[override] gamma={args.gamma:.2f}")

    model.eval()
    model.route_gate_alpha = 0.8

    # ---- load image & infer road_prob ----
    rgb = _load_rgb_from_tif(args.tif)
    H, W = rgb.shape[:2]
    print(f"[tif] {args.tif}  shape={H}x{W}")

    road_prob_np = sliding_window_inference(
        rgb, model, device, patch_size=int(cfg.PATCH_SIZE),
    ).astype(np.float32)
    road_prob_t = torch.from_numpy(road_prob_np).to(
        device, torch.float32,
    ).unsqueeze(0).clamp(1e-6, 1.0 - 1e-6)

    # ---- predict distance matrix via model.forward_distance_matrix ----
    nodes_yx = _parse_nodes(args.nodes)
    nodes_t = torch.from_numpy(nodes_yx).to(device, torch.long)

    with torch.no_grad():
        D = model.forward_distance_matrix(
            nodes_t,
            road_prob=road_prob_t,
            ds=int(args.ds),
            eik_iters=int(args.eik_iters),
            pool_mode=args.pool_mode,
            iter_floor_c=args.iter_floor_c,
            iter_floor_f=args.iter_floor_f,
        )

    D_np = D.cpu().numpy().astype(np.float32)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, D_np)
    print(f"[saved] {args.out}  shape={D_np.shape}  "
          f"min/mean/max={D_np.min():.3f}/{D_np.mean():.3f}/{D_np.max():.3f}")


if __name__ == "__main__":
    main()
