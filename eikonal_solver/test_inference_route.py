"""
SAM-Road inference on a single GeoTIFF.

Modes
-----
center  : crop the central 512x512 patch and run a single forward pass.
          Produces a 3-panel figure: satellite | probability mask | Eikonal field.

sliding : slide a 512x512 window across the full image with configurable stride
          (default 256 px = 50% overlap).  Overlapping predictions are blended
          with a Hanning (cosine) weight window so patch-boundary seams vanish.
          With --route (default ON), uses multigrid Eikonal to plan a path.

The model loading, routing parameters and Eikonal solver are aligned with
eval_ranking_accuracy.py so that visualised paths correspond to the same
distance metric used for quantitative evaluation.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from model_multigrid import (
    SAMRoute,
    _eikonal_soft_sweeping_diff,
    _eikonal_soft_sweeping_diff_init,
)
from gradcheck_route_loss_v2_multigrid_fullmap import (
    GradcheckConfig,
    _load_lightning_ckpt,
    _detect_smooth_decoder,
    _detect_patch_size,
    _load_rgb_from_tif,
    _maxpool_prob,
    _make_src_mask,
    _mix_eikonal_euclid,
    _load_npz_nodes,
    _normxy_to_yx,
    sliding_window_inference,
    smooth_prob,
)
from dataclasses import replace as dc_replace

_CKPT_DEFAULT = os.path.join(
    _PROJECT_ROOT,
    "training_outputs", "finetune_demo", "checkpoints",
    "best_lora_pos8.0_dice0.5_thin4.0.ckpt",
)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_tif(tif_path: str) -> np.ndarray:
    """Load a GeoTIFF and return uint8 RGB array [H, W, 3]."""
    img = tifffile.imread(tif_path)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[0] >= 3:
        img = img[:3, :, :].transpose(1, 2, 0)
    if img.dtype == np.uint16:
        img = (img / 256.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Point utilities
# ---------------------------------------------------------------------------

def _parse_yx(s: str) -> tuple[int, int]:
    """Parse 'y,x' (or 'y x') into integer (y, x)."""
    import re as _re
    parts = [p for p in _re.split(r"[,\s]+", s.strip()) if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate '{s}'. Use 'y,x' (e.g. '120,450').")
    y, x = int(float(parts[0])), int(float(parts[1]))
    return y, x


def _pick_two_points(
    img: np.ndarray,
    road_prob: "np.ndarray | None" = None,
    title: str = "Click START then END (left mouse). Press Enter when done.",
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Interactive picking (matplotlib ginput). Returns (src_yx, tgt_yx)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    if road_prob is not None:
        ax.imshow(road_prob, cmap="magma", vmin=0, vmax=1, alpha=0.35)
    ax.set_title(title)
    ax.axis("off")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise RuntimeError("Point picking cancelled or insufficient points. Need exactly 2 clicks.")
    src = (int(round(pts[0][1])), int(round(pts[0][0])))
    tgt = (int(round(pts[1][1])), int(round(pts[1][0])))
    H, W = img.shape[:2]
    src = (max(0, min(H - 1, src[0])), max(0, min(W - 1, src[1])))
    tgt = (max(0, min(H - 1, tgt[0])), max(0, min(W - 1, tgt[1])))
    return src, tgt


def _snap_to_road(
    road_prob: np.ndarray,
    yx: tuple[int, int],
    *,
    radius: int = 0,
    th: float = 0.5,
) -> tuple[int, int]:
    """Snap a clicked point to the nearest high-probability road pixel in a local window."""
    if radius <= 0:
        return yx
    H, W = road_prob.shape[:2]
    y, x = yx
    y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
    x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
    win = road_prob[y0:y1, x0:x1]
    if win.size == 0:
        return yx
    idx = int(np.argmax(win))
    wy, wx = np.unravel_index(idx, win.shape)
    if float(win[wy, wx]) < float(th):
        return yx
    return (y0 + int(wy), x0 + int(wx))


# ---------------------------------------------------------------------------
# Path tracing (gradient descent on distance field T)
# ---------------------------------------------------------------------------

def _trace_path_descend_T(
    T: np.ndarray,
    src_yx: tuple[int, int],
    tgt_yx: tuple[int, int],
    *,
    connectivity8: bool = True,
    max_steps: "int | None" = None,
) -> np.ndarray:
    """Backtrack a shortest path by descending the Eikonal distance field T."""
    H, W = T.shape
    sy, sx = src_yx
    ty, tx = tgt_yx
    max_steps = max_steps or int(H * W)

    if connectivity8:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    y, x = ty, tx
    path = [(y, x)]
    eps = 1e-6

    for _ in range(int(max_steps)):
        if (y == sy) and (x == sx):
            break
        cur = float(T[y, x])
        best = None
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            val = float(T[ny, nx])
            if val < cur - eps:
                if (best is None) or (val < best[0]):
                    best = (val, ny, nx)
        if best is None:
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                val = float(T[ny, nx])
                if (best is None) or (val < best[0]):
                    best = (val, ny, nx)
        if best is None:
            break
        y, x = best[1], best[2]
        path.append((y, x))

    path = path[::-1]  # src -> tgt
    return np.asarray(path, dtype=np.int32)


# ---------------------------------------------------------------------------
# Multigrid Eikonal solver (aligned with eval_ranking_accuracy.py)
# ---------------------------------------------------------------------------

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

    T_init = F.interpolate(
        T_c.unsqueeze(1), size=(Hf, Wf), mode="bilinear", align_corners=False,
    ).squeeze(1).clamp_min(0.0).clamp_max(large_val)
    src_fc = (all_src_yx.long() // ds).clamp(min=0)
    src_fc[:, 0] = src_fc[:, 0].clamp(0, Hf - 1)
    src_fc[:, 1] = src_fc[:, 1].clamp(0, Wf - 1)
    src_mask_f = _make_src_mask(B, Hf, Wf, src_fc, device)
    T_init = torch.where(src_mask_f, torch.zeros_like(T_init), T_init)

    cost_f_b = cost_f.expand(B, -1, -1)
    actual_iters_f = int(max(mg_itf, int(max(Hf, Wf) * iter_floor_f)))
    cfg_f = dc_replace(cfg, h=float(ds), n_iters=actual_iters_f, monotone=False)
    T_fine = _eikonal_soft_sweeping_diff_init(
        cost_f_b, src_mask_f, cfg_f, T_init,
        checkpoint_chunk=ckpt_chunk, gate_alpha=gate_alpha,
    )
    return T_fine


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------

def run_inference_on_tif(
    tif_path,
    ckpt_path=None,
    save_path=None,
    mode="sliding",
    stride=256,
    smooth_sigma=0.0,
    route=True,
    pick_points=False,
    src=None,
    tgt=None,
    snap_radius=25,
    snap_th=0.5,
    max_path_steps=None,
    # --- eval-aligned parameters ---
    tsp_load_ckpt="",
    cost_net=False,
    cost_net_ch=16,
    downsample=16,
    eik_iters=40,
    gate_override=1.0,
    alpha_override=50,
    gamma_override=2.0,
    iter_floor_c=1.2,
    iter_floor_f=0.65,
    gt_mask="",
    npz_case=None,
    p_count=20,
):
    ckpt_path = ckpt_path or _CKPT_DEFAULT
    save_path = save_path or "tif_inference_route_result.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load model (aligned with eval_ranking_accuracy.py) ---
    sd = _load_lightning_ckpt(ckpt_path)
    cfg = GradcheckConfig()
    has_lora = any("attn.qkv.linear_a_q" in k or "attn.qkv.linear_a_v" in k for k in sd)
    if has_lora:
        cfg.ENCODER_LORA = True
        cfg.FREEZE_ENCODER = False
        print(f"  LoRA detected in checkpoint -> ENCODER_LORA=True")
    cfg.USE_SMOOTH_DECODER = _detect_smooth_decoder(sd)
    ps = _detect_patch_size(sd)
    if ps is not None:
        cfg.PATCH_SIZE = ps
    cfg.ROUTE_EIK_ITERS = eik_iters
    cfg.ROUTE_EIK_DOWNSAMPLE = downsample
    if cost_net:
        cfg.ROUTE_COST_NET = True
        cfg.ROUTE_COST_NET_CH = cost_net_ch

    model = SAMRoute(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    for p_param in model.image_encoder.parameters():
        p_param.requires_grad_(False)

    if tsp_load_ckpt and os.path.isfile(tsp_load_ckpt):
        rckpt = torch.load(tsp_load_ckpt, map_location=device, weights_only=False)
        with torch.no_grad():
            if "cost_log_alpha" in rckpt:
                model.cost_log_alpha.copy_(rckpt["cost_log_alpha"].to(device))
            if "cost_log_gamma" in rckpt:
                model.cost_log_gamma.copy_(rckpt["cost_log_gamma"].to(device))
            if "eik_gate_logit" in rckpt:
                model.eik_gate_logit.copy_(rckpt["eik_gate_logit"].to(device))
        if "cost_net_state_dict" in rckpt and getattr(model, "cost_net", None) is not None:
            model.cost_net.load_state_dict(rckpt["cost_net_state_dict"])
        print(f"[loaded] routing ckpt: {tsp_load_ckpt}")

    with torch.no_grad():
        if gate_override >= 0:
            logit = 10.0 if gate_override >= 0.9999 else math.log(
                gate_override / max(1.0 - gate_override, 1e-9))
            model.eik_gate_logit.fill_(logit)
            print(f"[override] gate={gate_override:.4f} (logit={logit:.2f})")
        if alpha_override > 0:
            model.cost_log_alpha.fill_(math.log(alpha_override))
            print(f"[override] alpha={alpha_override:.1f}")
        if gamma_override > 0:
            model.cost_log_gamma.fill_(math.log(gamma_override))
            print(f"[override] gamma={gamma_override:.2f}")

    model.eval()
    model.route_gate_alpha = 0.8

    # --- load image ---
    print(f"Reading image: {tif_path}")
    img_array = _load_rgb_from_tif(tif_path)
    h, w = img_array.shape[:2]
    print(f"[tif] {tif_path}  shape={h}x{w}")

    # ------------------------------------------------------------------
    # Mode: center  — single 512x512 centre crop + Eikonal field
    # ------------------------------------------------------------------
    if mode == "center":
        ps_val = cfg.PATCH_SIZE
        start_y = max(0, (h - ps_val) // 2)
        start_x = max(0, (w - ps_val) // 2)
        img_crop = img_array[start_y:start_y + ps_val, start_x:start_x + ps_val]

        if img_crop.shape[0] < ps_val or img_crop.shape[1] < ps_val:
            pad = np.zeros((ps_val, ps_val, 3), dtype=np.uint8)
            pad[:img_crop.shape[0], :img_crop.shape[1]] = img_crop
            img_crop = pad

        rgb_tensor = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).to(device)

        print("Generating probability mask (center crop)...")
        with torch.no_grad():
            _, mask_scores = model._predict_mask_logits_scores(rgb_tensor)
            road_prob_tensor = mask_scores[..., 1]
            road_prob_raw = road_prob_tensor[0].cpu().numpy()

        road_prob = smooth_prob(road_prob_raw, smooth_sigma)

        if src is not None and tgt is not None:
            src_yx0 = _parse_yx(src) if isinstance(src, str) else tuple(src)
            tgt_yx0 = _parse_yx(tgt) if isinstance(tgt, str) else tuple(tgt)
        elif pick_points:
            src_yx0, tgt_yx0 = _pick_two_points(
                img_crop, road_prob,
                title="CENTER mode -- click START then END (coordinates within 512x512 crop)"
            )
        else:
            y_coords, x_coords = np.where(road_prob > 0.5)
            if len(y_coords) > 10:
                quarter = len(y_coords) // 4
                src_yx0 = (int(y_coords[quarter]), int(x_coords[quarter]))
                tgt_yx0 = (int(y_coords[-quarter - 1]), int(x_coords[-quarter - 1]))
            else:
                src_yx0 = (64, 64)
                tgt_yx0 = (448, 448)

        src_yx0 = _snap_to_road(road_prob, src_yx0, radius=int(snap_radius), th=float(snap_th))
        tgt_yx0 = _snap_to_road(road_prob, tgt_yx0, radius=int(snap_radius), th=float(snap_th))

        src_yx = torch.tensor([list(src_yx0)], dtype=torch.long, device=device)
        tgt_yx = torch.tensor([list(tgt_yx0)], dtype=torch.long, device=device)
        print(f"  src={src_yx.cpu().tolist()}  tgt={tgt_yx.cpu().tolist()}")

        print("Running Eikonal solver...")
        with torch.no_grad():
            from eikonal import make_source_mask, eikonal_soft_sweeping

            B, H_f, W_f = road_prob_tensor.shape
            cost = model._road_prob_to_cost(road_prob_tensor).to(dtype=torch.float32)
            src_mask = make_source_mask(H_f, W_f, src_yx.view(B, 1, 2))

            iters_val = int(eik_iters) if eik_iters is not None else int(model.route_eik_cfg.n_iters)
            eik_cfg = dc_replace(model.route_eik_cfg, n_iters=iters_val, h=1.0, mode="hard_eval")

            T = eikonal_soft_sweeping(cost, src_mask, eik_cfg)
            yy = tgt_yx[:, 0].clamp(0, H_f - 1)
            xx = tgt_yx[:, 1].clamp(0, W_f - 1)
            dist = T[torch.arange(B, device=device), yy, xx]
            dist_field = T[0].detach().cpu().numpy()

        path_yx = None
        if route:
            path_yx = _trace_path_descend_T(
                dist_field, src_yx0, tgt_yx0, max_steps=max_path_steps
            )

        print("Generating figure...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_crop)
        axes[0].set_title("Cropped Satellite Image")
        axes[0].plot(src_yx[0, 1].cpu(), src_yx[0, 0].cpu(), "g*", markersize=15, label="Start")
        axes[0].plot(tgt_yx[0, 1].cpu(), tgt_yx[0, 0].cpu(), "r*", markersize=15, label="End")
        if path_yx is not None:
            axes[0].plot(path_yx[:, 1], path_yx[:, 0], "-", linewidth=2.0, label="Route")
        axes[0].legend()
        axes[0].axis("off")

        sigma_str = f" (sigma={smooth_sigma})" if smooth_sigma > 0 else ""
        im1 = axes[1].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title(f"Predicted Probability Mask{sigma_str}")
        if path_yx is not None:
            axes[1].plot(path_yx[:, 1], path_yx[:, 0], "-", linewidth=2.0)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].axis("off")

        valid_dist = dist_field[dist_field < 1e5]
        vmax = np.percentile(valid_dist, 95) if len(valid_dist) > 0 else 1000
        im2 = axes[2].imshow(dist_field, cmap="viridis", vmax=vmax)
        axes[2].set_title(f"Eikonal Cost Field\n(Target Cost: {dist[0].item():.2f})")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis("off")

    # ------------------------------------------------------------------
    # Mode: sliding — full-image sliding window + multigrid Eikonal
    # ------------------------------------------------------------------
    elif mode == "sliding":
        # --- road_prob from GT mask, cache, or model inference ---
        if gt_mask and os.path.isfile(gt_mask):
            _gt_img = np.array(Image.open(gt_mask))
            if _gt_img.ndim == 3:
                _gt_img = _gt_img[:, :, 0]
            road_prob_np = (_gt_img > 0).astype(np.float32)
            if road_prob_np.shape != (h, w):
                road_prob_np = np.array(
                    Image.fromarray(road_prob_np).resize((w, h), Image.BILINEAR)
                )
            print(f"[GT mask] {gt_mask}  road_ratio={road_prob_np.mean():.4f}")
        else:
            cache = os.path.join("/tmp/vis_inference", "road_prob.npy")
            if os.path.isfile(cache):
                road_prob_np = np.load(cache).astype(np.float32)
                print(f"[cache] loaded: {cache}")
            else:
                print(f"Running sliding-window inference (stride={stride}, sigma={smooth_sigma})...")
                road_prob_np = sliding_window_inference(
                    img_array, model, device,
                    patch_size=int(cfg.PATCH_SIZE),
                    stride=stride,
                    smooth_sigma=smooth_sigma,
                    verbose=True,
                )
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.save(cache, road_prob_np)
                print(f"[cache] saved: {cache}")

        road_prob = road_prob_np

        if route:
            # --- determine src / tgt ---
            if npz_case is not None:
                tif_name = os.path.basename(tif_path)
                location_key = tif_name.replace("crop_", "").replace("_z16.tif", "").replace(".tif", "")
                npz_name = f"distance_dataset_all_{location_key}_p{p_count}.npz"
                npz_path = os.path.join(os.path.dirname(tif_path), npz_name)
                coords, udist, H_npz, W_npz = _load_npz_nodes(npz_path)
                ci = int(npz_case) if npz_case < coords.shape[0] else 0
                xy_norm = coords[ci]
                yx = _normxy_to_yx(xy_norm, h, w)
                src_yx0 = (int(yx[0, 0]), int(yx[0, 1]))
                tgt_yx0 = (int(yx[1, 0]), int(yx[1, 1]))
                print(f"[npz] case={ci}  nodes={yx.shape[0]}  src={src_yx0}  tgt={tgt_yx0}")
            elif src is not None and tgt is not None:
                src_yx0 = _parse_yx(src) if isinstance(src, str) else tuple(src)
                tgt_yx0 = _parse_yx(tgt) if isinstance(tgt, str) else tuple(tgt)
            elif pick_points:
                src_yx0, tgt_yx0 = _pick_two_points(
                    img_array, road_prob,
                    title="SLIDING mode -- click START then END (full-image coordinates)"
                )
            else:
                y_coords, x_coords = np.where(road_prob > 0.5)
                if len(y_coords) > 10:
                    quarter = len(y_coords) // 4
                    src_yx0 = (int(y_coords[quarter]), int(x_coords[quarter]))
                    tgt_yx0 = (int(y_coords[-quarter - 1]), int(x_coords[-quarter - 1]))
                else:
                    src_yx0 = (h // 4, w // 4)
                    tgt_yx0 = (3 * h // 4, 3 * w // 4)

            src_yx0 = _snap_to_road(road_prob, src_yx0, radius=int(snap_radius), th=float(snap_th))
            tgt_yx0 = _snap_to_road(road_prob, tgt_yx0, radius=int(snap_radius), th=float(snap_th))
            print(f"  src={src_yx0}  tgt={tgt_yx0}")

            # --- multigrid Eikonal solve (aligned with eval) ---
            ds = downsample
            mg_f = 4
            eps = 1e-6
            road_prob_t = torch.from_numpy(road_prob).to(device, torch.float32).unsqueeze(0)
            road_prob_t = road_prob_t.clamp(eps, 1.0 - eps)

            with torch.no_grad():
                prob_f, _, _ = _maxpool_prob(road_prob_t, ds)
                cost_f = model._road_prob_to_cost(prob_f).to(torch.float32)
                prob_c, _, _ = _maxpool_prob(road_prob_t, ds * mg_f)
                cost_c = model._road_prob_to_cost(prob_c).to(torch.float32)

            mg_itc = max(20, int(eik_iters * 0.25))
            mg_itf = max(20, int(eik_iters * 0.80))

            all_src_yx = torch.tensor([[src_yx0[0], src_yx0[1]]], dtype=torch.long, device=device)

            print("Running multigrid Eikonal solver...")
            with torch.no_grad():
                T_maps = _solve_all_sources_once(
                    model, cost_f, cost_c, all_src_yx,
                    model.route_eik_cfg, ds, mg_f, mg_itc, mg_itf, device,
                    iter_floor_c=iter_floor_c,
                    iter_floor_f=iter_floor_f,
                )

            T_fine = T_maps[0].detach().cpu().numpy()
            Hf, Wf = T_fine.shape

            # query distance at target
            tgt_fc_y = min(tgt_yx0[0] // ds, Hf - 1)
            tgt_fc_x = min(tgt_yx0[1] // ds, Wf - 1)
            dist_val = float(T_fine[tgt_fc_y, tgt_fc_x])

            # also compute Euclidean for _mix_eikonal_euclid
            euc_dist = math.sqrt((tgt_yx0[0] - src_yx0[0]) ** 2 + (tgt_yx0[1] - src_yx0[1]) ** 2)
            gate_logit = model.eik_gate_logit
            T_eik_t = torch.tensor([dist_val], device=device, dtype=torch.float32)
            d_euc_t = torch.tensor([euc_dist], device=device, dtype=torch.float32)
            mixed_dist = _mix_eikonal_euclid(
                T_eik_t.unsqueeze(0), d_euc_t.unsqueeze(0), gate_logit,
            ).squeeze(0).detach().cpu().item()

            print(f"Route distance: Eikonal={dist_val:.2f}  Euclidean={euc_dist:.2f}  "
                  f"Mixed={mixed_dist:.2f}  (ds={ds}, grid={Hf}x{Wf})")

            # --- trace path on fine-scale grid, map to original resolution ---
            src_grid = (min(src_yx0[0] // ds, Hf - 1), min(src_yx0[1] // ds, Wf - 1))
            tgt_grid = (tgt_fc_y, tgt_fc_x)
            path_grid = _trace_path_descend_T(
                T_fine, src_grid, tgt_grid, max_steps=max_path_steps,
            )
            path_yx = np.zeros_like(path_grid)
            path_yx[:, 0] = np.clip(path_grid[:, 0] * ds, 0, h - 1)
            path_yx[:, 1] = np.clip(path_grid[:, 1] * ds, 0, w - 1)

            # --- visualisation (3-panel) ---
            print("Generating figure...")
            fig, axes = plt.subplots(1, 3, figsize=(22, 8))

            axes[0].imshow(img_array)
            axes[0].plot(src_yx0[1], src_yx0[0], "g*", markersize=14, label="Start")
            axes[0].plot(tgt_yx0[1], tgt_yx0[0], "r*", markersize=14, label="End")
            axes[0].plot(path_yx[:, 1], path_yx[:, 0], "-", linewidth=2.0, label="Route")
            axes[0].set_title("Full Satellite + Planned Route")
            axes[0].legend()
            axes[0].axis("off")

            sigma_str = f" (sigma={smooth_sigma})" if smooth_sigma > 0 else ""
            im1 = axes[1].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
            axes[1].plot(path_yx[:, 1], path_yx[:, 0], "-", linewidth=2.0)
            axes[1].set_title(f"Road Probability + Route{sigma_str}")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].axis("off")

            valid = T_fine[T_fine < 1e8]
            vmax = np.percentile(valid, 95) if valid.size > 0 else 1000
            im2 = axes[2].imshow(T_fine, cmap="viridis", vmax=vmax)
            axes[2].set_title(f"Multigrid Eikonal (fine, ds={ds})\n"
                              f"Eik={dist_val:.1f}  Mixed={mixed_dist:.1f}")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            axes[2].axis("off")

        else:
            # --- no route: 2-panel visualisation ---
            print("Generating figure...")
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            axes[0].imshow(img_array)
            axes[0].set_title("Full Satellite Image")
            axes[0].axis("off")

            sigma_str = f" (sigma={smooth_sigma})" if smooth_sigma > 0 else ""
            im1 = axes[1].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
            axes[1].set_title(f"Predicted Road Probability (Sliding Window){sigma_str}")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].axis("off")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'center' or 'sliding'.")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved -> {os.path.abspath(save_path)}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="SAM-Road inference + multigrid Eikonal route visualisation (aligned with eval)")

    # --- model / checkpoint ---
    p.add_argument(
        "--ckpt", type=str, default=_CKPT_DEFAULT,
        help="SAM-Road checkpoint path (encoder + decoder).",
    )
    p.add_argument("--tsp_load_ckpt", type=str, default="",
                   help="Routing checkpoint (.pt) with alpha/gamma/gate/cost_net weights.")
    p.add_argument("--cost_net", action="store_true",
                   help="Enable neural residual cost net.")
    p.add_argument("--cost_net_ch", type=int, default=16)

    # --- image / data ---
    p.add_argument(
        "--tif", type=str,
        default="Gen_dataset_V2/Gen_dataset/19.940688_110.276704/"
                "00_20.021516_110.190699_3000.0/crop_20.021516_110.190699_3000.0_z16.tif",
        help="Target TIF path",
    )
    p.add_argument("--gt_mask", type=str, default="",
                   help="Path to GT road mask image. If given, use as road_prob instead of model.")

    # --- mode / inference ---
    p.add_argument("--output", type=str, default="tif_inference_route_result.png")
    p.add_argument("--mode", type=str, default="sliding", choices=["center", "sliding"])
    p.add_argument("--stride", type=int, default=256,
                   help="Sliding-window stride in pixels (default 256 = 50%% overlap)")
    p.add_argument("--smooth_sigma", type=float, default=0.0,
                   help="Gaussian post-processing sigma (0 = off).")

    # --- route planning ---
    p.add_argument("--route", action="store_true", default=True)
    p.add_argument("--no-route", dest="route", action="store_false")
    p.add_argument("--pick_points", action="store_true",
                   help="Interactively click two points on the image.")
    p.add_argument("--src", type=str, default=None, help="Start point 'y,x' (pixels).")
    p.add_argument("--tgt", type=str, default=None, help="End point 'y,x' (pixels).")
    p.add_argument("--snap_radius", type=int, default=25)
    p.add_argument("--snap_th", type=float, default=0.5)
    p.add_argument("--max_path_steps", type=int, default=None)

    # --- Eikonal parameters (aligned with eval defaults) ---
    p.add_argument("--downsample", type=int, default=16,
                   help="Eikonal grid downsample factor. 16 is optimal (SWEEP 3).")
    p.add_argument("--eik_iters", type=int, default=40,
                   help="Eikonal iteration hint (actual count controlled by iter_floor).")
    p.add_argument("--gate_override", type=float, default=1.0,
                   help="Eikonal/Euclidean gate (0-1). 1.0=pure Eikonal. -1=checkpoint value.")
    p.add_argument("--alpha_override", type=float, default=50,
                   help="Cost alpha. 50 is optimal. -1=checkpoint value.")
    p.add_argument("--gamma_override", type=float, default=2.0,
                   help="Cost gamma. 2.0 is optimal. -1=checkpoint value.")
    p.add_argument("--iter_floor_c", type=float, default=1.2,
                   help="Coarse iteration floor multiplier.")
    p.add_argument("--iter_floor_f", type=float, default=0.65,
                   help="Fine iteration floor multiplier.")

    # --- NPZ case selection (optional) ---
    p.add_argument("--npz_case", type=int, default=None,
                   help="Case index from NPZ to select src/tgt nodes.")
    p.add_argument("--p_count", type=int, default=20,
                   help="Node count variant for NPZ filename (20/50/100).")

    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    run_inference_on_tif(
        args.tif,
        ckpt_path=args.ckpt,
        save_path=args.output,
        mode=args.mode,
        stride=args.stride,
        smooth_sigma=args.smooth_sigma,
        route=args.route,
        pick_points=args.pick_points,
        src=args.src,
        tgt=args.tgt,
        snap_radius=args.snap_radius,
        snap_th=args.snap_th,
        max_path_steps=args.max_path_steps,
        tsp_load_ckpt=args.tsp_load_ckpt,
        cost_net=args.cost_net,
        cost_net_ch=args.cost_net_ch,
        downsample=args.downsample,
        eik_iters=args.eik_iters,
        gate_override=args.gate_override,
        alpha_override=args.alpha_override,
        gamma_override=args.gamma_override,
        iter_floor_c=args.iter_floor_c,
        iter_floor_f=args.iter_floor_f,
        gt_mask=args.gt_mask,
        npz_case=args.npz_case,
        p_count=args.p_count,
    )
