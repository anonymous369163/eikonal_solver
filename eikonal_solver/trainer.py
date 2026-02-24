"""
trainer.py — SAMRoute training entry point (Phase 1 and Phase 2).

Usage:
    cd /home/yuepeng/code/mmdm_V3/MMDataset
    python eikonal_solver/trainer.py \
        --config eikonal_solver/config_phase1.yaml \
        --data_root Gen_dataset_V2/Gen_dataset \
        --gpus 1 --epochs 200

Phase 2 (distance supervision) is activated when the config has
ROUTE_LAMBDA_DIST > 0 and --include_dist is passed.
"""

from __future__ import annotations

import math
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAM_REPO = os.path.join(_HERE, "../sam_road_repo")
for _sp in [_HERE,
            os.path.join(_SAM_REPO, "segment-anything-road"),
            os.path.join(_SAM_REPO, "sam"),
            _SAM_REPO]:
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
if sys.path[0] != _HERE:
    sys.path.remove(_HERE)
    sys.path.insert(0, _HERE)

import torch
import yaml
from addict import Dict as aDict

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

from model import SAMRoute
from dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Spearman evaluation callback (delegates heavy lifting to evaluator)
# ---------------------------------------------------------------------------

class SpearmanEvalCallback(pl.Callback):
    """Periodically evaluate Spearman on a held-out image, log to TensorBoard.

    Computes both model-predicted and Euclidean-baseline Spearman coefficients.
    When a new best mean Spearman is achieved, saves a path-visualization PNG.
    """

    def __init__(self, image_path: str, eval_every_n_epochs: int = 10,
                 k: int = 10, min_in_patch: int = 3, downsample: int = 1024):
        super().__init__()
        self.image_path = image_path
        self.eval_every = eval_every_n_epochs
        self.k = k
        self.min_in_patch = min_in_patch
        self.downsample = downsample
        self._rgb_ds = None
        self._nodes_yx = None
        self._info = None
        self._scale = 1.0
        self._rp_cache: dict = {}
        self._gt_mask_ds = None
        self._best_spearman = float("-inf")  # track best for visualization trigger

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if self.eval_every <= 0 or epoch % self.eval_every != 0:
            return
        if not os.path.isfile(self.image_path):
            return
        try:
            # Infer visualization output directory from trainer
            vis_dir = os.path.join(trainer.default_root_dir, "eval_vis")
            metrics = self._evaluate(pl_module, epoch, vis_dir)
            for key, val in metrics.items():
                pl_module.log(key, val, prog_bar=(key == "eval_spearman"),
                              on_epoch=True, sync_dist=False)
            print(
                f"\n[SpearmanEval ep={epoch}] "
                f"Model Spearman={metrics['eval_spearman']:+.4f}  "
                f"Euc Spearman={metrics['eval_spearman_euc']:+.4f}  "
                f"Kendall={metrics['eval_kendall']:+.4f}  "
                f"PW={metrics['eval_pw_acc']:.4f}  "
                f"n_anchors={metrics['eval_n_anchors']:.0f}"
                + ("  ★ NEW BEST — visualization saved" if metrics.get("_new_best") else "")
            )
        except Exception as exc:
            import traceback
            print(f"\n[SpearmanEval ep={epoch}] error: {exc}")
            traceback.print_exc()

    @torch.no_grad()
    def _ensure_loaded(self, device: torch.device) -> None:
        if self._rgb_ds is not None:
            return
        from evaluator import load_tif_image
        from dataset import load_nodes_from_npz
        import glob as _glob
        import torch.nn.functional as F
        from PIL import Image as _PILImage

        rgb = load_tif_image(self.image_path).to(device)
        _, _, H, W = rgb.shape
        scale = 1.0
        if self.downsample > 0 and max(H, W) > self.downsample:
            scale = self.downsample / max(H, W)
            rgb = F.interpolate(rgb, size=(max(1, int(H * scale)), max(1, int(W * scale))),
                                mode="bilinear", align_corners=False)
        self._rgb_ds = rgb
        self._scale = scale

        # Load GT road mask from the same region directory as the TIF.
        # Prefer the original thin roadnet PNG (most truthful GT).
        region_dir = os.path.dirname(self.image_path)
        gt_pngs = sorted(_glob.glob(os.path.join(region_dir, "roadnet_*.png")))
        gt_pngs = [p for p in gt_pngs if "normalized" not in os.path.basename(p)]
        if gt_pngs:
            import numpy as _np
            _arr = _np.array(_PILImage.open(gt_pngs[0]))
            _mask = _arr[:, :, 0].astype(_np.float32) / 255.0  # [H, W] float32
            gt_t = torch.from_numpy(_mask).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
            if scale != 1.0:
                _, _, H_ds, W_ds = rgb.shape
                gt_t = F.interpolate(gt_t, size=(H_ds, W_ds), mode="bilinear",
                                     align_corners=False)
            self._gt_mask_ds = gt_t.squeeze()  # [H_ds, W_ds]
        else:
            self._gt_mask_ds = None

        rp_dummy = torch.zeros(H, W, device=device)
        nodes_orig, info = load_nodes_from_npz(
            self.image_path, rp_dummy, p_count=20, snap=False, verbose=False)
        _, _, H_ds, W_ds = rgb.shape
        if scale != 1.0:
            nodes_yx = (nodes_orig.float() * scale).long()
            nodes_yx[:, 0].clamp_(0, H_ds - 1)
            nodes_yx[:, 1].clamp_(0, W_ds - 1)
        else:
            nodes_yx = nodes_orig
        self._nodes_yx = nodes_yx.to(device)
        self._info = info

    @torch.no_grad()
    def _evaluate(self, model, epoch: int, vis_dir: str) -> dict:
        import numpy as np
        import torch.nn.functional as F
        from evaluator import pick_anchor_and_knn, spearmanr, kendall_tau, pairwise_order_acc

        device = next(model.parameters()).device
        self._ensure_loaded(device)
        self._rp_cache.clear()

        PATCH = int(model.config.get("PATCH_SIZE", 512))
        margin = int(model.config.get("ROUTE_ROI_MARGIN", 64))
        n_iters = int(model.config.get("ROUTE_EIK_ITERS", 20))
        min_ds = int(model.config.get("ROUTE_EIK_DOWNSAMPLE", 8))

        nodes_yx = self._nodes_yx
        info = self._info
        case = info["case_idx"]
        meta_px = int(info["meta_sat_height_px"])
        pscale = 1.0 / self._scale if self._scale != 1.0 else 1.0
        _, _, H_ds, W_ds = self._rgb_ds.shape
        N = nodes_yx.shape[0]

        sp_acc, kd_acc, pw_acc, sp_euc_acc = [], [], [], []
        # Track the anchor with the highest single-anchor model Spearman for visualization
        best_anchor_sp = float("-inf")
        best_vis_data = None

        for anc_idx in range(N):
            _, nn_global = pick_anchor_and_knn(nodes_yx, k=self.k, anchor_idx=anc_idx)
            anc = nodes_yx[anc_idx]
            ay, ax = int(anc[0]), int(anc[1])
            py0 = max(0, min(ay - PATCH // 2, H_ds - PATCH))
            px0 = max(0, min(ax - PATCH // 2, W_ds - PATCH))
            py1, px1 = py0 + PATCH, px0 + PATCH
            tgts_g = nodes_yx[nn_global]
            in_p = ((tgts_g[:, 0] >= py0) & (tgts_g[:, 0] < py1) &
                    (tgts_g[:, 1] >= px0) & (tgts_g[:, 1] < px1))
            keep = torch.where(in_p)[0]
            if keep.numel() < self.min_in_patch:
                continue
            nn_kept = nn_global[keep]
            tgt_patch = tgts_g[keep] - torch.tensor([py0, px0], device=device)
            anc_patch = torch.tensor([ay - py0, ax - px0], device=device).long()

            # Cache road-prob per patch (patch boundaries as key)
            key = (py0, py1, px0, px1)
            if key not in self._rp_cache:
                rgb_p = self._rgb_ds[:, :, py0:py1, px0:px1]
                if rgb_p.shape[2] < PATCH or rgb_p.shape[3] < PATCH:
                    rgb_p = F.pad(rgb_p, (0, PATCH - rgb_p.shape[3], 0, PATCH - rgb_p.shape[2]))
                rgb_hwc = (rgb_p * 255.0).squeeze(0).permute(1, 2, 0).unsqueeze(0)
                _, ms = model._predict_mask_logits_scores(rgb_hwc)
                self._rp_cache[key] = ms[0, :, :, 1].detach()
            rp = self._rp_cache[key]

            # One Eikonal solve covering all K targets → read K T-values
            T_raw = _eikonal_one_src_k_tgts(
                rp, anc_patch, tgt_patch, model, device, margin, n_iters, min_ds)
            T_norm = T_raw * pscale / max(meta_px, 1)
            nn_np = nn_kept.cpu().numpy().astype(int)
            gt_arr = np.asarray(info["undirected_dist_norm"][case][anc_idx, nn_np], dtype=np.float64)
            euc_arr = np.asarray(info["euclidean_dist_norm"][case][anc_idx, nn_np], dtype=np.float64)

            valid = np.isfinite(T_norm) & (T_norm < 1e3) & np.isfinite(gt_arr)
            if valid.sum() < 3:
                continue

            pv, gv = T_norm[valid], gt_arr[valid]
            anc_sp = spearmanr(pv, gv)
            sp_acc.append(anc_sp)
            kd_acc.append(kendall_tau(pv, gv))
            pw_acc.append(pairwise_order_acc(pv, gv))

            # Euclidean baseline Spearman
            valid_euc = valid & np.isfinite(euc_arr)
            if valid_euc.sum() >= 3:
                sp_euc_acc.append(spearmanr(euc_arr[valid_euc], gt_arr[valid_euc]))

            # Track best anchor for visualization
            if anc_sp > best_anchor_sp:
                best_anchor_sp = anc_sp
                best_vis_data = {
                    "anc_idx": anc_idx,
                    "anc_patch": anc_patch.clone(),
                    "tgt_patch": tgt_patch.clone(),
                    "nn_np": nn_np.copy(),
                    "rp_key": key,
                    "py0": py0, "px0": px0,
                }

        if not sp_acc:
            return dict(eval_spearman=float("nan"), eval_spearman_euc=float("nan"),
                        eval_kendall=float("nan"), eval_pw_acc=float("nan"),
                        eval_n_anchors=0.0)

        mean_sp = float(np.mean(sp_acc))
        result = dict(
            eval_spearman=mean_sp,
            eval_spearman_euc=float(np.mean(sp_euc_acc)) if sp_euc_acc else float("nan"),
            eval_kendall=float(np.mean(kd_acc)),
            eval_pw_acc=float(np.nanmean(pw_acc)),
            eval_n_anchors=float(len(sp_acc)),
        )

        # Visualize paths when a new best mean Spearman is achieved
        new_best = mean_sp > self._best_spearman
        if new_best and best_vis_data is not None:
            self._best_spearman = mean_sp
            result["_new_best"] = True
            try:
                self._visualize_best_anchor(model, best_vis_data, epoch,
                                            mean_sp, vis_dir, device, margin,
                                            n_iters, min_ds, self._gt_mask_ds)
            except Exception as ve:
                print(f"\n[SpearmanEval] visualization error: {ve}")
        return result

    @torch.no_grad()
    def _visualize_best_anchor(self, model, vis_data: dict, epoch: int,
                               mean_sp: float, vis_dir: str,
                               device: torch.device, margin: int,
                               n_iters: int = 20, min_ds: int = 8,
                               gt_mask=None) -> None:
        """Run same-params Eikonal as evaluation on the best anchor, save visualization."""
        import numpy as np
        import torch.nn.functional as F
        from eikonal import EikonalConfig, eikonal_soft_sweeping
        from evaluator import trace_path_from_T, calc_path_len

        rp = self._rp_cache[vis_data["rp_key"]]
        anc = vis_data["anc_patch"]
        tgts = vis_data["tgt_patch"]
        py0, px0 = vis_data["py0"], vis_data["px0"]
        nn_np = vis_data["nn_np"]
        K = tgts.shape[0]
        H_r, W_r = rp.shape

        # Single large ROI covering all targets for a clean T-field
        if K > 0:
            span_max = int(torch.max(torch.abs(tgts.float() - anc.float())).item())
        else:
            span_max = 0
        half = span_max + margin
        P = max(2 * half + 1, 256)
        y0 = int(anc[0]) - half; x0 = int(anc[1]) - half
        y1 = y0 + P; x1 = x0 + P
        yy0, xx0 = max(y0, 0), max(x0, 0)
        yy1, xx1 = min(y1, H_r), min(x1, W_r)

        patch_full = torch.zeros(P, P, device=device, dtype=rp.dtype)
        patch_full[yy0-y0:yy1-y0, xx0-x0:xx1-x0] = rp[yy0:yy1, xx0:xx1]

        sr_y = max(0, min(int(anc[0]) - y0, P - 1))
        sr_x = max(0, min(int(anc[1]) - x0, P - 1))

        # Use the SAME downsample params as the Spearman evaluation so that
        # T values and paths in the visualization match actual model behavior.
        ds = max(min_ds, int(math.ceil(P / max(n_iters, 1))))
        if ds > 1:
            P_pad = int(math.ceil(P / ds) * ds)
            patch_ds = patch_full
            if P_pad > P:
                patch_ds = F.pad(patch_full, (0, P_pad - P, 0, P_pad - P), value=0.0)
            roi_c = F.max_pool2d(patch_ds[None, None], kernel_size=ds, stride=ds).squeeze()
            cost_c = model._road_prob_to_cost(roi_c)
            Pc = cost_c.shape[0]
            sc_y = max(0, min(sr_y // ds, Pc - 1))
            sc_x = max(0, min(sr_x // ds, Pc - 1))
            smask = torch.zeros(1, Pc, Pc, dtype=torch.bool, device=device)
            smask[0, sc_y, sc_x] = True
            cfg = EikonalConfig(n_iters=n_iters, h=float(ds), tau_min=0.03,
                                tau_branch=0.05, tau_update=0.03,
                                use_redblack=True, monotone=True)
            T_coarse = eikonal_soft_sweeping(cost_c.unsqueeze(0), smask, cfg)
            if T_coarse.dim() == 3:
                T_coarse = T_coarse[0]
            # Upsample back to P×P so path-tracing coordinates align with the patch
            T_patch = F.interpolate(
                T_coarse[None, None], size=(P, P),
                mode="bilinear", align_corners=False).squeeze()
            # Preserve the h-scaling: T_coarse is in units of ds pixels, keep as-is
        else:
            cost_c = model._road_prob_to_cost(patch_full)
            smask = torch.zeros(1, P, P, dtype=torch.bool, device=device)
            smask[0, sr_y, sr_x] = True
            cfg = EikonalConfig(n_iters=n_iters, h=1.0, tau_min=0.03,
                                tau_branch=0.05, tau_update=0.03,
                                use_redblack=True, monotone=True)
            T_patch = eikonal_soft_sweeping(cost_c.unsqueeze(0), smask, cfg)
            if T_patch.dim() == 3:
                T_patch = T_patch[0]

        # Backtrace paths
        # y0/x0 are offsets in PATCH space; adding py0/px0 converts to full-image space.
        src_rel = torch.tensor([sr_y, sr_x], device=device, dtype=torch.long)
        paths_global, t_vals = [], []
        for ki in range(K):
            tr_y = max(0, min(int(tgts[ki, 0]) - y0, P - 1))
            tr_x = max(0, min(int(tgts[ki, 1]) - x0, P - 1))
            tval = float(T_patch[tr_y, tr_x].item())
            t_vals.append(tval)
            tgt_rel = torch.tensor([tr_y, tr_x], device=device, dtype=torch.long)
            pp = trace_path_from_T(T_patch, src_rel, tgt_rel, device, diag=True)
            if len(pp) >= 2:
                paths_global.append([(p[0] + y0 + py0, p[1] + x0 + px0) for p in pp])
            else:
                paths_global.append(None)

        # Build global coordinate tensors for the plotter
        anc_global = torch.tensor([int(anc[0]) + py0, int(anc[1]) + px0],
                                  device=device, dtype=torch.long)
        tgts_global = tgts + torch.tensor([py0, px0], device=device)

        # Build a full-size road_prob (H_ds × W_ds) with the patch placed at its
        # correct position so that scale_img = H_ds/H_ds = 1.0 inside the plotter
        # and all coordinate arithmetic stays in the same space.
        H_ds, W_ds = self._rgb_ds.shape[-2:]
        road_prob_full = torch.zeros(H_ds, W_ds, device=rp.device, dtype=rp.dtype)
        py1_clip = min(py0 + H_r, H_ds)
        px1_clip = min(px0 + W_r, W_ds)
        road_prob_full[py0:py1_clip, px0:px1_clip] = rp[:py1_clip - py0, :px1_clip - px0]

        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, f"ep{epoch:03d}_sp{mean_sp:+.4f}.png")

        _plot_paths_standalone(
            rgb=self._rgb_ds,
            road_prob=road_prob_full,
            gt_mask=gt_mask,
            src=anc_global,
            tgts=tgts_global,
            T_patch=T_patch,
            t_y0=y0 + py0, t_x0=x0 + px0, P=P,
            patch_y0=py0, patch_x0=px0, patch_h=H_r, patch_w=W_r,
            paths_global=paths_global,
            t_vals=t_vals,
            nn_np=nn_np,
            save_path=save_path,
        )
        print(f"  [vis] {save_path}")


def _plot_paths_standalone(rgb, road_prob, src, tgts, T_patch,
                           t_y0, t_x0, P,
                           patch_y0, patch_x0, patch_h, patch_w,
                           paths_global, t_vals, nn_np, save_path,
                           gt_mask=None):
    """Two-panel path visualization: GT road (left) vs Predicted road + T-field + paths (right).

    Coordinate convention (all in full-downsampled-image pixel space):
      t_y0/t_x0/P  – top-left + size of the Eikonal T-field ROI.
      patch_y0/patch_x0/patch_h/patch_w – extent of road_prob PATCH (white box).
      paths_global – list of [(y,x), ...] in full-image pixel space.
      src, tgts    – (y,x) tensors in full-image pixel space.
      gt_mask      – optional [H_img, W_img] tensor with GT road probability in [0,1].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np

    if rgb.dim() == 4:
        rgb = rgb[0]
    img_np = (rgb.detach().float().cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    H_img, W_img = img_np.shape[:2]
    road_np = road_prob.detach().float().cpu().numpy()
    T_np = T_patch.detach().float().cpu().numpy()
    K = tgts.shape[0]

    # ── T-field: clip to PATCH, log-normalize ──────────────────────────────
    t_y1, t_x1 = t_y0 + P, t_x0 + P
    iy0 = max(0, t_y0, patch_y0)
    ix0 = max(0, t_x0, patch_x0)
    iy1 = min(H_img, t_y1, patch_y0 + patch_h)
    ix1 = min(W_img, t_x1, patch_x0 + patch_w)
    T_full = np.full((H_img, W_img), np.nan, dtype=np.float32)
    if iy1 > iy0 and ix1 > ix0:
        sy0, sx0 = iy0 - t_y0, ix0 - t_x0
        T_full[iy0:iy1, ix0:ix1] = T_np[sy0:sy0 + (iy1 - iy0), sx0:sx0 + (ix1 - ix0)]

    finite = np.isfinite(T_full)
    if finite.any():
        raw_min = float(np.min(T_full[finite]))
        raw_max = float(np.percentile(T_full[finite], 99))
        T_shifted = np.where(finite, T_full - raw_min, np.nan)
        T_display = np.log1p(T_shifted)
        vmin_log, vmax_log = 0.0, float(np.log1p(max(raw_max - raw_min, 1e-3)))
    else:
        T_display = T_full
        vmin_log, vmax_log = 0.0, 1.0

    # ── Patch boundary box ─────────────────────────────────────────────────
    bx0 = max(0, patch_x0);               by0 = max(0, patch_y0)
    bx1 = min(W_img, patch_x0 + patch_w); by1 = min(H_img, patch_y0 + patch_h)

    colors = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))
    sy = int(src[0].item());  sx_img = int(src[1].item())

    has_gt = gt_mask is not None
    n_cols = 2 if has_gt else 1
    fig_w = 14 * n_cols
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 14))
    if n_cols == 1:
        axes = [axes]

    def _draw_nodes_and_paths(ax, draw_paths=True):
        ax.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
                color="white", lw=1.5, ls="--", alpha=0.75, zorder=8)
        if draw_paths:
            for ki, path in enumerate(paths_global):
                if path is not None:
                    ys = [p[0] for p in path]
                    xs = [p[1] for p in path]
                    ax.plot(xs, ys, color=colors[ki], lw=2.0, alpha=0.85, zorder=10,
                            path_effects=[pe.Stroke(linewidth=3.5, foreground="black",
                                                    alpha=0.5), pe.Normal()])
        for ki in range(K):
            ty = int(tgts[ki, 0].item());  tx = int(tgts[ki, 1].item())
            tval = t_vals[ki]
            lbl = (f"node {nn_np[ki]}  T={tval:.0f}"
                   if np.isfinite(tval) else f"node {nn_np[ki]}")
            ax.scatter([tx], [ty], s=220, marker="x", color=colors[ki],
                       linewidths=2.5, zorder=14, label=lbl if draw_paths else "_")
        ax.scatter([sx_img], [sy], s=300, marker="o", color="lime",
                   edgecolors="black", linewidths=2.5, zorder=15,
                   label="Source" if draw_paths else "_")
        ax.axis("off")

    # ── Left panel: GT road ────────────────────────────────────────────────
    if has_gt:
        ax_gt = axes[0]
        gt_np = gt_mask.detach().float().cpu().numpy()
        ax_gt.imshow(img_np)
        masked_gt = np.ma.masked_where(gt_np < 0.1, gt_np)
        ax_gt.imshow(masked_gt, cmap="Greens", vmin=0, vmax=1, alpha=0.65)
        _draw_nodes_and_paths(ax_gt, draw_paths=False)
        ax_gt.set_title("GT Road Mask", fontsize=12)
        ax_pred = axes[1]
    else:
        ax_pred = axes[0]

    # ── Right panel (or only panel): Predicted road + T-field + paths ──────
    ax_pred.imshow(img_np)
    masked_road = np.ma.masked_where(road_np < 0.1, road_np)
    ax_pred.imshow(masked_road, cmap="Reds", vmin=0, vmax=1, alpha=0.35)
    T_masked = np.ma.masked_invalid(T_display)
    ax_pred.imshow(T_masked, cmap="plasma", vmin=vmin_log, vmax=vmax_log, alpha=0.55)
    _draw_nodes_and_paths(ax_pred, draw_paths=True)
    ax_pred.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8.5, framealpha=0.9)
    ax_pred.set_title("Predicted Road + T-field (log-scale) + Paths", fontsize=12)

    fig.suptitle("SAMRoute Evaluation  (GT road=green  |  Predicted road=red  |  T-field=plasma log-scale)",
                 fontsize=11, y=1.01)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Eikonal evaluation helper: one-source → K-targets, single solve
# ---------------------------------------------------------------------------

def _eikonal_one_src_k_tgts(
    rp: "torch.Tensor",        # [H_r, W_r] road-prob patch (no grad)
    anc_patch: "torch.Tensor", # [2] anchor coords in patch space (long)
    tgt_patch: "torch.Tensor", # [K, 2] target coords in patch space (long)
    model,
    device: "torch.device",
    margin: int,
    n_iters: int,
    min_ds: int,
) -> "np.ndarray":             # [K] float64
    """One Eikonal solve per anchor covering all K targets; read K T-values.

    Replaces the previous per-target inner loop (K separate small-ROI solves).
    The coarse grid size is the same as before (~n_iters × n_iters) because
    ds = max(min_ds, ceil(P / n_iters)) bounds P_c ≈ n_iters regardless of P.
    Total solves per evaluation: N (anchors) instead of N×K.
    """
    import math
    import numpy as np
    import torch.nn.functional as F
    from eikonal import EikonalConfig, eikonal_soft_sweeping

    K = tgt_patch.shape[0]
    if K == 0:
        return np.empty(0, dtype=np.float64)

    H_r, W_r = rp.shape

    span_max = int(torch.max(torch.abs(tgt_patch.float() - anc_patch.float())).item())
    half = span_max + margin
    P = max(2 * half + 1, 64)

    y0 = int(anc_patch[0]) - half;  x0 = int(anc_patch[1]) - half
    y1 = y0 + P;                    x1 = x0 + P
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
        roi_c = F.max_pool2d(roi[None, None], kernel_size=ds, stride=ds).squeeze()
        cost = model._road_prob_to_cost(roi_c)
        Pc = cost.shape[0]
        sc_y = max(0, min(sr_y // ds, Pc - 1))
        sc_x = max(0, min(sr_x // ds, Pc - 1))
        smask = torch.zeros(1, Pc, Pc, dtype=torch.bool, device=device)
        smask[0, sc_y, sc_x] = True
        cfg = EikonalConfig(n_iters=n_iters, h=float(ds), tau_min=0.03,
                            tau_branch=0.05, tau_update=0.03,
                            use_redblack=True, monotone=True)
        T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)[0]
        T_vals = []
        for ki in range(K):
            tr_y = max(0, min((int(tgt_patch[ki, 0]) - y0) // ds, Pc - 1))
            tr_x = max(0, min((int(tgt_patch[ki, 1]) - x0) // ds, Pc - 1))
            T_vals.append(float(T[tr_y, tr_x].item()))
    else:
        cost = model._road_prob_to_cost(roi)
        smask = torch.zeros(1, P, P, dtype=torch.bool, device=device)
        smask[0, sr_y, sr_x] = True
        cfg = EikonalConfig(n_iters=n_iters, h=1.0, tau_min=0.03,
                            tau_branch=0.05, tau_update=0.03,
                            use_redblack=True, monotone=True)
        T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)[0]
        T_vals = []
        for ki in range(K):
            tr_y = max(0, min(int(tgt_patch[ki, 0]) - y0, P - 1))
            tr_x = max(0, min(int(tgt_patch[ki, 1]) - x0, P - 1))
            T_vals.append(float(T[tr_y, tr_x].item()))

    return np.array(T_vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> aDict:
    with open(path) as f:
        return aDict(yaml.safe_load(f))


def build_model(config: aDict, pretrained_ckpt: str | None) -> SAMRoute:
    model = SAMRoute(config)
    if pretrained_ckpt and os.path.isfile(pretrained_ckpt):
        print(f"[train] Loading pretrained weights from: {pretrained_ckpt}")
        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[train]  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    else:
        if pretrained_ckpt:
            print(f"[train] WARNING: PRETRAINED_CKPT not found: {pretrained_ckpt}")
        print("[train] Training from SAM backbone weights only.")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAMRoute trainer")
    p.add_argument("--config", default=os.path.join(_HERE, "config_phase1.yaml"))
    p.add_argument("--data_root", default=os.path.join(_HERE, "../Gen_dataset_V2/Gen_dataset"))
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--output_dir", default=os.path.join(_HERE, "../training_outputs/sam_route"))
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--samples_per_region", type=int, default=50)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--include_dist", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_cached_features", action="store_true")
    p.add_argument("--no_preload", action="store_true")
    p.add_argument("--preload_workers", type=int, default=8)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--road_dilation_radius", type=int, default=0)
    p.add_argument("--eval_every_n_epochs", type=int, default=0)
    p.add_argument("--eval_image_path", type=str, default="")
    p.add_argument("--eval_k", type=int, default=10)
    p.add_argument("--eval_min_in_patch", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.TRAIN_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.workers is not None:
        config.DATA_WORKER_NUM = args.workers

    config_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ("SAM_CKPT_PATH", "PRETRAINED_CKPT"):
        val = str(config.get(key, "") or "")
        if val and not os.path.isabs(val):
            config[key] = os.path.normpath(os.path.join(config_dir, val))
    if args.ckpt_path:
        config.PRETRAINED_CKPT = os.path.abspath(args.ckpt_path)
    if args.no_pretrained:
        config.PRETRAINED_CKPT = ""

    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Data root:    {args.data_root}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Epochs:       {config.TRAIN_EPOCHS}")
    print(f"  Batch size:   {config.BATCH_SIZE}")
    print(f"  include_dist: {args.include_dist}")
    print(f"  road_dil_r:   {args.road_dilation_radius}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        root_dir=args.data_root,
        patch_size=int(config.PATCH_SIZE),
        batch_size=int(config.BATCH_SIZE),
        num_workers=int(config.DATA_WORKER_NUM),
        include_dist=args.include_dist,
        val_fraction=args.val_fraction,
        samples_per_region=args.samples_per_region,
        use_cached_features=not args.no_cached_features,
        preload_to_ram=not args.no_preload,
        preload_workers=args.preload_workers,
        k_targets=int(config.get("ROUTE_K_TARGETS", 4)),
        road_dilation_radius=args.road_dilation_radius,
        min_in_patch=int(config.get("ROUTE_MIN_IN_PATCH", 2)),
    )
    print(f"[train] Train batches/epoch: {len(train_loader)}  Val: {len(val_loader)}")

    pretrained = str(config.get("PRETRAINED_CKPT", "") or "")
    model = build_model(config, pretrained if pretrained else None)

    if args.compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile() applied.")
        except Exception as e:
            print(f"[train] torch.compile() skipped: {e}")

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="samroute-{epoch:02d}-{val_seg_loss:.4f}",
        monitor="val_seg_loss", mode="min", save_top_k=3, save_last=True,
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="step")]

    if args.eval_every_n_epochs > 0 and args.eval_image_path:
        callbacks.append(SpearmanEvalCallback(
            image_path=args.eval_image_path,
            eval_every_n_epochs=args.eval_every_n_epochs,
            k=args.eval_k, min_in_patch=args.eval_min_in_patch,
        ))
        print(f"[train] SpearmanEvalCallback: every {args.eval_every_n_epochs} epochs")

    trainer = pl.Trainer(
        max_epochs=int(config.TRAIN_EPOCHS),
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=args.output_dir, name="tensorboard"),
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume)
    print(f"\n[train] Done. Best checkpoint: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
