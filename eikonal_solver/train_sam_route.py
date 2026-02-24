"""
SAMRoute training script — Phase 1 (road segmentation only).

Usage:
    cd /home/yuepeng/code/mmdm_V3/MMDataset
    python eikonal_solver/train_sam_route.py \
        --config eikonal_solver/config_sam_route_demo.yaml \
        --data_root Gen_dataset_V2/Gen_dataset \
        --gpus 1 --epochs 5

Phase 2 (distance supervision) is activated automatically when the config has
ROUTE_LAMBDA_DIST > 0 and include_dist=True is passed (via --include_dist flag).
"""

from __future__ import annotations

import math
import os
import sys
import argparse

# ---------------------------------------------------------------------------
# sys.path: ensure eikonal_solver imports take priority over sam_road_repo
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SAM_REPO  = os.path.join(_HERE, "../sam_road_repo")
_SAM_SAM   = os.path.join(_SAM_REPO, "sam")
_SAM_SEG   = os.path.join(_SAM_REPO, "segment-anything-road")

for _sp in [_HERE, _SAM_SEG, _SAM_SAM, _SAM_REPO]:
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
# eikonal_solver must come first to avoid shadowing by sam_road_repo/utils.py
if sys.path[0] != _HERE:
    sys.path.remove(_HERE)
    sys.path.insert(0, _HERE)

import torch
import yaml
from addict import Dict as aDict

# Use `lightning` (the newer unified package) which is what model.py's LightningModule
# comes from.  pytorch_lightning v2+ is just a thin re-export of lightning.pytorch,
# but the isinstance check inside pl.Trainer uses the originating class object, so both
# the model and the Trainer must come from the same import path.
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

from model import SAMRoute
from train_dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Spearman evaluation callback
# ---------------------------------------------------------------------------

class SpearmanEvalCallback(pl.Callback):
    """Periodically evaluate Spearman rank correlation on a held-out image.

    Every `eval_every_n_epochs` epochs the callback:
      1. Runs the current model on 512×512 patches around each NPZ anchor.
      2. Solves non-differentiable Eikonal (20 iters, ds≥8) for K neighbours.
      3. Computes Spearman / Kendall / pairwise-accuracy vs GT road distances.
      4. Logs `eval_spearman`, `eval_kendall`, `eval_pw_acc` to TensorBoard.
    """

    def __init__(
        self,
        image_path: str,
        eval_every_n_epochs: int = 10,
        k: int = 19,
        min_in_patch: int = 3,
        downsample: int = 1024,
    ):
        super().__init__()
        self.image_path = image_path
        self.eval_every = eval_every_n_epochs
        self.k = k
        self.min_in_patch = min_in_patch
        self.downsample = downsample
        # Cached across epochs (loaded once, road_prob cleared each eval)
        self._rgb_ds = None
        self._nodes_yx = None
        self._info = None
        self._scale = 1.0
        self._rp_cache: dict = {}

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if self.eval_every <= 0 or epoch % self.eval_every != 0:
            return
        if not os.path.isfile(self.image_path):
            print(f"\n[SpearmanEval] image not found: {self.image_path}")
            return
        try:
            metrics = self._evaluate(pl_module)
            for key, val in metrics.items():
                pl_module.log(key, val, prog_bar=(key == "eval_spearman"),
                              on_epoch=True, sync_dist=False)
            print(
                f"\n[SpearmanEval ep={epoch}] "
                f"Spearman={metrics['eval_spearman']:+.4f}  "
                f"Kendall={metrics['eval_kendall']:+.4f}  "
                f"PW={metrics['eval_pw_acc']:.4f}  "
                f"n_anchors={metrics['eval_n_anchors']:.0f}"
            )
        except Exception as exc:
            import traceback
            print(f"\n[SpearmanEval ep={epoch}] error: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ensure_loaded(self, device: torch.device) -> None:
        """Load image and nodes once; reuse across eval epochs."""
        if self._rgb_ds is not None:
            return
        import torch.nn.functional as F
        from feature_extractor import load_tif_image
        from load_nodes_from_npz import load_nodes_from_npz

        rgb = load_tif_image(self.image_path).to(device)
        _, _, H, W = rgb.shape
        scale = 1.0
        if self.downsample > 0 and max(H, W) > self.downsample:
            scale = self.downsample / max(H, W)
            rgb = F.interpolate(rgb, size=(max(1, int(H * scale)),
                                           max(1, int(W * scale))),
                                mode="bilinear", align_corners=False)
        self._rgb_ds = rgb
        self._scale = scale
        _, _, H_ds, W_ds = rgb.shape

        # Use a zero road_prob for snapping (fast; snap=False to avoid SAM call)
        rp_dummy = torch.zeros(H, W, device=device)
        nodes_orig, info = load_nodes_from_npz(
            self.image_path, rp_dummy, p_count=20, snap=False, verbose=False)
        if scale != 1.0:
            nodes_yx = (nodes_orig.float() * scale).long()
            nodes_yx[:, 0].clamp_(0, H_ds - 1)
            nodes_yx[:, 1].clamp_(0, W_ds - 1)
        else:
            nodes_yx = nodes_orig
        self._nodes_yx = nodes_yx.to(device)
        self._info = info

    @torch.no_grad()
    def _patch_rp(self, model, py0: int, py1: int, px0: int, px1: int,
                  PATCH: int, device: torch.device) -> torch.Tensor:
        """Road-prob for one patch, cached per unique bounds."""
        import torch.nn.functional as F
        key = (py0, py1, px0, px1)
        if key not in self._rp_cache:
            rgb_p = self._rgb_ds[:, :, py0:py1, px0:px1]
            if rgb_p.shape[2] < PATCH or rgb_p.shape[3] < PATCH:
                rgb_p = F.pad(rgb_p, (0, PATCH - rgb_p.shape[3],
                                       0, PATCH - rgb_p.shape[2]), value=0.0)
            rgb_hwc = (rgb_p * 255.0).squeeze(0).permute(1, 2, 0).unsqueeze(0)
            _, ms = model._predict_mask_logits_scores(rgb_hwc)
            self._rp_cache[key] = ms[0, :, :, 1].detach()
        return self._rp_cache[key]

    @torch.no_grad()
    def _eikonal_k(self, model, rp: torch.Tensor, src: torch.Tensor,
                   tgts: torch.Tensor, n_iters: int, min_ds: int,
                   margin: int, device: torch.device):
        """Non-diff Eikonal from src to each of K targets. Returns np.ndarray [K]."""
        import numpy as np
        import torch.nn.functional as F
        from fast_sweeping import EikonalConfig, eikonal_soft_sweeping
        H, W = rp.shape
        T_vals = []
        for ki in range(tgts.shape[0]):
            tgt = tgts[ki].long()
            span = int(torch.max(torch.abs(tgt.float() - src.float())).item())
            half = span + margin
            P = max(2 * half + 1, 64)
            y0 = int(src[0]) - half;  x0 = int(src[1]) - half
            y1 = y0 + P;              x1 = x0 + P
            yy0 = max(y0, 0); xx0 = max(x0, 0)
            yy1 = min(y1, H); xx1 = min(x1, W)
            roi = F.pad(rp[yy0:yy1, xx0:xx1],
                        (xx0-x0, x1-xx1, yy0-y0, y1-yy1), value=0.0)
            sr_y = max(0, min(int(src[0])-y0, P-1))
            sr_x = max(0, min(int(src[1])-x0, P-1))
            tr_y = max(0, min(int(tgt[0])-y0, P-1))
            tr_x = max(0, min(int(tgt[1])-x0, P-1))
            ds = max(min_ds, math.ceil(P / max(n_iters, 1)))
            if ds > 1:
                P_pad = math.ceil(P / ds) * ds
                if P_pad > P:
                    roi = F.pad(roi, (0, P_pad-P, 0, P_pad-P), value=0.0)
                roi_c = torch.nn.functional.avg_pool2d(
                    roi.unsqueeze(0).unsqueeze(0), kernel_size=ds, stride=ds).squeeze()
                cost = model._road_prob_to_cost(roi_c)
                Pc = cost.shape[0]
                sc_y = max(0, min(sr_y//ds, Pc-1)); sc_x = max(0, min(sr_x//ds, Pc-1))
                tc_y = max(0, min(tr_y//ds, Pc-1)); tc_x = max(0, min(tr_x//ds, Pc-1))
                smask = torch.zeros(1, Pc, Pc, dtype=torch.bool, device=device)
                smask[0, sc_y, sc_x] = True
                cfg = EikonalConfig(n_iters=n_iters, h=float(ds),
                                    tau_min=0.03, tau_branch=0.05,
                                    tau_update=0.03, use_redblack=True, monotone=True)
                T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)
                T_vals.append(float(T[0, tc_y, tc_x].item()))
            else:
                cost = model._road_prob_to_cost(roi)
                smask = torch.zeros(1, P, P, dtype=torch.bool, device=device)
                smask[0, sr_y, sr_x] = True
                cfg = EikonalConfig(n_iters=n_iters, h=1.0,
                                    tau_min=0.03, tau_branch=0.05,
                                    tau_update=0.03, use_redblack=True, monotone=True)
                T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)
                T_vals.append(float(T[0, tr_y, tr_x].item()))
        return T_vals

    @torch.no_grad()
    def _evaluate(self, model) -> dict:
        import numpy as np
        from utils import pick_anchor_and_knn, spearmanr, kendall_tau, pairwise_order_acc

        device = next(model.parameters()).device
        self._ensure_loaded(device)
        self._rp_cache.clear()   # model weights changed → recompute road_prob

        PATCH  = int(model.config.get("PATCH_SIZE", 512))
        margin = int(model.config.get("ROUTE_ROI_MARGIN", 64))
        n_iters = int(model.config.get("ROUTE_EIK_ITERS", 20))
        min_ds  = int(model.config.get("ROUTE_EIK_DOWNSAMPLE", 8))

        nodes_yx = self._nodes_yx
        info     = self._info
        case     = info["case_idx"]
        meta_px  = int(info["meta_sat_height_px"])
        pscale   = 1.0 / self._scale if self._scale != 1.0 else 1.0
        _, _, H_ds, W_ds = self._rgb_ds.shape

        spearman_acc, kendall_acc, pw_acc = [], [], []
        N = nodes_yx.shape[0]

        for anc_idx in range(N):
            _, nn_global = pick_anchor_and_knn(nodes_yx, k=self.k, anchor_idx=anc_idx)
            anc = nodes_yx[anc_idx]
            ay, ax = int(anc[0]), int(anc[1])

            py0 = max(0, min(ay - PATCH // 2, H_ds - PATCH))
            px0 = max(0, min(ax - PATCH // 2, W_ds - PATCH))
            py1, px1 = py0 + PATCH, px0 + PATCH

            tgts_global = nodes_yx[nn_global]
            in_p = ((tgts_global[:, 0] >= py0) & (tgts_global[:, 0] < py1) &
                    (tgts_global[:, 1] >= px0) & (tgts_global[:, 1] < px1))
            keep = torch.where(in_p)[0]
            if keep.numel() < self.min_in_patch:
                continue

            nn_kept   = nn_global[keep]
            tgt_patch = tgts_global[keep] - torch.tensor([py0, px0], device=device)
            anc_patch = torch.tensor([ay - py0, ax - px0], device=device).long()

            rp = self._patch_rp(model, py0, py1, px0, px1, PATCH, device)
            T_raw = np.array(self._eikonal_k(
                model, rp, anc_patch, tgt_patch, n_iters, min_ds, margin, device),
                dtype=np.float64)
            T_norm = T_raw * pscale / max(meta_px, 1)

            nn_np  = nn_kept.cpu().numpy().astype(int)
            gt_arr = np.asarray(
                info["undirected_dist_norm"][case][anc_idx, nn_np], dtype=np.float64)
            valid  = np.isfinite(T_norm) & (T_norm < 1e3) & np.isfinite(gt_arr)
            if valid.sum() < 3:
                continue

            pv, gv = T_norm[valid], gt_arr[valid]
            spearman_acc.append(spearmanr(pv, gv))
            kendall_acc.append(kendall_tau(pv, gv))
            pw_acc.append(pairwise_order_acc(pv, gv))

        if not spearman_acc:
            return dict(eval_spearman=float("nan"), eval_kendall=float("nan"),
                        eval_pw_acc=float("nan"), eval_n_anchors=0.0)
        return dict(
            eval_spearman=float(np.mean(spearman_acc)),
            eval_kendall =float(np.mean(kendall_acc)),
            eval_pw_acc  =float(np.nanmean(pw_acc)),
            eval_n_anchors=float(len(spearman_acc)),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> aDict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return aDict(raw)


def load_pretrained_weights(model: SAMRoute, ckpt_path: str) -> None:
    """
    Load a SAMRoad / SAMRoute checkpoint into the model.

    Accepts both:
    - Lightning .ckpt files (contain "state_dict" key)
    - Raw PyTorch .pth / .pt state-dict files
    """
    print(f"[train] Loading pretrained weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[train]  Missing keys   : {len(missing)}")
    print(f"[train]  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[train]  (first 5 missing): {missing[:5]}")


def build_model(config: aDict, pretrained_ckpt: str | None) -> SAMRoute:
    """Instantiate SAMRoute (which already loads SAM weights via __init__)."""
    model = SAMRoute(config)
    if pretrained_ckpt and os.path.isfile(pretrained_ckpt):
        load_pretrained_weights(model, pretrained_ckpt)
    else:
        if pretrained_ckpt:
            print(f"[train] WARNING: PRETRAINED_CKPT not found: {pretrained_ckpt}")
        print("[train] Training from SAM backbone weights only.")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAMRoute training demo")
    p.add_argument("--config",       default=os.path.join(_HERE, "config_sam_route_demo.yaml"),
                   help="Path to YAML config file")
    p.add_argument("--data_root",    default=os.path.join(_HERE, "../Gen_dataset_V2/Gen_dataset"),
                   help="Root of Gen_dataset folder")
    p.add_argument("--ckpt_path",    default=None,
                   help="Override PRETRAINED_CKPT in config (path to .ckpt/.pth)")
    p.add_argument("--output_dir",   default=os.path.join(_HERE, "../training_outputs/sam_route"),
                   help="Directory for checkpoints and logs")
    p.add_argument("--gpus",         type=int, default=1,
                   help="Number of GPUs (0 for CPU)")
    p.add_argument("--epochs",       type=int, default=None,
                   help="Override TRAIN_EPOCHS in config")
    p.add_argument("--batch_size",   type=int, default=None,
                   help="Override BATCH_SIZE in config")
    p.add_argument("--workers",      type=int, default=None,
                   help="Override DATA_WORKER_NUM in config")
    p.add_argument("--samples_per_region", type=int, default=50,
                   help="Virtual samples per region per epoch")
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction of cities used for validation")
    p.add_argument("--include_dist", action="store_true",
                   help="Include distance supervision (requires ROUTE_LAMBDA_DIST > 0)")
    p.add_argument("--resume",       default=None,
                   help="Resume from a Lightning checkpoint (path to .ckpt)")
    p.add_argument("--precision",    default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"],
                   help="Training precision")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Ignore PRETRAINED_CKPT and train from SAM weights only")
    p.add_argument("--no_cached_features", action="store_true",
                   help="Disable pre-computed encoder features; always run encoder forward")
    p.add_argument("--no_preload", action="store_true",
                   help="Disable pre-loading data into RAM (use lazy/memmap loading instead)")
    p.add_argument("--preload_workers", type=int, default=8,
                   help="Number of threads for parallel pre-loading (default 8)")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile() to the model for ~15%% GPU speedup (PyTorch 2.0+)")
    p.add_argument("--road_dilation_radius", type=int, default=0,
                   help="Use normalized road masks with this dilation radius (0=off). "
                        "Run precompute_normalized_masks.py first.")
    # --- Periodic Spearman evaluation ---
    p.add_argument("--eval_every_n_epochs", type=int, default=0,
                   help="Evaluate Spearman rank correlation every N epochs (0=disabled). "
                        "Requires --eval_image_path.")
    p.add_argument("--eval_image_path", type=str, default="",
                   help="Path to a held-out .tif image with a matching "
                        "distance_dataset_all_*_p20.npz for Spearman evaluation.")
    p.add_argument("--eval_k", type=int, default=19,
                   help="Number of KNN neighbours to evaluate per anchor (default 19).")
    p.add_argument("--eval_min_in_patch", type=int, default=3,
                   help="Skip anchors with fewer than this many in-patch targets (default 3).")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Config ---
    config = load_config(args.config)
    if args.epochs is not None:
        config.TRAIN_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.workers is not None:
        config.DATA_WORKER_NUM = args.workers

    # Resolve relative paths inside config relative to the config file's dir
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ("SAM_CKPT_PATH", "PRETRAINED_CKPT"):
        val = str(config.get(key, "") or "")
        if val and not os.path.isabs(val):
            config[key] = os.path.normpath(os.path.join(config_dir, val))

    # CLI overrides
    if args.ckpt_path:
        config.PRETRAINED_CKPT = os.path.abspath(args.ckpt_path)
    if args.no_pretrained:
        config.PRETRAINED_CKPT = ""

    print("=" * 60)
    print(f"[train] Config:       {args.config}")
    print(f"[train] Data root:    {args.data_root}")
    print(f"[train] Output dir:   {args.output_dir}")
    print(f"[train] Epochs:       {config.TRAIN_EPOCHS}")
    print(f"[train] Batch size:   {config.BATCH_SIZE}")
    print(f"[train] Workers:      {config.DATA_WORKER_NUM}")
    print(f"[train] Patch size:   {config.PATCH_SIZE}")
    print(f"[train] FREEZE_ENC:   {config.FREEZE_ENCODER}")
    print(f"[train] lambda_seg:   {config.ROUTE_LAMBDA_SEG}")
    print(f"[train] lambda_dist:  {config.ROUTE_LAMBDA_DIST}")
    print(f"[train] include_dist: {args.include_dist}")
    print(f"[train] road_dil_r:   {args.road_dilation_radius}")
    print(f"[train] Precision:    {args.precision}")
    print(f"[train] GPUs:         {args.gpus}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Dataloaders ---
    use_cached  = not args.no_cached_features
    preload_ram = not args.no_preload
    train_loader, val_loader = build_dataloaders(
        root_dir=args.data_root,
        patch_size=int(config.PATCH_SIZE),
        batch_size=int(config.BATCH_SIZE),
        num_workers=int(config.DATA_WORKER_NUM),
        include_dist=args.include_dist,
        val_fraction=args.val_fraction,
        samples_per_region=args.samples_per_region,
        use_cached_features=use_cached,
        preload_to_ram=preload_ram,
        preload_workers=args.preload_workers,
        k_targets=int(config.get("ROUTE_K_TARGETS", 4)),
        road_dilation_radius=args.road_dilation_radius,
        min_in_patch=int(config.get("ROUTE_MIN_IN_PATCH", 2)),
    )
    print(f"[train] Cached features: {use_cached}  |  Pre-load to RAM: {preload_ram}")
    print(f"[train] Train batches/epoch: {len(train_loader)}")
    print(f"[train] Val   batches/epoch: {len(val_loader)}")

    # --- Model ---
    pretrained = str(config.get("PRETRAINED_CKPT", "") or "")
    model = build_model(config, pretrained if pretrained else None)

    # torch.compile — fuses kernels in the map decoder for ~15-20% GPU speedup.
    # Falls back silently on PyTorch < 2.0 or when dynamo is unsupported.
    if args.compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile() applied.")
        except Exception as e:
            print(f"[train] torch.compile() skipped: {e}")

    # --- Callbacks & Logger ---
    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="samroute-{epoch:02d}-{val_seg_loss:.4f}",
        monitor="val_seg_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tb_logger  = TensorBoardLogger(
        save_dir=args.output_dir,
        name="tensorboard",
    )

    callbacks_list = [ckpt_callback, lr_monitor]

    if args.eval_every_n_epochs > 0 and args.eval_image_path:
        spearman_cb = SpearmanEvalCallback(
            image_path=args.eval_image_path,
            eval_every_n_epochs=args.eval_every_n_epochs,
            k=args.eval_k,
            min_in_patch=args.eval_min_in_patch,
        )
        callbacks_list.append(spearman_cb)
        print(f"[train] SpearmanEvalCallback: every {args.eval_every_n_epochs} epochs "
              f"on {args.eval_image_path}")

    # --- Trainer ---
    accelerator = "gpu" if args.gpus > 0 else "cpu"
    devices     = args.gpus if args.gpus > 0 else 1

    trainer = pl.Trainer(
        max_epochs=int(config.TRAIN_EPOCHS),
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=callbacks_list,
        logger=tb_logger,
        log_every_n_steps=10,
        val_check_interval=1.0,       # validate once per epoch
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    # --- Fit ---
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    print(f"\n[train] Done. Best checkpoint: {ckpt_callback.best_model_path}")


if __name__ == "__main__":
    main()
