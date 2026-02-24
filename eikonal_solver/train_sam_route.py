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

    # --- Trainer ---
    accelerator = "gpu" if args.gpus > 0 else "cpu"
    devices     = args.gpus if args.gpus > 0 else 1

    trainer = pl.Trainer(
        max_epochs=int(config.TRAIN_EPOCHS),
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=[ckpt_callback, lr_monitor],
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
