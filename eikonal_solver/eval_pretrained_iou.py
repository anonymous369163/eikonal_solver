"""
Evaluate road segmentation IoU of the PRETRAINED model (before any Phase 1 training).

This gives the baseline IoU on your validation set — useful to compare against
training progress (e.g., if you see IoU 0.07, is that better or worse than baseline?).

Usage:
    cd /home/yuepeng/code/mmdm_V3/MMDataset
    python eikonal_solver/eval_pretrained_iou.py
    python eikonal_solver/eval_pretrained_iou.py --ckpt_path path/to/other.ckpt --max_batches 100
    python eikonal_solver/eval_pretrained_iou.py --config eikonal_solver/config_normalized_masks.yaml \
        --ckpt_path training_outputs/phase1_r8_lora_dice/checkpoints/last.ckpt \
        --road_dilation_radius 8
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAM_REPO = os.path.join(_HERE, "../sam_road_repo")
_SAM_SEG = os.path.join(_SAM_REPO, "segment-anything-road")
_SAM_SAM = os.path.join(_SAM_REPO, "sam")
for _sp in [_HERE, _SAM_SEG, _SAM_SAM, _SAM_REPO]:
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
# eikonal_solver must be first so we import SAMRoute from eikonal_solver/model.py
if sys.path[0] != _HERE:
    if _HERE in sys.path:
        sys.path.remove(_HERE)
    sys.path.insert(0, _HERE)

import torch
import yaml
from addict import Dict as aDict
from torchmetrics.classification import BinaryJaccardIndex

from model import SAMRoute
from train_dataset import build_dataloaders


def load_config(path: str) -> aDict:
    with open(path) as f:
        return aDict(yaml.safe_load(f))


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate pretrained model IoU on validation set")
    p.add_argument("--config", default=os.path.join(_HERE, "config_sam_route_demo.yaml"))
    p.add_argument("--data_root", default=os.path.join(_HERE, "../Gen_dataset_V2/Gen_dataset"))
    p.add_argument("--ckpt_path", default=None, help="Pretrained checkpoint (default: from config)")
    p.add_argument("--max_batches", type=int, default=None,
                   help="Limit evaluation to N batches (default: all)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_cached_features", action="store_true",
                   help="Disable cached encoder features (slower but no precompute needed)")
    p.add_argument("--road_dilation_radius", type=int, default=0,
                   help="Must match training config: 0=original masks, 3/8=normalized masks")
    p.add_argument("--gpus", type=int, default=1)
    args = p.parse_args()

    config = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ("SAM_CKPT_PATH", "PRETRAINED_CKPT"):
        val = str(config.get(key, "") or "")
        if val and not os.path.isabs(val):
            config[key] = os.path.normpath(os.path.join(config_dir, val))

    ckpt = args.ckpt_path or config.get("PRETRAINED_CKPT", "")
    if not ckpt or not os.path.isfile(ckpt):
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print("  Set PRETRAINED_CKPT in config or pass --ckpt_path")
        sys.exit(1)

    print("=" * 60)
    print("  Evaluate PRETRAINED model IoU (before Phase 1 training)")
    print("=" * 60)
    print(f"  Config:        {args.config}")
    print(f"  Data root:     {args.data_root}")
    print(f"  Checkpoint:    {ckpt}")
    print(f"  Max batches:   {args.max_batches or 'all'}")
    print(f"  road_dil_r:    {args.road_dilation_radius}")
    print("=" * 60)

    _, val_loader = build_dataloaders(
        root_dir=args.data_root,
        patch_size=int(config.PATCH_SIZE),
        batch_size=args.batch_size,
        num_workers=args.workers,
        include_dist=False,
        val_fraction=0.1,
        samples_per_region=50,
        use_cached_features=not args.no_cached_features,
        preload_to_ram=True,
        preload_workers=4,
        road_dilation_radius=args.road_dilation_radius,
    )

    model = SAMRoute(config)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if args.max_batches is not None and n_batches >= args.max_batches:
                break

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            loss_seg, road_prob = model._seg_forward(batch)
            road_mask = batch["road_mask"].to(device, dtype=torch.float32)
            road_gt = (road_mask > 0.5).to(torch.long)
            iou_metric.update(road_prob, road_gt)  # preds: probs; target: int (same as model validation_step)
            n_batches += 1

            if n_batches % 20 == 0:
                print(f"  Processed {n_batches} batches ...")

    iou = iou_metric.compute().item()
    print("\n" + "=" * 60)
    print(f"  Val Road IoU (pretrained): {iou:.4f}")
    print(f"  (evaluated on {n_batches} batches)")
    print("=" * 60)
    print("\n  Compare with your training val_road_iou. If pretrained IoU is")
    print("  similar or higher than 0.07, training may need more epochs or")
    print("  the dataset may differ significantly from the pretraining data.")


if __name__ == "__main__":
    main()
