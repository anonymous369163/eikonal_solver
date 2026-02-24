"""
Visualize segmentation with IoU and seg_loss overlaid.

Loads validation samples, runs SAMRoute forward, computes IoU/seg_loss,
and saves a figure: [RGB | GT mask | Pred mask] with metrics in title.

Use this to judge: is poor IoU (e.g. 0.07) due to bad input, or is the metric
itself misleading for this task?

Usage:
    cd /home/yuepeng/code/mmdm_V3/MMDataset
    python eikonal_solver/visualize_with_iou.py
    python eikonal_solver/visualize_with_iou.py --no_cached_features   # show Input RGB column (slower)
    python eikonal_solver/visualize_with_iou.py --ckpt_path path/to/ckpt.ckpt --n_samples 8
    python eikonal_solver/visualize_with_iou.py --config eikonal_solver/config_normalized_masks.yaml \
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
if sys.path[0] != _HERE:
    if _HERE in sys.path:
        sys.path.remove(_HERE)
    sys.path.insert(0, _HERE)

import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from addict import Dict as aDict
from torchmetrics.classification import BinaryJaccardIndex

from model import SAMRoute
from train_dataset import build_dataloaders


def load_config(path: str) -> aDict:
    with open(path) as f:
        return aDict(yaml.safe_load(f))


def main():
    import argparse
    p = argparse.ArgumentParser(description="Visualize segmentation with IoU and seg_loss")
    p.add_argument("--config", default=os.path.join(_HERE, "config_sam_route_demo.yaml"))
    p.add_argument("--data_root", default=os.path.join(_HERE, "../Gen_dataset_V2/Gen_dataset"))
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--n_samples", type=int, default=4,
                   help="Number of samples to visualize (in a 2x2 or 2x4 grid)")
    p.add_argument("--save_path", default=os.path.join(_HERE, "images", "vis_with_iou.png"))
    p.add_argument("--no_cached_features", action="store_true",
                   help="Disable cached features to load RGB (slower); needed to show input column")
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
        sys.exit(1)

    use_cached = not args.no_cached_features
    _, val_loader = build_dataloaders(
        root_dir=args.data_root,
        patch_size=int(config.PATCH_SIZE),
        batch_size=args.n_samples,
        num_workers=2,
        include_dist=False,
        val_fraction=0.1,
        samples_per_region=50,
        use_cached_features=use_cached,
        preload_to_ram=True,
        preload_workers=2,
        road_dilation_radius=args.road_dilation_radius,
    )

    model = SAMRoute(config)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()

    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch = next(iter(val_loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.no_grad():
        loss_seg, road_prob = model._seg_forward(batch)

    road_mask = batch["road_mask"].to(torch.float32)
    road_gt = (road_mask > 0.5).to(torch.long)
    road_pred = (road_prob > 0.5)  # for viz
    iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)
    iou_metric.update(road_prob, road_gt)  # preds: probs; target: int (same as model validation_step)
    iou_val = iou_metric.compute().item()

    n = min(args.n_samples, batch["road_mask"].shape[0])
    n_cols = 3 if "rgb" in batch else 2
    fig, axes = plt.subplots(n, n_cols, figsize=(5 * n_cols, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        idx = 0
        if "rgb" in batch:
            rgb = batch["rgb"][i].cpu().numpy()
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            axes[i, idx].imshow(rgb)
            axes[i, idx].set_title("Input RGB")
            axes[i, idx].axis("off")
            idx += 1

        gt = road_gt[i].cpu().numpy()
        axes[i, idx].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i, idx].set_title("GT road mask")
        axes[i, idx].axis("off")
        idx += 1

        pred = road_pred[i].float().cpu().numpy()
        axes[i, idx].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[i, idx].set_title("Pred road mask")
        axes[i, idx].axis("off")

    title = f"IoU={iou_val:.4f}  seg_loss={loss_seg.item():.4f}  (ckpt: {os.path.basename(ckpt)})"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("=" * 60)
    print(f"  IoU:      {iou_val:.4f}")
    print(f"  seg_loss: {loss_seg.item():.4f}")
    print(f"  Saved:    {args.save_path}")
    print("=" * 60)
    print("  Open the saved image to compare GT vs Pred (and RGB if --no_cached_features)")


if __name__ == "__main__":
    main()
