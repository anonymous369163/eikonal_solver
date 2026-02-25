"""
eval_compare_spearman.py  —  Compare Spearman ranking across multiple checkpoints.

Usage:
    # Quick demo (2 images, default checkpoints):
    python eikonal_solver/eval_compare_spearman.py

    # Custom checkpoints and images:
    python eikonal_solver/eval_compare_spearman.py \
        --ckpts "Pretrained:checkpoints/cityscale_vitb_512_e10.ckpt" \
                "NewFinetuned:training_outputs/phase1_spearman_track/checkpoints/samroute-epoch=89-val_seg_loss=0.8235.ckpt" \
        --images Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/crop_34.350697_108.914569_3000.0_z16.tif \
    --n_images 5 --k 15 --save_csv results/spearman_compare.csv

Arguments:
    --ckpts       One or more "Label:path" pairs (label must not contain spaces).
    --config      Path to config yaml (default: eikonal_solver/config_phase1.yaml).
    --images      Explicit list of TIF image paths to evaluate.
    --data_root   Root to auto-discover crop_*.tif images (used if --images not given).
    --n_images    Max number of images to evaluate (default: all found).
    --k           Number of KNN anchors per node (default: 15).
    --min_in_patch  Min targets that must fall inside the patch (default: 3).
    --downsample  Long-edge cap for image (default: 1024).
    --save_csv    If provided, write per-image results to this CSV path.
    --device      cuda / cpu (default: cuda).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sys-path setup (mirror evaluator.py)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in [str(_REPO / "sam_road_repo" / "segment-anything-road"),
           str(_REPO / "sam_road_repo" / "sam"),
           str(_REPO / "sam_road_repo")]:
    if _p not in sys.path:
        sys.path.append(_p)
_here_str = str(_HERE)
if _here_str in sys.path:
    sys.path.remove(_here_str)
sys.path.insert(0, _here_str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_crop_images(data_root: str, limit: Optional[int] = None) -> List[str]:
    """Recursively find all crop_*.tif files under data_root."""
    root = Path(data_root)
    images = sorted(root.rglob("crop_*.tif"))
    if limit:
        images = images[:limit]
    return [str(p) for p in images]


def _fmt(v) -> str:
    if v is None:
        return "   N/A  "
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "   N/A  "
    import math
    if math.isnan(f):
        return "  NaN   "
    return f"{f:+.4f}"


def _delta_str(new_val, baseline_val) -> str:
    """Return Δ relative to baseline, colored if terminal supports it."""
    try:
        d = float(new_val) - float(baseline_val)
    except (TypeError, ValueError):
        import math
        try:
            if math.isnan(float(new_val)) or math.isnan(float(baseline_val)):
                return "    NaN "
        except Exception:
            pass
        return "    N/A "
    import math
    if math.isnan(d):
        return "    NaN "
    sign = "▲" if d > 0 else ("▼" if d < 0 else " ")
    color_on = "\033[32m" if d > 0 else ("\033[31m" if d < 0 else "")
    color_off = "\033[0m" if (d != 0) else ""
    return f"{color_on}{sign}{abs(d):.4f}{color_off}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare Spearman ranking across multiple SAMRoute checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpts", nargs="+", metavar="LABEL:PATH",
        help="One or more 'Label:path' pairs. Label must not contain colons.",
    )
    parser.add_argument(
        "--config",
        default=str(_HERE / "config_phase1.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--images", nargs="*", metavar="TIF_PATH",
        help="Explicit list of TIF image paths.",
    )
    parser.add_argument(
        "--data_root",
        default=str(_REPO / "Gen_dataset_V2" / "Gen_dataset"),
        help="Root for auto-discovery of crop_*.tif images.",
    )
    parser.add_argument("--n_images", type=int, default=None,
                        help="Max images to evaluate.")
    parser.add_argument("--k", type=int, default=15,
                        help="KNN anchors per node.")
    parser.add_argument("--min_in_patch", type=int, default=3)
    parser.add_argument("--downsample", type=int, default=1024)
    parser.add_argument("--save_csv", default=None,
                        help="Write per-image results to this CSV path.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # ---- default checkpoints ----
    if not args.ckpts:
        defaults = [
            ("Pretrained",
             str(_REPO / "checkpoints" / "cityscale_vitb_512_e10.ckpt")),
            ("phase1_r5_seg (epoch73)",
             str(_REPO / "training_outputs" / "phase1_r5_seg" / "checkpoints" /
                 "samroute-epoch=73-val_seg_loss=0.6838.ckpt")),
            ("phase1_spearman_track (epoch89)",
             str(_REPO / "training_outputs" / "phase1_spearman_track" / "checkpoints" /
                 "samroute-epoch=89-val_seg_loss=0.8235.ckpt")),
        ]
        ckpt_pairs: List[Tuple[str, str]] = [(lbl, p) for lbl, p in defaults
                                              if Path(p).exists()]
        if not ckpt_pairs:
            print("[ERROR] No default checkpoints found. Use --ckpts LABEL:PATH ...")
            sys.exit(1)
    else:
        ckpt_pairs = []
        for item in args.ckpts:
            parts = item.split(":", 1)
            if len(parts) != 2:
                print(f"[ERROR] Bad --ckpts item (expected 'Label:path'): {item}")
                sys.exit(1)
            label, path = parts
            if not Path(path).exists():
                print(f"[WARN]  Checkpoint not found, skipping: {path}")
                continue
            ckpt_pairs.append((label, path))
        if not ckpt_pairs:
            print("[ERROR] No valid checkpoints after filtering.")
            sys.exit(1)

    # ---- image list ----
    if args.images:
        image_paths = [p for p in args.images if Path(p).exists()]
        missing = [p for p in args.images if not Path(p).exists()]
        for m in missing:
            print(f"[WARN]  Image not found, skipping: {m}")
    else:
        image_paths = discover_crop_images(args.data_root, limit=args.n_images)
    if args.n_images:
        image_paths = image_paths[:args.n_images]
    if not image_paths:
        print("[ERROR] No valid images found. Use --images or --data_root.")
        sys.exit(1)

    # ---- print plan ----
    print("=" * 70)
    print("  Spearman Comparison across Checkpoints")
    print("=" * 70)
    print(f"  Config  : {args.config}")
    print(f"  Images  : {len(image_paths)}")
    for i, (lbl, p) in enumerate(ckpt_pairs):
        marker = "[baseline]" if i == 0 else f"[vs baseline]"
        print(f"  {marker:14s} {lbl}: {Path(p).name}")
    print(f"  k={args.k}  min_in_patch={args.min_in_patch}  downsample={args.downsample}")
    print("=" * 70)

    from evaluator import SAMRouteEvaluator

    # ---- load all evaluators ----
    evaluators: Dict[str, SAMRouteEvaluator] = {}
    for label, ckpt_path in ckpt_pairs:
        print(f"\n[Loading] {label}")
        evaluators[label] = SAMRouteEvaluator(args.config, ckpt_path, device=args.device)

    # ---- evaluate ----
    # per_image[img_path][label] = result_dict
    per_image: Dict[str, Dict[str, dict]] = {}

    for img_idx, img_path in enumerate(image_paths):
        img_name = Path(img_path).parent.name + "/" + Path(img_path).name
        print(f"\n{'─'*70}")
        print(f"[Image {img_idx+1}/{len(image_paths)}]  {img_name}")
        per_image[img_path] = {}
        for label, _ in ckpt_pairs:
            print(f"  > {label}")
            t0 = time.time()
            try:
                res = evaluators[label].eval_spearman(
                    img_path, k=args.k,
                    min_in_patch=args.min_in_patch,
                    downsample=args.downsample,
                )
            except Exception as e:
                print(f"    [ERROR] {e}")
                res = {"spearman": float("nan"), "kendall": float("nan"),
                       "pw_acc": float("nan"), "n_anchors": 0}
            per_image[img_path][label] = res

        # ---- per-image comparison ----
        baseline_label, _ = ckpt_pairs[0]
        baseline = per_image[img_path][baseline_label]
        print(f"\n  Per-image summary [{img_name}]:")
        col_w = max(len(lbl) for lbl, _ in ckpt_pairs) + 2
        header = f"  {'Checkpoint':<{col_w}}  {'Spearman':>9}  {'Kendall':>9}  {'PW-Acc':>9}  {'Anchors':>7}  {'Δ Spearman':>11}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, (label, _) in enumerate(ckpt_pairs):
            r = per_image[img_path][label]
            delta = ("  [baseline]" if i == 0
                     else f"  {_delta_str(r.get('spearman'), baseline.get('spearman'))}")
            print(f"  {label:<{col_w}}  {_fmt(r.get('spearman')):>9}"
                  f"  {_fmt(r.get('kendall')):>9}"
                  f"  {_fmt(r.get('pw_acc')):>9}"
                  f"  {r.get('n_anchors', 0):>7}{delta}")

    # ---- aggregate summary ----
    import numpy as np

    print(f"\n{'='*70}")
    print("  AGGREGATE SUMMARY (mean across all images)")
    print(f"{'='*70}")
    col_w = max(len(lbl) for lbl, _ in ckpt_pairs) + 2
    header = f"  {'Checkpoint':<{col_w}}  {'Spearman':>9}  {'Kendall':>9}  {'PW-Acc':>9}  {'Images':>7}  {'Δ Spearman':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    agg: Dict[str, dict] = {}
    for label, _ in ckpt_pairs:
        sp_vals, kd_vals, pw_vals, img_count = [], [], [], 0
        for img_path in image_paths:
            if img_path not in per_image:
                continue
            r = per_image[img_path].get(label, {})
            sp, kd, pw = r.get("spearman"), r.get("kendall"), r.get("pw_acc")
            if sp is not None and np.isfinite(float(sp)):
                sp_vals.append(float(sp))
            if kd is not None and np.isfinite(float(kd)):
                kd_vals.append(float(kd))
            if pw is not None and np.isfinite(float(pw)):
                pw_vals.append(float(pw))
            img_count += 1
        agg[label] = {
            "spearman": float(np.mean(sp_vals)) if sp_vals else float("nan"),
            "kendall":  float(np.mean(kd_vals)) if kd_vals else float("nan"),
            "pw_acc":   float(np.mean(pw_vals)) if pw_vals else float("nan"),
            "n_images": img_count,
        }

    baseline_label, _ = ckpt_pairs[0]
    for i, (label, _) in enumerate(ckpt_pairs):
        r = agg[label]
        delta = ("  [baseline]" if i == 0
                 else f"  {_delta_str(r['spearman'], agg[baseline_label]['spearman'])}")
        print(f"  {label:<{col_w}}  {_fmt(r['spearman']):>9}"
              f"  {_fmt(r['kendall']):>9}"
              f"  {_fmt(r['pw_acc']):>9}"
              f"  {r['n_images']:>7}{delta}")

    # ---- CSV export ----
    if args.save_csv:
        save_path = Path(args.save_csv)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        labels = [lbl for lbl, _ in ckpt_pairs]
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            header_row = ["image"]
            for lbl in labels:
                header_row += [f"{lbl}_spearman", f"{lbl}_kendall",
                               f"{lbl}_pw_acc", f"{lbl}_n_anchors"]
            writer.writerow(header_row)
            for img_path in image_paths:
                row = [img_path]
                for lbl in labels:
                    r = per_image.get(img_path, {}).get(lbl, {})
                    row += [r.get("spearman", ""), r.get("kendall", ""),
                            r.get("pw_acc", ""), r.get("n_anchors", "")]
                writer.writerow(row)
            # aggregate row
            agg_row = ["[AGGREGATE MEAN]"]
            for lbl in labels:
                r = agg[lbl]
                agg_row += [r["spearman"], r["kendall"], r["pw_acc"], r["n_images"]]
            writer.writerow(agg_row)
        print(f"\n  CSV saved → {save_path}")

    print(f"\n{'='*70}\n  Done.\n{'='*70}")


if __name__ == "__main__":
    main()
