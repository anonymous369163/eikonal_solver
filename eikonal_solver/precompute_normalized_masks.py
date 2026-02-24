"""
Precompute uniform-width road masks for all regions.

Problem: the original roadnet PNG has variable road widths — main roads are
thicker than side streets.  This creates inconsistent supervision: the model
sees different-width roads and may bias toward thicker/easier roads.

Solution (per region):
  1. Binarize the road mask at 0.5.
  2. Skeletonize → 1-pixel-wide road centrelines.
  3. Dilate with a fixed radius → all roads are uniformly (2r+1) pixels wide.
  4. Save as  roadnet_normalized_r{radius}.png  (R=G=B=255 for road, alpha=255).

Usage:
    cd /home/yuepeng/code/mmdm_V3/MMDataset
    python eikonal_solver/precompute_normalized_masks.py
    python eikonal_solver/precompute_normalized_masks.py --radius 3 --workers 16
    python eikonal_solver/precompute_normalized_masks.py --dry_run   # count regions only
"""

from __future__ import annotations

import os
import sys
import glob
import time
import argparse
import threading
from typing import List

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, disk, binary_dilation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_region_dirs(root_dir: str) -> List[str]:
    """Return leaf region dirs that contain a roadnet PNG."""
    regions = []
    for city in sorted(os.listdir(root_dir)):
        city_path = os.path.join(root_dir, city)
        if not os.path.isdir(city_path):
            continue
        for region in sorted(os.listdir(city_path)):
            rpath = os.path.join(city_path, region)
            if not os.path.isdir(rpath):
                continue
            pngs = glob.glob(os.path.join(rpath, "roadnet_*.png"))
            if pngs:
                regions.append(rpath)
    return regions


def normalize_road_mask(mask_float: np.ndarray, radius: int) -> np.ndarray:
    """
    Convert a float [0,1] road mask to a uniform-width binary mask.

    Steps:
      1. Binarize at 0.5 → bool [H, W]
      2. Skeletonize → 1-pixel centrelines
      3. Dilate with disk(radius) → uniform width = (2*radius+1) px

    Returns float32 [H, W] in {0.0, 1.0}.
    """
    binary = mask_float > 0.5
    if not binary.any():
        return np.zeros_like(mask_float)
    skel    = skeletonize(binary)
    dilated = binary_dilation(skel, disk(radius))
    return dilated.astype(np.float32)


def process_region(region_path: str, radius: int, overwrite: bool) -> str:
    """Process one region and save the normalized mask. Returns status string."""
    pngs = glob.glob(os.path.join(region_path, "roadnet_*.png"))
    if not pngs:
        return f"SKIP (no PNG): {region_path}"

    src_png = sorted(pngs)[0]
    out_png = os.path.join(region_path, f"roadnet_normalized_r{radius}.png")

    if os.path.exists(out_png) and not overwrite:
        return f"EXISTS: {out_png}"

    arr = np.array(Image.open(src_png))[:, :, 0].astype(np.float32) / 255.0
    normalized = normalize_road_mask(arr, radius)

    # Save as RGBA PNG (R=G=B=road value * 255, A=255) matching original format
    road_uint8 = (normalized * 255).astype(np.uint8)
    rgba = np.stack([road_uint8, road_uint8, road_uint8,
                     np.full_like(road_uint8, 255)], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(out_png)
    return f"OK: {out_png}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Precompute uniform-width road masks")
    p.add_argument("--data_root", default="Gen_dataset_V2/Gen_dataset")
    p.add_argument("--radius", type=int, default=3,
                   help="Dilation radius in pixels. Width = 2*radius+1. "
                        "r=2→5px, r=3→7px (default), r=4→9px, r=5→11px")
    p.add_argument("--workers", type=int, default=16,
                   help="Number of parallel threads")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if output file already exists")
    p.add_argument("--dry_run", action="store_true",
                   help="Only count regions, don't compute")
    args = p.parse_args()

    regions = _find_region_dirs(args.data_root)
    print(f"Found {len(regions)} regions under {args.data_root}")
    print(f"Dilation radius: {args.radius}  →  road width = {2*args.radius+1} px")
    print(f"Output file name: roadnet_normalized_r{args.radius}.png")
    if args.dry_run:
        return

    t_start = time.time()
    results  = [None] * len(regions)
    done_cnt = [0]
    lock     = threading.Lock()

    def _worker(indices):
        for i in indices:
            results[i] = process_region(regions[i], args.radius, args.overwrite)
            with lock:
                done_cnt[0] += 1
                n = done_cnt[0]
            if n % 50 == 0 or n == len(regions):
                elapsed = time.time() - t_start
                eta = elapsed / n * (len(regions) - n) if n > 0 else 0
                print(f"  {n}/{len(regions)} done  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    n = len(regions)
    chunk = max(1, (n + args.workers - 1) // args.workers)
    threads = []
    for w in range(args.workers):
        s, e = w * chunk, min((w + 1) * chunk, n)
        if s >= n:
            break
        t = threading.Thread(target=_worker, args=(range(s, e),), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - t_start
    ok  = sum(1 for r in results if r and r.startswith("OK"))
    skp = sum(1 for r in results if r and r.startswith(("EXISTS", "SKIP")))
    print(f"\nDone in {elapsed:.1f}s  |  Computed: {ok}  |  Skipped: {skp}")
    print(f"Average: {elapsed/max(ok,1):.2f}s per region")

    # Density stats on a sample
    sample_pngs = glob.glob(os.path.join(
        args.data_root, "*", "*", f"roadnet_normalized_r{args.radius}.png"))[:5]
    if sample_pngs:
        densities = []
        for png in sample_pngs:
            arr = np.array(Image.open(png))[:, :, 0].astype(float) / 255.0
            densities.append((arr > 0.5).mean())
        print(f"\nSample road densities after normalization (r={args.radius}):")
        for png, d in zip(sample_pngs, densities):
            print(f"  {os.path.basename(os.path.dirname(png))}: {d:.4f}")
        print(f"  Mean: {np.mean(densities):.4f}  "
              f"→  recommended ROAD_POS_WEIGHT ≈ {(1-np.mean(densities))/np.mean(densities):.1f}")


if __name__ == "__main__":
    main()
