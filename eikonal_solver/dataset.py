"""
MMRouteDataset  —  PyTorch Dataset for SAMRoute training.

Scans all region folders under `Gen_dataset_V2/Gen_dataset`, where each region
provides:
  - crop_*.tif          : RGB satellite image (typically ~2673×2673)
  - roadnet_*.png       : road segmentation mask (RGBA; R channel used as label)
  - distance_dataset_all_*_p{N}.npz  : matched node coordinates + GT distances

For each __getitem__ call the dataset:
  1. Picks a region (one large TIF + mask).
  2. Randomly crops a patch_size × patch_size window.
  3. Optionally samples one valid node-pair inside the window and returns the
     corresponding GT road distance (for future distance supervision).

Returns a dict:
    rgb:       [H, W, 3]  float32  values in [0, 255]  (SAMRoad convention)
    road_mask: [H, W]     float32  values in [0, 1]

Optional (only when include_dist=True and a valid pair is found):
    src_yx:   [2]   int64  source pixel (y, x) within the cropped patch
    tgt_yx:   [2]   int64  target pixel (y, x) within the cropped patch
    gt_dist:  []    float32  GT road distance in pixels (undirected_dist_norm
                             × image_height_px, giving pixel-scale path length).
                             Filtered to reject pairs whose GT distance exceeds
                             the ROI-reachable range (detour ratio or absolute cap).
"""

from __future__ import annotations

import os
import glob
import random
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Iterator

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_region_dirs(root_dir: str) -> List[str]:
    """Return all leaf region directories that contain a TIF and a PNG."""
    regions = []
    for city in sorted(os.listdir(root_dir)):
        city_path = os.path.join(root_dir, city)
        if not os.path.isdir(city_path):
            continue
        for region in sorted(os.listdir(city_path)):
            region_path = os.path.join(city_path, region)
            if not os.path.isdir(region_path):
                continue
            tifs = glob.glob(os.path.join(region_path, "crop_*.tif"))
            pngs = glob.glob(os.path.join(region_path, "roadnet_*.png"))
            if tifs and pngs:
                regions.append(region_path)
    return regions


def _load_rgb_from_tif(tif_path: str) -> np.ndarray:
    """Load a GeoTIFF as uint8 RGB array [H, W, 3]."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            # rasterio returns [C, H, W]; take first 3 bands
            bands = src.read()[:3]          # [3, H, W]
            arr = np.moveaxis(bands, 0, -1) # [H, W, 3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    except Exception:
        # Fallback: PIL (drops geo metadata but fine for training)
        img = Image.open(tif_path).convert("RGB")
        return np.array(img, dtype=np.uint8)


def _load_road_mask(png_path: str) -> np.ndarray:
    """Load road mask from RGBA PNG; returns float32 [H, W] in [0, 1]."""
    img = Image.open(png_path)
    arr = np.array(img)          # [H, W, 4]  uint8
    # R=G=B in this dataset; take R channel as road probability
    mask = arr[:, :, 0].astype(np.float32) / 255.0
    return mask


def _load_npz_nodes(npz_path: str):
    """Return (coords_norm, undirected_dist_norm, H, W) from an NPZ file."""
    d = np.load(npz_path, allow_pickle=True)
    coords = d["matched_node_norm"]          # (B, N, 2)  (x_norm, y_norm) bottom-left
    udist  = d["undirected_dist_norm"]       # (B, N, N)
    H = int(d["meta_sat_height_px"][0])
    W = int(d["meta_sat_width_px"][0])
    return coords, udist, H, W


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MMRouteDataset(Dataset):
    """
    Parameters
    ----------
    root_dir : str
        Path to the Gen_dataset folder (contains city sub-folders).
    patch_size : int
        Square patch side length in pixels. Default 512 (matches SAM ViT-B).
    include_dist : bool
        If True, attempt to sample a node pair inside each patch and return
        src_yx / tgt_yx / gt_dist. Falls back to seg-only when no valid pair.
    npz_variant : str
        "p20" or "p50" — which NPZ problem size to use for node pairs.
    split : str
        "train" or "val". Cities are split at the city level.
    val_fraction : float
        Fraction of cities reserved for validation. Default 0.1.
    samples_per_region : int
        Virtual dataset size multiplier per region (controls epoch length).
    augment : bool
        Apply random horizontal/vertical flips during training.
    seed : int
        Random seed for train/val split.
    """

    def __init__(
        self,
        root_dir: str,
        patch_size: int = 512,
        include_dist: bool = False,
        npz_variant: str = "p20",
        split: str = "train",
        val_fraction: float = 0.1,
        samples_per_region: int = 50,
        augment: bool = True,
        seed: int = 42,
        use_cached_features: bool = True,
        preload_to_ram: bool = True,
        preload_workers: int = 8,
        k_targets: int = 4,
        road_dilation_radius: int = 0,
        min_in_patch: int = 2,
    ):
        super().__init__()
        self.patch_size          = patch_size
        self.include_dist        = include_dist
        self.min_in_patch        = max(1, int(min_in_patch))
        self.npz_variant         = npz_variant
        self.split               = split
        self.samples_per_region  = samples_per_region
        self.augment             = augment and (split == "train")
        self.use_cached_features = use_cached_features
        self.k_targets           = max(1, int(k_targets))
        self.road_dilation_radius = int(road_dilation_radius)

        all_regions = _find_region_dirs(root_dir)
        if not all_regions:
            raise RuntimeError(f"No valid region dirs found under {root_dir}")

        # Split at city level (parent directory of each region)
        cities = sorted(set(os.path.dirname(r) for r in all_regions))
        rng = random.Random(seed)
        rng.shuffle(cities)
        n_val = max(1, int(len(cities) * val_fraction))
        val_cities  = set(cities[:n_val])
        train_cities = set(cities[n_val:])

        chosen_cities = val_cities if split == "val" else train_cities
        self.regions: List[str] = [
            r for r in all_regions if os.path.dirname(r) in chosen_cities
        ]
        if not self.regions:
            raise RuntimeError(
                f"No regions found for split='{split}' (val_fraction={val_fraction}). "
                f"Total cities: {len(cities)}, chosen: {len(chosen_cities)}"
            )

        # Pre-load metadata (avoid re-opening files on every __getitem__)
        self._meta: List[Dict[str, Any]] = []
        n_with_feat = 0
        n_with_norm = 0
        for region_path in self.regions:
            tif_path = sorted(glob.glob(os.path.join(region_path, "crop_*.tif")))[0]
            # Prefer pre-computed normalized mask when road_dilation_radius > 0
            r = self.road_dilation_radius
            orig_png = sorted(glob.glob(os.path.join(region_path, "roadnet_*.png")))
            orig_png = [p for p in orig_png if "normalized" not in p]
            orig_png_path = orig_png[0] if orig_png else None
            norm_path = os.path.join(region_path, f"roadnet_normalized_r{r}.png") if r > 0 else None
            if norm_path and os.path.exists(norm_path):
                png_path = norm_path
                n_with_norm += 1
            else:
                png_path = orig_png_path
            npz_paths = sorted(
                glob.glob(os.path.join(region_path, f"*_{npz_variant}.npz"))
            )
            # Pre-computed SAM encoder feature map (samroad_feat_full_*.npy)
            feat_paths = sorted(
                glob.glob(os.path.join(region_path, "samroad_feat_full_*.npy"))
            )
            has_feat = bool(feat_paths) and use_cached_features
            if has_feat:
                n_with_feat += 1
            # When dual_target is active, store original thin mask path separately.
            # png_thin is set only when road_dilation_radius > 0 AND the
            # original (non-normalized) mask differs from the thick mask.
            png_thin = orig_png_path if (r > 0 and png_path != orig_png_path) else None
            self._meta.append({
                "tif":  tif_path,
                "png":  png_path,
                "png_thin": png_thin,
                "npz":  npz_paths[0] if npz_paths else None,
                "feat": feat_paths[0] if has_feat else None,
                # lazy-loaded arrays stored here on first access
                "_rgb":    None,
                "_mask":   None,
                "_mask_thin": None,
                "_feat":   None,   # (C, H_feat, W_feat) float32
                "_coords": None,
                "_udist":  None,
                "_H":      None,
                "_W":      None,
            })
        if use_cached_features:
            print(f"[MMRouteDataset] {n_with_feat}/{len(self.regions)} regions "
                  f"have pre-computed encoder features (samroad_feat_full_*.npy). "
                  + ("Encoder will be skipped during training." if n_with_feat > 0
                     else "Will fall back to encoder forward."))
        if self.road_dilation_radius > 0:
            width = 2 * self.road_dilation_radius + 1
            print(f"[MMRouteDataset] Normalized road masks (r={self.road_dilation_radius}, "
                  f"width={width}px): {n_with_norm}/{len(self.regions)} regions. "
                  + (f"Run precompute_normalized_masks.py --radius {self.road_dilation_radius} "
                     f"to generate missing ones."
                     if n_with_norm < len(self.regions) else "All regions covered."))

        self._total = len(self.regions) * samples_per_region

        # Pre-load all masks and encoder feature maps into RAM using multi-threading.
        # Eliminates HDD random-access latency at training time entirely.
        # Typical footprint per region: ~7MB mask + ~28MB feat ≈ 35MB
        # For 400 regions: ~14GB — well within the available 217GB RAM.
        if preload_to_ram:
            self._preload_all(preload_workers)

    # ------------------------------------------------------------------
    # Pre-load all data into RAM
    # ------------------------------------------------------------------

    def _preload_all(self, n_workers: int = 8) -> None:
        """
        Load all mask + feature-map arrays into RAM using a thread pool.

        Uses threads (not processes) because numpy/PIL I/O releases the GIL,
        so multi-threading achieves full disk parallelism without the overhead
        of multiprocessing.  All loaded arrays are set as numpy read-only to
        avoid accidental copies when forked into DataLoader worker processes
        (Linux copy-on-write semantics).
        """
        n = len(self._meta)
        # Estimate memory footprint (features stored as float16 → halved)
        sample_feat_mb = 256 * 168 * 168 * 2 / 1024**2   # ~14 MB (float16)
        sample_mask_mb = 2700 * 2700 * 4 / 1024**2        # ~28 MB (float32)
        est_gb = n * (sample_feat_mb + sample_mask_mb) / 1024
        print(f"[MMRouteDataset] Pre-loading {n} regions into RAM "
              f"(estimated ~{est_gb:.1f} GB) using {n_workers} threads …")

        # Check available memory
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemAvailable" in line:
                        avail_gb = int(line.split()[1]) / 1024**2
                        break
            if est_gb > avail_gb * 0.8:
                print(f"[MMRouteDataset] WARNING: estimated {est_gb:.1f} GB exceeds "
                      f"80% of available {avail_gb:.1f} GB — skipping pre-load.")
                return
        except Exception:
            pass

        loaded = [0]
        lock = threading.Lock()

        def _load_one(idx: int) -> None:
            meta = self._meta[idx]
            # Load mask (always needed)
            arr = _load_road_mask(meta["png"])
            arr.flags.writeable = False
            meta["_mask"] = arr
            # Also load thin mask for dual-target training if available
            if meta["png_thin"] is not None:
                arr_thin = _load_road_mask(meta["png_thin"])
                arr_thin.flags.writeable = False
                meta["_mask_thin"] = arr_thin
            # Load encoder feature (when available). Store as float16 in RAM to
            # halve memory bandwidth; the model casts to its compute dtype anyway.
            if meta["feat"] is not None:
                feat = np.load(meta["feat"]).astype(np.float16)
                feat.flags.writeable = False
                meta["_feat"] = feat
            with lock:
                loaded[0] += 1
                done = loaded[0]
            if done % 50 == 0 or done == n:
                print(f"[MMRouteDataset]   {done}/{n} regions loaded …")

        threads = []
        chunk = max(1, (n + n_workers - 1) // n_workers)
        for w in range(n_workers):
            start = w * chunk
            end   = min(start + chunk, n)
            if start >= n:
                break
            t = threading.Thread(target=lambda s=start, e=end: [_load_one(i) for i in range(s, e)], daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        print(f"[MMRouteDataset] Pre-load complete. "
              f"{n} regions in RAM ({n_workers} threads).")

    # ------------------------------------------------------------------
    # Lazy loading helpers (fall back when preload_to_ram=False)
    # ------------------------------------------------------------------

    def _get_rgb(self, idx: int) -> np.ndarray:
        meta = self._meta[idx]
        if meta["_rgb"] is None:
            meta["_rgb"] = _load_rgb_from_tif(meta["tif"])
        return meta["_rgb"]

    def _get_mask(self, idx: int) -> np.ndarray:
        meta = self._meta[idx]
        if meta["_mask"] is None:
            meta["_mask"] = _load_road_mask(meta["png"])
        return meta["_mask"]

    def _get_mask_thin(self, idx: int) -> "np.ndarray | None":
        meta = self._meta[idx]
        if meta.get("png_thin") is None:
            return None
        if meta["_mask_thin"] is None:
            meta["_mask_thin"] = _load_road_mask(meta["png_thin"])
        return meta["_mask_thin"]

    def _get_feat(self, idx: int) -> "np.ndarray | None":
        """Return pre-computed SAM encoder feature map (C, H_feat, W_feat) or None.

        Uses np.memmap so the OS only reads the pages actually accessed, reducing
        disk I/O from ~28MB (full map) to ~800KB (32×32 crop) per sample.
        """
        meta = self._meta[idx]
        if meta["feat"] is None:
            return None
        if meta["_feat"] is None:
            # Use mmap_mode='r' so the OS only reads the pages we actually access.
            # For a 32×32 crop from a (256,168,168) map, this reads ~800KB instead
            # of the full 28MB, dramatically reducing disk I/O per sample.
            meta["_feat"] = np.load(meta["feat"], mmap_mode='r')
        return meta["_feat"]

    def _get_nodes(self, idx: int):
        meta = self._meta[idx]
        if meta["_coords"] is None and meta["npz"] is not None:
            coords, udist, H, W = _load_npz_nodes(meta["npz"])
            meta["_coords"] = coords
            meta["_udist"]  = udist
            meta["_H"]      = H
            meta["_W"]      = W
        return (
            meta["_coords"], meta["_udist"],
            meta["_H"],      meta["_W"],
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, flat_idx: int) -> Dict[str, Any]:
        region_idx = flat_idx % len(self.regions)

        # When cached encoder features are available, skip loading the large TIF
        # (rgb is only needed by the encoder, which is bypassed).  We use the
        # image dimensions from the mask instead.
        feat_full = self._get_feat(region_idx)
        has_feat  = feat_full is not None

        mask_full = self._get_mask(region_idx)    # [H_full, W_full]
        H_full, W_full = mask_full.shape[:2]

        if not has_feat:
            rgb_full = self._get_rgb(region_idx)  # [H_full, W_full, 3]
        else:
            rgb_full = None  # will not be used

        ps = self.patch_size

        # Anchor-centered crop when include_dist=True and nodes are available;
        # otherwise fall back to random crop (no dist supervision).
        anchor_idx: "int | None" = None
        case_idx:   "int | None" = None
        max_y = max(0, H_full - ps)
        max_x = max(0, W_full - ps)

        if self.include_dist:
            anchor_result = self._pick_anchor(region_idx, H_full, W_full)
            if anchor_result is not None:
                anchor_idx, case_idx, anchor_py, anchor_px = anchor_result
                y0 = max(0, min(anchor_py - ps // 2, max_y))
                x0 = max(0, min(anchor_px - ps // 2, max_x))
            else:
                y0 = random.randint(0, max_y)
                x0 = random.randint(0, max_x)
        else:
            y0 = random.randint(0, max_y)
            x0 = random.randint(0, max_x)

        y1 = min(y0 + ps, H_full)
        x1 = min(x0 + ps, W_full)

        # Crop mask — single contiguous copy (no redundant .copy() on flips later)
        mask_patch = np.ascontiguousarray(mask_full[y0:y1, x0:x1])

        pad_y = ps - mask_patch.shape[0]
        pad_x = ps - mask_patch.shape[1]
        if pad_y > 0 or pad_x > 0:
            mask_patch = np.pad(mask_patch, ((0, pad_y), (0, pad_x)))

        # Thin (original) mask for dual-target training
        mask_thin_full = self._get_mask_thin(region_idx)
        mask_thin_patch = None
        if mask_thin_full is not None:
            mask_thin_patch = np.ascontiguousarray(mask_thin_full[y0:y1, x0:x1])
            if pad_y > 0 or pad_x > 0:
                mask_thin_patch = np.pad(mask_thin_patch, ((0, pad_y), (0, pad_x)))

        if rgb_full is not None:
            rgb_patch = np.ascontiguousarray(rgb_full[y0:y1, x0:x1])
            if pad_y > 0 or pad_x > 0:
                rgb_patch = np.pad(rgb_patch, ((0, pad_y), (0, pad_x), (0, 0)))
        else:
            rgb_patch = None

        # Convert to tensors first — flips will be done on torch side (zero-copy view)
        mask_t = torch.from_numpy(mask_patch)
        mask_thin_t = torch.from_numpy(mask_thin_patch) if mask_thin_patch is not None else None
        rgb_t  = torch.from_numpy(rgb_patch.astype(np.float32)) if rgb_patch is not None else None

        flip_h = self.augment and (random.random() > 0.5)
        flip_v = self.augment and (random.random() > 0.5)
        if flip_h:
            mask_t = mask_t.flip(-1)            # flip W in [H, W]
            if mask_thin_t is not None:
                mask_thin_t = mask_thin_t.flip(-1)
            if rgb_t is not None:
                rgb_t = rgb_t.flip(1)            # flip W in [H, W, 3]
        if flip_v:
            mask_t = mask_t.flip(-2)            # flip H in [H, W]
            if mask_thin_t is not None:
                mask_thin_t = mask_thin_t.flip(-2)
            if rgb_t is not None:
                rgb_t = rgb_t.flip(0)            # flip H in [H, W, 3]

        sample: Dict[str, Any] = {"road_mask": mask_t}
        if mask_thin_t is not None:
            sample["road_mask_thin"] = mask_thin_t
        if rgb_t is not None:
            sample["rgb"] = rgb_t

        # Pre-computed encoder feature crop (stride-16 spatial resolution).
        if has_feat:
            stride    = 16
            feat_size = ps // stride               # 32 for patch_size=512
            fy0       = y0 // stride
            fx0       = x0 // stride
            C, FH, FW = feat_full.shape
            fy1 = min(fy0 + feat_size, FH)
            fx1 = min(fx0 + feat_size, FW)
            feat_crop = np.ascontiguousarray(feat_full[:, fy0:fy1, fx0:fx1])
            pad_fy = feat_size - feat_crop.shape[1]
            pad_fx = feat_size - feat_crop.shape[2]
            if pad_fy > 0 or pad_fx > 0:
                feat_crop = np.pad(feat_crop, ((0, 0), (0, pad_fy), (0, pad_fx)))
            feat_t = torch.from_numpy(feat_crop)   # [C, Hf, Wf]
            if flip_h:
                feat_t = feat_t.flip(-1)            # W in CHW
            if flip_v:
                feat_t = feat_t.flip(-2)            # H in CHW
            sample["encoder_feat"] = feat_t

        # Optional: distance supervision (anchor-centered, K targets per sample)
        if self.include_dist and anchor_idx is not None:
            pair = self._sample_anchor_targets(
                region_idx, anchor_idx, case_idx, y0, x0, y1, x1, H_full, W_full)
            if pair is not None:
                src_yx, tgt_yx_k, gt_dist_k = pair
                # Apply the same flips used on the image to the coordinates.
                # src_yx / tgt_yx_k are [y, x] in patch space [0, ps-1].
                # flip_h mirrors W:  x_new = ps - 1 - x
                # flip_v mirrors H:  y_new = ps - 1 - y
                valid_k = gt_dist_k > 0   # [K] numpy bool mask for valid targets
                if flip_h:
                    src_yx[1] = ps - 1 - src_yx[1]
                    tgt_yx_k[valid_k, 1] = ps - 1 - tgt_yx_k[valid_k, 1]
                if flip_v:
                    src_yx[0] = ps - 1 - src_yx[0]
                    tgt_yx_k[valid_k, 0] = ps - 1 - tgt_yx_k[valid_k, 0]
                sample["src_yx"]  = torch.tensor(src_yx,    dtype=torch.long)
                sample["tgt_yx"]  = torch.tensor(tgt_yx_k,  dtype=torch.long)    # [K, 2]
                sample["gt_dist"] = torch.tensor(gt_dist_k,  dtype=torch.float32) # [K]

        return sample

    # Max allowed ratio of GT road distance to Euclidean distance.
    # Pairs exceeding this are likely to have shortest paths that exit the
    # training ROI, creating an unresolvable pred > gt bias.
    _MAX_GT_DETOUR_RATIO = 3.0
    # GT road distance must not exceed this multiple of patch_size.
    _MAX_GT_PATCH_RATIO = 1.5

    def _pick_anchor(
        self,
        region_idx: int,
        H_full: int,
        W_full: int,
    ) -> "Optional[Tuple[int, int, int, int]]":
        """Pick a random node as anchor for anchor-centered cropping.

        Guarantees that the chosen anchor has at least self.min_in_patch
        other nodes inside the PATCH_SIZE×PATCH_SIZE window centered on it,
        so that the distance branch always gets ≥ min_in_patch supervision
        signals per sample. Falls back to a random anchor after MAX_TRIES
        attempts (avoids stalling on sparse regions).

        Returns (anchor_idx, case_idx, anchor_py, anchor_px) or None if no
        node data is available for this region.
        """
        coords, udist, H_npz, W_npz = self._get_nodes(region_idx)
        if coords is None:
            return None

        n_cases, N, _ = coords.shape
        case_idx   = random.randint(0, n_cases - 1)
        xy_norm    = coords[case_idx]   # (N, 2) (x_norm, y_norm) bottom-left
        ps         = self.patch_size
        min_nb     = self.min_in_patch

        # Pre-compute all node pixel coords for quick in-patch counting
        all_px = np.round(xy_norm[:, 0] * W_full).astype(int)
        all_py = np.round((1.0 - xy_norm[:, 1]) * H_full).astype(int)

        MAX_TRIES = max(N * 3, 30)
        candidates = list(range(N))
        random.shuffle(candidates)

        for attempt in range(MAX_TRIES):
            anchor_idx = candidates[attempt % N]
            apx = all_px[anchor_idx]
            apy = all_py[anchor_idx]

            # Patch bounds (same clamping as __getitem__)
            max_y = max(0, H_full - ps); max_x = max(0, W_full - ps)
            py0   = max(0, min(apy - ps // 2, max_y))
            px0   = max(0, min(apx - ps // 2, max_x))
            py1   = py0 + ps; px1 = px0 + ps

            # Count other nodes inside this patch
            in_patch = (
                (all_px >= px0) & (all_px < px1) &
                (all_py >= py0) & (all_py < py1)
            )
            in_patch[anchor_idx] = False   # exclude self
            if in_patch.sum() >= min_nb:
                return anchor_idx, case_idx, apy, apx

        # Fallback: return a random anchor without the count guarantee
        anchor_idx = random.randint(0, N - 1)
        return anchor_idx, case_idx, int(all_py[anchor_idx]), int(all_px[anchor_idx])

    def _sample_anchor_targets(
        self,
        region_idx: int,
        anchor_idx: int,
        case_idx: int,
        y0: int, x0: int, y1: int, x1: int,
        H_full: int, W_full: int,
    ) -> "Optional[Tuple[List[int], np.ndarray, np.ndarray]]":
        """Sample up to K nearest-neighbor targets for the given anchor node.

        All candidate targets must lie inside the patch window [y0:y1, x0:x1]
        and pass the same filtering criteria as the old _sample_node_pair.
        Targets are sorted by Euclidean distance to the anchor so that
        the most relevant (closest) neighbors are supervised first.

        Returns:
            (src_yx, tgt_yx_k, gt_dist_k) where
              src_yx   : list [y, x]  anchor coords in patch space
              tgt_yx_k : np.ndarray [K, 2]  target coords in patch space
                         (-1, -1) for padded (invalid) entries
              gt_dist_k: np.ndarray [K]  GT road distances in pixels
                         -1.0 for padded (invalid) entries
            or None if the anchor itself is outside the patch (shouldn't happen
            with anchor-centered crop, but guarded for safety).
        """
        coords, udist, H_npz, W_npz = self._get_nodes(region_idx)
        if coords is None:
            return None

        n_cases, N, _ = coords.shape
        xy_norm = coords[case_idx]               # (N, 2)

        px_x = xy_norm[:, 0] * W_full
        px_y = (1.0 - xy_norm[:, 1]) * H_full
        px   = np.stack([px_x, px_y], axis=1)   # (N, 2)  (x, y) top-left pixel

        # Anchor in patch-relative coords
        anc_x = px[anchor_idx, 0]
        anc_y = px[anchor_idx, 1]
        ps    = self.patch_size

        src_y = max(0, min(ps - 1, int(round(anc_y)) - y0))
        src_x = max(0, min(ps - 1, int(round(anc_x)) - x0))
        src_yx = [src_y, src_x]

        # Find other nodes inside the patch window (anchor excluded)
        inside = (
            (px[:, 0] >= x0) & (px[:, 0] < x1) &
            (px[:, 1] >= y0) & (px[:, 1] < y1)
        )
        inside_idx = np.where(inside)[0]
        inside_idx = inside_idx[inside_idx != anchor_idx]

        gt_cap     = ps * self._MAX_GT_PATCH_RATIO
        detour_cap = self._MAX_GT_DETOUR_RATIO

        # Sort by Euclidean distance to anchor (nearest first)
        if len(inside_idx) > 0:
            euclid = np.sqrt(
                (px[inside_idx, 0] - anc_x) ** 2
                + (px[inside_idx, 1] - anc_y) ** 2
            )
            inside_idx = inside_idx[np.argsort(euclid)]

        # Collect up to K valid targets
        K       = self.k_targets
        tgt_list: List[List[int]] = []
        dist_list: List[float]    = []

        for ni in inside_idx:
            if len(tgt_list) >= K:
                break
            d_norm = float(udist[case_idx, anchor_idx, ni])
            if d_norm <= 0 or not np.isfinite(d_norm):
                continue
            gt_dist_px = d_norm * H_full
            if gt_dist_px > gt_cap:
                continue
            euclid_px = float(np.sqrt(
                (px[anchor_idx, 0] - px[ni, 0]) ** 2
                + (px[anchor_idx, 1] - px[ni, 1]) ** 2
            ))
            if euclid_px > 1.0 and gt_dist_px / euclid_px > detour_cap:
                continue
            tgt_y = max(0, min(ps - 1, int(round(px[ni, 1])) - y0))
            tgt_x = max(0, min(ps - 1, int(round(px[ni, 0])) - x0))
            tgt_list.append([tgt_y, tgt_x])
            dist_list.append(gt_dist_px)

        if not tgt_list:
            return None

        # Pad to exactly K entries with sentinel values
        while len(tgt_list) < K:
            tgt_list.append([-1, -1])
            dist_list.append(-1.0)

        tgt_yx_k  = np.array(tgt_list,  dtype=np.int64)    # [K, 2]
        gt_dist_k = np.array(dist_list, dtype=np.float32)  # [K]
        return src_yx, tgt_yx_k, gt_dist_k

    def _sample_node_pair(
        self,
        region_idx: int,
        y0: int, x0: int, y1: int, x1: int,
        H_full: int, W_full: int,
    ) -> Optional[Tuple[List[int], List[int], float]]:
        """
        Sample one (src, tgt) pair whose pixel coords both fall inside the crop
        window [y0:y1, x0:x1] **and** whose GT road distance is compatible with
        the ROI-based Eikonal solver used during training.

        Filtering criteria (reject pair if ANY triggers):
          1. GT road distance > patch_size * _MAX_GT_PATCH_RATIO
             → path almost certainly exits the ROI, Eikonal cannot reproduce it
          2. GT / Euclidean > _MAX_GT_DETOUR_RATIO
             → road detours heavily, high chance the shortest path leaves ROI

        Returns (src_yx_in_patch, tgt_yx_in_patch, gt_dist_pixels) or None.

        Node coords in NPZ: (x_norm, y_norm) with bottom-left origin.
        Pixel conversion:
            px_x = x_norm * W_full
            px_y = (1 - y_norm) * H_full
        """
        coords, udist, H_npz, W_npz = self._get_nodes(region_idx)
        if coords is None:
            return None

        # coords: (n_cases, N, 2)
        n_cases, N, _ = coords.shape

        # Pick a random case
        case_idx = random.randint(0, n_cases - 1)
        xy_norm = coords[case_idx]  # (N, 2)  (x, y) bottom-left

        # Convert to pixel coords in the full image
        px_x = xy_norm[:, 0] * W_full
        px_y = (1.0 - xy_norm[:, 1]) * H_full
        px   = np.stack([px_x, px_y], axis=1)  # (N, 2)  (x, y) top-left pixel

        # Find nodes inside crop window
        inside = (
            (px[:, 0] >= x0) & (px[:, 0] < x1) &
            (px[:, 1] >= y0) & (px[:, 1] < y1)
        )
        inside_idx = np.where(inside)[0]
        if len(inside_idx) < 2:
            return None

        ps = self.patch_size
        gt_cap = ps * self._MAX_GT_PATCH_RATIO
        detour_cap = self._MAX_GT_DETOUR_RATIO

        # Pick two distinct inside nodes with non-zero, finite GT dist
        inside_idx = inside_idx.copy()
        np.random.shuffle(inside_idx)
        for i in range(len(inside_idx)):
            for j in range(i + 1, len(inside_idx)):
                ni, nj = inside_idx[i], inside_idx[j]
                d_norm = float(udist[case_idx, ni, nj])
                if d_norm <= 0 or not np.isfinite(d_norm):
                    continue

                # undirected_dist_norm = d_meters / meta_distance_ref_m (bbox_max)
                # For square images: d_norm * H_full = d_meters * pixels_per_meter
                gt_dist_px = d_norm * H_full

                if gt_dist_px > gt_cap:
                    continue

                euclid_px = np.sqrt(
                    (px[ni, 0] - px[nj, 0]) ** 2
                    + (px[ni, 1] - px[nj, 1]) ** 2
                )
                if euclid_px > 1.0 and gt_dist_px / euclid_px > detour_cap:
                    continue

                src_yx_patch = [
                    int(round(px[ni, 1])) - y0,
                    int(round(px[ni, 0])) - x0,
                ]
                tgt_yx_patch = [
                    int(round(px[nj, 1])) - y0,
                    int(round(px[nj, 0])) - x0,
                ]
                src_yx_patch = [
                    max(0, min(ps - 1, src_yx_patch[0])),
                    max(0, min(ps - 1, src_yx_patch[1])),
                ]
                tgt_yx_patch = [
                    max(0, min(ps - 1, tgt_yx_patch[0])),
                    max(0, min(ps - 1, tgt_yx_patch[1])),
                ]
                return src_yx_patch, tgt_yx_patch, gt_dist_px
        return None


# ---------------------------------------------------------------------------
# Threaded prefetch loader — eliminates IPC overhead when data is in RAM
# ---------------------------------------------------------------------------

class ThreadedLoader:
    """
    A DataLoader replacement that uses Python threads instead of processes.

    When all data is already pre-loaded into the parent process's RAM, the
    standard multiprocessing DataLoader is suboptimal because:
      - Worker processes send batches back to the main process via IPC Queue,
        which requires copying tensors through shared memory (~10ms per batch)

    This loader uses `num_workers` daemon threads instead.  Threads share the
    parent's address space directly, so tensors are passed *by reference* —
    zero-copy, zero-serialisation.

    Architecture:
      - `num_workers` threads each pull from a shared index queue and place
        completed batches into a result queue (bounded by num_prefetch).
      - The main thread reads from the result queue and yields to PyTorch Lightning.

    Interface: compatible with PyTorch Lightning (iter, len, __len__).
    """

    def __init__(
        self,
        dataset: MMRouteDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 2,
        num_prefetch: int = 4,
        drop_last: bool = False,
        pin_memory: bool = True,
        collate_fn=None,
    ):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.num_workers  = max(1, num_workers)
        self.num_prefetch = num_prefetch
        self.drop_last    = drop_last
        self.pin_memory   = pin_memory and torch.cuda.is_available()
        self._collate     = collate_fn or _default_collate

        n = len(dataset)
        if drop_last:
            self._n_batches = n // batch_size
        else:
            self._n_batches = (n + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            random.shuffle(indices)
        if self.drop_last:
            indices = indices[: self._n_batches * self.batch_size]

        # Build per-batch index lists
        batch_list = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

        # Shared queues
        idx_q: "queue.Queue[list | None]" = queue.Queue()
        out_q: "queue.Queue[Dict | None]" = queue.Queue(maxsize=self.num_prefetch)

        # Populate index queue
        for b in batch_list:
            idx_q.put(b)
        for _ in range(self.num_workers):
            idx_q.put(None)  # sentinel per worker

        finished = threading.Semaphore(0)

        def _worker():
            while True:
                batch_idx = idx_q.get()
                if batch_idx is None:
                    finished.release()
                    return
                items = [self.dataset[i] for i in batch_idx]
                collated = self._collate(items)
                if self.pin_memory:
                    collated = {
                        k: v.pin_memory() if torch.is_tensor(v) else v
                        for k, v in collated.items()
                    }
                out_q.put(collated)

        workers = [
            threading.Thread(target=_worker, daemon=True)
            for _ in range(self.num_workers)
        ]
        for w in workers:
            w.start()

        yielded = 0
        while yielded < len(batch_list):
            try:
                item = out_q.get(timeout=60)
                yield item
                yielded += 1
            except queue.Empty:
                pass

        for w in workers:
            w.join()


def _default_collate(batch):
    """Collate a list of dicts into a dict of stacked tensors."""
    keys = {k for s in batch for k in s}
    out = {}
    for k in keys:
        vals = [s[k] for s in batch if k in s]
        if len(vals) == len(batch):
            out[k] = torch.stack(vals, dim=0)
    return out


# ---------------------------------------------------------------------------
# Convenience DataLoader builders
# ---------------------------------------------------------------------------

def build_dataloaders(
    root_dir: str,
    patch_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    include_dist: bool = False,
    npz_variant: str = "p20",
    val_fraction: float = 0.1,
    samples_per_region: int = 50,
    seed: int = 42,
    use_cached_features: bool = True,
    preload_to_ram: bool = True,
    preload_workers: int = 8,
    k_targets: int = 4,
    road_dilation_radius: int = 0,
    min_in_patch: int = 2,
) -> Tuple["DataLoader | ThreadedLoader", "DataLoader | ThreadedLoader"]:
    """Build train and val DataLoaders for SAMRoute training.

    When preload_to_ram=True, returns ThreadedLoader instances instead of
    standard DataLoaders.  ThreadedLoader uses threads (not processes) to
    prefetch batches, eliminating the ~11ms IPC overhead per batch that
    the multiprocessing DataLoader incurs even when data is already in RAM.

    road_dilation_radius: if > 0, uses pre-computed normalized road masks
        (roadnet_normalized_r{radius}.png). Run precompute_normalized_masks.py first.
    """
    train_ds = MMRouteDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        include_dist=include_dist,
        npz_variant=npz_variant,
        split="train",
        val_fraction=val_fraction,
        samples_per_region=samples_per_region,
        augment=True,
        seed=seed,
        use_cached_features=use_cached_features,
        preload_to_ram=preload_to_ram,
        preload_workers=preload_workers,
        k_targets=k_targets,
        road_dilation_radius=road_dilation_radius,
        min_in_patch=min_in_patch,
    )
    val_ds = MMRouteDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        include_dist=include_dist,
        npz_variant=npz_variant,
        split="val",
        val_fraction=val_fraction,
        samples_per_region=samples_per_region,
        augment=False,
        seed=seed,
        use_cached_features=use_cached_features,
        preload_to_ram=preload_to_ram,
        preload_workers=preload_workers,
        k_targets=k_targets,
        road_dilation_radius=road_dilation_radius,
        min_in_patch=min_in_patch,
    )

    if preload_to_ram:
        # Use thread-based prefetch: no IPC overhead since data is in shared RAM
        train_loader = ThreadedLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, num_prefetch=num_workers * 2,
            drop_last=True, pin_memory=True,
        )
        val_loader = ThreadedLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=max(1, num_workers // 2), num_prefetch=4,
            drop_last=False, pin_memory=True,
        )
        print(f"[build_dataloaders] Using ThreadedLoader (no IPC, {num_workers} workers). "
              f"train={len(train_loader)} batches, val={len(val_loader)} batches.")
    else:
        def _collate(batch):
            keys_all = {k for s in batch for k in s}
            out = {}
            for k in keys_all:
                vals = [s[k] for s in batch if k in s]
                if len(vals) == len(batch):
                    out[k] = torch.stack(vals, dim=0)
            return out

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            collate_fn=_collate, persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            collate_fn=_collate, persistent_workers=(num_workers > 0),
        )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else (
        os.path.join(os.path.dirname(__file__), "../Gen_dataset_V2/Gen_dataset")
    )
    print(f"Testing MMRouteDataset on: {root}")

    ds = MMRouteDataset(
        root_dir=root,
        patch_size=512,
        include_dist=True,
        npz_variant="p20",
        split="train",
        val_fraction=0.1,
        samples_per_region=5,
    )
    val_ds = MMRouteDataset(
        root_dir=root,
        patch_size=512,
        include_dist=True,
        npz_variant="p20",
        split="val",
        val_fraction=0.1,
        samples_per_region=5,
    )
    print(f"Train regions: {len(ds.regions)}   samples: {len(ds)}")
    print(f"Val   regions: {len(val_ds.regions)}   samples: {len(val_ds)}")

    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype} "
              f"min={v.float().min():.2f} max={v.float().max():.2f}")


# ---------------------------------------------------------------------------
# Node loading from NPZ (merged from load_nodes_from_npz.py)
# ---------------------------------------------------------------------------

def _normalize_prob_map_npz(x: torch.Tensor) -> torch.Tensor:
    """Normalize probability map to [0, 1]."""
    x = x.float()
    mx = float(x.max().item()) if x.numel() else 0.0
    if mx > 1.5:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def snap_nodes_vectorized(
    nodes_yx: torch.Tensor,
    road_map: torch.Tensor,
    threshold: float = 0.3,
    win: int = 30,
) -> torch.Tensor:
    """Snap nodes to nearby high-probability road pixels (vectorized)."""
    N = nodes_yx.shape[0]
    H, W = road_map.shape
    device = nodes_yx.device
    if N == 0:
        return nodes_yx

    current_probs = road_map[nodes_yx[:, 0], nodes_yx[:, 1]]
    mask_good = current_probs > threshold
    if mask_good.all():
        return nodes_yx

    range_t = torch.arange(-win, win + 1, device=device)
    dy, dx = torch.meshgrid(range_t, range_t, indexing='ij')
    offsets = torch.stack([dy.flatten(), dx.flatten()], dim=1)

    sample_coords = nodes_yx.unsqueeze(1) + offsets.unsqueeze(0)
    sample_coords[..., 0].clamp_(0, H - 1)
    sample_coords[..., 1].clamp_(0, W - 1)

    candidate_probs = road_map[sample_coords[..., 0], sample_coords[..., 1]]
    max_vals, max_indices = candidate_probs.max(dim=1)
    best_candidates = sample_coords[torch.arange(N, device=device), max_indices]

    mask_improve = (~mask_good) & (max_vals > threshold)
    final_nodes = nodes_yx.clone()
    if mask_improve.any():
        final_nodes[mask_improve] = best_candidates[mask_improve]
    return final_nodes


def load_nodes_from_npz(
    tif_path: str,
    road_prob: torch.Tensor,
    *,
    p_count: int = 20,
    key: str = "matched_node_norm",
    col_order: str = "xy",
    snap: bool = True,
    snap_threshold: float = 0.1,
    snap_win: int = 10,
    verbose: bool = True,
    case_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load ground-truth road-network nodes from a distance NPZ file.

    Automatically constructs the NPZ path from the TIF filename.
    Converts normalized bottom-left coordinates to pixel top-left (y, x).
    Optionally snaps nodes to nearby high-probability road pixels.
    """
    road = _normalize_prob_map_npz(road_prob)
    H, W = road.shape[0], road.shape[1]

    tif_p = Path(tif_path)
    tif_name = tif_p.name
    location_key = tif_name.replace("crop_", "").replace("_z16.tif", "").replace(".tif", "")

    npz_name = f"distance_dataset_all_{location_key}_p{p_count}.npz"
    npz_p = tif_p.parent / npz_name

    if not npz_p.exists():
        raise FileNotFoundError(f"[NPZ] Missing expected file: {npz_p}")

    with np.load(npz_p, allow_pickle=True) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' not found in {npz_p}")
        nodes = np.asarray(z[key])
        meta_sat_height_px = int(z["meta_sat_height_px"])
        meta_sat_width_px = int(z["meta_sat_width_px"])
        euclidean_dist_norm = z["euclidean_dist_norm"]
        undirected_dist_norm = z["undirected_dist_norm"]

    if nodes.ndim == 3:
        if case_idx is None:
            case_idx = np.random.randint(0, nodes.shape[0])
        nodes = nodes[case_idx]

    if nodes.size == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=road_prob.device), {"N": 0}

    x_norm, y_norm = nodes[:, 0], nodes[:, 1]
    x_pix = np.rint(x_norm * (W - 1)).astype(np.int64)
    x_pix = np.clip(x_pix, 0, W - 1)
    y_pix = np.rint((1.0 - y_norm) * (H - 1)).astype(np.int64)
    y_pix = np.clip(y_pix, 0, H - 1)

    nodes_yx = torch.stack([
        torch.from_numpy(y_pix),
        torch.from_numpy(x_pix),
    ], dim=1).to(road_prob.device, dtype=torch.long)

    if snap:
        nodes_yx = snap_nodes_vectorized(nodes_yx, road, snap_threshold, snap_win)

    if verbose:
        print(f"[NPZ] Selected: {npz_name} (N={len(nodes_yx)}, p={p_count})")
        print(f"[META] Height: {meta_sat_height_px}, Width: {meta_sat_width_px}")

    info_dict = {
        "npz_path": npz_p,
        "p_count": p_count,
        "N": len(nodes_yx),
        "meta_sat_height_px": meta_sat_height_px,
        "meta_sat_width_px": meta_sat_width_px,
        "euclidean_dist_norm": euclidean_dist_norm,
        "undirected_dist_norm": undirected_dist_norm,
        "case_idx": case_idx,
    }
    return nodes_yx, info_dict
