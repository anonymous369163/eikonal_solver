import torch
import numpy as np
import os
import glob
import random
from collections import OrderedDict
 
from PIL import Image 

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


##########################################################################################
# Region name to coordinate mapping (same as GIMF-P)
##########################################################################################
REGION_MAPPING = {
    'haikou': '19.940688_110.276704',
    'nanning': '22.823258_108.374618',
    'guangzhou': '23.203150_113.352068',
    'kunming': '24.984335_102.683337',
    'fuzhou': '26.062311_119.304663',
    'guiyang': '26.592264_106.671947',
    'changsha': '28.196750_112.977367',
    'chongqing': '29.558731_106.522964',
    'lasa': '29.657377_91.117333',
    'hangzhou': '30.256330_120.159448',
    'wuhan': '30.596069_114.297691',
    'chengdu': '30.657398_104.065859',
    'shanghai': '31.240186_121.496062',
    'hefei': '31.844155_117.280057',
    'xian': '34.269888_108.947180',
    'zhengzhou': '34.746600_113.625300',
    'lanzhou': '36.085171_103.709368',
    'xining': '36.628353_101.765799',
    'jinan': '36.675109_117.022236',
    'taiyuan': '37.849791_112.548392',
    'shijiazhuang': '38.042805_114.514893',
    'yinchuan': '38.484236_106.226103',
    'tianjin': '39.132612_117.199202',
    'beijing': '39.906357_116.391299',
    'hohhot': '40.801862_111.680011',
    'shenyang': '41.796906_123.410659',
    'urumqi': '43.858585_87.577676',
    'changchun': '43.887200_125.320755',
    'harbin': '45.750000_126.633331',
}

COORD_TO_NAME = {v: k for k, v in REGION_MAPPING.items()}


def get_all_regions_from_dataset(dataset_root):
    """Scan dataset directory to get all available region coordinates."""
    if not os.path.exists(dataset_root):
        return []
    regions = []
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path) and '_' in item and item.replace('.', '').replace('_', '').replace('-', '').isdigit():
            regions.append(item)
    return sorted(regions)


def get_selected_regions(regions_arg, dataset_root=None):
    """Parse regions argument and return list of region coordinates."""
    regions_arg = regions_arg.lower().strip()
    
    if regions_arg == 'all':
        if dataset_root and os.path.exists(dataset_root):
            scanned = get_all_regions_from_dataset(dataset_root)
            if scanned:
                print(f"[INFO] Auto-scanned {len(scanned)} regions from dataset directory")
                return scanned
        print(f"[INFO] Using REGION_MAPPING ({len(REGION_MAPPING)} regions)")
        return list(REGION_MAPPING.values())
    
    selected = []
    for region in regions_arg.split(','):
        region = region.strip()
        if region in REGION_MAPPING:
            selected.append(REGION_MAPPING[region])
        elif '_' in region and region.replace('.', '').replace('_', '').replace('-', '').isdigit():
            selected.append(region)
        else:
            raise ValueError(f"Unknown region: {region}. Available: {list(REGION_MAPPING.keys())} or 'all'")
    
    return selected


def build_datasets_from_regions(dataset_root, regions, folder_indices, npz_suffix='p20'):
    """
    Build DATASETS list by scanning specified regions and folders.
    Similar to GIMF-P but returns satellite_tif_path instead of satdino paths.
    """
    datasets = []
    
    for region_coord in regions:
        region_path = os.path.join(dataset_root, region_coord)
        
        if not os.path.exists(region_path):
            print(f"Warning: Region path does not exist: {region_path}")
            continue
        
        subfolders = sorted(glob.glob(os.path.join(region_path, '*_*_*_*')))
        
        for subfolder in subfolders:
            folder_name = os.path.basename(subfolder)
            
            try:
                folder_idx = int(folder_name.split('_')[0])
            except (ValueError, IndexError):
                continue
            
            if folder_idx not in folder_indices:
                continue
            
            # Find npz files
            npz_files = glob.glob(os.path.join(subfolder, f'distance_dataset_train_*_{npz_suffix}.npz'))
            if not npz_files:
                npz_files = glob.glob(os.path.join(subfolder, f'distance_dataset_all_*_{npz_suffix}.npz'))
            
            if not npz_files:
                print(f"Warning: Missing npz files in {subfolder}")
                continue
            
            # Find satellite tif file
            sat_tif_files = glob.glob(os.path.join(subfolder, 'crop_*.tif'))
            satellite_tif_path = sat_tif_files[0] if sat_tif_files else None
            
            parts = folder_name.split('_')
            if len(parts) >= 3:
                lat, lon = parts[1], parts[2]
                loc_name = f"loc_{lat}_{lon}"
            else:
                loc_name = folder_name
            
            region_name = COORD_TO_NAME.get(region_coord, region_coord[:10])
            
            datasets.append({
                'name': f'{region_name}_{loc_name}',
                'npz_path': npz_files[0],
                'satellite_tif_path': satellite_tif_path,
                'region': region_coord,
            })
    
    return datasets


def build_datasets_with_random_split(dataset_root, regions, split_seed=None, npz_suffix='p20'):
    """
    Build train and test datasets with random split per city.
    Each city independently and randomly selects 1 location as test, rest as train.
    """
    import random as rnd
    
    if split_seed is not None:
        rnd.seed(split_seed)
    
    train_datasets = []
    test_datasets = []
    split_info = {}
    
    for region_coord in regions:
        region_path = os.path.join(dataset_root, region_coord)
        region_name = COORD_TO_NAME.get(region_coord, region_coord[:10])
        
        if not os.path.exists(region_path):
            print(f"Warning: Region path does not exist: {region_path}")
            continue
        
        subfolders = sorted(glob.glob(os.path.join(region_path, '*_*_*_*')))
        
        valid_folders = []
        for subfolder in subfolders:
            folder_name = os.path.basename(subfolder)
            
            try:
                folder_idx = int(folder_name.split('_')[0])
            except (ValueError, IndexError):
                continue
            
            npz_files = glob.glob(os.path.join(subfolder, f'distance_dataset_train_*_{npz_suffix}.npz'))
            if not npz_files:
                npz_files = glob.glob(os.path.join(subfolder, f'distance_dataset_all_*_{npz_suffix}.npz'))
            
            if not npz_files:
                continue
            
            sat_tif_files = glob.glob(os.path.join(subfolder, 'crop_*.tif'))
            satellite_tif_path = sat_tif_files[0] if sat_tif_files else None
            
            parts = folder_name.split('_')
            if len(parts) >= 3:
                lat, lon = parts[1], parts[2]
                loc_name = f"loc_{lat}_{lon}"
            else:
                loc_name = folder_name
            
            valid_folders.append({
                'folder_idx': folder_idx,
                'name': f'{region_name}_{loc_name}',
                'npz_path': npz_files[0],
                'satellite_tif_path': satellite_tif_path,
                'region': region_coord,
            })
        
        if len(valid_folders) < 2:
            print(f"Warning: Region {region_name} has less than 2 valid folders, skipping")
            continue
        
        test_idx = rnd.randint(0, len(valid_folders) - 1)
        test_folder = valid_folders[test_idx]
        split_info[region_name] = test_folder['folder_idx']
        
        for i, folder in enumerate(valid_folders):
            if i == test_idx:
                test_datasets.append(folder)
            else:
                train_datasets.append(folder)
    
    return train_datasets, test_datasets, split_info


##########################################################################################
# Manifest-based dataset split (Train / Test1 / Test2)
##########################################################################################

def get_default_split_manifest_path(dataset_root: str, npz_suffix: str) -> str:
    """Default manifest path under dataset_root."""
    safe_suffix = str(npz_suffix).replace(os.sep, '_')
    return os.path.join(dataset_root, f"split_manifest_{safe_suffix}.json")


def _list_immediate_subdirs(parent_dir: str) -> List[str]:
    if not os.path.isdir(parent_dir):
        return []
    items = []
    for name in os.listdir(parent_dir):
        if name.startswith('.'):
            continue
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p):
            items.append(name)
    return sorted(items)


def _find_npz_file(subgraph_dir: str, npz_suffix: str) -> Optional[str]:
    """Find the first matching npz file for a given node-size suffix inside a subgraph folder."""
    patterns = [
        f"distance_dataset_train_*_{npz_suffix}.npz",
        f"distance_dataset_all_*_{npz_suffix}.npz",
    ]
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(subgraph_dir, pat)))
        if files:
            return files[0]
    return None


def _find_satellite_tif(subgraph_dir: str) -> Optional[str]:
    """Find satellite tif path (crop_*.tif) inside a subgraph folder."""
    tif_files = sorted(glob.glob(os.path.join(subgraph_dir, "crop_*.tif")))
    return tif_files[0] if tif_files else None


def _safe_relpath(path: Optional[str], start: str) -> Optional[str]:
    if path is None:
        return None
    try:
        return os.path.relpath(path, start)
    except Exception:
        return path


def _resolve_path(path: Optional[str], start: str) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(start, path))


def _get_npz_num_cases(npz_path: str) -> int:
    """Get number of cases (instances) in a npz file."""
    data = np.load(npz_path, allow_pickle=True)
    if 'matched_node_norm' not in data.files:
        raise KeyError(f"'matched_node_norm' not found in {npz_path}. Keys={list(data.files)}")
    return int(data['matched_node_norm'].shape[0])


def create_dataset_split_manifest(
    dataset_root: str,
    npz_suffix: str,
    manifest_path: Optional[str] = None,
    test1_subgraph_threshold: int = 10,
    test2_case_ratio: float = 0.2,
    seed: Optional[int] = 0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Create a manifest that freezes dataset split for:
      - Train: first (1 - test2_case_ratio) cases from every subgraph except test1 subgraphs
      - Test1: one randomly selected subgraph from cities with (#subgraphs > test1_subgraph_threshold)
      - Test2: remaining test2_case_ratio cases from the same subgraphs used by Train

    Notes:
      - City = each immediate folder under dataset_root
      - Subgraph = each immediate folder under city folder (no recursion)
      - Case split is **order-preserving** (no shuffle) and aligned across subgraphs using a global cutoff.
    """
    dataset_root = os.path.abspath(dataset_root)
    if manifest_path is None:
        manifest_path = get_default_split_manifest_path(dataset_root, npz_suffix)
    manifest_path = os.path.abspath(manifest_path)

    if logger is None:
        logger = logging.getLogger(__name__)

    if not (0.0 < float(test2_case_ratio) < 1.0):
        raise ValueError(f"test2_case_ratio must be in (0,1), got {test2_case_ratio}")

    rng = random.Random(seed) if seed is not None else random

    cities = _list_immediate_subdirs(dataset_root)
    if not cities:
        raise FileNotFoundError(f"No city folders found under dataset_root={dataset_root}")

    cities_info: Dict[str, Any] = {}
    shared_subgraph_case_counts: List[int] = []
    test1_cities: List[str] = []

    for city in cities:
        city_dir = os.path.join(dataset_root, city)
        subgraphs = _list_immediate_subdirs(city_dir)

        subgraph_entries: List[Dict[str, Any]] = []
        for sg in subgraphs:
            sg_dir = os.path.join(city_dir, sg)
            npz_path = _find_npz_file(sg_dir, npz_suffix=npz_suffix)
            if npz_path is None:
                continue

            sat_path = _find_satellite_tif(sg_dir)

            try:
                total_cases = _get_npz_num_cases(npz_path)
            except Exception as e:
                logger.warning(f"[Split Manifest] Failed to read npz cases: {npz_path} ({e}), skip this subgraph.")
                continue

            subgraph_entries.append({
                'subgraph': sg,
                'npz_path': _safe_relpath(npz_path, dataset_root),
                'satellite_tif_path': _safe_relpath(sat_path, dataset_root),
                'total_cases': int(total_cases),
            })

        if not subgraph_entries:
            # city has no valid subgraphs for this node-size
            continue

        # Sort subgraphs for stable manifest structure
        subgraph_entries = sorted(subgraph_entries, key=lambda x: x['subgraph'])

        test1_subgraph: Optional[str] = None
        if len(subgraph_entries) > int(test1_subgraph_threshold):
            test1_subgraph = rng.choice(subgraph_entries)['subgraph']
            test1_cities.append(city)

        # collect shared subgraph case counts (exclude test1 subgraph)
        for sg in subgraph_entries:
            if test1_subgraph is not None and sg['subgraph'] == test1_subgraph:
                continue
            shared_subgraph_case_counts.append(int(sg['total_cases']))

        cities_info[city] = {
            'num_subgraphs': len(subgraph_entries),
            'test1_subgraph': test1_subgraph,
            'subgraphs': subgraph_entries,
        }

    if not cities_info:
        raise RuntimeError(
            f"No valid city/subgraph found under {dataset_root} for npz_suffix={npz_suffix}. "
            f"Please check dataset structure and file patterns."
        )

    if not shared_subgraph_case_counts:
        raise RuntimeError(
            "No shared subgraphs found for Train/Test2 after applying Test1 selection. "
            "Try lowering test1_subgraph_threshold."
        )

    aligned_case_count = int(min(shared_subgraph_case_counts))
    max_case_count = int(max(shared_subgraph_case_counts))
    if max_case_count != aligned_case_count:
        logger.warning(
            f"[Split Manifest] Shared subgraphs have different #cases (min={aligned_case_count}, max={max_case_count}). "
            f"To keep case indices aligned across subgraphs, only the first {aligned_case_count} cases "
            f"will be used for Train/Test2 in every shared subgraph."
        )

    train_case_count = int(aligned_case_count * (1.0 - float(test2_case_ratio)))
    # guard
    train_case_count = max(0, min(train_case_count, aligned_case_count))

    manifest: Dict[str, Any] = {
        'manifest_version': 1,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_root': dataset_root,
        'npz_suffix': str(npz_suffix),
        'test1_subgraph_threshold': int(test1_subgraph_threshold),
        'test2_case_ratio': float(test2_case_ratio),
        'seed': seed,
        'aligned_case_count': aligned_case_count,
        'train_case_count': train_case_count,
        'test2_case_count': aligned_case_count - train_case_count,
        'test1_cities': sorted(test1_cities),
        'cities': cities_info,
    }

    # persist
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"[Split Manifest] Created manifest: {manifest_path}")
    logger.info(
        f"[Split Manifest] Train/Test2 aligned_case_count={aligned_case_count}, "
        f"train_case_count={train_case_count}, test2_case_count={aligned_case_count - train_case_count}"
    )
    if test1_cities:
        logger.info(f"[Split Manifest] Test1 cities ({len(test1_cities)}): {sorted(test1_cities)}")
    else:
        logger.info("[Split Manifest] No Test1 cities (no city has subgraphs > threshold).")

    return manifest


def load_dataset_split_manifest(manifest_path: str) -> Dict[str, Any]:
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    return manifest


def load_or_create_dataset_split_manifest(
    dataset_root: str,
    npz_suffix: str,
    manifest_path: Optional[str] = None,
    test1_subgraph_threshold: int = 10,
    test2_case_ratio: float = 0.2,
    mode: str = 'load',
    seed: Optional[int] = 0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    mode:
      - 'load'  : must exist, never modify (default)
      - 'create': create new manifest; error if already exists
      - 'update': overwrite existing manifest (re-create)
    """
    dataset_root = os.path.abspath(dataset_root)
    if manifest_path is None:
        manifest_path = get_default_split_manifest_path(dataset_root, npz_suffix)
    manifest_path = os.path.abspath(manifest_path)

    if logger is None:
        logger = logging.getLogger(__name__)

    mode = str(mode).lower().strip()
    if mode not in ('load', 'create', 'update'):
        raise ValueError(f"Unknown mode: {mode}. Use 'load' | 'create' | 'update'.")

    exists = os.path.exists(manifest_path)

    if mode == 'load':
        if not exists:
            raise FileNotFoundError(
                f"Split manifest not found: {manifest_path}\n"
                f"Set mode='create' to create it once, then keep mode='load' afterwards."
            )
        manifest = load_dataset_split_manifest(manifest_path)

        # check consistency
        if os.path.abspath(str(manifest.get('dataset_root', ''))) != dataset_root:
            logger.warning(
                f"[Split Manifest] dataset_root mismatch:\n"
                f"  manifest: {manifest.get('dataset_root')}\n"
                f"  current : {dataset_root}\n"
                f"Proceeding, but be careful if you moved the dataset folder."
            )

        if str(manifest.get('npz_suffix')) != str(npz_suffix):
            raise ValueError(
                f"npz_suffix mismatch between manifest and current config:\n"
                f"  manifest: {manifest.get('npz_suffix')}\n"
                f"  current : {npz_suffix}\n"
                f"Use mode='update' to regenerate manifest."
            )

        if int(manifest.get('test1_subgraph_threshold')) != int(test1_subgraph_threshold) or \
           abs(float(manifest.get('test2_case_ratio')) - float(test2_case_ratio)) > 1e-9:
            raise ValueError(
                "Split hyper-parameters mismatch between manifest and current config.\n"
                f"  manifest: threshold={manifest.get('test1_subgraph_threshold')}, ratio={manifest.get('test2_case_ratio')}\n"
                f"  current : threshold={test1_subgraph_threshold}, ratio={test2_case_ratio}\n"
                "Use mode='update' to regenerate manifest."
            )

        return manifest

    if mode == 'create' and exists:
        raise FileExistsError(
            f"Split manifest already exists: {manifest_path}\n"
            f"Use mode='load' to reuse it, or mode='update' to overwrite it."
        )

    # mode == 'create' (not exists) or mode == 'update'
    return create_dataset_split_manifest(
        dataset_root=dataset_root,
        npz_suffix=npz_suffix,
        manifest_path=manifest_path,
        test1_subgraph_threshold=test1_subgraph_threshold,
        test2_case_ratio=test2_case_ratio,
        seed=seed,
        logger=logger,
    )


def build_datasets_from_split_manifest(
    manifest: Dict[str, Any],
    dataset_root: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build (TRAIN_DATASETS, TEST1_DATASETS, TEST2_DATASETS) from manifest.

    - TRAIN_DATASETS and TEST2_DATASETS share the same subgraph folders (and thus same images),
      but use different case_range.
    - TEST1_DATASETS uses different subgraph folders (images do NOT overlap with train).
    """
    if dataset_root is None:
        dataset_root = str(manifest.get('dataset_root', ''))
    dataset_root = os.path.abspath(dataset_root)

    aligned_case_count = int(manifest['aligned_case_count'])
    train_case_count = int(manifest['train_case_count'])

    train_datasets: List[Dict[str, Any]] = []
    test1_datasets: List[Dict[str, Any]] = []
    test2_datasets: List[Dict[str, Any]] = []

    cities_info = manifest.get('cities', {})
    for city, cinfo in sorted(cities_info.items(), key=lambda x: x[0]):
        test1_subgraph = cinfo.get('test1_subgraph', None)
        subgraphs = cinfo.get('subgraphs', [])

        for sg in subgraphs:
            sg_name = sg['subgraph']
            npz_path = _resolve_path(sg.get('npz_path'), dataset_root)
            sat_path = _resolve_path(sg.get('satellite_tif_path'), dataset_root)
            total_cases = int(sg.get('total_cases', 0))

            base_entry = {
                'name': f'{city}_{sg_name}',
                'npz_path': npz_path,
                'satellite_tif_path': sat_path,
                'region': city,          # keep key name 'region' for compatibility
                'city': city,
                'subgraph': sg_name,
                'total_cases': total_cases,
            }

            if test1_subgraph is not None and sg_name == test1_subgraph:
                # Test1: full subgraph
                e = dict(base_entry)
                e['case_range'] = (0, total_cases)
                test1_datasets.append(e)
            else:
                # Train/Test2: aligned case ranges
                e_train = dict(base_entry)
                e_train['case_range'] = (0, train_case_count)
                train_datasets.append(e_train)

                e_test2 = dict(base_entry)
                e_test2['case_range'] = (train_case_count, aligned_case_count)
                test2_datasets.append(e_test2)

    return train_datasets, test1_datasets, test2_datasets



def get_random_problems(batch_size, problem_size, num_objectives=1):
    coord_dim = 2 if num_objectives == 1 else 4
    return torch.rand(size=(batch_size, problem_size, coord_dim))


def augment_xy_data_by_8_fold(xy_data):
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    dat = {}
    dat[0] = torch.cat((x, y), dim=2)
    dat[1] = torch.cat((1-x, y), dim=2)
    dat[2] = torch.cat((x, 1-y), dim=2)
    dat[3] = torch.cat((1-x, 1-y), dim=2)
    dat[4] = torch.cat((y, x), dim=2)
    dat[5] = torch.cat((1-y, x), dim=2)
    dat[6] = torch.cat((y, 1-x), dim=2)
    dat[7] = torch.cat((1-y, 1-x), dim=2)
    dat_aug = [dat[i] for i in range(8)]
    return torch.cat(dat_aug, dim=0)

def augment_xy_data_by_64_fold_2obj(xy_data):

    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]

    dat1 = {}
    dat2 = {}

    dat_aug = []

    dat1[0] = torch.cat((x1, y1), dim=2)
    dat1[1]= torch.cat((1-x1, y1), dim=2)
    dat1[2] = torch.cat((x1, 1-y1), dim=2)
    dat1[3] = torch.cat((1-x1, 1-y1), dim=2)
    dat1[4]= torch.cat((y1, x1), dim=2)
    dat1[5] = torch.cat((1-y1, x1), dim=2)
    dat1[6] = torch.cat((y1, 1-x1), dim=2)
    dat1[7] = torch.cat((1-y1, 1-x1), dim=2)

    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1]= torch.cat((1-x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1-y2), dim=2)
    dat2[3] = torch.cat((1-x2, 1-y2), dim=2)
    dat2[4]= torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1-y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1-x2), dim=2)
    dat2[7] = torch.cat((1-y2, 1-x2), dim=2)

    for i in range(8):
        for j in range(8):
            dat = torch.cat((dat1[i], dat2[j]), dim=2)
            dat_aug.append(dat)

    aug_problems = torch.cat(dat_aug, dim=0)

    return aug_problems


##########################################################################################
# MultiDatasetLoaderWithSatellite: dataset loader with train/test split + satellite tif
##########################################################################################

class MultiDatasetLoaderWithSatellite:
    """
    Multi-dataset loader with satellite image support.

    This loader supports **frozen dataset splits** via the optional field:
      - case_range: (start_idx, end_idx)  # end_idx is exclusive

    so that Train/Test2 can share the same subgraph (same satellite image path),
    but use different case index ranges.
    """

    def __init__(self, datasets, switch_interval=50, load_satellite=True, sat_cache_size=10):
        """
        Initialize with a list of dataset configurations.

        Args:
            datasets: List of dicts, each containing:
                - 'name': Dataset name
                - 'npz_path': Path to the NPZ file
                - 'satellite_tif_path': Path to the satellite tif image (optional)
                - 'region': City name / identifier (optional, kept for compatibility)
                - 'case_range': Optional tuple (start, end) selecting subset by index (end exclusive)
            switch_interval: Number of batches to use same dataset before switching
            load_satellite: Whether to load satellite images
            sat_cache_size: LRU cache size for satellite images
        """
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.switch_interval = switch_interval
        self.load_satellite = load_satellite
        self.sat_cache_size = sat_cache_size

        # Track current dataset and batch count for interval-based switching
        self.current_dataset_idx = 0
        self.batches_since_switch = 0

        # Satellite image LRU cache
        self._sat_cache = OrderedDict()

        # Load all NPZ data into memory
        self.data_cache = {}
        total_instances = 0

        for i, ds in enumerate(datasets):
            print(f"Loading dataset {i+1}/{self.num_datasets}: {ds.get('name', f'ds_{i}')}...")
            data = np.load(ds['npz_path'], allow_pickle=True)

            dist_key = 'undirected_dist_norm' if 'undirected_dist_norm' in data.files else 'euclidean_dist_norm'
            if 'matched_node_norm' not in data.files:
                raise KeyError(f"'matched_node_norm' not found in {ds['npz_path']}. Keys={list(data.files)}")

            problems_all = data['matched_node_norm']
            dist_all = data[dist_key]
            full_total = int(problems_all.shape[0])

            # Subset selection by case_range (order-preserving, no shuffle)
            start_idx = 0
            end_idx = full_total
            if 'case_range' in ds and ds['case_range'] is not None:
                try:
                    start_idx = int(ds['case_range'][0])
                    end_idx = int(ds['case_range'][1]) if ds['case_range'][1] is not None else full_total
                except Exception as e:
                    raise ValueError(f"Invalid case_range for dataset {ds.get('name')}: {ds.get('case_range')} ({e})")
            start_idx = max(0, min(start_idx, full_total))
            end_idx = max(start_idx, min(end_idx, full_total))
            subset_total = int(end_idx - start_idx)

            if subset_total <= 0:
                raise ValueError(
                    f"Dataset {ds.get('name')} has empty subset after applying case_range={ds.get('case_range')} "
                    f"(full_total={full_total}, start={start_idx}, end={end_idx})."
                )

            self.data_cache[i] = {
                'name': ds.get('name', f'ds_{i}'),
                'problems': problems_all,
                'dist_matrix': dist_all,
                'satellite_tif_path': ds.get('satellite_tif_path', None),
                'full_total_instances': full_total,
                'index_start': start_idx,
                'index_end': end_idx,
                'total_instances': subset_total,
            }
            total_instances += subset_total
            print(f"  Loaded {subset_total}/{full_total} instances (index range [{start_idx}, {end_idx}))")

        self.total_instances = total_instances
        print(f"All {self.num_datasets} datasets loaded! Total usable instances: {total_instances}")
        print(f"Dataset switch interval: {switch_interval} batches")

    def _load_sat_image_uint8(self, tif_path):
        """Load satellite image as CPU uint8 tensor (1, 3, H, W)."""
        if tif_path is None or not os.path.exists(tif_path):
            return None
        img = Image.open(tif_path).convert('RGB')
        arr = np.asarray(img, dtype=np.uint8)
        chw = np.transpose(arr, (2, 0, 1)).copy()
        tensor = torch.from_numpy(chw).unsqueeze(0).contiguous()
        return tensor

    def _get_sat_image_cached(self, tif_path):
        """Get satellite image with LRU caching."""
        if not self.load_satellite or tif_path is None:
            return None

        if self.sat_cache_size == 0:
            return self._load_sat_image_uint8(tif_path)

        if tif_path in self._sat_cache:
            self._sat_cache.move_to_end(tif_path, last=True)
            return self._sat_cache[tif_path]

        tensor = self._load_sat_image_uint8(tif_path)
        if tensor is not None:
            self._sat_cache[tif_path] = tensor
            self._sat_cache.move_to_end(tif_path, last=True)
            while len(self._sat_cache) > self.sat_cache_size:
                self._sat_cache.popitem(last=False)
        return tensor

    def sample_batch(self, batch_size):
        """
        Sample a batch of problems from datasets (random within the selected case_range).

        Returns:
            problems: torch.Tensor (batch_size, problem_size, 2)
            dist_matrix: torch.Tensor (batch_size, problem_size, problem_size)
            sat_img: torch.Tensor (1, 3, H, W) uint8 or None
            dataset_name: str
        """
        if self.num_datasets <= 0:
            raise RuntimeError("No datasets provided to MultiDatasetLoaderWithSatellite")

        # Switch dataset based on interval
        if self.batches_since_switch >= self.switch_interval:
            self.current_dataset_idx = random.randint(0, self.num_datasets - 1)
            self.batches_since_switch = 0

        self.batches_since_switch += 1
        ds_idx = self.current_dataset_idx
        ds = self.data_cache[ds_idx]

        total = int(ds['total_instances'])
        start = int(ds['index_start'])

        # Random sampling within the usable subset
        rand_offsets = np.random.randint(0, total, size=batch_size)
        random_indices = rand_offsets + start

        problems = ds['problems'][random_indices]
        dist_matrix = ds['dist_matrix'][random_indices]

        # Load satellite image (shared per subgraph)
        sat_img = self._get_sat_image_cached(ds['satellite_tif_path'])

        return (
            torch.tensor(problems, dtype=torch.float32),
            torch.tensor(dist_matrix, dtype=torch.float32),
            sat_img,
            ds['name']
        )

    def get_dataset_info(self):
        """Return information about loaded datasets."""
        info = []
        for i, ds in self.data_cache.items():
            info.append({
                'index': i,
                'name': ds['name'],
                'total_instances': ds['total_instances'],
                'full_total_instances': ds['full_total_instances'],
                'index_start': ds['index_start'],
                'index_end': ds['index_end'],
                'satellite_tif_path': ds['satellite_tif_path'],
            })
        return info

    def __len__(self):
        return int(self.total_instances)

class ValidationDatasetLoader:
    """
    Loader for validation datasets with optimal solutions.

    Supports optional 'case_range' in each dataset entry to validate on a subset
    (e.g., Test2 uses the last 20% cases).
    """

    def __init__(self, datasets, load_satellite=True, sat_cache_size=10):
        """
        Initialize with a list of dataset configurations.

        Args:
            datasets: List of dicts, each containing:
                - 'name': Dataset name
                - 'npz_path': Path to the NPZ file
                - 'satellite_tif_path': Path to the satellite tif image (optional)
                - 'case_range': Optional tuple (start, end) selecting subset by index (end exclusive)
            load_satellite: Whether to load satellite images
            sat_cache_size: LRU cache size for satellite images
        """
        self.datasets = datasets
        self.load_satellite = load_satellite
        self.sat_cache_size = sat_cache_size

        # Satellite image LRU cache
        self._sat_cache = OrderedDict()

        # Load validation data with optimal solutions
        self.optimal_data = {}  # {dataset_name: {sample_indices, optimal_distances_norm, ...}}
        self.source_data = {}   # {dataset_name: {matched_node_norm, dist_matrix, ...}}

        for ds in datasets:
            name = ds.get('name', None) or os.path.basename(ds.get('npz_path', 'dataset'))
            npz_path = ds['npz_path']

            if not os.path.exists(npz_path):
                print(f'Warning: Source dataset not found for {name}: {npz_path}')
                continue

            src_data = np.load(npz_path, allow_pickle=True)

            # choose subset indices
            if 'tsp_obj_norm_undirected' in src_data.files:
                full_num = int(len(src_data['tsp_obj_norm_undirected']))
            else:
                print(f'Warning: No optimal solution available for {name}, skipping')
                continue

            start_idx = 0
            end_idx = full_num
            if 'case_range' in ds and ds['case_range'] is not None:
                try:
                    start_idx = int(ds['case_range'][0])
                    end_idx = int(ds['case_range'][1]) if ds['case_range'][1] is not None else full_num
                except Exception as e:
                    raise ValueError(f"Invalid case_range for {name}: {ds.get('case_range')} ({e})")
            start_idx = max(0, min(start_idx, full_num))
            end_idx = max(start_idx, min(end_idx, full_num))
            sample_indices = np.arange(start_idx, end_idx)

            if sample_indices.size == 0:
                print(f'Warning: Empty case_range for {name}, skipping')
                continue

            # Store optimal solutions subset
            self.optimal_data[name] = {
                'sample_indices': sample_indices,
                'optimal_tours': (
                    src_data['tsp_route_idx_undirected'][sample_indices, :-1]
                    if 'tsp_route_idx_undirected' in src_data.files else None
                ),
                'optimal_distances_norm': src_data['tsp_obj_norm_undirected'][sample_indices],
            }
            print(f'Loaded {len(sample_indices)}/{full_num} optimal solutions for {name} (range [{start_idx}, {end_idx}))')

            dist_key = 'undirected_dist_norm' if 'undirected_dist_norm' in src_data.files else 'euclidean_dist_norm'
            if 'matched_node_norm' not in src_data.files:
                print(f'Warning: matched_node_norm not found for {name}, skipping')
                self.optimal_data.pop(name, None)
                continue

            self.source_data[name] = {
                'matched_node_norm': src_data['matched_node_norm'][sample_indices],
                'dist_matrix': src_data[dist_key][sample_indices],
                'satellite_tif_path': ds.get('satellite_tif_path', None),
            }

        self.dataset_names = list(self.optimal_data.keys())
        print(f'ValidationDatasetLoader initialized with {len(self.dataset_names)} datasets')

    def _load_sat_image_uint8(self, tif_path):
        """Load satellite image as CPU uint8 tensor (1, 3, H, W)."""
        if tif_path is None or not os.path.exists(tif_path):
            return None
        img = Image.open(tif_path).convert('RGB')
        arr = np.asarray(img, dtype=np.uint8)
        chw = np.transpose(arr, (2, 0, 1)).copy()
        tensor = torch.from_numpy(chw).unsqueeze(0).contiguous()
        return tensor

    def _get_sat_image_cached(self, tif_path):
        """Get satellite image with LRU caching."""
        if not self.load_satellite or tif_path is None:
            return None

        if self.sat_cache_size == 0:
            return self._load_sat_image_uint8(tif_path)

        if tif_path in self._sat_cache:
            self._sat_cache.move_to_end(tif_path, last=True)
            return self._sat_cache[tif_path]

        tensor = self._load_sat_image_uint8(tif_path)
        if tensor is not None:
            self._sat_cache[tif_path] = tensor
            self._sat_cache.move_to_end(tif_path, last=True)
            while len(self._sat_cache) > self.sat_cache_size:
                self._sat_cache.popitem(last=False)
        return tensor

    def get_validation_batch(self, dataset_name, batch_indices):
        """
        Get a batch of validation problems.

        Args:
            dataset_name: Name of the dataset
            batch_indices: Indices within the subset (0 to subset_size-1)

        Returns:
            problems: torch.Tensor (batch_size, problem_size, 2)
            distance_matrix: torch.Tensor (batch_size, problem_size, problem_size)
            optimal_distances: np.ndarray (batch_size,)
            sat_img: torch.Tensor (1, 3, H, W) or None
        """
        src = self.source_data[dataset_name]
        opt = self.optimal_data[dataset_name]

        problems = torch.from_numpy(src['matched_node_norm'][batch_indices]).float()
        distance_matrix = torch.from_numpy(src['dist_matrix'][batch_indices]).float()
        optimal_distances = opt['optimal_distances_norm'][batch_indices]
        sat_img = self._get_sat_image_cached(src['satellite_tif_path'])

        return problems, distance_matrix, optimal_distances, sat_img

    def get_num_samples(self, dataset_name):
        """Get number of samples for a dataset."""
        return len(self.optimal_data[dataset_name]['sample_indices'])
