#!/usr/bin/env python3
"""
GPU Batch Size Validation for e2e_decoder mode.

Runs one forward+backward step per batch size and reports peak GPU memory
or OOM. Use to determine if larger batch sizes (e.g. 32, 64) fit on GPU.

Prerequisites:
  - precompute_encoder_cache.py (encoder_cache.pt)
  - precompute_road_prob.py (road_prob_cache.npz for fast-step fallback)

Usage:
  python validate_batch_size_gpu.py
  python validate_batch_size_gpu.py --batch_sizes 16 32 64 128
  python validate_batch_size_gpu.py --gpu 1
"""
import os
import sys
import argparse
import logging

# Path setup (same as train_motsp_n20.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def find_project_root(start_path, marker='MMDataset'):
    current = os.path.abspath(start_path)
    for _ in range(10):
        if os.path.isdir(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not find project root (directory containing 'MMDataset')")

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import torch
import math

from utils.utils import create_logger
from MOTSProblemDef import (
    get_default_split_manifest_path,
    load_or_create_dataset_split_manifest,
    build_datasets_from_split_manifest,
)
from MOTSPTrainer import TSPTrainer as Trainer

# Config (same as train_motsp_n20)
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'MMDataset', 'Gen_dataset_V2', 'Gen_dataset')
NPZ_SUFFIX = 'p20'
TEST1_SUBGRAPH_THRESHOLD = 10
TEST2_CASE_RATIO = 0.2
SPLIT_MANIFEST_MODE = 'load'
SPLIT_MANIFEST_SEED = 0
SPLIT_MANIFEST_PATH = get_default_split_manifest_path(DATASET_ROOT, NPZ_SUFFIX)
VALIDATION_SPLIT = 'test1'


def _check_encoder_cache(datasets):
    """Check encoder_cache.pt exists for all datasets."""
    missing = []
    for ds in datasets:
        tif_path = ds.get('satellite_tif_path')
        if not tif_path:
            continue
        cache_path = os.path.join(os.path.dirname(tif_path), 'encoder_cache.pt')
        if not os.path.exists(cache_path):
            missing.append(ds.get('name', 'unknown'))
    return missing


def main():
    parser = argparse.ArgumentParser(description='GPU batch size validation for e2e_decoder')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16, 32, 64],
                        help='Batch sizes to test')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')
    args = parser.parse_args()

    batch_sizes = sorted(args.batch_sizes)
    max_b = max(batch_sizes)

    # Logger
    create_logger(log_file={'desc': 'validate_batch_size', 'filename': 'validate_batch_size'})
    logger = logging.getLogger('root')

    # Load manifest and datasets
    manifest = load_or_create_dataset_split_manifest(
        dataset_root=DATASET_ROOT,
        npz_suffix=NPZ_SUFFIX,
        manifest_path=SPLIT_MANIFEST_PATH,
        test1_subgraph_threshold=TEST1_SUBGRAPH_THRESHOLD,
        test2_case_ratio=TEST2_CASE_RATIO,
        mode=SPLIT_MANIFEST_MODE,
        seed=SPLIT_MANIFEST_SEED,
        logger=logger,
    )
    TRAIN_DATASETS, TEST1_DATASETS, TEST2_DATASETS = build_datasets_from_split_manifest(
        manifest=manifest,
        dataset_root=DATASET_ROOT,
    )

    if not TRAIN_DATASETS:
        logger.error("No train datasets. Cannot run validation.")
        sys.exit(1)

    # Check encoder cache coverage
    missing = _check_encoder_cache(TRAIN_DATASETS)
    if missing:
        logger.error(
            f"Encoder cache missing for {len(missing)} datasets. "
            f"Run: python precompute_encoder_cache.py --device cuda"
        )
        logger.error(f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        sys.exit(1)

    # e2e_decoder config (no batch_size cap)
    env_params = {
        'problem_size': 20,
        'pomo_size': 20,
        'use_custom_dataset': True,
        'datasets': TRAIN_DATASETS,
        'switch_interval': 10,
        'load_satellite': True,
        'sat_cache_size': 10,
    }

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'hyper_hidden_dim': 256,
        'in_channels': 1,
        'patch_size': 16,
        'pixel_density': 10,
        'fusion_layer_num': 3,
        'bn_num': 10,
        'bn_img_num': 10,
        'use_satellite': True,
        'sat_grid': 14,
        'sat_encoder_type': 'sam_road',
        'sam_road_ckpt_name': 'cityscale_vitb_512_e10.ckpt',
        'sam_road_cfg_name': 'toponet_vitb_512_cityscale.yaml',
        'sam_road_overlap': 64,
        'sam_road_patch_batch': 4,
        'use_distance_in_encoder': True,
        'distance_use_stats': True,
        'encoder_mode': 'e2e_eikonal',
        'e2e_eikonal_ckpt': 'training_outputs/fulldataset_seg_dist_lora_v2/checkpoints/best_seg_dist_lora_v2.ckpt',
        'e2e_eikonal_ds': 32,
        'e2e_eikonal_eik_iters': 40,
        'e2e_eikonal_pool_mode': 'max',
        'e2e_eikonal_iter_floor_c': 1.2,
        'e2e_eikonal_iter_floor_f': 0.65,
        'e2e_eikonal_freeze_encoder': True,
        'e2e_eikonal_train_cost_net': False,
        'e2e_train_decoder': True,
        'e2e_decoder_lr': 1e-4,
        'e2e_grad_clip_norm': 1.0,
        'e2e_samroute_lr': 1e-3,
        'e2e_parallel_cases': True,
        'e2e_use_compile': True,
        'e2e_grad_interval': 10,
    }
    model_params['img_size'] = math.ceil(
        env_params['problem_size'] ** (1 / 2) * model_params['pixel_density'] / model_params['patch_size']
    ) * model_params['patch_size']
    env_params['img_size'] = model_params['img_size']
    env_params['patch_size'] = model_params['patch_size']
    env_params['in_channels'] = model_params['in_channels']

    optimizer_params = {
        'optimizer': {'lr': 1e-4, 'weight_decay': 1e-6},
        'scheduler': {'milestones': [180,], 'gamma': 0.1},
    }

    trainer_params = {
        'use_cuda': True,
        'use_custom_dataset': True,
        'cuda_device_num': args.gpu,
        'dec_method': 'WS',
        'epochs': 1,
        'train_episodes': 100,
        'train_batch_size': 16,
        'validation_interval': 0,
        'validation_batch_size': 64,
        'validation_datasets': TEST1_DATASETS if VALIDATION_SPLIT.lower() == 'test1' else TEST2_DATASETS,
        'early_stop_patience': 0,
        'save_best_model': False,
        'logging': {'model_save_interval': 999, 'img_save_interval': 999,
                    'log_image_params_1': {}, 'log_image_params_2': {}},
        'model_load': {'enable': False},
    }

    # Create trainer (loads SAMRoute, NCO, dataset_loader)
    logger.info("Loading models and dataset...")
    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )

    if not trainer._encoder_cache_available:
        logger.error("Trainer did not enable encoder cache (e2e_decoder). Check encoder_cache.pt.")
        sys.exit(1)

    # Validation loop
    print("\n" + "=" * 60)
    print("GPU Batch Size Validation (e2e_decoder)")
    print("=" * 60)

    for B in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            trainer._train_one_batch(B)
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"B={B}: OK, peak={peak_gb:.2f} GB")
        except torch.cuda.OutOfMemoryError as e:
            print(f"B={B}: OOM")
        except Exception as e:
            print(f"B={B}: ERROR - {e}")
        torch.cuda.empty_cache()

    print("=" * 60)


if __name__ == "__main__":
    main()
