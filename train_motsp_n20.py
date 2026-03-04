##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Find project root (MMDM) by looking for a marker directory
def find_project_root(start_path, marker='MMDataset'):
    """Find project root by searching upward for a marker directory."""
    current = os.path.abspath(start_path)
    for _ in range(10):  # Limit search depth
        if os.path.isdir(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # Reached filesystem root
            break
        current = parent
    return None

PROJECT_ROOT = find_project_root(SCRIPT_DIR)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not find project root (directory containing 'MMDataset')")

sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
import math
from MOTSPTrainer import TSPTrainer as Trainer

##########################################################################################
# Dataset split configuration (Train / Test1 / Test2) with MANIFEST
##########################################################################################
from MOTSProblemDef import (
    get_default_split_manifest_path,
    load_or_create_dataset_split_manifest,
    build_datasets_from_split_manifest,
)

# Dataset root (城市目录)
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'MMDataset', 'Gen_dataset_V2', 'Gen_dataset')

# Node-size suffix of the npz file to use (e.g., p20 / p50 / p100)
NPZ_SUFFIX = 'p20'

# Split hyper-parameters (写入 manifest 后将被固化)
TEST1_SUBGRAPH_THRESHOLD = 10   # City with subgraph_count > threshold: choose 1 subgraph as Test1
TEST2_CASE_RATIO = 0.2         # Last 20% cases (order-preserving) become Test2; first 80% become Train

# Manifest settings:
#   - 'load'  : must exist, never modify (recommended for almost all runs)
#   - 'create': create once when manifest is missing
#   - 'update': overwrite existing manifest (only when you explicitly want to change the split)
SPLIT_MANIFEST_MODE = 'load'   # 'load' | 'create' | 'update'
SPLIT_MANIFEST_SEED = 0        # only used when creating/updating manifest

SPLIT_MANIFEST_PATH = get_default_split_manifest_path(DATASET_ROOT, NPZ_SUFFIX)

# Choose which test split to use for validation during training
# (You can switch to 'test2' if you prefer in-subgraph validation.)
VALIDATION_SPLIT = 'test1'     # 'test1' | 'test2'

# NOTE:
# We will build TRAIN_DATASETS / TEST1_DATASETS / TEST2_DATASETS inside main(),
# after logger is created, so that split decisions are recorded in the log file.
TRAIN_DATASETS = []
TEST1_DATASETS = []
TEST2_DATASETS = []


##########################################################################################
# parameters
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'use_custom_dataset': True,
    # Dataset list with train/test split (same as GIMF-P)
    'datasets': TRAIN_DATASETS,
    'switch_interval': 10,      # Batches before switching to next dataset
    # Satellite image loading (lazy + LRU cache)
    'load_satellite': True,
    'sat_cache_size': 10,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
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
    # Satellite branch (placeholder encoder; planned to be replaced with SAM-Road)
    'use_satellite': True,
    'sat_grid': 14,  # satellite tokens = sat_grid^2
    'sat_encoder_type': 'sam_road',  # 'placeholder' | 'sam_road'
    # SAM-Road settings (use local repo + local checkpoints)
    # If not set, code will auto-find `sam_road_repo/` and `checkpoints/*.ckpt` from project root.
    'sam_road_ckpt_name': 'cityscale_vitb_512_e10.ckpt',
    'sam_road_cfg_name': 'toponet_vitb_512_cityscale.yaml',
    'sam_road_overlap': 64,
    'sam_road_patch_batch': 4,
    # Scheme 1: inject GT distance matrix into encoder (teacher; default off)
    'use_distance_in_encoder': False,
    'distance_use_stats': True,
    # Encoder mode:
    #   'graph_image_fusion' (default): graph + xy_img + satellite two-stream fusion
    #   'graph_only'                  : pure graph encoder + distance matrix, no image branch
    'encoder_mode': 'graph_image_fusion',
}

model_params['img_size'] = math.ceil(
    env_params['problem_size'] ** (1 / 2) * model_params['pixel_density'] / model_params['patch_size']) * model_params[
                               'patch_size']
env_params['img_size'] = model_params['img_size']
env_params['patch_size'] = model_params['patch_size']
env_params['in_channels'] = model_params['in_channels']

optimizer_params = {
    'optimizer': {
        'lr': 1e-4, 
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [180,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'use_custom_dataset': env_params['use_custom_dataset'],
    'cuda_device_num': CUDA_DEVICE_NUM,
    'dec_method': 'WS',
    'epochs': 400,
    'train_episodes': 100 * 1000,
    'train_batch_size': 256,
    
    # Validation settings (uses TEST_DATASETS from random split, same as GIMF-P)
    'validation_interval': 10,        # Validate every N epochs (set to 0 to disable)
    'validation_batch_size': 64,      # Batch size for validation
    'validation_datasets': [],  # Use held-out test locations for validation
    
    # Early stopping and best model saving
    'early_stop_patience': 0,         # Stop if no improvement for N validations (0 to disable)
    'save_best_model': True,          # Save best model based on validation gap
    
    'logging': {
        'model_save_interval': 5,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        'epoch': 1,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    # Logger first (so split decisions are recorded in log file)
    create_logger(**logger_params)
    logger = logging.getLogger('root')

    # Load/Create split manifest (frozen split)
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

    global TRAIN_DATASETS, TEST1_DATASETS, TEST2_DATASETS
    TRAIN_DATASETS, TEST1_DATASETS, TEST2_DATASETS = build_datasets_from_split_manifest(
        manifest=manifest,
        dataset_root=DATASET_ROOT,
    )

    # Log split summary (especially Test1 cities)
    logger.info(
        f"[Dataset Split] Train subgraphs: {len(TRAIN_DATASETS)}, "
        f"Test1 subgraphs: {len(TEST1_DATASETS)}, Test2 subgraphs: {len(TEST2_DATASETS)}"
    )
    logger.info(f"[Dataset Split] Test1 cities: {manifest.get('test1_cities', [])}")

    # Auto-configure satellite loading based on encoder_mode
    if model_params.get('encoder_mode', 'graph_image_fusion') == 'graph_only':
        env_params['load_satellite'] = False
        model_params['use_satellite'] = False
        logger.info("[Encoder Mode] graph_only: satellite loading disabled automatically.")
    else:
        logger.info("[Encoder Mode] graph_image_fusion: satellite loading enabled.")

    # Apply datasets to training config
    env_params['datasets'] = TRAIN_DATASETS
    if VALIDATION_SPLIT.lower() == 'test2':
        trainer_params['validation_datasets'] = TEST2_DATASETS
    else:
        trainer_params['validation_datasets'] = TEST1_DATASETS

    _print_config()

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params
    )

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info(f'DEBUG_MODE: {DEBUG_MODE}')
    logger.info(f'USE_CUDA: {USE_CUDA}, CUDA_DEVICE_NUM: {CUDA_DEVICE_NUM}')
    [logger.info(f'{g_key}{globals()[g_key]}') for g_key in globals() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
    main()
