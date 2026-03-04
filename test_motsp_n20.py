##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Test mode: 'single' for single-objective TSP, 'multi' for multi-objective MOTSP
TEST_MODE = 'single'  # 'single' or 'multi'

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

# Find project root by looking for a marker directory ('MMDataset')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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


##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
import math
from MOTSPTester import TSPTester as Tester
from MOTSProblemDef import (
    get_random_problems,
    MultiDatasetLoaderWithSatellite,
    get_default_split_manifest_path,
    load_or_create_dataset_split_manifest,
    build_datasets_from_split_manifest,
)

# Multi-objective imports (only needed for TEST_MODE='multi')
try:
    from utils.cal_pareto_demo import Pareto_sols
    from utils.cal_ps_hv import cal_ps_hv
except ImportError:
    pass  # Not needed for single-objective testing

##########################################################################################
import time

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
}

# Test data configuration
# Set use_road_network_test=True to test with road network distance data
test_data_params = {
    'use_road_network_test': True,  # True: use npz with road distance, False: use Euclidean test data
    # Path to test data folder (npz files) - used when use_road_network_test=True
    'test_data_path': os.path.join(PROJECT_ROOT, 'MMDataset', 'Gen_dataset_V2', 'Gen_dataset'),
    'test_episodes_from_npz': 200,  # How many instances to sample for testing
}

# Frozen dataset split (must match training manifest)
TEST1_SUBGRAPH_THRESHOLD = 10   # same as training
TEST2_CASE_RATIO = 0.2         # same as training
SPLIT_MANIFEST_MODE = 'load'   # do NOT change unless you explicitly want to create/update manifest
SPLIT_MANIFEST_SEED = 0        # only used when creating/updating manifest

# Which split to evaluate on
EVAL_SPLIT = 'test1'            # 'test1' | 'test2'

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
    'in_channels': 1,  # Should match training config (1 for single objective)
    'patch_size': 16,
    'pixel_density': 10,
    'fusion_layer_num': 3,
    'bn_num': 10,
    'bn_img_num': 10,
}

model_params['img_size'] = math.ceil(
    env_params['problem_size'] ** (1 / 2) * model_params['pixel_density'] / model_params['patch_size']) * model_params[
                               'patch_size']
env_params['img_size'] = model_params['img_size']
env_params['patch_size'] = model_params['patch_size']
env_params['in_channels'] = model_params['in_channels']

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    "dec_method": "WS",
    'model_load': {
        'path': './result/20260203_132425_train__tsp_n20',  # Your trained model
        'info': "Single-objective TSP with road network distance",
        'epoch': 180,  # Change to your saved epoch (5, 10, 15, 20, ...)
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    'aug_factor': 1, #64,
    'aug_batch_size': 200,
}
if tester_params['aug_factor'] > 1:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################
def load_test_data_from_npz(dataset_root, problem_size, num_episodes, eval_split='test1'):
    """
    Load test data from NPZ files (with road network distance) using the **frozen manifest split**.

    eval_split:
      - 'test1': held-out subgraphs (1 subgraph per city for cities with many subgraphs)
      - 'test2': held-out cases (last TEST2_CASE_RATIO cases) from the same subgraphs used in training
    """
    npz_suffix = f'p{problem_size}'

    logger = logging.getLogger('root')

    manifest_path = get_default_split_manifest_path(dataset_root, npz_suffix)
    manifest = load_or_create_dataset_split_manifest(
        dataset_root=dataset_root,
        npz_suffix=npz_suffix,
        manifest_path=manifest_path,
        test1_subgraph_threshold=TEST1_SUBGRAPH_THRESHOLD,
        test2_case_ratio=TEST2_CASE_RATIO,
        mode=SPLIT_MANIFEST_MODE,
        seed=SPLIT_MANIFEST_SEED,
        logger=logger,
    )

    train_datasets, test1_datasets, test2_datasets = build_datasets_from_split_manifest(
        manifest=manifest,
        dataset_root=dataset_root,
    )

    if eval_split.lower() == 'test2':
        eval_datasets = test2_datasets
    else:
        eval_datasets = test1_datasets

    logger.info(
        f"[Test Data] eval_split={eval_split} | "
        f"train_subgraphs={len(train_datasets)}, test1_subgraphs={len(test1_datasets)}, test2_subgraphs={len(test2_datasets)}"
    )

    # Use dataset loader (no satellite image needed for testing)
    loader = MultiDatasetLoaderWithSatellite(
        datasets=eval_datasets,
        switch_interval=1,   # switch every call for diversity
        load_satellite=False,
    )

    # Sample instances across datasets (call multiple times so switch_interval takes effect)
    problems_chunks = []
    dist_chunks = []

    remaining = int(num_episodes)
    chunk_size = min(256, remaining) if remaining > 0 else 0

    while remaining > 0:
        bs = min(chunk_size, remaining)
        problems, dist_matrix, _, _ = loader.sample_batch(bs)
        problems_chunks.append(problems)
        dist_chunks.append(dist_matrix)
        remaining -= bs

    problems = torch.cat(problems_chunks, dim=0)
    distance_matrix = torch.cat(dist_chunks, dim=0)

    logger.info(f"[Test Data] Loaded {problems.shape[0]} instances from {eval_split}")
    logger.info(f"  Problems shape: {tuple(problems.shape)}")
    logger.info(f"  Distance matrix shape: {tuple(distance_matrix.shape)}")

    return problems, distance_matrix




def main(n_sols = 101):
    if tester_params['aug_factor'] == 1:
        sols_floder = f"PMOCO_mean_sols_n{env_params['problem_size']}.txt"
        pareto_fig = f"PMOCO_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"PMOCO_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"PMOCO_hv_n{env_params['problem_size']}.txt"
    else:
        sols_floder = f"PMOCO(aug)_mean_sols_n{env_params['problem_size']}.txt"
        pareto_fig = f"PMOCO(aug)_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"PMOCO(aug)_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"PMOCO(aug)_hv_n{env_params['problem_size']}.txt"


    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()

    timer_start = time.time()
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    
    copy_all_src(tester.result_folder)
    
    # Load test data based on configuration
    distance_matrix = None
    if test_data_params.get('use_road_network_test', False):
        # Load from npz files with road network distance
        test_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            test_data_params['test_data_path']
        ))
        num_episodes = test_data_params.get('test_episodes_from_npz', tester_params['test_episodes'])
        shared_problem, distance_matrix = load_test_data_from_npz(
            test_path,
            env_params['problem_size'],
            num_episodes,
            eval_split=EVAL_SPLIT
        )
        shared_problem = shared_problem.to(device=CUDA_DEVICE_NUM)
        distance_matrix = distance_matrix.to(device=CUDA_DEVICE_NUM)
        # Update test_episodes to match loaded data
        tester_params['test_episodes'] = num_episodes
        print(f"[Test Mode] Using ROAD NETWORK distance for evaluation")
    else:
        # Load from pt file (Euclidean distance)
        test_path = f"./data/testdata_tsp_size{env_params['problem_size']}.pt"
        shared_problem = torch.load(test_path).to(device=CUDA_DEVICE_NUM)
        print(f"[Test Mode] Using EUCLIDEAN distance for evaluation")

    ref = np.array([20,20])    #20
    # ref2 = np.array([35,35])   #50
    #ref = np.array([65,65])   #100

    batch_size = shared_problem.shape[0]
    sols = np.zeros([batch_size, n_sols, 2])
    total_test_time = 0
    for i in range(n_sols):
        pref = torch.zeros(2).cuda()
        pref[0] = 1 - i / (n_sols - 1)
        pref[1] = i / (n_sols - 1)
        pref = pref / torch.sum(pref)

        test_timer_start = time.time()
        aug_score = tester.run(shared_problem, pref, distance_matrix=distance_matrix)
        test_timer_end = time.time()
        total_test_time += test_timer_end - test_timer_start
        print('Ins{:d} Test Time(s): {:.4f}'.format(i, test_timer_end - test_timer_start))

        sols[:, i, 0] = np.array(aug_score[0].flatten())
        sols[:, i, 1] = np.array(aug_score[1].flatten())

    timer_end = time.time()
    total_time = timer_end - timer_start

    max_obj1 = sols.reshape(-1, 2)[:, 0].max()
    max_obj2 = sols.reshape(-1, 2)[:, 1].max()
    txt2 = F"{tester.result_folder}/max_cost_n{env_params['problem_size']}.txt"
    f = open(
        txt2,
        'a')
    f.write(f"MAX OBJ1:{max_obj1}\n")
    f.write(f"MAX OBJ2:{max_obj2}\n")
    f.close()

    
    # MOTSP 20
    single_task = [3.83, 3.83]
    
    # MOTSP 50
    #single_task = [5.69, 5.69]
    
    # MOTSP 100
    #single_task = [7.76, 7.76]
    
    fig = plt.figure()

    sols_mean = sols.mean(0)
    plt.axvline(single_task[0],linewidth=3 , alpha = 0.25)
    plt.axhline(single_task[1],linewidth=3,alpha = 0.25, label = 'Single Objective TSP (Concorde)')
    plt.plot(sols_mean[:,0],sols_mean[:,1], marker = 'o', c = 'C1',ms = 3,  label='PSL-MOCO (Ours)')

    plt.legend()
    plt.savefig(F"{tester.result_folder}/{pareto_fig}")

    np.savetxt(F"{tester.result_folder}/{sols_floder}", sols_mean,
               delimiter='\t', fmt="%.4f\t%.4f")


    nd_sort = Pareto_sols(p_size=env_params['problem_size'], pop_size=sols.shape[0], obj_num=sols.shape[2])
    sols_t = torch.Tensor(sols)
    nd_sort.update_PE(objs=sols_t)
    p_sols, p_sols_num, _ = nd_sort.show_PE()
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)


    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hvs.mean()))
    print('NDS: {:.4f}'.format(p_sols_num.float().mean()))
    print('Avg Test Time(s): {:.4f}\n'.format(total_test_time))

    np.savetxt(F"{tester.result_folder}/{all_sols_floder}", sols.reshape(-1, 2),
               delimiter='\t', fmt="%.4f\t%.4f")
    np.savetxt(F"{tester.result_folder}/{hv_floder}", hvs,
               delimiter='\t', fmt="%.4f")

    if tester_params['aug_factor'] == 1:
        f = open(
            F"{tester.result_folder}/PMOCO-TSP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO-TSP{env_params['problem_size']}\n")
    else:
        f = open(
            F"{tester.result_folder}/PMOCO(aug)-TSP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO(aug)-TSP{env_params['problem_size']}\n")


    f.write(f"MOTSP_2obj Type1\n")
    f.write(f"Model Path: {tester_params['model_load']['path']}\n")
    f.write(f"Model Epoch: {tester_params['model_load']['epoch']}\n")
    f.write(f"Hyper Hidden Dim: {model_params['hyper_hidden_dim']}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
    f.write('Test Time(s): {:.4f}\n'.format(total_test_time))
    f.write('Run Time(s): {:.4f}\n'.format(total_time))
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))
    f.write('NDS: {:.4f}\n'.format(p_sols_num.float().mean()))
    f.write(f"Ref Point:[{ref[0]},{ref[1]}] \n")
    f.write(f"Distance Type: {'Road Network' if test_data_params.get('use_road_network_test', False) else 'Euclidean'}\n")
    f.write(f"Info: {tester_params['model_load']['info']}\n")
    f.close()


##########################################################################################
def main_single_objective():
    """
    Simplified single-objective TSP testing.
    Only evaluates total path length (road network or Euclidean distance).
    No Pareto front or HV calculation needed.
    """
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()

    timer_start = time.time()
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    
    copy_all_src(tester.result_folder)
    
    # Load test data based on configuration
    distance_matrix = None
    if test_data_params.get('use_road_network_test', False):
        # Load from npz files with road network distance
        test_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            test_data_params['test_data_path']
        ))
        num_episodes = test_data_params.get('test_episodes_from_npz', tester_params['test_episodes'])
        shared_problem, distance_matrix = load_test_data_from_npz(
            test_path,
            env_params['problem_size'],
            num_episodes,
            eval_split=EVAL_SPLIT
        )
        shared_problem = shared_problem.to(device=CUDA_DEVICE_NUM)
        distance_matrix = distance_matrix.to(device=CUDA_DEVICE_NUM)
        tester_params['test_episodes'] = num_episodes
        print(f"[Test Mode] Single-objective with ROAD NETWORK distance")
    else:
        # Load from pt file (Euclidean distance)
        test_path = f"./data/testdata_tsp_size{env_params['problem_size']}.pt"
        shared_problem = torch.load(test_path).to(device=CUDA_DEVICE_NUM)
        print(f"[Test Mode] Single-objective with EUCLIDEAN distance")

    batch_size = shared_problem.shape[0]
    
    # Single preference: [1.0, 0.0] means only optimize the first objective (path length)
    pref = torch.tensor([1.0, 0.0]).cuda()
    
    test_timer_start = time.time()
    aug_score = tester.run(shared_problem, pref, distance_matrix=distance_matrix)
    test_timer_end = time.time()
    test_time = test_timer_end - test_timer_start

    timer_end = time.time()
    total_time = timer_end - timer_start
    
    # Extract results
    path_lengths = aug_score[0].flatten()  # Only first objective matters for single-objective
    
    avg_path_length = path_lengths.mean().item()
    min_path_length = path_lengths.min().item()
    max_path_length = path_lengths.max().item()
    std_path_length = path_lengths.std().item()
    
    # Print results
    print("\n" + "="*60)
    print("Single-Objective TSP Test Results")
    print("="*60)
    print(f"Model: {tester_params['model_load']['path']}")
    print(f"Epoch: {tester_params['model_load']['epoch']}")
    print(f"Distance Type: {'Road Network' if test_data_params.get('use_road_network_test', False) else 'Euclidean'}")
    print(f"Test Instances: {batch_size}")
    print("-"*60)
    print(f"Average Path Length: {avg_path_length:.4f}")
    print(f"Min Path Length: {min_path_length:.4f}")
    print(f"Max Path Length: {max_path_length:.4f}")
    print(f"Std Path Length: {std_path_length:.4f}")
    print("-"*60)
    print(f"Test Time: {test_time:.4f}s")
    print(f"Total Time: {total_time:.4f}s")
    print("="*60 + "\n")
    
    # Save results to file
    result_file = f"{tester.result_folder}/single_obj_result_n{env_params['problem_size']}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Single-Objective TSP Test Results\n")
        f.write(f"="*50 + "\n")
        f.write(f"Model Path: {tester_params['model_load']['path']}\n")
        f.write(f"Model Epoch: {tester_params['model_load']['epoch']}\n")
        f.write(f"Distance Type: {'Road Network' if test_data_params.get('use_road_network_test', False) else 'Euclidean'}\n")
        f.write(f"Test Instances: {batch_size}\n")
        f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
        f.write(f"-"*50 + "\n")
        f.write(f"Average Path Length: {avg_path_length:.4f}\n")
        f.write(f"Min Path Length: {min_path_length:.4f}\n")
        f.write(f"Max Path Length: {max_path_length:.4f}\n")
        f.write(f"Std Path Length: {std_path_length:.4f}\n")
        f.write(f"-"*50 + "\n")
        f.write(f"Test Time: {test_time:.4f}s\n")
        f.write(f"Total Time: {total_time:.4f}s\n")
    
    # Save all path lengths
    np.savetxt(f"{tester.result_folder}/path_lengths_n{env_params['problem_size']}.txt", 
               path_lengths.numpy(), fmt="%.4f")
    
    print(f"Results saved to: {result_file}")
    
    return avg_path_length


##########################################################################################
if __name__ == "__main__":
    if TEST_MODE == 'single':
        main_single_objective()
    else:
        main()  # Multi-objective test
