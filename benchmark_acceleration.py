#!/usr/bin/env python3
"""
Comprehensive benchmark for e2e_eikonal acceleration strategies.

Tests combinations of:
  - ds:                 16 / 32 / 64
  - eik_iters:          20 / 30 / 40
  - parallel_cases:     True / False
  - disable_checkpoint: True / False

For each configuration, measures:
  - Forward time
  - Backward time
  - GPU memory peak
  - Accuracy vs NPZ (relative error, pairwise ordering)
  - Gradient norm (confirm gradient flow is intact)
"""
import os, sys, time, gc
import torch
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

EIKONAL_DIR = os.path.join(SCRIPT_DIR, 'eikonal_solver')
if EIKONAL_DIR not in sys.path:
    sys.path.insert(0, EIKONAL_DIR)

device = torch.device('cuda')
torch.cuda.set_device(0)


# ======================================================================
# Helper: load SAMRoute
# ======================================================================
def load_samroute():
    from model_multigrid import SAMRoute
    from gradcheck_route_loss_v2_multigrid_fullmap import _load_lightning_ckpt
    from types import SimpleNamespace

    ckpt_path = os.path.join(SCRIPT_DIR, 'training_outputs', 'fulldataset_seg_dist_lora_v2',
                             'checkpoints', 'best_seg_dist_lora_v2.ckpt')
    sd = _load_lightning_ckpt(ckpt_path)
    has_lora = any("attn.qkv.linear_a_q" in k for k in sd)
    pe = sd.get('image_encoder.pos_embed')
    patch_size = int(pe.shape[1]) * 16 if pe is not None else 512
    w7 = sd.get('map_decoder.7.weight')
    use_smooth = w7 is not None and w7.shape[1] == 32

    cfg = SimpleNamespace(
        SAM_VERSION='vit_b', PATCH_SIZE=patch_size, NO_SAM=False,
        USE_SAM_DECODER=False, USE_SMOOTH_DECODER=use_smooth,
        ENCODER_LORA=has_lora, LORA_RANK=4, FREEZE_ENCODER=True,
        FOCAL_LOSS=False, TOPONET_VERSION='default',
        SAM_CKPT_PATH=os.path.join(SCRIPT_DIR, 'sam_road_repo', 'sam_ckpts', 'sam_vit_b_01ec64.pth'),
        ROUTE_COST_MODE='add', ROUTE_ADD_ALPHA=20.0, ROUTE_ADD_GAMMA=2.0,
        ROUTE_ADD_BLOCK_ALPHA=0.0, ROUTE_BLOCK_TH=0.0, ROUTE_ROI_MARGIN=64,
        ROUTE_COST_NET=False, ROUTE_COST_NET_CH=8, ROUTE_COST_NET_USE_COORD=False,
    )
    if has_lora:
        cfg.FREEZE_ENCODER = False
        for k, v in sd.items():
            if "linear_a_q.weight" in k:
                cfg.LORA_RANK = v.shape[0]
                break

    model = SAMRoute(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.requires_grad_(False)
    model.cost_log_alpha.requires_grad_(True)
    model.cost_log_gamma.requires_grad_(True)
    model.eik_gate_logit.requires_grad_(True)
    model.eval()
    return model, patch_size


# ======================================================================
# Helper: load test data
# ======================================================================
def find_test_data(dataset_root):
    for city in sorted(os.listdir(dataset_root)):
        city_dir = os.path.join(dataset_root, city)
        if not os.path.isdir(city_dir):
            continue
        for sg in sorted(os.listdir(city_dir)):
            sg_dir = os.path.join(city_dir, sg)
            if not os.path.isdir(sg_dir):
                continue
            tifs = [f for f in os.listdir(sg_dir) if f.startswith('crop_') and f.endswith('.tif')]
            npzs = [f for f in os.listdir(sg_dir) if f.endswith('_p20.npz')]
            if tifs and npzs:
                return os.path.join(sg_dir, tifs[0]), os.path.join(sg_dir, npzs[0])
    return None, None


def prepare_road_prob(tif_path, model, patch_size):
    from gradcheck_route_loss_v2_multigrid_fullmap import sliding_window_inference as swi
    from PIL import Image
    img = Image.open(tif_path).convert('RGB')
    img_np = np.asarray(img, dtype=np.uint8)
    with torch.no_grad():
        rp_np = swi(img_np, model, device, patch_size=patch_size, verbose=False).astype(np.float32)
    rp = torch.from_numpy(rp_np).to(device, torch.float32).unsqueeze(0).clamp(1e-6, 1.0 - 1e-6)
    return rp, img_np.shape[0], img_np.shape[1]


def prepare_batch_nodes(node_coords_xy, B, H_img, W_img):
    batch_nodes_yx = []
    for b in range(B):
        xy = node_coords_xy[b]
        x_norm = torch.from_numpy(xy[:, 0].copy()).to(device, torch.float32)
        y_norm = torch.from_numpy(xy[:, 1].copy()).to(device, torch.float32)
        y_pix = torch.round((1.0 - y_norm) * (H_img - 1)).long()
        x_pix = torch.round(x_norm * (W_img - 1)).long()
        batch_nodes_yx.append(torch.stack([y_pix, x_pix], dim=-1))
    return batch_nodes_yx


# ======================================================================
# Accuracy metrics
# ======================================================================
def compute_accuracy_metrics(D_online, npz_dist_norm, B, N):
    """Compare online distances with NPZ ground truth."""
    D_np = D_online.detach().cpu().numpy()
    rel_errors = []
    pw_correct_list = []

    for b in range(B):
        d_pred = D_np[b]
        d_gt = npz_dist_norm[b]
        mask = ~np.eye(N, dtype=bool) & np.isfinite(d_gt) & (d_gt > 0)
        if not mask.any():
            continue
        rel = np.abs(d_pred[mask] - d_gt[mask]) / np.maximum(d_gt[mask], 1e-6)
        rel_errors.append(rel.mean())

        # pairwise ordering accuracy
        triu_idx = np.triu_indices(N, k=1)
        gt_flat = d_gt[triu_idx]
        pred_flat = d_pred[triu_idx]
        valid = np.isfinite(gt_flat) & np.isfinite(pred_flat) & (gt_flat > 0) & (pred_flat > 0)
        if valid.sum() < 2:
            continue
        gt_f = gt_flat[valid]
        pr_f = pred_flat[valid]
        n_pairs = 0
        n_correct = 0
        for i in range(len(gt_f)):
            for j in range(i + 1, len(gt_f)):
                gt_ord = np.sign(gt_f[i] - gt_f[j])
                pr_ord = np.sign(pr_f[i] - pr_f[j])
                if gt_ord != 0:
                    n_pairs += 1
                    if gt_ord == pr_ord:
                        n_correct += 1
        if n_pairs > 0:
            pw_correct_list.append(n_correct / n_pairs)

    return {
        'rel_err_mean': np.mean(rel_errors) if rel_errors else float('nan'),
        'pairwise_acc': np.mean(pw_correct_list) if pw_correct_list else float('nan'),
    }


# ======================================================================
# Single benchmark run
# ======================================================================
def run_benchmark(model, road_prob, batch_nodes_yx, H_img, *,
                  ds, eik_iters, parallel_cases, disable_checkpoint,
                  do_backward=True, warmup=True):
    """Run one configuration and return timing + memory stats."""
    B = len(batch_nodes_yx)
    kwargs = dict(
        road_prob=road_prob.float(),
        ds=ds, eik_iters=eik_iters, pool_mode='max',
        iter_floor_c=1.2, iter_floor_f=0.65,
        parallel_cases=parallel_cases,
        disable_checkpoint=disable_checkpoint,
    )

    # warmup run (avoids CUDA kernel compilation delays)
    if warmup:
        model.zero_grad()
        with torch.amp.autocast("cuda", enabled=False):
            D_w = model.forward_distance_matrix_batch(batch_nodes_yx, **kwargs)
        if do_backward:
            D_w.sum().backward()
        del D_w
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # --- forward ---
    model.zero_grad()
    t0 = time.time()
    with torch.amp.autocast("cuda", enabled=False):
        D = model.forward_distance_matrix_batch(batch_nodes_yx, **kwargs)
    torch.cuda.synchronize()
    t_fwd = time.time() - t0

    D_out = D.detach().clone()
    mem_fwd = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- backward ---
    t_bwd = 0.0
    grad_norm = 0.0
    if do_backward:
        loss = D.sum()
        torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t0

        for p in [model.cost_log_alpha, model.cost_log_gamma, model.eik_gate_logit]:
            if p.grad is not None:
                grad_norm += p.grad.data.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

    mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # normalize distances
    D_norm = D_out.float() / float(H_img)

    return {
        't_fwd': t_fwd,
        't_bwd': t_bwd,
        't_total': t_fwd + t_bwd,
        'mem_fwd_mb': mem_fwd,
        'mem_peak_mb': mem_peak,
        'grad_norm': grad_norm,
        'D_norm': D_norm,
    }


# ======================================================================
# Main benchmark
# ======================================================================
def main():
    print("=" * 80)
    print("E2E EIKONAL ACCELERATION BENCHMARK")
    print("=" * 80)

    # --- setup ---
    project_root = os.path.dirname(SCRIPT_DIR)
    if not os.path.isdir(os.path.join(project_root, 'MMDataset')):
        project_root = SCRIPT_DIR
    dataset_root = os.path.join(project_root, 'MMDataset', 'Gen_dataset_V2', 'Gen_dataset')
    if not os.path.isdir(dataset_root):
        dataset_root = os.path.join(SCRIPT_DIR, 'Gen_dataset_V2', 'Gen_dataset')

    print("\nLoading SAMRoute model...")
    model, patch_size = load_samroute()
    print(f"  cost_log_alpha = {model.cost_log_alpha.item():.4f}")
    print(f"  cost_log_gamma = {model.cost_log_gamma.item():.4f}")
    print(f"  eik_gate_logit = {model.eik_gate_logit.item():.4f}")

    print("\nFinding test data...")
    tif_path, npz_path = find_test_data(dataset_root)
    print(f"  TIF: {tif_path}")
    print(f"  NPZ: {npz_path}")

    print("\nComputing road_prob (one-time cost)...")
    t0 = time.time()
    road_prob, H_img, W_img = prepare_road_prob(tif_path, model, patch_size)
    print(f"  road_prob shape: {road_prob.shape}, time: {time.time()-t0:.2f}s")

    data = np.load(npz_path, allow_pickle=True)
    node_coords_xy = data['matched_node_norm']
    N = node_coords_xy.shape[1]

    npz_dist_norm = None
    for key in ['undirected_dist_norm', 'distance_matrix', 'dist_matrix']:
        if key in data:
            npz_dist_norm = data[key].astype(np.float64)
            print(f"  NPZ distance key: '{key}', shape: {npz_dist_norm.shape}, "
                  f"range: [{npz_dist_norm.min():.4f}, {npz_dist_norm.max():.4f}]")
            break

    B = 4
    batch_nodes_yx = prepare_batch_nodes(node_coords_xy, B, H_img, W_img)
    print(f"\n  Image: {H_img}x{W_img}, N={N}, B={B}")

    # --- configurations to test ---
    configs = [
        # label, ds, eik_iters, parallel, no_ckpt
        ("Baseline (ds=16, serial, ckpt)",      16, 40, False, False),
        ("ds=16, parallel, ckpt",               16, 40, True,  False),
        ("ds=16, serial, NO ckpt",              16, 40, False, True),
        ("ds=16, parallel, NO ckpt",            16, 40, True,  True),
        ("ds=32, serial, ckpt",                 32, 40, False, False),
        ("ds=32, parallel, ckpt",               32, 40, True,  False),
        ("ds=32, parallel, NO ckpt",            32, 40, True,  True),
        ("ds=64, parallel, ckpt",               64, 40, True,  False),
        ("ds=64, parallel, NO ckpt",            64, 40, True,  True),
        # eik_iters sweep (parallel + ds=32)
        ("ds=32, parallel, iters=30",           32, 30, True,  False),
        ("ds=32, parallel, iters=20",           32, 20, True,  False),
        # eik_iters sweep (parallel + ds=16)
        ("ds=16, parallel, iters=30",           16, 30, True,  False),
        ("ds=16, parallel, iters=20",           16, 20, True,  False),
    ]

    results = []
    print("\n" + "=" * 120)
    header = f"{'Configuration':<40s} | {'Fwd(s)':>7s} | {'Bwd(s)':>7s} | {'Total':>7s} | {'MemPk':>7s} | {'GradN':>8s} | {'RelErr':>7s} | {'PW_Acc':>7s}"
    print(header)
    print("-" * 120)

    for label, ds, eik_iters, parallel, no_ckpt in configs:
        torch.cuda.empty_cache()
        gc.collect()

        try:
            r = run_benchmark(
                model, road_prob, batch_nodes_yx, H_img,
                ds=ds, eik_iters=eik_iters,
                parallel_cases=parallel,
                disable_checkpoint=no_ckpt,
            )

            acc = {'rel_err_mean': float('nan'), 'pairwise_acc': float('nan')}
            if npz_dist_norm is not None:
                acc = compute_accuracy_metrics(r['D_norm'], npz_dist_norm[:B], B, N)

            row = {
                'label': label,
                **r,
                **acc,
            }
            results.append(row)

            print(f"{label:<40s} | {r['t_fwd']:7.3f} | {r['t_bwd']:7.3f} | "
                  f"{r['t_total']:7.3f} | {r['mem_peak_mb']:6.0f}M | "
                  f"{r['grad_norm']:8.4f} | {acc['rel_err_mean']:7.4f} | "
                  f"{acc['pairwise_acc']:7.4f}")
        except Exception as e:
            print(f"{label:<40s} | FAILED: {e}")
            results.append({'label': label, 'error': str(e)})

    # --- training time estimates ---
    print("\n" + "=" * 80)
    print("TRAINING TIME ESTIMATES (B=4, episodes=4000, epochs=400)")
    print("=" * 80)
    batches_per_epoch = 4000 // B
    epochs = 400
    nco_overhead = 0.05

    for r in results:
        if 'error' in r:
            continue
        t = r['t_total'] + nco_overhead
        t_epoch = batches_per_epoch * t
        t_total = t_epoch * epochs
        speedup = 1.0
        baseline_t = next((x['t_total'] for x in results
                          if x.get('label', '').startswith('Baseline')), None)
        if baseline_t:
            speedup = (baseline_t + nco_overhead) / t
        print(f"  {r['label']:<40s}: {t:.2f}s/batch -> {t_total/3600:.1f}h total "
              f"(speedup: {speedup:.1f}x)")

    print("\nDone.")


if __name__ == "__main__":
    main()
