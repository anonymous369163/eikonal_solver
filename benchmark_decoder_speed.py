#!/usr/bin/env python3
"""
Benchmark: measure decoder forward/backward speed for _compute_road_prob_differentiable.

Tests:
  1. Current serial path (one patch at a time)
  2. Batched path (chunked map_decoder)
  3. Numerical consistency between the two
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
    model.image_encoder = None
    model.requires_grad_(False)
    model.cost_log_alpha.requires_grad_(True)
    model.cost_log_gamma.requires_grad_(True)
    model.eik_gate_logit.requires_grad_(True)
    for p in model.map_decoder.parameters():
        p.requires_grad_(True)
    return model


def load_encoder_cache(cache_path):
    data = torch.load(cache_path, map_location='cpu', weights_only=False)
    features_stacked = data['features']  # [N, 256, 32, 32] float16
    patches_info = data['patches_info']
    win2d = data['win2d']
    H, W = data['H'], data['W']
    return patches_info, features_stacked.float(), win2d, H, W


def road_prob_serial(model, patches_info, features_list, win2d, H, W, device):
    """Current serial implementation (one patch at a time)."""
    win_t = torch.from_numpy(win2d).to(device)
    prob_sum = torch.zeros(H, W, device=device)
    weight_sum = torch.zeros(H, W, device=device)

    for (y0, x0, ph, pw), feat in zip(patches_info, features_list):
        feat_gpu = feat.to(device)
        _, ms = model._predict_mask_logits_scores(None, encoder_feat=feat_gpu)
        prob = ms[0, :, :, 1]

        prob_sum[y0:y0 + ph, x0:x0 + pw] = (
            prob_sum[y0:y0 + ph, x0:x0 + pw] +
            prob[:ph, :pw] * win_t[:ph, :pw])
        weight_sum[y0:y0 + ph, x0:x0 + pw] = (
            weight_sum[y0:y0 + ph, x0:x0 + pw] +
            win_t[:ph, :pw])

    safe_weight = weight_sum.clamp(min=1e-6)
    road_prob = (prob_sum / safe_weight).unsqueeze(0)
    return road_prob.clamp(1e-6, 1.0 - 1e-6)


def road_prob_batched(model, patches_info, features_stacked, win2d, H, W, device, chunk_size=32):
    """Batched implementation: one GPU transfer + chunked map_decoder."""
    all_feat = features_stacked.to(device)  # [N, 256, 32, 32]
    win_t = torch.from_numpy(win2d).to(device)

    all_probs = []
    for i in range(0, all_feat.shape[0], chunk_size):
        chunk = all_feat[i:i + chunk_size]
        logits = model.map_decoder(chunk)              # [C, 2, H_patch, W_patch]
        probs = torch.sigmoid(logits[:, 1, :, :])      # [C, H_patch, W_patch]
        all_probs.append(probs)
    all_probs = torch.cat(all_probs, dim=0)  # [N, H_patch, W_patch]

    prob_sum = torch.zeros(H, W, device=device)
    weight_sum = torch.zeros(H, W, device=device)

    for idx, (y0, x0, ph, pw) in enumerate(patches_info):
        prob_sum[y0:y0 + ph, x0:x0 + pw] += all_probs[idx, :ph, :pw] * win_t[:ph, :pw]
        weight_sum[y0:y0 + ph, x0:x0 + pw] += win_t[:ph, :pw]

    safe_weight = weight_sum.clamp(min=1e-6)
    road_prob = (prob_sum / safe_weight).unsqueeze(0)
    return road_prob.clamp(1e-6, 1.0 - 1e-6)


def benchmark(fn, model, n_warmup=2, n_runs=5, backward=True):
    """Benchmark a road_prob function with forward + optional backward."""
    times_fwd = []
    times_bwd = []

    for i in range(n_warmup + n_runs):
        model.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        rp = fn()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if backward:
            loss = rp.sum()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
        else:
            t2 = t1

        if i >= n_warmup:
            times_fwd.append(t1 - t0)
            times_bwd.append(t2 - t1)

    fwd_ms = np.mean(times_fwd) * 1000
    bwd_ms = np.mean(times_bwd) * 1000
    return fwd_ms, bwd_ms


def find_encoder_cache():
    import glob
    files = sorted(glob.glob(os.path.join(
        SCRIPT_DIR, 'Gen_dataset_V2', 'Gen_dataset', '*', '*', 'encoder_cache.pt')))
    if not files:
        raise RuntimeError("No encoder_cache.pt found")
    mid = len(files) // 2
    return files[mid]


def main():
    print("=" * 70)
    print(" Decoder Speed Benchmark")
    print("=" * 70)

    print("\n[1] Loading SAMRoute (decoder only)...")
    model = load_samroute()
    model.train()
    print("   Done.")

    print("\n[2] Loading encoder cache...")
    cache_path = find_encoder_cache()
    patches_info, features_stacked, win2d, H, W = load_encoder_cache(cache_path)
    N = features_stacked.shape[0]
    print(f"   {os.path.basename(os.path.dirname(cache_path))}: {H}x{W}, {N} patches")

    features_list = [features_stacked[i:i+1] for i in range(N)]

    print("\n" + "=" * 70)
    print(" A. Serial decoder (current implementation)")
    print("=" * 70)

    # Forward + backward with grad
    fwd_ms, bwd_ms = benchmark(
        lambda: road_prob_serial(model, patches_info, features_list, win2d, H, W, device),
        model, n_warmup=1, n_runs=3, backward=True)
    print(f"   [with grad]  Forward: {fwd_ms:.1f} ms, Backward: {bwd_ms:.1f} ms, Total: {fwd_ms+bwd_ms:.1f} ms")

    # Forward only, no_grad
    with torch.no_grad():
        fwd_ng, _ = benchmark(
            lambda: road_prob_serial(model, patches_info, features_list, win2d, H, W, device),
            model, n_warmup=1, n_runs=3, backward=False)
    print(f"   [no_grad]    Forward: {fwd_ng:.1f} ms")

    rp_serial = road_prob_serial(model, patches_info, features_list, win2d, H, W, device).detach()

    print("\n" + "=" * 70)
    print(" B. Batched decoder (chunk_size=32)")
    print("=" * 70)

    fwd_ms32, bwd_ms32 = benchmark(
        lambda: road_prob_batched(model, patches_info, features_stacked, win2d, H, W, device, chunk_size=32),
        model, n_warmup=1, n_runs=3, backward=True)
    print(f"   [with grad]  Forward: {fwd_ms32:.1f} ms, Backward: {bwd_ms32:.1f} ms, Total: {fwd_ms32+bwd_ms32:.1f} ms")

    with torch.no_grad():
        fwd_ng32, _ = benchmark(
            lambda: road_prob_batched(model, patches_info, features_stacked, win2d, H, W, device, chunk_size=32),
            model, n_warmup=1, n_runs=3, backward=False)
    print(f"   [no_grad]    Forward: {fwd_ng32:.1f} ms")

    rp_batched32 = road_prob_batched(model, patches_info, features_stacked, win2d, H, W, device, chunk_size=32).detach()

    print("\n" + "=" * 70)
    print(" C. Batched decoder (chunk_size=64)")
    print("=" * 70)

    fwd_ms64, bwd_ms64 = benchmark(
        lambda: road_prob_batched(model, patches_info, features_stacked, win2d, H, W, device, chunk_size=64),
        model, n_warmup=1, n_runs=3, backward=True)
    print(f"   [with grad]  Forward: {fwd_ms64:.1f} ms, Backward: {bwd_ms64:.1f} ms, Total: {fwd_ms64+bwd_ms64:.1f} ms")

    with torch.no_grad():
        fwd_ng64, _ = benchmark(
            lambda: road_prob_batched(model, patches_info, features_stacked, win2d, H, W, device, chunk_size=64),
            model, n_warmup=1, n_runs=3, backward=False)
    print(f"   [no_grad]    Forward: {fwd_ng64:.1f} ms")

    print("\n" + "=" * 70)
    print(" D. Numerical consistency")
    print("=" * 70)

    diff = (rp_serial - rp_batched32).abs()
    print(f"   Serial vs Batched(32): max_diff={diff.max().item():.2e}, mean_diff={diff.mean().item():.2e}")

    print("\n" + "=" * 70)
    print(" E. Summary")
    print("=" * 70)
    print(f"   {'Method':<25} {'Fwd(ms)':>8} {'Bwd(ms)':>8} {'Total(ms)':>10} {'Speedup':>8}")
    print(f"   {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    serial_total = fwd_ms + bwd_ms
    batch32_total = fwd_ms32 + bwd_ms32
    batch64_total = fwd_ms64 + bwd_ms64

    print(f"   {'Serial (grad)':<25} {fwd_ms:>8.1f} {bwd_ms:>8.1f} {serial_total:>10.1f} {'1.0x':>8}")
    print(f"   {'Batched/32 (grad)':<25} {fwd_ms32:>8.1f} {bwd_ms32:>8.1f} {batch32_total:>10.1f} {serial_total/batch32_total:>7.1f}x")
    print(f"   {'Batched/64 (grad)':<25} {fwd_ms64:>8.1f} {bwd_ms64:>8.1f} {batch64_total:>10.1f} {serial_total/batch64_total:>7.1f}x")
    print(f"   {'Serial (no_grad)':<25} {fwd_ng:>8.1f} {'N/A':>8} {fwd_ng:>10.1f} {'---':>8}")
    print(f"   {'Batched/32 (no_grad)':<25} {fwd_ng32:>8.1f} {'N/A':>8} {fwd_ng32:>10.1f} {fwd_ng/fwd_ng32:>7.1f}x")
    print(f"   {'Batched/64 (no_grad)':<25} {fwd_ng64:>8.1f} {'N/A':>8} {fwd_ng64:>10.1f} {fwd_ng/fwd_ng64:>7.1f}x")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n   Peak GPU memory: {peak_mem:.2f} GB")


if __name__ == '__main__':
    main()
