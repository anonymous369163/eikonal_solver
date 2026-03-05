#!/usr/bin/env python3
"""
Pre-compute road probability maps for all satellite images and save to disk.

Each satellite TIF in Gen_dataset_V2/Gen_dataset/*/*/crop_*.tif gets a
corresponding road_prob_cache.npz saved in the same directory.

This allows the e2e_eikonal training to skip the SAMRoute encoder entirely
(~350MB GPU memory, ~4s per first inference), loading pre-computed road_prob
from disk in ~0.01s.

Metadata stored in each .npz:
  - road_prob: float16 array (H, W)
  - ckpt_path: checkpoint used for inference
  - timestamp: ISO 8601 creation time

Usage:
    python precompute_road_prob.py [--device cuda] [--ckpt PATH] [--force]
"""
import os, sys, time, argparse, glob, hashlib, datetime
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

EIKONAL_DIR = os.path.join(SCRIPT_DIR, 'eikonal_solver')
if EIKONAL_DIR not in sys.path:
    sys.path.insert(0, EIKONAL_DIR)


def load_samroute(ckpt_path, device):
    from model_multigrid import SAMRoute
    from gradcheck_route_loss_v2_multigrid_fullmap import _load_lightning_ckpt
    from types import SimpleNamespace

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
    model.eval()
    model.requires_grad_(False)
    return model, patch_size


def main():
    parser = argparse.ArgumentParser(description='Pre-compute road probability maps')
    parser.add_argument('--device', default='cuda', help='torch device')
    parser.add_argument('--ckpt', default=os.path.join(
        SCRIPT_DIR, 'training_outputs', 'fulldataset_seg_dist_lora_v2',
        'checkpoints', 'best_seg_dist_lora_v2.ckpt'))
    parser.add_argument('--force', action='store_true', help='overwrite existing caches')
    parser.add_argument('--dataset-root', default=os.path.join(
        SCRIPT_DIR, 'Gen_dataset_V2', 'Gen_dataset'))
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(0)

    tif_files = sorted(glob.glob(os.path.join(args.dataset_root, '*', '*', 'crop_*.tif')))
    print(f"Found {len(tif_files)} satellite TIF files")

    if not tif_files:
        print("No TIF files found, exiting.")
        return

    existing = 0
    to_process = []
    for tif in tif_files:
        cache_path = os.path.join(os.path.dirname(tif), 'road_prob_cache.npz')
        if os.path.exists(cache_path) and not args.force:
            existing += 1
        else:
            to_process.append(tif)

    print(f"  Already cached: {existing}")
    print(f"  To process: {len(to_process)}")

    if not to_process:
        print("All road_prob maps already cached. Use --force to recompute.")
        return

    ckpt_abs = os.path.abspath(args.ckpt)
    ckpt_basename = os.path.basename(ckpt_abs)
    print(f"\nLoading SAMRoute from {ckpt_abs}...")
    model, patch_size = load_samroute(ckpt_abs, device)
    print(f"SAMRoute loaded (patch_size={patch_size})")

    from gradcheck_route_loss_v2_multigrid_fullmap import sliding_window_inference as swi
    from PIL import Image

    total_time = 0.0
    for idx, tif_path in enumerate(to_process):
        cache_path = os.path.join(os.path.dirname(tif_path), 'road_prob_cache.npz')
        rel_path = os.path.relpath(tif_path, args.dataset_root)

        t0 = time.time()
        img = Image.open(tif_path).convert('RGB')
        img_np = np.asarray(img, dtype=np.uint8)
        H, W = img_np.shape[:2]

        with torch.no_grad():
            road_prob_np = swi(
                img_np, model, device,
                patch_size=patch_size, verbose=False,
            ).astype(np.float16)

        np.savez_compressed(
            cache_path,
            road_prob=road_prob_np,
            ckpt_path=np.array(ckpt_abs, dtype=object),
            ckpt_name=np.array(ckpt_basename, dtype=object),
            timestamp=np.array(datetime.datetime.now().isoformat(), dtype=object),
        )
        elapsed = time.time() - t0
        total_time += elapsed

        file_size = os.path.getsize(cache_path) / (1024 * 1024)
        eta = (total_time / (idx + 1)) * (len(to_process) - idx - 1)
        print(f"  [{idx+1}/{len(to_process)}] {rel_path}: "
              f"{H}x{W} -> {file_size:.1f}MB, {elapsed:.1f}s "
              f"(ETA: {eta/60:.0f}min)")

    print(f"\nDone! Processed {len(to_process)} images in {total_time:.0f}s "
          f"({total_time/60:.1f}min)")
    print(f"Total cached: {existing + len(to_process)}")
    print(f"Checkpoint used: {ckpt_abs}")


if __name__ == '__main__':
    main()
