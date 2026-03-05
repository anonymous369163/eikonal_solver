#!/usr/bin/env python3
"""
Quick benchmark: measure per-batch wall time for e2e_eikonal mode.
Runs DEBUG_MODE with encoder_mode='e2e_eikonal', times each component.
"""
import os, sys, time, torch, numpy as np

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
    return os.path.abspath(start_path)

PROJECT_ROOT = find_project_root(SCRIPT_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'MMDataset', 'Gen_dataset_V2', 'Gen_dataset')

EIKONAL_DIR = os.path.join(SCRIPT_DIR, 'eikonal_solver')
if EIKONAL_DIR not in sys.path:
    sys.path.insert(0, EIKONAL_DIR)

device = torch.device('cuda')
torch.cuda.set_device(0)

# --- Load SAMRoute ---
from model_multigrid import SAMRoute
from gradcheck_route_loss_v2_multigrid_fullmap import (
    sliding_window_inference as swi,
    _load_lightning_ckpt, _detect_smooth_decoder, _detect_patch_size,
)
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
print(f"SAMRoute loaded on {device}")

# --- Find test data ---
test_tif, test_npz = None, None
for city in os.listdir(DATASET_ROOT):
    city_dir = os.path.join(DATASET_ROOT, city)
    if not os.path.isdir(city_dir): continue
    for sg in os.listdir(city_dir):
        sg_dir = os.path.join(city_dir, sg)
        if not os.path.isdir(sg_dir): continue
        tifs = [f for f in os.listdir(sg_dir) if f.startswith('crop_') and f.endswith('.tif')]
        npzs = [f for f in os.listdir(sg_dir) if f.endswith('_p20.npz')]
        if tifs and npzs:
            test_tif = os.path.join(sg_dir, tifs[0])
            test_npz = os.path.join(sg_dir, npzs[0])
            break
    if test_tif: break

print(f"TIF: {test_tif}")
print(f"NPZ: {test_npz}")

# --- Load image and NPZ ---
from PIL import Image
img = Image.open(test_tif).convert('RGB')
img_np = np.asarray(img, dtype=np.uint8)
H_img, W_img = img_np.shape[:2]

data = np.load(test_npz)
node_coords_xy = data['matched_node_norm']  # [total_cases, N, 2]
N = node_coords_xy.shape[1]

# ============================================================
# Benchmark 1: road_prob computation (first call vs cached)
# ============================================================
print("\n=== Benchmark: road_prob (sliding_window_inference) ===")
torch.cuda.synchronize()
t0 = time.time()
road_prob_np = swi(img_np, model, device, patch_size=patch_size, verbose=False).astype(np.float32)
torch.cuda.synchronize()
t_road_prob_first = time.time() - t0
road_prob = torch.from_numpy(road_prob_np).to(device, torch.float32).unsqueeze(0).clamp(1e-6, 1.0 - 1e-6)
print(f"  First call: {t_road_prob_first:.2f}s")

torch.cuda.synchronize()
t0 = time.time()
road_prob_np2 = swi(img_np, model, device, patch_size=patch_size, verbose=False).astype(np.float32)
torch.cuda.synchronize()
t_road_prob_second = time.time() - t0
print(f"  Second call (no Python cache): {t_road_prob_second:.2f}s")
print(f"  (With LRU cache this would be ~0ms)")

# ============================================================
# Benchmark 2: forward_distance_matrix_batch for various B
# ============================================================
print("\n=== Benchmark: forward_distance_matrix_batch ===")

for B in [1, 2, 4, 8]:
    if B > node_coords_xy.shape[0]:
        break
    batch_nodes_yx = []
    for b in range(B):
        xy = node_coords_xy[b]
        x_norm = torch.from_numpy(xy[:, 0]).to(device, torch.float32)
        y_norm = torch.from_numpy(xy[:, 1]).to(device, torch.float32)
        y_pix = torch.round((1.0 - y_norm) * (H_img - 1)).long()
        x_pix = torch.round(x_norm * (W_img - 1)).long()
        batch_nodes_yx.append(torch.stack([y_pix, x_pix], dim=-1))

    # ds=16 (current default)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.amp.autocast("cuda", enabled=False):
        D16 = model.forward_distance_matrix_batch(
            batch_nodes_yx, road_prob=road_prob.float(),
            ds=16, eik_iters=40, pool_mode='max',
            iter_floor_c=1.2, iter_floor_f=0.65,
        )
    torch.cuda.synchronize()
    t_ds16 = time.time() - t0

    # ds=32 (optimized)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.amp.autocast("cuda", enabled=False):
        D32 = model.forward_distance_matrix_batch(
            batch_nodes_yx, road_prob=road_prob.float(),
            ds=32, eik_iters=40, pool_mode='max',
            iter_floor_c=1.2, iter_floor_f=0.65,
        )
    torch.cuda.synchronize()
    t_ds32 = time.time() - t0

    print(f"  B={B}: ds=16 -> {t_ds16:.3f}s ({t_ds16/B:.3f}s/case), "
          f"ds=32 -> {t_ds32:.3f}s ({t_ds32/B:.3f}s/case), "
          f"speedup={t_ds16/max(t_ds32,1e-6):.1f}x")

# ============================================================
# Benchmark 3: backward through distance_matrix
# ============================================================
print("\n=== Benchmark: backward (grad through Eikonal) ===")
for B, ds in [(4, 16), (4, 32)]:
    batch_nodes_yx = []
    for b in range(B):
        xy = node_coords_xy[b]
        x_norm = torch.from_numpy(xy[:, 0]).to(device, torch.float32)
        y_norm = torch.from_numpy(xy[:, 1]).to(device, torch.float32)
        y_pix = torch.round((1.0 - y_norm) * (H_img - 1)).long()
        x_pix = torch.round(x_norm * (W_img - 1)).long()
        batch_nodes_yx.append(torch.stack([y_pix, x_pix], dim=-1))

    model.zero_grad()
    with torch.amp.autocast("cuda", enabled=False):
        D = model.forward_distance_matrix_batch(
            batch_nodes_yx, road_prob=road_prob.float(),
            ds=ds, eik_iters=40, pool_mode='max',
            iter_floor_c=1.2, iter_floor_f=0.65,
        )
    loss = D.sum()
    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bw = time.time() - t0
    print(f"  B={B}, ds={ds}: backward = {t_bw:.3f}s")

# ============================================================
# Summary: estimate full training time
# ============================================================
print("\n" + "=" * 70)
print("TRAINING TIME ESTIMATES")
print("=" * 70)

# Measure one full forward+backward cycle for B=4, ds=16
B = 4
batch_nodes_yx = []
for b in range(B):
    xy = node_coords_xy[b]
    x_norm = torch.from_numpy(xy[:, 0]).to(device, torch.float32)
    y_norm = torch.from_numpy(xy[:, 1]).to(device, torch.float32)
    y_pix = torch.round((1.0 - y_norm) * (H_img - 1)).long()
    x_pix = torch.round(x_norm * (W_img - 1)).long()
    batch_nodes_yx.append(torch.stack([y_pix, x_pix], dim=-1))

model.zero_grad()
torch.cuda.synchronize()
t_total_start = time.time()
with torch.amp.autocast("cuda", enabled=False):
    D = model.forward_distance_matrix_batch(
        batch_nodes_yx, road_prob=road_prob.float(),
        ds=16, eik_iters=40, pool_mode='max',
        iter_floor_c=1.2, iter_floor_f=0.65,
    )
loss = D.sum()
loss.backward()
torch.cuda.synchronize()
t_total_ds16 = time.time() - t_total_start

model.zero_grad()
torch.cuda.synchronize()
t_total_start = time.time()
with torch.amp.autocast("cuda", enabled=False):
    D = model.forward_distance_matrix_batch(
        batch_nodes_yx, road_prob=road_prob.float(),
        ds=32, eik_iters=40, pool_mode='max',
        iter_floor_c=1.2, iter_floor_f=0.65,
    )
loss = D.sum()
loss.backward()
torch.cuda.synchronize()
t_total_ds32 = time.time() - t_total_start

# Current config after auto-adjust: B=4, episodes=4000, epochs=400
batches_per_epoch = 4000 // 4  # = 1000
epochs = 400

# Add ~50ms for NCO forward+backward per batch
nco_overhead = 0.05

for ds, t_batch in [(16, t_total_ds16 + nco_overhead), (32, t_total_ds32 + nco_overhead)]:
    t_epoch = batches_per_epoch * t_batch
    t_total = t_epoch * epochs
    print(f"\nds={ds}, B=4, episodes=4000, epochs=400:")
    print(f"  Per batch (Eikonal fwd+bwd + NCO): {t_batch:.2f}s")
    print(f"  Per epoch ({batches_per_epoch} batches): {t_epoch:.0f}s = {t_epoch/60:.1f}min")
    print(f"  Full training ({epochs} epochs): {t_total:.0f}s = {t_total/3600:.1f}h")

print(f"\nFor comparison, graph_only (B=256, episodes=100k, epochs=400):")
t_go_batch = 0.1
batches_go = 100000 // 256
t_go_epoch = batches_go * t_go_batch
t_go_total = t_go_epoch * epochs
print(f"  Per epoch ({batches_go} batches x {t_go_batch}s): {t_go_epoch:.0f}s = {t_go_epoch/60:.1f}min")
print(f"  Full training: {t_go_total:.0f}s = {t_go_total/3600:.1f}h")
