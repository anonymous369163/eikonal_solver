"""
Finetune SAMRoute map_decoder on Gen_dataset_V2/Gen_dataset (multi-image batch training).

Optimized for multi-GPU (DDP), large batch, TF32, cached features, high-throughput data loading.
Supports Single-Region Overfitting for debugging.
"""
import argparse
import os
import glob
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataclasses import replace as dc_replace

# 开启 TF32：4090 等 Ada 架构上矩阵乘法可加速 2~3 倍，几乎无精度损失
torch.set_float32_matmul_precision("high")

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

from eikonal import EikonalConfig, eikonal_soft_sweeping, make_source_mask
from model_multigrid import SAMRoute, _prob_to_cost_additive
from dataset import build_dataloaders

# ==========================================
# 1. 训练配置
# ==========================================
class TrainConfig:
    def __init__(self):
        self.SAM_VERSION = 'vit_b'
        self.PATCH_SIZE = 512
        self.NO_SAM = False
        self.USE_SAM_DECODER = False
        # When True, the final ConvTranspose2d(32→2) is replaced by
        # ConvTranspose2d(32→32) + Conv2d(32→2, k=3) to blend ViT token-grid
        # boundaries and eliminate 16-px checkerboard artifacts.
        self.USE_SMOOTH_DECODER = False
        self.ENCODER_LORA = False
        self.LORA_RANK = 4   # 仅当 ENCODER_LORA=True 时生效
        self.FREEZE_ENCODER = True   # 绝对冻结 SAM Encoder，只训练 Decoder（LoRA 模式必须 False）
        self.FOCAL_LOSS = False
        self.TOPONET_VERSION = 'default'

        _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        _PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
        self.SAM_CKPT_PATH = os.path.join(_PROJECT_ROOT, "sam_road_repo", "sam_ckpts", "sam_vit_b_01ec64.pth")

        # 路由/损失配置
        self.ROUTE_COST_MODE = 'add'
        self.ROUTE_ADD_ALPHA = 20.0
        self.ROUTE_ADD_GAMMA = 2.0
        self.ROUTE_ADD_BLOCK_ALPHA = 0.0
        self.ROUTE_BLOCK_TH = 0.0
        self.ROAD_POS_WEIGHT = 5.0

        self.ROAD_DICE_WEIGHT = 0.8
        self.ROAD_DUAL_TARGET = True
        self.ROAD_THIN_BOOST = 10.0

        self.ROUTE_LAMBDA_SEG = 1.0
        self.ROUTE_LAMBDA_DIST = 0.0
        self.BASE_LR = 5e-4
        self.ENCODER_LR_FACTOR = 0.1

        # neural residual cost net (disabled by default for finetune)
        self.ROUTE_COST_NET = False
        self.ROUTE_COST_NET_CH = 8

        # Eikonal solver configuration (active when ROUTE_LAMBDA_DIST > 0)
        # Aligned with eval/test SWEEP-tuned optimal values
        self.ROUTE_EIK_ITERS = 40
        self.ROUTE_EIK_DOWNSAMPLE = 16
        self.ROUTE_EIK_MODE = "soft_train"
        self.ROUTE_CAP_MODE = "tanh"
        self.ROUTE_CAP_MULT = 10.0
        self.ROUTE_CKPT_CHUNK = 10
        self.ROUTE_DIST_NORM_PX = 512.0
        self.ROUTE_GATE_ALPHA = 0.8
        self.ROUTE_ROI_MARGIN = 64
        self.ROUTE_EIK_WARMUP_EPOCHS = 5
        self.ROUTE_EIK_ITERS_MIN = 15
        self.ROUTE_DIST_WARMUP_STEPS = 500

        # Multigrid warm-start (coarse→fine, matching eval/test pipeline)
        self.ROUTE_MULTIGRID = True
        self.ROUTE_MG_FACTOR = 4
        self.ROUTE_MG_ITERS_COARSE = 0
        self.ROUTE_MG_ITERS_FINE = 0
        self.ROUTE_MG_DETACH_COARSE = False

        self.LR_MILESTONES = [150]
        self.LR_SCHEDULER = "multistep"   # "multistep" | "cosine"
        self.LR_WARMUP_EPOCHS = 0
        self.LR_MIN = 1e-6
        self.LEARNABLE_BLOCK = False

    def get(self, key, default):
        return getattr(self, key, default)

# ==========================================
# 1b. Standalone GT distance pre-computation (CPU-only, no model dependency)
# ==========================================

def _maxpool_2d(prob: torch.Tensor, ds: int) -> torch.Tensor:
    """Max-pool a [H, W] or [1, H, W] tensor by factor *ds*."""
    ds = max(1, int(ds))
    if ds == 1:
        return prob
    if prob.dim() == 2:
        prob = prob.unsqueeze(0)
    B, H, W = prob.shape
    pad_h = (ds - (H % ds)) % ds
    pad_w = (ds - (W % ds)) % ds
    if pad_h or pad_w:
        prob = F.pad(prob, (0, pad_w, 0, pad_h), value=0.0)
    return F.max_pool2d(prob.unsqueeze(1), kernel_size=ds, stride=ds).squeeze(1)


def _precompute_gt_dist_for_patch(
    road_mask_np: np.ndarray,
    *,
    k_targets: int = 8,
    src_onroad_p: float = 0.9,
    tgt_onroad_p: float = 0.9,
    min_euclid_px: int = 64,
    max_euclid_px: int = 0,
    teacher_alpha: float = 20.0,
    teacher_gamma: float = 2.0,
    eik_iters: int = 200,
    eik_downsample: int = 4,
    gt_cap_factor: float = 0.85,
    detour_cap: float = 12.0,
    prob_eps: float = 1e-4,
    large_val: float = 1e6,
    teacher_mask_thin: np.ndarray | None = None,
    use_thin: bool = False,
) -> dict:
    """Pre-compute GT distance for one patch on CPU. Returns CPU tensors."""
    mask_np = teacher_mask_thin if (use_thin and teacher_mask_thin is not None) else road_mask_np
    H, W = mask_np.shape[:2]
    mask_t = torch.tensor((mask_np > 127).astype(np.float32), dtype=torch.float32).unsqueeze(0)  # [1,H,W]

    # --- sample src + K targets ---
    w_flat = mask_t[0].reshape(-1)  # [H*W]
    has_road = bool(w_flat.sum().item() > 0)
    k = k_targets
    oversample = max(8, 2 * k)

    if has_road and random.random() < src_onroad_p:
        idx = torch.multinomial(w_flat, num_samples=1, replacement=True)[0]
        sy, sx = (idx // W).long().item(), (idx % W).long().item()
    else:
        sy = random.randint(0, H - 1)
        sx = random.randint(0, W - 1)
    src_yx = torch.tensor([sy, sx], dtype=torch.long)  # [2]

    cand_n = k * oversample
    n_road = int(cand_n * tgt_onroad_p)
    n_rand = cand_n - n_road
    cand_parts = []
    if has_road and n_road > 0:
        idx_r = torch.multinomial(w_flat, num_samples=n_road, replacement=True)
        cand_parts.append(torch.stack([(idx_r // W).long(), (idx_r % W).long()], dim=1))
    if n_rand > 0:
        y_u = torch.randint(0, H, (n_rand,), dtype=torch.long)
        x_u = torch.randint(0, W, (n_rand,), dtype=torch.long)
        cand_parts.append(torch.stack([y_u, x_u], dim=1))
    cand_yx = torch.cat(cand_parts, dim=0) if len(cand_parts) > 1 else cand_parts[0]

    d = torch.norm(cand_yx.float() - src_yx.float(), dim=1)
    mask_d = d >= float(min_euclid_px)
    if max_euclid_px > 0:
        mask_d = mask_d & (d <= float(max_euclid_px))
    good = cand_yx[mask_d]
    if good.numel() == 0:
        good = cand_yx
    take = min(k, good.shape[0])
    tgt_yx = torch.full((k, 2), -1, dtype=torch.long)
    tgt_yx[:take] = good[:take]
    if take < k:
        y_f = torch.randint(0, H, (k - take,), dtype=torch.long)
        x_f = torch.randint(0, W, (k - take,), dtype=torch.long)
        tgt_yx[take:k] = torch.stack([y_f, x_f], dim=1)

    # --- teacher Eikonal on CPU ---
    ds = max(1, eik_downsample)
    prob = mask_t.clamp(prob_eps, 1.0 - prob_eps)
    prob_ds = _maxpool_2d(prob, ds)  # [1, Hf, Wf]
    Hf, Wf = prob_ds.shape[-2], prob_ds.shape[-1]

    cost = _prob_to_cost_additive(
        prob_ds, alpha=teacher_alpha, gamma=teacher_gamma,
        block_th=0.0, block_alpha=0.0, block_smooth=50.0, eps=1e-6,
    ).to(dtype=torch.float32)

    src_c = (src_yx // ds).clamp(min=0)
    src_c[0].clamp_(0, Hf - 1)
    src_c[1].clamp_(0, Wf - 1)
    src_mask = make_source_mask(Hf, Wf, src_c.view(1, 1, 2))

    cfg_t = EikonalConfig(
        n_iters=eik_iters, h=float(ds), mode="hard_eval", monotone=True,
        large_val=large_val,
    )
    with torch.no_grad():
        T = eikonal_soft_sweeping(cost, src_mask, cfg_t)  # [1,Hf,Wf]

    tgt_c = (tgt_yx // ds)
    tgt_c_y = tgt_c[:, 0].clamp(0, Hf - 1)
    tgt_c_x = tgt_c[:, 1].clamp(0, Wf - 1)
    lin = (tgt_c_y * Wf + tgt_c_x).view(1, k)
    gt_dist = torch.gather(T.view(1, -1), 1, lin).squeeze(0)  # [K]

    # --- filter invalid pairs ---
    euclid = torch.norm(tgt_yx.float() - src_yx.float(), dim=1).clamp_min(1.0)
    cap_px = float(W) * gt_cap_factor
    valid = (
        (gt_dist > 0)
        & (gt_dist < cap_px)
        & (gt_dist < large_val * 0.9)
        & ((gt_dist / euclid) < detour_cap)
    )
    tgt_yx_out = torch.where(valid[:, None], tgt_yx, torch.full_like(tgt_yx, -1))
    gt_dist_out = torch.where(valid, gt_dist, torch.full_like(gt_dist, -1.0))

    return {
        'src_yx': src_yx,        # [2]
        'tgt_yx': tgt_yx_out,    # [K, 2]
        'gt_dist': gt_dist_out,  # [K]
    }


# ==========================================
# 2. 单图过拟合专属 Dataset (极速版)
# ==========================================
class SingleRegionDataset(Dataset):
    def __init__(self, region_dir, config, samples_per_epoch=200,
                 dist_cache_cfg: dict | None = None):
        self.patch_size = config.PATCH_SIZE
        self.steps = samples_per_epoch
        self.config = config
        self.region_dir = region_dir

        print(f"Loading single-region data: {region_dir}")
        # 1. TIF
        tifs = glob.glob(os.path.join(region_dir, "crop_*.tif"))
        img_array = tifffile.imread(tifs[0])
        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            img_array = img_array[:, :, :3]
        elif img_array.ndim == 3 and img_array.shape[0] >= 3: 
            img_array = img_array[:3, :, :].transpose(1, 2, 0)
        if img_array.dtype == np.uint16:
            img_array = (img_array / 256.0).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        self.img_array = img_array

        # 2. Masks
        masks_thick = glob.glob(os.path.join(region_dir, "roadnet_normalized_*.png"))
        masks_skel = glob.glob(os.path.join(region_dir, "roadnet_skeleton*.png"))
        masks_orig = [m for m in glob.glob(os.path.join(region_dir, "roadnet_*.png"))
                      if "normalized" not in m and "skeleton" not in m]

        mask_path = masks_thick[0] if masks_thick else masks_orig[0]
        self.mask_array = np.array(Image.open(mask_path).convert('L'))

        if config.ROAD_DUAL_TARGET:
            if masks_skel:
                self.thin_mask_array = np.array(Image.open(masks_skel[0]).convert('L'))
            elif masks_orig:
                self.thin_mask_array = np.array(Image.open(masks_orig[0]).convert('L'))
            else:
                self.thin_mask_array = None
        else:
            self.thin_mask_array = None

        # 3. Cached features
        feats = glob.glob(os.path.join(region_dir, "samroad_feat_full_*.npy"))
        if getattr(config, 'ENCODER_LORA', False):
            self.feat_array = None
            print("LoRA mode: skip cached_feature, compute encoder online")
        elif feats:
            self.feat_array = np.load(feats[0]).astype(np.float32)
            print("Found cached_feature, fast finetuning enabled")
        else:
            self.feat_array = None
            print("No cached_feature found, will run encoder online")

        # 4. Pre-compute GT distance cache (CPU-only, one-time cost)
        self._dist_cache = None
        self._patch_positions = None
        if dist_cache_cfg is not None:
            h, w = self.img_array.shape[:2]
            ps = self.patch_size
            self._patch_positions = []
            for _ in range(samples_per_epoch):
                py = random.randint(0, max(0, h - ps))
                px = random.randint(0, max(0, w - ps))
                py = (py // 16) * 16
                px = (px // 16) * 16
                self._patch_positions.append((py, px))
            self._build_dist_cache(dist_cache_cfg)

    def _build_dist_cache(self, cfg: dict):
        """Pre-compute teacher GT distances for all fixed patches."""
        ps = self.patch_size
        use_thin = cfg.get('use_thin', False)
        cfg = {k: v for k, v in cfg.items() if k != 'use_thin'}

        import hashlib, json
        _hash_keys = sorted(cfg.items())
        _hash_keys.append(('use_thin', use_thin))
        param_hash = hashlib.md5(
            json.dumps(_hash_keys, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        cache_tag = (f"gt_dist_cache_ps{ps}"
                     f"_k{cfg.get('k_targets',8)}"
                     f"_ds{cfg.get('eik_downsample',4)}"
                     f"_n{len(self._patch_positions)}"
                     f"_{param_hash}.pt")
        cache_path = os.path.join(self.region_dir, cache_tag)

        if os.path.exists(cache_path):
            print(f"Loading GT distance cache from {cache_path} ...")
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
            self._patch_positions = data['positions']
            self._dist_cache = data['cache']
            print(f"Loaded {len(self._dist_cache)} cached patches.")
            return

        print(f"Pre-computing GT distances for {len(self._patch_positions)} patches (CPU) ...")
        t0 = time.time()
        cache = []
        for i, (y, x) in enumerate(self._patch_positions):
            mask_crop = self.mask_array[y:y+ps, x:x+ps]
            pad_y = max(0, ps - mask_crop.shape[0])
            pad_x = max(0, ps - mask_crop.shape[1])
            if pad_y > 0 or pad_x > 0:
                mask_crop = np.pad(mask_crop, ((0, pad_y), (0, pad_x)))

            thin_crop = None
            if use_thin and self.thin_mask_array is not None:
                thin_crop = self.thin_mask_array[y:y+ps, x:x+ps]
                if pad_y > 0 or pad_x > 0:
                    thin_crop = np.pad(thin_crop, ((0, pad_y), (0, pad_x)))

            result = _precompute_gt_dist_for_patch(
                mask_crop, teacher_mask_thin=thin_crop, use_thin=use_thin, **cfg)
            cache.append(result)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(self._patch_positions)}] {elapsed:.1f}s")

        self._dist_cache = cache
        elapsed = time.time() - t0
        print(f"Pre-computation done in {elapsed:.1f}s. Saving to {cache_path}")
        torch.save({'positions': list(self._patch_positions), 'cache': cache}, cache_path)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        h, w = self.img_array.shape[:2]
        ps = self.patch_size
        if self._patch_positions is not None:
            y, x = self._patch_positions[idx]
        else:
            y = random.randint(0, max(0, h - ps))
            x = random.randint(0, max(0, w - ps))
            y = (y // 16) * 16
            x = (x // 16) * 16

        img_crop = self.img_array[y : y + ps, x : x + ps]
        mask_crop = self.mask_array[y : y + ps, x : x + ps]

        pad_y = max(0, ps - img_crop.shape[0])
        pad_x = max(0, ps - img_crop.shape[1])
        if pad_y > 0 or pad_x > 0:
            img_crop = np.pad(img_crop, ((0, pad_y), (0, pad_x), (0, 0)))
            mask_crop = np.pad(mask_crop, ((0, pad_y), (0, pad_x)))

        rgb_t = torch.tensor(img_crop, dtype=torch.float32)

        if self.feat_array is None:
            rgb_f = rgb_t / 255.0
            brightness = 0.8 + torch.rand(1).item() * 0.4
            contrast_shift = (torch.rand(1).item() - 0.5) * 0.2
            rgb_f = (rgb_f * brightness + contrast_shift).clamp(0.0, 1.0)
            rgb_t = rgb_f * 255.0

        sample = {
            'rgb': rgb_t,
            'road_mask': (torch.tensor(mask_crop, dtype=torch.float32) > 0).float()
        }

        if self.thin_mask_array is not None:
            thin_crop = self.thin_mask_array[y : y + ps, x : x + ps]
            if pad_y > 0 or pad_x > 0:
                thin_crop = np.pad(thin_crop, ((0, pad_y), (0, pad_x)))
            sample['road_mask_thin'] = (torch.tensor(thin_crop, dtype=torch.float32) > 0).float()

        if self.feat_array is not None:
            stride = 16
            feat_size = ps // stride
            fy, fx = y // stride, x // stride
            feat_crop = self.feat_array[:, fy : fy + feat_size, fx : fx + feat_size]
            pad_fy = max(0, feat_size - feat_crop.shape[1])
            pad_fx = max(0, feat_size - feat_crop.shape[2])
            if pad_fy > 0 or pad_fx > 0:
                feat_crop = np.pad(feat_crop, ((0, 0), (0, pad_fy), (0, pad_fx)))
            sample['encoder_feat'] = torch.tensor(feat_crop)

        if self._dist_cache is not None:
            c = self._dist_cache[idx]
            sample['src_yx'] = c['src_yx']
            sample['tgt_yx'] = c['tgt_yx']
            sample['gt_dist'] = c['gt_dist']

        return sample


class SAMRouteGTMaskTeacher(SAMRoute):
    """Patch-level distance supervision using GT roadmap inside each patch.

    If the dataloader does not provide (src_yx, tgt_yx, gt_dist), this module
    samples 1 source + K targets per patch, computes pseudo-GT distances by
    running *hard-eval* Eikonal on a GT roadmap mask, then delegates to the
    standard SAMRoute training_step/validation_step (which runs differentiable
    Eikonal on the predicted roadmap).
    """

    def __init__(
        self,
        config,
        *,
        k_targets: int = 8,
        src_onroad_p: float = 0.9,
        tgt_onroad_p: float = 0.9,
        min_euclid_px: int = 64,
        max_euclid_px: int = 0,
        teacher_iters=None,
        teacher_alpha: float = 20.0,
        teacher_gamma: float = 2.0,
        teacher_mask: str = "thick",  # thick | thin
        gt_cap_factor: float = 0.85,
        detour_cap: float = 12.0,
        prob_eps: float = 1e-4,
    ):
        super().__init__(config)
        self._k_targets = int(k_targets)
        self._src_onroad_p = float(src_onroad_p)
        self._tgt_onroad_p = float(tgt_onroad_p)
        self._min_euclid_px = int(min_euclid_px)
        self._max_euclid_px = int(max_euclid_px)
        self._teacher_iters = int(teacher_iters) if teacher_iters is not None else None
        self._teacher_alpha = float(teacher_alpha)
        self._teacher_gamma = float(teacher_gamma)
        self._teacher_mask = str(teacher_mask)
        self._gt_cap_factor = float(gt_cap_factor)
        self._detour_cap = float(detour_cap)
        self._prob_eps = float(prob_eps)

    @staticmethod
    def _maxpool_bhw(prob_bhw: torch.Tensor, ds: int) -> torch.Tensor:
        ds = max(1, int(ds))
        if ds == 1:
            return prob_bhw
        B, H, W = prob_bhw.shape
        pad_h = (ds - (H % ds)) % ds
        pad_w = (ds - (W % ds)) % ds
        if pad_h or pad_w:
            prob_bhw = F.pad(prob_bhw, (0, pad_w, 0, pad_h), value=0.0)
        return F.max_pool2d(prob_bhw.unsqueeze(1), kernel_size=ds, stride=ds).squeeze(1)

    def _sample_src_tgts(self, road_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (src_yx, tgt_yx_k) on the *full-res* patch grid."""
        device = road_mask.device
        B, H, W = road_mask.shape
        k = self._k_targets
        w_b = (road_mask > 0.5).to(torch.float32)

        src_yx = torch.empty((B, 2), dtype=torch.long, device=device)
        tgt_yx = torch.full((B, k, 2), -1, dtype=torch.long, device=device)

        oversample = max(8, int(2 * k))

        for b in range(B):
            w = w_b[b].reshape(-1)
            has_road = bool((w.sum() > 0).item())

            # src
            if has_road and (torch.rand((), device=device) < self._src_onroad_p):
                idx = torch.multinomial(w, num_samples=1, replacement=True)[0]
                sy = (idx // W).long(); sx = (idx % W).long()
            else:
                sy = torch.randint(0, H, (1,), device=device, dtype=torch.long)[0]
                sx = torch.randint(0, W, (1,), device=device, dtype=torch.long)[0]
            src = torch.stack([sy, sx], dim=0)
            src_yx[b] = src

            # targets: biased mixture (on-road + uniform), then filter by min euclid
            cand_n = k * oversample
            n_road = int(cand_n * self._tgt_onroad_p)
            n_rand = cand_n - n_road
            cand = []
            if has_road and n_road > 0:
                idx_r = torch.multinomial(w, num_samples=n_road, replacement=True)
                y_r = (idx_r // W).long(); x_r = (idx_r % W).long()
                cand.append(torch.stack([y_r, x_r], dim=1))
            if n_rand > 0:
                y_u = torch.randint(0, H, (n_rand,), device=device, dtype=torch.long)
                x_u = torch.randint(0, W, (n_rand,), device=device, dtype=torch.long)
                cand.append(torch.stack([y_u, x_u], dim=1))
            cand_yx = torch.cat(cand, dim=0) if len(cand) > 1 else cand[0]

            d = torch.norm(cand_yx.to(torch.float32) - src.to(torch.float32), dim=1)
            mask_d = d >= float(self._min_euclid_px)
            if self._max_euclid_px > 0:
                mask_d = mask_d & (d <= float(self._max_euclid_px))
            good = cand_yx[mask_d]
            if good.numel() == 0:
                good = cand_yx  # fallback: allow close pairs
            take = min(k, good.shape[0])
            tgt_yx[b, :take] = good[:take]
            if take < k:
                # fill rest uniformly
                y_f = torch.randint(0, H, (k - take,), device=device, dtype=torch.long)
                x_f = torch.randint(0, W, (k - take,), device=device, dtype=torch.long)
                tgt_yx[b, take:k] = torch.stack([y_f, x_f], dim=1)

        return src_yx, tgt_yx

    def _teacher_gt_dist(self, road_mask: torch.Tensor, src_yx: torch.Tensor, tgt_yx: torch.Tensor) -> torch.Tensor:
        """Hard-eval Eikonal on GT roadmap; returns gt_dist[B,K] (pixels)."""
        device = road_mask.device
        B, H, W = road_mask.shape
        K = tgt_yx.shape[1]

        ds = max(1, int(getattr(self, 'route_eik_downsample', 4)))
        eps = float(self._prob_eps)

        prob = road_mask.to(torch.float32).clamp(0.0, 1.0).clamp(eps, 1.0 - eps)
        prob_ds = self._maxpool_bhw(prob, ds)
        Hf, Wf = prob_ds.shape[-2], prob_ds.shape[-1]

        # NOTE: teacher cost is FIXED (alpha/gamma from args) to avoid moving targets.
        cost = _prob_to_cost_additive(
            prob_ds,
            alpha=float(self._teacher_alpha),
            gamma=float(self._teacher_gamma),
            block_th=float(self.route_block_th),
            block_alpha=float(self.route_add_block_alpha),
            block_smooth=float(self.route_add_block_smooth),
            eps=float(self.route_eps),
        ).to(dtype=torch.float32)

        src_c = (src_yx // ds).clamp(min=0)
        src_c[:, 0].clamp_(0, Hf - 1)
        src_c[:, 1].clamp_(0, Wf - 1)
        src_mask = make_source_mask(Hf, Wf, src_c.view(B, 1, 2))

        n_iters = int(self._teacher_iters) if self._teacher_iters is not None else int(self.route_eik_cfg.n_iters)
        cfg_t = dc_replace(self.route_eik_cfg, h=float(ds), n_iters=n_iters, mode="hard_eval", monotone=True)

        with torch.no_grad():
            T = eikonal_soft_sweeping(cost, src_mask, cfg_t)  # [B,Hf,Wf]

        # gather target distances
        tgt_c = (tgt_yx // ds)
        tgt_c_y = tgt_c[..., 0].clamp(0, Hf - 1)
        tgt_c_x = tgt_c[..., 1].clamp(0, Wf - 1)
        lin = (tgt_c_y * Wf + tgt_c_x).view(B, K)
        Tflat = T.view(B, -1)
        gt = torch.gather(Tflat, 1, lin)
        return gt

    def _teacher_gt_field_on_roi(
        self,
        road_mask: torch.Tensor,       # [B, H, W]  full-res GT road mask
        field_meta: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard-eval Eikonal on GT mask cropped to the SAME ROI as student.

        Replicates the exact ROI crop, padding, and pooling that the student
        solver used, ensuring domain-identical boundary conditions.

        Returns:
            T_gt:         [B, tube_h, tube_w]  teacher distance field
            road_mask_ds: [B, tube_h, tube_w]  bool, downsampled GT road mask
        """
        device = road_mask.device
        B, H, W = road_mask.shape
        geom       = field_meta["geom"]
        off_f      = field_meta["off_f"]
        ds_common  = int(field_meta["ds_common"])
        P_c        = int(field_meta["P_c"])
        tube_h     = int(field_meta["tube_h"])
        tube_w     = int(field_meta["tube_w"])
        src_f_yx   = field_meta["src_f_yx"]
        eps        = float(self._prob_eps)

        n_iters = int(self._teacher_iters) if self._teacher_iters is not None else int(self.route_eik_cfg.n_iters)

        T_gt_list = []
        road_ds_list = []

        for b in range(B):
            y0, x0, P, sr_y, sr_x, _ds_b, _valid_mask = geom[b]
            y1, x1 = y0 + P, x0 + P

            mask_b = road_mask[b].to(torch.float32)
            yy0 = max(y0, 0); xx0 = max(x0, 0)
            yy1 = min(y1, H); xx1 = min(x1, W)

            patch = F.pad(
                mask_b[yy0:yy1, xx0:xx1],
                (xx0 - x0, x1 - xx1, yy0 - y0, y1 - yy1),
                value=0.0,
            )

            P_pad_f = int(np.ceil(P / ds_common) * ds_common)
            if P_pad_f > P:
                patch = F.pad(patch, (0, P_pad_f - P, 0, P_pad_f - P), value=0.0)

            patch_ds = F.max_pool2d(
                patch.unsqueeze(0).unsqueeze(0), kernel_size=ds_common, stride=ds_common
            ).squeeze(0).squeeze(0)

            ph, pw = patch_ds.shape
            if ph < P_c or pw < P_c:
                patch_ds = F.pad(patch_ds, (0, P_c - pw, 0, P_c - ph), value=0.0)

            road_ds_full = (patch_ds > 0.5)

            yoff, xoff = off_f[b]
            road_roi = road_ds_full[yoff:yoff + tube_h, xoff:xoff + tube_w]
            dh = tube_h - road_roi.shape[0]
            dw = tube_w - road_roi.shape[1]
            if dh > 0 or dw > 0:
                road_roi = F.pad(road_roi.float(), (0, dw, 0, dh), value=0.0).bool()

            prob_roi = road_roi.float().clamp(eps, 1.0 - eps)
            cost_roi = _prob_to_cost_additive(
                prob_roi.unsqueeze(0),
                alpha=float(self._teacher_alpha),
                gamma=float(self._teacher_gamma),
                block_th=float(self.route_block_th),
                block_alpha=float(self.route_add_block_alpha),
                block_smooth=float(self.route_add_block_smooth),
                eps=float(self.route_eps),
            ).to(dtype=torch.float32).squeeze(0)

            sf_y, sf_x = src_f_yx[b]
            sf_y_roi = sf_y - yoff
            sf_x_roi = sf_x - xoff
            sf_y_roi = max(0, min(sf_y_roi, tube_h - 1))
            sf_x_roi = max(0, min(sf_x_roi, tube_w - 1))

            src_mask_b = torch.zeros(1, tube_h, tube_w, dtype=torch.bool, device=device)
            src_mask_b[0, sf_y_roi, sf_x_roi] = True

            actual_iters = max(n_iters, int(max(tube_h, tube_w) * 1.5))
            cfg_t = dc_replace(
                self.route_eik_cfg,
                h=float(ds_common),
                n_iters=actual_iters,
                mode="hard_eval",
                monotone=True,
            )

            with torch.no_grad():
                T_b = eikonal_soft_sweeping(cost_roi.unsqueeze(0), src_mask_b, cfg_t).squeeze(0)

            T_gt_list.append(T_b)
            road_ds_list.append(road_roi)

        T_gt = torch.stack(T_gt_list)
        road_mask_ds = torch.stack(road_ds_list)
        return T_gt, road_mask_ds

    def _inject_dist(self, batch: dict) -> dict:
        road_mask = batch["road_mask"]
        if self._teacher_mask == "thin" and ("road_mask_thin" in batch):
            road_mask = batch["road_mask_thin"]

        src_yx, tgt_yx = self._sample_src_tgts(road_mask)
        gt_dist = self._teacher_gt_dist(road_mask, src_yx, tgt_yx)

        # filter unreachable/extreme-detour pairs; mark as padding (-1)
        euclid = torch.norm(
            tgt_yx.to(torch.float32) - src_yx[:, None, :].to(torch.float32),
            dim=-1,
        ).clamp_min(1.0)
        cap_px = float(road_mask.shape[-1]) * float(self._gt_cap_factor)
        large_val = float(getattr(self.route_eik_cfg, "large_val", 1e6))
        valid = (
            (gt_dist > 0)
            & (gt_dist < cap_px)
            & (gt_dist < large_val * 0.9)
            & ((gt_dist / euclid) < float(self._detour_cap))
        )
        batch["src_yx"] = src_yx
        batch["tgt_yx"] = torch.where(valid[..., None], tgt_yx, torch.full_like(tgt_yx, -1))
        batch["gt_dist"] = torch.where(valid, gt_dist, torch.full_like(gt_dist, -1.0))
        return batch

    def _compute_dense_field_loss(self, batch: dict) -> torch.Tensor:
        """Compute dense distance field loss using stored field_meta from solver."""
        field_meta = self._last_field_meta
        if field_meta is None:
            return torch.tensor(0.0, device=batch["road_mask"].device)

        road_mask = batch["road_mask"]
        if self._teacher_mask == "thin" and ("road_mask_thin" in batch):
            road_mask = batch["road_mask_thin"]

        T_pred = field_meta["T_pred"]
        T_gt, road_ds = self._teacher_gt_field_on_roi(road_mask, field_meta)

        B = T_pred.shape[0]
        large_val = float(getattr(self.route_eik_cfg, "large_val", 1e6))
        ds_common = int(field_meta["ds_common"])
        cap_px = float(max(field_meta["tube_h"], field_meta["tube_w"]) * ds_common) * self._dense_field_cap_factor
        min_pix = self._dense_field_min_pixels

        all_pred, all_gt = [], []
        n_valid_samples = 0
        for b in range(B):
            M = road_ds[b] & (T_gt[b] < large_val * 0.9) & (T_gt[b] < cap_px)
            n_valid = int(M.sum().item())
            if n_valid < min_pix:
                continue
            tp = T_pred[b][M]
            tg = T_gt[b][M]
            tp_max = tp.max().detach() + 1e-6
            tg_max = tg.max() + 1e-6
            all_pred.append(tp / tp_max)
            all_gt.append(tg / tg_max)
            n_valid_samples += 1

        if not all_pred:
            return torch.tensor(0.0, device=road_mask.device)

        loss_field = F.huber_loss(
            torch.cat(all_pred), torch.cat(all_gt).detach(), delta=2.0
        )
        return loss_field

    def training_step(self, batch, batch_idx):
        if self.route_lambda_dist > 0 and ("src_yx" not in batch):
            batch = self._inject_dist(batch)
        self._last_field_meta = None
        loss = super().training_step(batch, batch_idx)

        if self._dense_field_loss and self._lambda_dist_field > 0:
            loss_field = self._compute_dense_field_loss(batch)
            lam_f = self._effective_lambda_dist() / max(self.route_lambda_dist, 1e-9) * self._lambda_dist_field if self.route_lambda_dist > 0 else self._lambda_dist_field
            field_contrib = lam_f * loss_field
            loss = loss + field_contrib
            self.log("train_field_loss", loss_field, on_step=True, on_epoch=False, prog_bar=True)
            self.log("field_loss_weighted", field_contrib, on_step=True, on_epoch=False)

        self._last_field_meta = None
        return loss

    def validation_step(self, batch, batch_idx):
        if self.route_lambda_dist > 0 and ("src_yx" not in batch):
            batch = self._inject_dist(batch)
        return super().validation_step(batch, batch_idx)

# ==========================================
# 3. 训练流程 (使用 Lightning Trainer)
# ==========================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"准备训练... 检测到设备: {device}")

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(_PROJECT_ROOT, args.data_root)
    PRETRAINED_CKPT = args.pretrained_ckpt if os.path.isabs(args.pretrained_ckpt) else os.path.join(_PROJECT_ROOT, args.pretrained_ckpt)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_PROJECT_ROOT, args.output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")

    os.makedirs(ckpt_dir, exist_ok=True)

    config = TrainConfig()
    if args.lr is not None:
        config.BASE_LR = args.lr
    if args.encoder_lora:
        config.ENCODER_LORA = True
        config.LORA_RANK = args.lora_rank
        config.FREEZE_ENCODER = False  # 避免外部冻结误伤 LoRA 参数
        print(f"LoRA 模式: rank={config.LORA_RANK}")
    if args.road_pos_weight is not None:
        config.ROAD_POS_WEIGHT = args.road_pos_weight
    if args.road_dice_weight is not None:
        config.ROAD_DICE_WEIGHT = args.road_dice_weight
    if args.road_thin_boost is not None:
        config.ROAD_THIN_BOOST = args.road_thin_boost
    if args.smooth_decoder:
        config.USE_SMOOTH_DECODER = True
        print("平滑 Decoder 已启用（末端 3×3 Conv 抗马赛克）")
    if args.lambda_dist is not None:
        config.ROUTE_LAMBDA_DIST = args.lambda_dist
    if args.eik_iters is not None:
        config.ROUTE_EIK_ITERS = args.eik_iters
    if args.eik_downsample is not None:
        config.ROUTE_EIK_DOWNSAMPLE = args.eik_downsample
    if args.eik_mode is not None:
        config.ROUTE_EIK_MODE = args.eik_mode
    if args.cap_mode is not None:
        config.ROUTE_CAP_MODE = args.cap_mode
    if args.no_multigrid:
        config.ROUTE_MULTIGRID = False
    if args.multigrid:
        config.ROUTE_MULTIGRID = True
    if args.mg_factor is not None:
        config.ROUTE_MG_FACTOR = args.mg_factor
    if args.ckpt_chunk is not None:
        config.ROUTE_CKPT_CHUNK = args.ckpt_chunk

    # LR scheduler config
    config.LR_SCHEDULER = args.lr_scheduler
    config.LR_WARMUP_EPOCHS = args.lr_warmup_epochs
    config.LR_MIN = args.lr_min

    # Cost param stability
    config.FREEZE_COST_PARAMS = args.freeze_cost_params
    config.COST_PARAM_LR_SCALE = args.cost_param_lr_scale

    # Learnable block params
    config.LEARNABLE_BLOCK = args.learnable_block
    if args.learnable_block:
        print("Learnable block params enabled: block_th and block_alpha will be trained")

    # Dense field loss config
    config.DENSE_FIELD_LOSS = args.dense_field_loss
    config.LAMBDA_DIST_FIELD = args.lambda_dist_field
    config.DENSE_FIELD_MIN_PIXELS = args.dense_field_min_pixels
    config.DENSE_FIELD_CAP_FACTOR = args.dense_field_cap_factor
    if args.dense_field_loss:
        print(f"Dense field loss enabled: lambda_dist_field={args.lambda_dist_field}, "
              f"min_pixels={args.dense_field_min_pixels}, cap_factor={args.dense_field_cap_factor}")

    if args.dist_supervision == "gtmask_random" and config.ROUTE_LAMBDA_DIST <= 0.0:
        # Avoid silently disabling the feature due to a missing flag.
        config.ROUTE_LAMBDA_DIST = 0.2
        print("[WARN] dist_supervision=gtmask_random but lambda_dist<=0; auto-set ROUTE_LAMBDA_DIST=0.2")

    if config.ROUTE_LAMBDA_DIST > 0:
        print(f"Distance loss enabled: lambda_dist={config.ROUTE_LAMBDA_DIST}, "
              f"eik_iters={config.ROUTE_EIK_ITERS}, ds={config.ROUTE_EIK_DOWNSAMPLE}, "
              f"mode={config.ROUTE_EIK_MODE}, cap={config.ROUTE_CAP_MODE}, "
              f"multigrid={config.ROUTE_MULTIGRID} (mg_factor={config.ROUTE_MG_FACTOR}), "
              f"dist_supervision={args.dist_supervision}")

    if args.dist_supervision == "gtmask_random":
        model = SAMRouteGTMaskTeacher(
            config,
            k_targets=args.dist_k_targets,
            src_onroad_p=args.dist_src_onroad_p,
            tgt_onroad_p=args.dist_tgt_onroad_p,
            min_euclid_px=args.dist_min_euclid_px,
            max_euclid_px=args.dist_max_euclid_px,
            teacher_iters=args.dist_teacher_iters,
            teacher_alpha=args.dist_teacher_alpha,
            teacher_gamma=args.dist_teacher_gamma,
            teacher_mask=args.dist_teacher_mask,
            gt_cap_factor=args.dist_gt_cap_factor,
            detour_cap=args.dist_detour_cap,
            prob_eps=args.dist_prob_eps,
        )
    else:
        model = SAMRoute(config)

    # 手动冻结本阶段不参与 Loss 的参数，满足 DDP 校验，从而使用标准 ddp 而非龟速 find_unused_parameters
    if config.FREEZE_ENCODER:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    if config.ROUTE_LAMBDA_DIST == 0.0:
        model.cost_log_alpha.requires_grad = False
        model.cost_log_gamma.requires_grad = False
        model.eik_gate_logit.requires_grad = False
        if hasattr(model, 'topo_net'):
            for p in model.topo_net.parameters():
                p.requires_grad = False

    # 1. 安全加载预训练权重（若模型结构变化导致无法加载，则从随机初始化继续训练）
    if os.path.isfile(PRETRAINED_CKPT):
        try:
            print(f"加载预训练权重: {PRETRAINED_CKPT}")
            ckpt = torch.load(PRETRAINED_CKPT, map_location='cpu', weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            print("✓ 预训练权重加载成功")
        except Exception as e:
            print(f"⚠️ 预训练权重加载失败（可能与 map_decoder 结构变更不兼容）: {e}")
            print("   将从随机初始化 Decoder 开始训练")
    else:
        print(f"⚠️ 未找到预训练权重 {PRETRAINED_CKPT}，从随机 Decoder 开始训练。")

    # Override cost params to inference-optimal values when frozen
    if config.FREEZE_COST_PARAMS and config.ROUTE_LAMBDA_DIST > 0:
        import math as _m
        with torch.no_grad():
            model.cost_log_alpha.fill_(_m.log(50.0))
            model.cost_log_gamma.fill_(_m.log(2.0))
            model.eik_gate_logit.fill_(10.0)  # sigmoid(10) ≈ 1.0
        print(f"[freeze_cost] Set alpha=50.0, gamma=2.0, gate≈1.0 (frozen)")

    # 2. 构建 DataLoader (自动路由：单图测试 或 全量训练)
    # Build dist_cache_cfg when gtmask_random + dist cache enabled
    dist_cache_cfg = None
    if (args.dist_supervision == "gtmask_random"
            and config.ROUTE_LAMBDA_DIST > 0
            and not args.no_dist_cache):
        dist_cache_cfg = dict(
            k_targets=args.dist_k_targets,
            src_onroad_p=args.dist_src_onroad_p,
            tgt_onroad_p=args.dist_tgt_onroad_p,
            min_euclid_px=args.dist_min_euclid_px,
            max_euclid_px=args.dist_max_euclid_px,
            teacher_alpha=args.dist_teacher_alpha,
            teacher_gamma=args.dist_teacher_gamma,
            eik_iters=args.dist_teacher_iters or config.ROUTE_EIK_ITERS,
            eik_downsample=config.ROUTE_EIK_DOWNSAMPLE,
            gt_cap_factor=args.dist_gt_cap_factor,
            detour_cap=args.dist_detour_cap,
            prob_eps=args.dist_prob_eps,
            use_thin=(args.dist_teacher_mask == "thin"),
        )

    if args.single_region_dir:
        print(f"\n=============================================")
        print(f"Single-region overfitting mode")
        print(f"=============================================\n")
        train_ds = SingleRegionDataset(
            args.single_region_dir, config, samples_per_epoch=200,
            dist_cache_cfg=dist_cache_cfg)
        val_ds = SingleRegionDataset(
            args.single_region_dir, config, samples_per_epoch=20,
            dist_cache_cfg=dist_cache_cfg)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"数据集根目录不存在: {data_root}")
        print("正在构建全量数据集并加载到内存...")
        train_loader, val_loader = build_dataloaders(
            root_dir=data_root,
            patch_size=config.PATCH_SIZE,
            batch_size=args.batch_size,
            num_workers=args.workers,
            include_dist=(config.ROUTE_LAMBDA_DIST > 0 and args.dist_supervision == "npz"),
            val_fraction=args.val_fraction,
            samples_per_region=args.samples_per_region,
            use_cached_features=args.use_cached_features,
            preload_to_ram=args.preload_to_ram,
            road_dilation_radius=args.road_dilation_radius,
        )

    # 3. 配置 Lightning 回调
    ckpt_filename = f'best_{args.run_name}' if args.run_name else 'best'
    ckpt_monitor = 'val_loss' if config.ROUTE_LAMBDA_DIST > 0 else 'val_seg_loss'
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_filename,
        save_top_k=1,
        monitor=ckpt_monitor,
        mode='min',
        save_last=True,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 训练日志（TensorBoard）
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard", version=None)

    # 4. 启动 Trainer
    use_gpu = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if use_gpu else 0
    devices = args.devices if args.devices is not None else (n_gpus or 1)
    use_ddp = use_gpu and devices > 1
    precision = "16-mixed" if use_gpu else "32"
    strategy = "ddp" if use_ddp else "auto"
    
    accum = args.accumulate_grad_batches
    eff_batch = args.batch_size * accum * (devices if isinstance(devices, int) else len(devices))
    if accum > 1:
        print(f"Gradient accumulation: {accum} steps, effective batch size = {eff_batch}")

    trainer_kwargs = dict(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10 if args.single_region_dir else 50,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=False,
        accumulate_grad_batches=accum,
    )
    if args.grad_clip is not None and args.grad_clip > 0:
        trainer_kwargs["gradient_clip_val"] = args.grad_clip
        trainer_kwargs["gradient_clip_algorithm"] = "norm"
        print(f"Gradient clipping enabled: max_norm={args.grad_clip}")
    trainer = pl.Trainer(**trainer_kwargs)
    
    if use_ddp:
        print(f"使用 {devices} 张 GPU 进行 DDP 分布式训练")

    print("\n🚀 开始训练...")
    trainer.fit(model, train_loader, val_loader)
    print(f"\n✅ 训练完成。最佳权重已保存在: {ckpt_dir}")
    print(f"   训练曲线: tensorboard --logdir {os.path.join(output_dir, 'tensorboard')} --port 6006")

# ==========================================
# 4. 命令行参数
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="SAMRoute finetune — 多图批量正式训练 (支持 DDP、TF32、cached features)",
        epilog="单图过拟合测试: python finetune_demo.py --single_region_dir path/to/region --epochs 100"
    )
    p.add_argument("--data_root", default="Gen_dataset_V2/Gen_dataset", help="数据集根目录")
    
    # 🎯 单图测试开关
    p.add_argument("--single_region_dir", type=str, default=None, 
                   help="如果提供单个图像的文件夹路径（包含 .tif 和 .png），则无视 data_root，只训练这一个区域！")
                   
    p.add_argument("--pretrained_ckpt", default="checkpoints/cityscale_vitb_512_e10.ckpt", help="预训练 SAM-Road 权重路径")
    p.add_argument("--output_dir", default="training_outputs/finetune_demo", help="输出目录")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=32, help="batch size，cached 模式显存占用低可开 32~64")
    p.add_argument("--lr", type=float, default=None, help="学习率，默认 5e-4 (配合 batch 16)")
    p.add_argument("--val_fraction", type=float, default=0.1, help="验证集城市占比 (0~1)")
    p.add_argument("--samples_per_region", type=int, default=50, help="每区域每 epoch 采样数")
    p.add_argument("--road_dilation_radius", type=int, default=3, help="归一化 mask 半径")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers，preload 时 4 足够")
    p.add_argument("--devices", type=int, default=1, help="GPU 数量，默认单卡训练")
    p.add_argument("--use_cached_features", action="store_true", help="使用预计算 samroad_feat_full_*.npy 跳过 Encoder")
    p.add_argument("--no_cached_features", action="store_true", help="关闭 cached features，始终跑 Encoder")
    p.add_argument("--no_preload", action="store_true", help="关闭 preload_to_ram")
    p.add_argument("--fast", action="store_true", help="快速模式：单卡 batch32 workers4，便于调试")
    # 参数扫描：ROAD_POS_WEIGHT / ROAD_DICE_WEIGHT / ROAD_THIN_BOOST
    p.add_argument("--road_pos_weight", type=float, default=None, help="BCE 正样本权重")
    p.add_argument("--road_dice_weight", type=float, default=None, help="Dice 损失权重，越大越强调细路")
    p.add_argument("--road_thin_boost", type=float, default=None, help="BCE 细路像素额外权重倍数 (默认 6.0)")
    p.add_argument("--run_name", type=str, default=None, help="参数扫描时指定，checkpoint 保存为 best_{run_name}.ckpt")
    # LoRA encoder 微调
    p.add_argument("--encoder_lora", action="store_true", help="开启 LoRA 微调 Encoder")
    p.add_argument("--lora_rank", type=int, default=4, help="LoRA rank，仅 --encoder_lora 时生效")
    # 平滑 Decoder（抗马赛克）
    p.add_argument("--smooth_decoder", action="store_true",
                   help="使用平滑 Decoder（末端加 3×3 Conv，消除 16px ViT token 格栅伪影）")
    # Eikonal distance loss
    p.add_argument("--lambda_dist", type=float, default=None,
                   help="Distance loss weight (default 0.0 = off). Set >0 to enable.")
    p.add_argument("--dist_supervision", type=str, default="npz",
                   choices=["npz", "gtmask_random"],
                   help="Distance supervision source: 'npz' uses dataset NPZ GT distances; "
                        "'gtmask_random' samples random pairs per patch and uses GT roadmap mask + hard-eval Eikonal.")
    p.add_argument("--dist_k_targets", type=int, default=8,
                   help="For gtmask_random: number of targets per patch (1 src + K targets => 1 Eikonal solve).")
    p.add_argument("--dist_src_onroad_p", type=float, default=0.9,
                   help="For gtmask_random: probability to sample src on GT road pixels.")
    p.add_argument("--dist_tgt_onroad_p", type=float, default=0.9,
                   help="For gtmask_random: probability to sample targets on GT road pixels.")
    p.add_argument("--dist_min_euclid_px", type=int, default=64,
                   help="For gtmask_random: enforce target Euclid distance >= this (fallback if impossible).")
    p.add_argument("--dist_max_euclid_px", type=int, default=0,
                   help="For gtmask_random: enforce target Euclid distance <= this. 0 = no limit.")
    p.add_argument("--dist_teacher_iters", type=int, default=None,
                   help="For gtmask_random: teacher hard-eval Eikonal iters (default = ROUTE_EIK_ITERS).")
    p.add_argument("--dist_teacher_alpha", type=float, default=20.0,
                   help="For gtmask_random: fixed teacher cost alpha (additive mode).")
    p.add_argument("--dist_teacher_gamma", type=float, default=2.0,
                   help="For gtmask_random: fixed teacher cost gamma (additive mode).")
    p.add_argument("--dist_teacher_mask", type=str, default="thick",
                   choices=["thick", "thin"],
                   help="For gtmask_random: which GT roadmap to use for teacher distances.")
    p.add_argument("--dist_gt_cap_factor", type=float, default=0.85,
                   help="For gtmask_random: drop teacher pairs with gt_dist > patch_size * factor.")
    p.add_argument("--dist_detour_cap", type=float, default=12.0,
                   help="For gtmask_random: drop teacher pairs with (gt_dist / euclid) > this.")
    p.add_argument("--dist_prob_eps", type=float, default=1e-4,
                   help="For gtmask_random: clamp GT prob to [eps, 1-eps] before cost conversion.")
    p.add_argument("--no_dist_cache", action="store_true",
                   help="Disable GT distance pre-computation cache; revert to online teacher Eikonal every step.")
    p.add_argument("--eik_iters", type=int, default=None, help="Eikonal iterations (default 40)")
    p.add_argument("--eik_downsample", type=int, default=None, help="Eikonal downsample factor (default 16)")
    p.add_argument("--eik_mode", type=str, default=None,
                   choices=["soft_train", "ste_train", "hard_eval"],
                   help="Eikonal solver mode (default soft_train)")
    p.add_argument("--cap_mode", type=str, default=None,
                   choices=["tanh", "clamp", "none"],
                   help="Distance cap mode (default tanh)")
    # Multigrid Eikonal (coarse→fine warm-start, aligned with eval/test)
    p.add_argument("--multigrid", action="store_true", default=None,
                   help="Enable multigrid Eikonal (default ON)")
    p.add_argument("--no_multigrid", action="store_true",
                   help="Disable multigrid Eikonal")
    p.add_argument("--mg_factor", type=int, default=None,
                   help="Multigrid coarse-to-fine factor (default 4)")
    p.add_argument("--ckpt_chunk", type=int, default=None,
                   help="Gradient checkpoint chunk size for Eikonal solver (default 10)")
    # Dense distance field loss
    p.add_argument("--dense_field_loss", action="store_true",
                   help="Enable dense distance field loss (compare full T_pred vs T_gt on road pixels).")
    p.add_argument("--lambda_dist_field", type=float, default=0.2,
                   help="Weight for dense field loss (default 0.2).")
    p.add_argument("--dense_field_min_pixels", type=int, default=10,
                   help="Skip dense field loss if fewer than this many valid pixels in ROI.")
    p.add_argument("--dense_field_cap_factor", type=float, default=0.85,
                   help="Cap for T_gt in dense field loss: ROI_size * factor.")
    # LR scheduler
    p.add_argument("--lr_scheduler", type=str, default="multistep",
                   choices=["multistep", "cosine"],
                   help="LR scheduler type: 'multistep' or 'cosine' (cosine annealing with optional warmup).")
    p.add_argument("--lr_warmup_epochs", type=int, default=0,
                   help="Linear warmup epochs before cosine decay (only for --lr_scheduler cosine).")
    p.add_argument("--lr_min", type=float, default=1e-6,
                   help="Minimum learning rate for cosine annealing.")
    # Gradient clipping & accumulation
    p.add_argument("--grad_clip", type=float, default=None,
                   help="Global gradient norm clipping value (e.g. 1.0). None = disabled.")
    p.add_argument("--accumulate_grad_batches", type=int, default=1,
                   help="Accumulate gradients over N batches to simulate larger effective batch size.")
    # Cost param stability
    p.add_argument("--freeze_cost_params", action="store_true",
                   help="Freeze Eikonal cost params (alpha/gamma/gate). Uses init values, no gradient.")
    p.add_argument("--cost_param_lr_scale", type=float, default=1.0,
                   help="LR multiplier for cost params (alpha/gamma/gate). E.g. 0.01 = 100x slower. Ignored if --freeze_cost_params.")
    # Learnable block params
    p.add_argument("--learnable_block", action="store_true",
                   help="Make block_th and block_alpha learnable parameters in the cost function.")
    args = p.parse_args()
    args.preload_to_ram = not args.no_preload
    args.use_cached_features = args.use_cached_features and not args.no_cached_features
    if args.encoder_lora:
        args.use_cached_features = False  # LoRA 必须在线计算，不可用缓存
    if args.fast:
        args.devices = args.devices or 1
        args.batch_size = 32
        args.workers = 4
        args.use_cached_features = True
        print("⚠️ --fast: 单卡 batch=32 workers=4 use_cached_features=True")
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)