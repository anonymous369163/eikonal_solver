"""
Satellite Encoder for SAM-Road (Final Optimized).
Changes:
1. Full GPU pipeline (no CPU transfers).
2. Robust normalization (mimics original _to_uint8 logic).
3. Memory buffer reuse with STRICT zeroing (Fixes padding pollution).
4. Localized Sigmoid check (Fixes cross-image state pollution).
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch 
import torch.nn.functional as F


# ============================================================
# Helper utils (Updated for GPU Robustness)
# ============================================================


def load_tif_image(image_path: str) -> torch.Tensor:
    """
    Load TIF/TIFF image, return [1, 3, H, W] float tensor in [0, 1].
    Strictly checks for .tif/.tiff extension.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.tif', '.tiff']:
        raise ValueError(f"Unsupported file format: {ext}. Only .tif or .tiff are allowed.")

    try:
        import rasterio
        with rasterio.open(image_path) as src:
            img = src.read()  # [C, H, W]
    except ImportError:
        import tifffile
        img = tifffile.imread(image_path)
        # Handle dimensions
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)
        elif img.ndim == 3:
            # Assume (H, W, C) if C is last and small
            if img.shape[2] <= 4:
                img = img.transpose(2, 0, 1)
    
    # Ensure 3 channels
    if img.shape[0] > 3:
        img = img[:3]
    
    img = img.astype(np.float32)
    # Normalize to [0, 1]
    if img.max() > 1.0:
        if img.max() > 255.0:
            img /= 65535.0
        else:
            img /= 255.0
            
    return torch.from_numpy(img).unsqueeze(0)


def _as_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)

def _hann2d(size: int, device: torch.device, eps: float = 1e-6) -> torch.Tensor:
    w = torch.hann_window(size, periodic=False, device=device, dtype=torch.float32)
    return (w[:, None] * w[None, :]).clamp_min_(eps)

def _full_coverage_coords(H: int, W: int, patch: int, stride: int) -> Tuple[list[int], list[int]]:
    y_max = max(H - patch, 0)
    x_max = max(W - patch, 0)
    ys = list(range(0, y_max + 1, stride)) or [0]
    xs = list(range(0, x_max + 1, stride)) or [0]
    if ys[-1] != y_max: ys.append(y_max)
    if xs[-1] != x_max: xs.append(x_max)
    return ys, xs

def _to_uint8_gpu(img: torch.Tensor) -> torch.Tensor:
    """
    [Optimization] GPU-based robust normalization.
    Mimics the logic of _to_uint8_cpu strictly to ensure input consistency.
    """
    if img.dtype == torch.uint8:
        return img
    
    # Handle uint16
    if img.dtype == torch.uint16:
        return (img.float() / 65535.0 * 255.0).round().clamp(0, 255).to(torch.uint8)

    # Handle float
    img_f = img.float()
    
    # Check max value. Note: .max() causes a sync, but we do it ONCE per full image.
    if img_f.numel() > 0:
        mx = img_f.max().item()
    else:
        mx = 0.0

    if mx <= 1.5:
        # 0-1 float -> 0-255
        img_f = img_f.clamp(0.0, 1.0) * 255.0
    elif mx <= 300.0:
        # 0-255 float -> clamp
        img_f = img_f.clamp(0.0, 255.0)
    else:
        # 0-65535 float -> 0-255
        img_f = (img_f.clamp(0.0, 65535.0) / 65535.0) * 255.0
        
    return img_f.round().to(torch.uint8)


# ============================================================
# SamRoadTiledInferencer (Highly Optimized & Fixed)
# ============================================================

class SamRoadTiledInferencer:
    def __init__(self, **cfg: Any):
        self.sat_grid: int = int(cfg.get("sat_grid", 14))
        self.overlap: int = int(cfg.get("sam_road_overlap", 64))
        self.patch_batch: int = int(cfg.get("sam_road_patch_batch", 4))
        self.ds: int = int(cfg.get("sam_road_downsample", 16))

        self.sam_road_repo_dir = cfg.get("sam_road_repo_dir", None)
        self.sam_road_ckpt_dir = cfg.get("sam_road_ckpt_dir", None)
        self.sam_road_ckpt_name = cfg.get("sam_road_ckpt_name", "cityscale_vitb_512_e10.ckpt")

        self._samroad = None
        self._samroad_cfg = None
        self._loaded_device: Optional[torch.device] = None

        self.enable_cache: bool = bool(cfg.get("enable_cache", False))
        self._cache_last_obj = None
        self._cache_out = None

    # ... (Loading Logic stays the same) ...
    @staticmethod
    def _find_upwards(start_dir: Path, marker: str, max_depth: int = 10) -> Optional[Path]:
        cur = start_dir.resolve()
        for _ in range(max_depth):
            if (cur / marker).exists(): return cur
            if cur.parent == cur: break
            cur = cur.parent
        return None

    def ensure_loaded(self, device: torch.device | str) -> None:
        device = _as_device(device)
        if self._samroad is not None and self._loaded_device == device:
            return

        here = Path(__file__).resolve()
        root = self._find_upwards(here.parent, "sam_road_repo")
        if root is None:
            root = here.parent if (here.parent / "sam_road_repo").exists() else here.parent.parent

        repo_dir = Path(self.sam_road_repo_dir or (root / "sam_road_repo")).resolve()
        ckpt_dir = Path(self.sam_road_ckpt_dir or (root / "checkpoints")).resolve()
        
        ckpt_path = ckpt_dir / self.sam_road_ckpt_name
        
        if not repo_dir.exists():
            raise RuntimeError(f"SAM-Road repo not found at {repo_dir}")
            
        cfg_name = "toponet_vitb_512_cityscale.yaml" if "cityscale" in self.sam_road_ckpt_name.lower() else "toponet_vitb_256_spacenet.yaml"
        cfg_path = repo_dir / "config" / cfg_name
        utils_path = repo_dir / "utils.py"
        sam_ckpt_path = repo_dir / "sam_ckpts" / "sam_vit_b_01ec64.pth"

        import importlib.util
        spec_utils = importlib.util.spec_from_file_location("sam_road_utils", str(utils_path))
        sam_road_utils = importlib.util.module_from_spec(spec_utils)
        spec_utils.loader.exec_module(sam_road_utils)

        for p in [repo_dir, repo_dir / "sam", repo_dir / "segment-anything-road"]:
            s_p = str(p.resolve())
            if s_p not in sys.path:
                sys.path.insert(0, s_p)

        from model import SAMRoad 

        cfg = sam_road_utils.load_config(str(cfg_path))
        if sam_ckpt_path.exists():
            cfg.SAM_CKPT_PATH = str(sam_ckpt_path)

        net = SAMRoad(cfg)
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt_data["state_dict"] if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data else ckpt_data
        net.load_state_dict(state_dict, strict=True)
        net.eval().to(device)

        self._samroad = net
        self._samroad_cfg = cfg
        self._loaded_device = device

    @torch.no_grad()
    def infer(self, rgb: torch.Tensor, device: torch.device | str, sat_grid: Optional[int] = None, return_mask: bool = True) -> Dict[str, Any]:
        """
        Optimized inference:
        1. Keeps data on GPU.
        2. Robust GPU-based normalization.
        3. AMP (Automatic Mixed Precision).
        4. Memory buffer reuse with STRICT cleaning.
        """
        device = _as_device(device)
        self.ensure_loaded(device)

        if rgb.dim() != 4 or rgb.size(1) != 3:
            raise ValueError(f"rgb must be [1,3,H,W], got {tuple(rgb.shape)}")

        if self.enable_cache and (rgb is self._cache_last_obj) and (self._cache_out is not None):
            return self._cache_out

        # [Optim] 1. Move to GPU once.
        x_gpu = rgb.to(device)
        
        # [Optim] 2. Robust Normalization on GPU
        x_u8 = _to_uint8_gpu(x_gpu)
        
        _, _, H, W = x_u8.shape
        G = int(sat_grid or self.sat_grid)
        patch = int(getattr(self._samroad_cfg, "PATCH_SIZE", 512))
        stride = max(patch - self.overlap, 1)
        ds = int(self.ds)

        Hf, Wf = int(math.ceil(H / ds)), int(math.ceil(W / ds))
        pf = patch // ds

        # [Optim] 3. Pre-allocate accumulators (FP32)
        feat_sum = torch.zeros((1, 256, Hf, Wf), device=device, dtype=torch.float32)
        w_sum = torch.zeros((1, 1, Hf, Wf), device=device, dtype=torch.float32)
        win_feat = _hann2d(pf, device=device)

        if return_mask:
            road_sum = torch.zeros((H, W), device=device, dtype=torch.float32)
            keypoint_sum = torch.zeros((H, W), device=device, dtype=torch.float32)
            mask_w_sum = torch.zeros((H, W), device=device, dtype=torch.float32)
            win_mask = _hann2d(patch, device=device)

        ys, xs = _full_coverage_coords(H, W, patch, stride)
        coords = [(x0, y0) for y0 in ys for x0 in xs]
        bs = max(int(self.patch_batch), 1)

        # [Optim] 4. Pre-allocate Batch Buffer
        patches_buffer = torch.zeros((bs, 3, patch, patch), device=device, dtype=torch.uint8)

        # [Optim] 5. Sigmoid State (Local per image)
        # Default to True to be safe. We check the first batch to see if it's actually Logits.
        _need_sigmoid = True
        _sigmoid_checked = False
        
        amp_enabled = (device.type == 'cuda')
        
        for i in range(0, len(coords), bs):
            batch_coords = coords[i:i + bs]
            current_bs = len(batch_coords)
            
            # [CRITICAL FIX] Clean buffer to prevent pollution from previous batch's residual data.
            # Especially important when edge patches are smaller than patch size.
            patches_buffer[:current_bs].zero_()
            
            valid_hw = []

            for b_idx, (x0, y0) in enumerate(batch_coords):
                y1, x1 = min(y0 + patch, H), min(x0 + patch, W)
                h, w = y1 - y0, x1 - x0
                patches_buffer[b_idx, :, :h, :w] = x_u8[0, :, y0:y1, x0:x1]
                valid_hw.append((h, w))

            # Uint8 -> Float -> Permute
            batch_input = patches_buffer[:current_bs].permute(0, 2, 3, 1).float()

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                mask_scores, img_emb = self._samroad.infer_masks_and_img_features(batch_input)

            # [Optim] Check Logits/Prob only once per image
            if return_mask and (mask_scores is not None) and not _sigmoid_checked: 
                ms_min = mask_scores.amin().item()
                ms_max = mask_scores.amax().item()
                # If values are outside [0, 1], it's definitely logits.
                if ms_min < -0.1 or ms_max > 1.1: 
                    _need_sigmoid = True
                else: 
                    _need_sigmoid = (ms_min < 0.0) or (ms_max > 1.0)
                _sigmoid_checked = True

            # Accumulate (FP32)
            for b, ((x0, y0), (h, w)) in enumerate(zip(batch_coords, valid_hw)):
                yf, xf = y0 // ds, x0 // ds
                phf, pwf = int(math.ceil(h / ds)), int(math.ceil(w / ds))

                feat_sum[0, :, yf:yf+phf, xf:xf+pwf] += img_emb[b, :, :phf, :pwf].float() * win_feat[:phf, :pwf]
                w_sum[0, 0, yf:yf+phf, xf:xf+pwf] += win_feat[:phf, :pwf]

                if return_mask and (mask_scores is not None):
                    ms = mask_scores[b, :h, :w, :]
                    if _need_sigmoid:
                        ms = torch.sigmoid(ms)
                    
                    wm = win_mask[:h, :w]
                    road_sum[y0:y0+h, x0:x0+w] += ms[..., 1].float() * wm
                    keypoint_sum[y0:y0+h, x0:x0+w] += ms[..., 0].float() * wm
                    mask_w_sum[y0:y0+h, x0:x0+w] += wm

        # Normalize & Output
        feat = feat_sum / (w_sum + 1e-6)
        pooled = F.adaptive_avg_pool2d(feat, (G, G))

        out = {
            "pooled_256": pooled.squeeze(0).cpu().numpy(),
            "patch_tokens_256": pooled.squeeze(0).permute(1, 2, 0).reshape(-1, 256).cpu().numpy(),
            "feat_full_256": feat.squeeze(0).cpu().numpy(),
        }

        if return_mask:
            out["road_mask"] = (road_sum / (mask_w_sum + 1e-6)).cpu().numpy()
            out["keypoint_mask"] = (keypoint_sum / (mask_w_sum + 1e-6)).cpu().numpy()

        if self.enable_cache:
            self._cache_last_obj = rgb
            self._cache_out = out

        return out