# load_nodes_from_npz_final.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch

def _normalize_prob_map(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    mx = float(x.max().item()) if x.numel() else 0.0
    if mx > 1.5:
        x = x / 255.0
    return x.clamp(0.0, 1.0)

def snap_nodes_vectorized(
    nodes_yx: torch.Tensor,
    road_map: torch.Tensor,
    threshold: float = 0.3,
    win: int = 30
) -> torch.Tensor:
    """向量化吸附逻辑 (保持不变)"""
    N = nodes_yx.shape[0]
    H, W = road_map.shape
    device = nodes_yx.device
    if N == 0: return nodes_yx

    current_probs = road_map[nodes_yx[:, 0], nodes_yx[:, 1]]
    mask_good = current_probs > threshold
    if mask_good.all(): return nodes_yx

    range_t = torch.arange(-win, win + 1, device=device)
    dy, dx = torch.meshgrid(range_t, range_t, indexing='ij')
    offsets = torch.stack([dy.flatten(), dx.flatten()], dim=1)  
    
    sample_coords = nodes_yx.unsqueeze(1) + offsets.unsqueeze(0)
    sample_coords[..., 0].clamp_(0, H - 1)
    sample_coords[..., 1].clamp_(0, W - 1)

    candidate_probs = road_map[sample_coords[..., 0], sample_coords[..., 1]]
    max_vals, max_indices = candidate_probs.max(dim=1)
    best_candidates = sample_coords[torch.arange(N, device=device), max_indices]

    mask_improve = (~mask_good) & (max_vals > threshold)
    final_nodes = nodes_yx.clone()
    if mask_improve.any():
        final_nodes[mask_improve] = best_candidates[mask_improve]
    return final_nodes

def load_nodes_from_npz(
    tif_path: str,
    road_prob: torch.Tensor,
    *,
    p_count: int = 20,           # 指定样本规模，如 20 或 50
    key: str = "matched_node_norm",
    col_order: str = "xy", 
    snap: bool = True,
    snap_threshold: float = 0.1,
    snap_win: int = 10,
    verbose: bool = True,
    case_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    针对特定数据集结构完善的版本：
    1. 根据 tif 文件名和 p_count 自动构建 npz 路径（不使用 glob）。
    2. 坐标转换：归一化左下角 -> 像素左上角。
    3. 向量化吸附优化。
    """
    road = _normalize_prob_map(road_prob)
    H, W = road.shape[0], road.shape[1]

    # --- 1. 精准构建 NPZ 路径 ---
    tif_p = Path(tif_path)
    tif_name = tif_p.name
    location_key = tif_name.replace("crop_", "").replace("_z16.tif", "").replace(".tif", "")
    
    # 构建目标文件名: distance_dataset_all_{location_key}_p{p_count}.npz
    npz_name = f"distance_dataset_all_{location_key}_p{p_count}.npz"
    npz_p = tif_p.parent / npz_name

    if not npz_p.exists():
        raise FileNotFoundError(f"[NPZ] Missing expected file: {npz_p}")

    # --- 2. 加载数据 ---
    with np.load(npz_p, allow_pickle=True) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' not found in {npz_p}")
        nodes = np.asarray(z[key])
        
        # 读取元数据参数  
        meta_sat_height_px = int(z["meta_sat_height_px"]) 
        meta_sat_width_px = int(z["meta_sat_width_px"])
        euclidean_dist_norm = z["euclidean_dist_norm"]
        undirected_dist_norm = z["undirected_dist_norm"]

    # --- 3. 处理维度与 Case ---
    if nodes.ndim == 3:
        if case_idx is None:
            case_idx = np.random.randint(0, nodes.shape[0])
        nodes = nodes[case_idx]
    
    if nodes.size == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=road_prob.device), {"N": 0}

    # --- 4. 坐标转换 (Fixed: Norm, Bottom-Left) --- 
    x_norm, y_norm = nodes[:, 0], nodes[:, 1] 

    # X: [0, 1] -> [0, W-1]
    x_pix = np.rint(x_norm * (W - 1)).astype(np.int64)
    x_pix = np.clip(x_pix, 0, W - 1)

    # Y: Bottom-Up [0, 1] -> Top-Down [0, H-1]
    y_pix = np.rint((1.0 - y_norm) * (H - 1)).astype(np.int64)
    y_pix = np.clip(y_pix, 0, H - 1)

    nodes_yx = torch.stack([
        torch.from_numpy(y_pix), 
        torch.from_numpy(x_pix)
    ], dim=1).to(road_prob.device, dtype=torch.long)

    # --- 5. 向量化吸附 ---
    if snap:
        nodes_yx = snap_nodes_vectorized(nodes_yx, road, snap_threshold, snap_win)

    if verbose:
        print(f"[NPZ] Selected: {npz_name} (N={len(nodes_yx)}, p={p_count})") 
        print(f"[META] Height: {meta_sat_height_px}, Width: {meta_sat_width_px}")

    # --- 6. 构建返回信息字典 ---
    info_dict = {
        "npz_path": npz_p, 
        "p_count": p_count, 
        "N": len(nodes_yx),
        "meta_sat_height_px": meta_sat_height_px,
        "meta_sat_width_px": meta_sat_width_px,
        "euclidean_dist_norm": euclidean_dist_norm,
        "undirected_dist_norm": undirected_dist_norm,
        "case_idx": case_idx
    }
    
    return nodes_yx, info_dict
