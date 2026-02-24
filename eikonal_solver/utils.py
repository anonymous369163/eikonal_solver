"""
Utils module for common functions shared across eikonal solver modules.
Contains shared constants, functions and classes extracted from duplicate code.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from matplotlib import pyplot as plt
import torch.nn.functional as F

# FIX: treat cfg.inf=1e9 as 'unreached' sentinel from solver
INF_SENTINEL = 1e9


# -----------------------------------------------------------------------------
# Common Helper Functions and Classes
# -----------------------------------------------------------------------------

def _round_up_int(v: int, m: int) -> int:
    """Round up integer v to the nearest multiple of m."""
    return ((v + (m - 1)) // m) * m


def _ensure_hw_tensor(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in (H, W) format."""
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        if x.size(0) == 1:
            return x[0]
        if x.size(-1) == 1:
            return x[..., 0]
    raise ValueError(f"road_prob must be 2D [H,W] (or squeezable), got shape={tuple(x.shape)}")


def _normalize_prob_map(x: torch.Tensor) -> torch.Tensor:
    """Normalize probability map to [0, 1] range."""
    x = x.float()
    mx = float(x.max().item()) if x.numel() else 0.0
    if mx > 1.5:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def _to_uint8_hwc(rgb: torch.Tensor) -> np.ndarray:
    """Convert tensor to uint8 HWC numpy array."""
    if isinstance(rgb, torch.Tensor):
        x = rgb.detach().cpu()
        if x.dim() == 4:
            x = x[0]
        if x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        x = x.numpy()
    else:
        x = np.asarray(rgb)

    if x.dtype != np.uint8:
        mx = float(x.max()) if x.size else 0.0
        if mx <= 1.5:
            x = (np.clip(x, 0, 1) * 255.0).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def load_ground_truth_road_map_common(tif_path: str, target_hw: Tuple[int, int], device) -> torch.Tensor:
    """
    加载真实路网图并强制对齐尺寸到 target_hw。
    """
    tif_p = Path(tif_path)
    base_name = tif_p.name.replace("crop_", "roadnet_").replace(".tif", "")
    parent_dir = tif_p.parent
    
    gt_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = parent_dir / (base_name + ext)
        if candidate.exists():
            gt_path = candidate
            break
    
    if gt_path is None:
        raise FileNotFoundError(f"[Debug] GT RoadNet not found in {parent_dir} for {base_name}")

    print(f"[Debug Mode] Loading GT RoadNet: {gt_path.name}")
    img = Image.open(gt_path).convert('L')
    w_orig, h_orig = img.size
    
    # 归一化并转为 Tensor (1, 1, H, W)
    gt_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    # 强制尺寸对齐
    if (h_orig != target_hw[0]) or (w_orig != target_hw[1]):
        print(f"  - Resizing GT from ({h_orig}, {w_orig}) to {target_hw}")
        gt_tensor = F.interpolate(gt_tensor, size=target_hw, mode='bilinear', align_corners=False)
    
    return gt_tensor.squeeze()


def trace_path_from_distance_field_common(
    T: torch.Tensor,
    source_yx: torch.Tensor,
    goal_yx: torch.Tensor,
    device,
    *,
    diag: bool = True,
    max_steps: int = 200000,
    stop_radius: int = 1,
):
    """
    More robust backtrace:
    - prefer descending
    - allow tiny plateau moves
    - prevent loops with visited
    """
    T = T.to(device)
    H, W = T.shape
    src = source_yx.to(device).long()
    dst = goal_yx.to(device).long()
    src[0].clamp_(0, H - 1); src[1].clamp_(0, W - 1)
    dst[0].clamp_(0, H - 1); dst[1].clamp_(0, W - 1)

    if diag:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]

    y, x = int(dst[0].item()), int(dst[1].item())
    path = [(y, x)]
    visited = {(y, x)}

    prev_val = float(T[y, x].item())
    # FIX: solver uses INF_SENTINEL for unreached cells; treat as invalid too
    if (not np.isfinite(prev_val)) or (prev_val >= 0.999 * float(INF_SENTINEL)):
        return path

    for _ in range(int(max_steps)):
        if abs(y - int(src[0])) <= stop_radius and abs(x - int(src[1])) <= stop_radius:
            break

        best_y, best_x = y, x
        best_val = prev_val

        # strictly better first
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if (ny, nx) in visited:
                continue
            v = float(T[ny, nx].item())
            if not np.isfinite(v):
                continue
            if v < best_val:
                best_val = v
                best_y, best_x = ny, nx

        # if no descending neighbor: take the smallest non-increasing step (plateau)
        if best_y == y and best_x == x:
            cand = []
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if (ny, nx) in visited:
                    continue
                v = float(T[ny, nx].item())
                if not np.isfinite(v):
                    continue
                cand.append((v, ny, nx))
            if not cand:
                break
            cand.sort(key=lambda t: t[0])
            best_val, best_y, best_x = cand[0]
            if best_val > prev_val + 1e-2:
                break

        y, x = best_y, best_x
        visited.add((y, x))
        path.append((y, x))
        prev_val = best_val

    return path


# -----------------------------------------------------------------------------
# Path Length Calculation
# -----------------------------------------------------------------------------

def calc_path_len_vec(path_list):
    """
    Calculate path length from a list of (y, x) coordinates.
    Input: path_list = [(y0,x0), (y1,x1), ..., (yn,xn)]
    Output: float length
    """
    if len(path_list) < 2:
        return 0.0
    
    # Convert to NumPy array (N, 2)
    pts = np.array(path_list, dtype=np.float32)
    
    # Calculate differences between adjacent points (dy, dx)
    diffs = pts[1:] - pts[:-1]
    
    # Calculate Euclidean distance sqrt(dx^2 + dy^2)
    segment_dists = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Sum all segments
    return float(np.sum(segment_dists))


# -----------------------------------------------------------------------------
# Anchor + KNN Selection
# -----------------------------------------------------------------------------

def pick_anchor_and_knn(
    nodes_yx: torch.Tensor,
    *,
    k: int = 10,
    anchor_idx: Optional[int] = None,
) -> Tuple[int, torch.Tensor]:
    """Select an anchor and its K nearest neighbors (excluding itself)."""
    if nodes_yx.ndim != 2 or nodes_yx.shape[1] != 2:
        raise ValueError(f"nodes_yx must be (N,2) yx, got {tuple(nodes_yx.shape)}")
    N = int(nodes_yx.shape[0])
    if N <= 1:
        raise ValueError(f"Need at least 2 nodes, got N={N}")
    k = int(min(k, N - 1))
    if anchor_idx is None:
        anchor_idx = int(torch.randint(low=0, high=N, size=(1,)).item())
    anchor_idx = int(max(0, min(anchor_idx, N - 1)))

    anchor = nodes_yx[anchor_idx].float()  # (2,)
    d = torch.norm(nodes_yx.float() - anchor[None, :], dim=1)  # (N,)
    d[anchor_idx] = float("inf")
    nn_idx = torch.topk(d, k=k, largest=False).indices.long()  # (k,)
    return anchor_idx, nn_idx


# -----------------------------------------------------------------------------
# Statistical Functions for Ranking Evaluation
# -----------------------------------------------------------------------------

def rankdata(x: np.ndarray) -> np.ndarray:
    """Compute ranks with average for ties."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def mask_finite(*arrs):
    """Create a mask for finite values across multiple arrays."""
    m = None
    for a in arrs:
        a = np.asarray(a, dtype=np.float64)
        mm = np.isfinite(a)
        m = mm if m is None else (m & mm)
    return m if m is not None else None


def spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12)
    return float((ra * rb).sum() / denom)


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Kendall tau-a correlation coefficient."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    if n < 2:
        return 0.0
    concord = 0
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            s = da * db
            if s > 0:
                concord += 1
            elif s < 0:
                discord += 1
    denom = n * (n - 1) / 2
    return float((concord - discord) / (denom + 1e-12))


def pairwise_order_acc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute pairwise ordering accuracy."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    n = len(pred)
    if n < 2:
        return 1.0
    total = 0
    ok = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            ok += int((pred[i] - pred[j]) * (gt[i] - gt[j]) > 0)
    return float(ok / max(total, 1))


# -----------------------------------------------------------------------------
# Formatting and Output Functions
# -----------------------------------------------------------------------------

def fmt_rank_table(
    nei_idx: np.ndarray, 
    pred: np.ndarray, 
    gt: np.ndarray, 
    euc: np.ndarray, 
    tvals: np.ndarray, 
    valid: np.ndarray
) -> str:
    """Format ranking results as a table string."""
    order_gt = np.argsort(gt)
    lines = ["idx\tpred\tgt\teuclid\tT(goal)\tvalid"]
    for t in order_gt:
        lines.append(
            f"{int(nei_idx[t])}\t{pred[t]:.6f}\t{gt[t]:.6f}\t{euc[t]:.6f}\t{tvals[t]:.3e}\t{bool(valid[t])}"
        )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Ranking Metrics Computation
# -----------------------------------------------------------------------------

def compute_ranking_metrics(
    nn_idx: torch.Tensor,
    pred_norm: np.ndarray,
    gt: np.ndarray,
    euc: np.ndarray,
    mask: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Compute Spearman, Kendall, pairwise accuracy, and top-1 indices."""
    n_valid = int(mask.sum()) if mask is not None else 0
    top1_gt = int(nn_idx[int(np.argmin(gt))].item())
    top1_euc = int(nn_idx[int(np.argmin(euc))].item())
    top1_pred = (
        int(nn_idx[int(np.nanargmin(pred_norm))].item())
        if np.isfinite(pred_norm).any()
        else -1
    )

    if n_valid >= 3:
        s_pred = spearmanr(pred_norm[mask], gt[mask])
        s_euc = spearmanr(euc[mask], gt[mask])
        k_pred = kendall_tau(pred_norm[mask], gt[mask])
        k_euc = kendall_tau(euc[mask], gt[mask])
        pw_pred = pairwise_order_acc(pred_norm[mask], gt[mask])
        pw_euc = pairwise_order_acc(euc[mask], gt[mask])
    else:
        s_pred = s_euc = k_pred = k_euc = pw_pred = pw_euc = float("nan")

    return {
        "s_pred": s_pred,
        "s_euc": s_euc,
        "k_pred": k_pred,
        "k_euc": k_euc,
        "pw_pred": pw_pred,
        "pw_euc": pw_euc,
        "top1_gt": top1_gt,
        "top1_pred": top1_pred,
        "top1_euc": top1_euc,
        "n_valid": n_valid,
    }