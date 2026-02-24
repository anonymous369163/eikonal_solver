"""
evaluator.py — Unified evaluation for SAMRoute.

Provides SAMRouteEvaluator class with four evaluation modes:
  - eval_iou:               validation-set road segmentation IoU
  - visualize_segmentation: side-by-side GT vs pred mask images
  - eval_spearman:          multi-anchor Spearman/Kendall/PW-accuracy
  - eval_single_anchor:     single-anchor eval + path visualization

Usage (CLI):
    python evaluator.py iou       --ckpt_path ... --data_root ...
    python evaluator.py spearman  --ckpt_path ... --image_path ...
    python evaluator.py visualize --ckpt_path ... --save_path ...
    python evaluator.py single    --ckpt_path ... --anchor_idx 3 --k 8
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# sys.path setup (same as other modules)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SAM_REPO = _HERE.parent / "sam_road_repo"
for _p in [str(_SAM_REPO / "segment-anything-road"),
           str(_SAM_REPO / "sam"),
           str(_SAM_REPO)]:
    if _p not in sys.path:
        sys.path.append(_p)
_here_str = str(_HERE)
if _here_str in sys.path:
    sys.path.remove(_here_str)
sys.path.insert(0, _here_str)


# ---------------------------------------------------------------------------
# TIF image loading (extracted from feature_extractor.py)
# ---------------------------------------------------------------------------

def load_tif_image(image_path: str) -> torch.Tensor:
    """Load TIF/TIFF image, return [1, 3, H, W] float tensor in [0, 1]."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.tif', '.tiff']:
        raise ValueError(f"Unsupported file format: {ext}. Only .tif or .tiff are allowed.")
    try:
        import rasterio
        with rasterio.open(image_path) as src:
            img = src.read()
    except ImportError:
        import tifffile
        img = tifffile.imread(image_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)
        elif img.ndim == 3 and img.shape[2] <= 4:
            img = img.transpose(2, 0, 1)
    if img.shape[0] > 3:
        img = img[:3]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 65535.0 if img.max() > 255.0 else img / 255.0
    return torch.from_numpy(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Ranking / statistics helpers (extracted from utils.py)
# ---------------------------------------------------------------------------

def _rankdata(x: np.ndarray) -> np.ndarray:
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


def spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ra, rb = _rankdata(a) - _rankdata(a).mean(), _rankdata(b) - _rankdata(b).mean()
    ra = _rankdata(a); ra = ra - ra.mean()
    rb = _rankdata(b); rb = rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n < 2:
        return 0.0
    c, d = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return float((c - d) / (n * (n - 1) / 2 + 1e-12))


def pairwise_order_acc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = np.asarray(pred), np.asarray(gt)
    n = len(pred)
    if n < 2:
        return 1.0
    total, ok = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            ok += int((pred[i] - pred[j]) * (gt[i] - gt[j]) > 0)
    return float(ok / max(total, 1))


def pick_anchor_and_knn(
    nodes_yx: torch.Tensor, *, k: int = 10, anchor_idx: Optional[int] = None,
) -> Tuple[int, torch.Tensor]:
    """Select an anchor and its K nearest neighbors (excluding itself)."""
    N = int(nodes_yx.shape[0])
    if N <= 1:
        raise ValueError(f"Need at least 2 nodes, got N={N}")
    k = int(min(k, N - 1))
    if anchor_idx is None:
        anchor_idx = int(torch.randint(low=0, high=N, size=(1,)).item())
    anchor_idx = int(max(0, min(anchor_idx, N - 1)))
    anchor = nodes_yx[anchor_idx].float()
    d = torch.norm(nodes_yx.float() - anchor[None, :], dim=1)
    d[anchor_idx] = float("inf")
    nn_idx = torch.topk(d, k=k, largest=False).indices.long()
    return anchor_idx, nn_idx


INF_SENTINEL = 1e9

def trace_path_from_T(
    T: torch.Tensor, source_yx: torch.Tensor, goal_yx: torch.Tensor,
    device, *, diag: bool = True, max_steps: int = 200_000, stop_radius: int = 1,
) -> List[Tuple[int, int]]:
    """Backtrace shortest path from goal to source on distance field T."""
    T = T.to(device)
    H, W = T.shape
    src = source_yx.to(device).long()
    dst = goal_yx.to(device).long()
    src[0].clamp_(0, H - 1); src[1].clamp_(0, W - 1)
    dst[0].clamp_(0, H - 1); dst[1].clamp_(0, W - 1)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    y, x = int(dst[0].item()), int(dst[1].item())
    path = [(y, x)]
    visited = {(y, x)}
    prev_val = float(T[y, x].item())
    if (not np.isfinite(prev_val)) or (prev_val >= 0.999 * INF_SENTINEL):
        return path
    for _ in range(int(max_steps)):
        if abs(y - int(src[0])) <= stop_radius and abs(x - int(src[1])) <= stop_radius:
            break
        best_y, best_x, best_val = y, x, prev_val
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W or (ny, nx) in visited:
                continue
            v = float(T[ny, nx].item())
            if np.isfinite(v) and v < best_val:
                best_val, best_y, best_x = v, ny, nx
        if best_y == y and best_x == x:
            cand = []
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W or (ny, nx) in visited:
                    continue
                v = float(T[ny, nx].item())
                if np.isfinite(v):
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


def calc_path_len(path: List[Tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    pts = np.array(path, dtype=np.float32)
    return float(np.sum(np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# SAMRouteEvaluator
# ---------------------------------------------------------------------------

class SAMRouteEvaluator:
    """Unified evaluator for SAMRoute models."""

    def __init__(self, config_path: str, ckpt_path: str, device: str = "cuda"):
        import yaml
        from addict import Dict as AdDict
        from model import SAMRoute

        with open(config_path) as f:
            self.config = AdDict(yaml.safe_load(f))

        config_dir = os.path.dirname(os.path.abspath(config_path))
        for key in ("SAM_CKPT_PATH", "PRETRAINED_CKPT"):
            val = str(self.config.get(key, "") or "")
            if val and not os.path.isabs(val):
                self.config[key] = os.path.normpath(os.path.join(config_dir, val))

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SAMRoute(self.config)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state.get("state_dict", state), strict=False)
        self.model.eval().to(self.device)
        self.ckpt_path = ckpt_path
        print(f"[Evaluator] Loaded {ckpt_path} on {self.device}")

    # ------------------------------------------------------------------
    # 1. IoU evaluation (replaces eval_pretrained_iou.py)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_iou(
        self, data_root: str, *, max_batches: Optional[int] = None,
        batch_size: int = 8, road_dilation_radius: int = 0,
    ) -> Dict[str, float]:
        from dataset import build_dataloaders
        from torchmetrics.classification import BinaryJaccardIndex

        _, val_loader = build_dataloaders(
            root_dir=data_root, patch_size=int(self.config.PATCH_SIZE),
            batch_size=batch_size, num_workers=4, include_dist=False,
            val_fraction=0.1, samples_per_region=50,
            use_cached_features=True, preload_to_ram=True, preload_workers=4,
            road_dilation_radius=road_dilation_radius,
        )
        iou_metric = BinaryJaccardIndex(threshold=0.5).to(self.device)
        n_batches = 0
        for batch in val_loader:
            if max_batches and n_batches >= max_batches:
                break
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            _, road_prob = self.model._seg_forward(batch)
            road_gt = (batch["road_mask"].to(self.device, dtype=torch.float32) > 0.5).long()
            iou_metric.update(road_prob, road_gt)
            n_batches += 1
        iou = iou_metric.compute().item()
        print(f"[eval_iou] IoU={iou:.4f}  ({n_batches} batches)")
        return {"iou": iou, "n_batches": n_batches}

    # ------------------------------------------------------------------
    # 2. Segmentation visualization (replaces visualize_with_iou.py)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualize_segmentation(
        self, data_root: str, *, n_samples: int = 4,
        save_path: str = "images/vis_seg.png", road_dilation_radius: int = 0,
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from dataset import build_dataloaders
        from torchmetrics.classification import BinaryJaccardIndex

        _, val_loader = build_dataloaders(
            root_dir=data_root, patch_size=int(self.config.PATCH_SIZE),
            batch_size=n_samples, num_workers=2, include_dist=False,
            val_fraction=0.1, samples_per_region=50,
            use_cached_features=False, preload_to_ram=True, preload_workers=2,
            road_dilation_radius=road_dilation_radius,
        )
        batch = next(iter(val_loader))
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        loss_seg, road_prob = self.model._seg_forward(batch)
        road_gt = (batch["road_mask"].float() > 0.5).long()
        iou_m = BinaryJaccardIndex(threshold=0.5).to(self.device)
        iou_m.update(road_prob, road_gt)
        iou_val = iou_m.compute().item()

        n = min(n_samples, road_prob.shape[0])
        n_cols = 3 if "rgb" in batch else 2
        fig, axes = plt.subplots(n, n_cols, figsize=(5 * n_cols, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        for i in range(n):
            idx = 0
            if "rgb" in batch:
                rgb = batch["rgb"][i].cpu().numpy()
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                axes[i, idx].imshow(rgb); axes[i, idx].set_title("Input RGB"); axes[i, idx].axis("off")
                idx += 1
            gt = road_gt[i].cpu().numpy()
            axes[i, idx].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[i, idx].set_title("GT mask"); axes[i, idx].axis("off")
            idx += 1
            pred = (road_prob[i] > 0.5).float().cpu().numpy()
            axes[i, idx].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[i, idx].set_title("Pred mask"); axes[i, idx].axis("off")

        fig.suptitle(f"IoU={iou_val:.4f}  seg_loss={loss_seg.item():.4f}  ({os.path.basename(self.ckpt_path)})", fontsize=11)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[visualize_segmentation] IoU={iou_val:.4f}  Saved: {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # 3. Multi-anchor Spearman evaluation (replaces eval_multi_anchor.py)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_spearman(
        self, image_path: str, *, k: int = 19, min_in_patch: int = 3,
        downsample: int = 1024,
    ) -> Dict[str, float]:
        from dataset import load_nodes_from_npz
        from eikonal import EikonalConfig, eikonal_soft_sweeping

        device = self.device
        PATCH = int(self.config.get("PATCH_SIZE", 512))
        n_iters = int(self.config.get("ROUTE_EIK_ITERS", 20))
        min_ds = int(self.config.get("ROUTE_EIK_DOWNSAMPLE", 8))
        margin = int(self.config.get("ROUTE_ROI_MARGIN", 64))

        rgb = load_tif_image(image_path).to(device)
        _, _, H, W = rgb.shape
        scale = 1.0
        if downsample > 0 and max(H, W) > downsample:
            scale = downsample / max(H, W)
            rgb = F.interpolate(rgb, size=(max(1, int(H * scale)), max(1, int(W * scale))),
                                mode="bilinear", align_corners=False)
        _, _, H_ds, W_ds = rgb.shape

        rp_dummy = torch.zeros(H, W, device=device)
        nodes_orig, info = load_nodes_from_npz(
            image_path, rp_dummy, p_count=20, snap=False, verbose=False)
        if scale != 1.0:
            nodes_yx = (nodes_orig.float() * scale).long()
            nodes_yx[:, 0].clamp_(0, H_ds - 1)
            nodes_yx[:, 1].clamp_(0, W_ds - 1)
        else:
            nodes_yx = nodes_orig
        nodes_yx = nodes_yx.to(device)
        N = nodes_yx.shape[0]
        case = info["case_idx"]
        meta_px = int(info["meta_sat_height_px"])
        pscale = 1.0 / scale if scale != 1.0 else 1.0

        rp_cache: Dict[tuple, torch.Tensor] = {}

        def _get_road_prob(py0, py1, px0, px1):
            key = (py0, py1, px0, px1)
            if key not in rp_cache:
                rgb_p = rgb[:, :, py0:py1, px0:px1]
                if rgb_p.shape[2] < PATCH or rgb_p.shape[3] < PATCH:
                    rgb_p = F.pad(rgb_p, (0, PATCH - rgb_p.shape[3], 0, PATCH - rgb_p.shape[2]))
                rgb_hwc = (rgb_p * 255.0).squeeze(0).permute(1, 2, 0).unsqueeze(0)
                _, ms = self.model._predict_mask_logits_scores(rgb_hwc)
                rp_cache[key] = ms[0, :, :, 1].detach()
            return rp_cache[key]

        def _eikonal_k(rp, src, tgts):
            """One Eikonal solve covering all K targets; read K T-values."""
            K = tgts.shape[0]
            if K == 0:
                return np.empty(0, dtype=np.float64)
            H_r, W_r = rp.shape

            span_max = int(torch.max(torch.abs(tgts.float() - src.float())).item())
            half = span_max + margin
            P = max(2 * half + 1, 64)
            y0 = int(src[0]) - half;  x0 = int(src[1]) - half
            y1 = y0 + P;              x1 = x0 + P
            yy0, xx0 = max(y0, 0), max(x0, 0)
            yy1, xx1 = min(y1, H_r), min(x1, W_r)
            roi = F.pad(rp[yy0:yy1, xx0:xx1],
                        (xx0 - x0, x1 - xx1, yy0 - y0, y1 - yy1), value=0.0)
            sr_y = max(0, min(int(src[0]) - y0, P - 1))
            sr_x = max(0, min(int(src[1]) - x0, P - 1))

            ds = max(min_ds, math.ceil(P / max(n_iters, 1)))
            if ds > 1:
                P_pad = math.ceil(P / ds) * ds
                if P_pad > P:
                    roi = F.pad(roi, (0, P_pad - P, 0, P_pad - P), value=0.0)
                roi_c = F.max_pool2d(roi[None, None], kernel_size=ds, stride=ds).squeeze()
                cost = self.model._road_prob_to_cost(roi_c)
                Pc = cost.shape[0]
                sc_y = max(0, min(sr_y // ds, Pc - 1))
                sc_x = max(0, min(sr_x // ds, Pc - 1))
                smask = torch.zeros(1, Pc, Pc, dtype=torch.bool, device=device)
                smask[0, sc_y, sc_x] = True
                cfg = EikonalConfig(n_iters=n_iters, h=float(ds), tau_min=0.03,
                                    tau_branch=0.05, tau_update=0.03,
                                    use_redblack=True, monotone=True)
                T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)[0]
                vals = []
                for ki in range(K):
                    tr_y = max(0, min((int(tgts[ki, 0]) - y0) // ds, Pc - 1))
                    tr_x = max(0, min((int(tgts[ki, 1]) - x0) // ds, Pc - 1))
                    vals.append(float(T[tr_y, tr_x].item()))
            else:
                cost = self.model._road_prob_to_cost(roi)
                smask = torch.zeros(1, P, P, dtype=torch.bool, device=device)
                smask[0, sr_y, sr_x] = True
                cfg = EikonalConfig(n_iters=n_iters, h=1.0, tau_min=0.03,
                                    tau_branch=0.05, tau_update=0.03,
                                    use_redblack=True, monotone=True)
                T = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)[0]
                vals = []
                for ki in range(K):
                    tr_y = max(0, min(int(tgts[ki, 0]) - y0, P - 1))
                    tr_x = max(0, min(int(tgts[ki, 1]) - x0, P - 1))
                    vals.append(float(T[tr_y, tr_x].item()))
            return np.array(vals, dtype=np.float64)

        sp_acc, kd_acc, pw_acc = [], [], []
        t0 = time.time()

        for anc_idx in range(N):
            _, nn_global = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=anc_idx)
            anc = nodes_yx[anc_idx]
            ay, ax = int(anc[0]), int(anc[1])
            py0 = max(0, min(ay - PATCH // 2, H_ds - PATCH))
            px0 = max(0, min(ax - PATCH // 2, W_ds - PATCH))
            py1, px1 = py0 + PATCH, px0 + PATCH

            tgts_global = nodes_yx[nn_global]
            in_p = ((tgts_global[:, 0] >= py0) & (tgts_global[:, 0] < py1) &
                    (tgts_global[:, 1] >= px0) & (tgts_global[:, 1] < px1))
            keep = torch.where(in_p)[0]
            if keep.numel() < min_in_patch:
                continue

            nn_kept = nn_global[keep]
            tgt_patch = tgts_global[keep] - torch.tensor([py0, px0], device=device)
            anc_patch = torch.tensor([ay - py0, ax - px0], device=device).long()

            rp = _get_road_prob(py0, py1, px0, px1)
            T_raw = _eikonal_k(rp, anc_patch, tgt_patch)
            T_norm = T_raw * pscale / max(meta_px, 1)

            nn_np = nn_kept.cpu().numpy().astype(int)
            gt_arr = np.asarray(info["undirected_dist_norm"][case][anc_idx, nn_np], dtype=np.float64)
            valid = np.isfinite(T_norm) & (T_norm < 1e3) & np.isfinite(gt_arr)
            if valid.sum() < 3:
                continue

            pv, gv = T_norm[valid], gt_arr[valid]
            sp_acc.append(spearmanr(pv, gv))
            kd_acc.append(kendall_tau(pv, gv))
            pw_acc.append(pairwise_order_acc(pv, gv))

        elapsed = time.time() - t0
        if not sp_acc:
            result = dict(spearman=float("nan"), kendall=float("nan"),
                          pw_acc=float("nan"), n_anchors=0, elapsed=elapsed)
        else:
            result = dict(
                spearman=float(np.mean(sp_acc)), kendall=float(np.mean(kd_acc)),
                pw_acc=float(np.nanmean(pw_acc)), n_anchors=len(sp_acc), elapsed=elapsed,
            )
        print(f"[eval_spearman] Spearman={result['spearman']:+.4f}  "
              f"Kendall={result['kendall']:+.4f}  PW={result['pw_acc']:.4f}  "
              f"anchors={result['n_anchors']}  time={elapsed:.1f}s")
        return result

    # ------------------------------------------------------------------
    # 4. Single-anchor eval + path visualization
    #    (replaces validate_sam_route.py and model.py __main__)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_single_anchor(
        self, image_path: str, *, anchor_idx: Optional[int] = None,
        k: int = 8, downsample: int = 1024, eik_iters: int = 700,
        margin: int = 96, save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        from dataset import load_nodes_from_npz
        from eikonal import EikonalConfig, eikonal_soft_sweeping

        device = self.device

        rgb = load_tif_image(image_path).to(device)
        _, _, H_orig, W_orig = rgb.shape
        rp_dummy = torch.zeros(H_orig, W_orig, device=device)
        nodes_orig, info = load_nodes_from_npz(
            image_path, rp_dummy, p_count=20, snap=False, verbose=True)
        if nodes_orig.shape[0] < 2:
            raise RuntimeError("Not enough nodes in NPZ (<2).")

        scale = 1.0
        if downsample > 0 and max(H_orig, W_orig) > downsample:
            scale = downsample / max(H_orig, W_orig)
        nH = max(1, int(H_orig * scale))
        nW = max(1, int(W_orig * scale))

        rgb_ds = F.interpolate(rgb, size=(nH, nW), mode="bilinear", align_corners=False)
        rgb_hwc = (rgb_ds * 255.0).squeeze(0).permute(1, 2, 0).unsqueeze(0)
        _, ms = self.model._predict_mask_logits_scores(rgb_hwc)
        road_prob = ms[0, :, :, 1].detach()
        H_ds, W_ds = road_prob.shape

        if scale != 1.0:
            nodes_yx = (nodes_orig.float() * scale).long()
            nodes_yx[:, 0].clamp_(0, H_ds - 1)
            nodes_yx[:, 1].clamp_(0, W_ds - 1)
        else:
            nodes_yx = nodes_orig
        nodes_yx = nodes_yx.to(device)

        anc_idx, nn_idx = pick_anchor_and_knn(nodes_yx, k=k, anchor_idx=anchor_idx)
        anc = nodes_yx[anc_idx]
        tgts = nodes_yx[nn_idx]
        K = tgts.shape[0]
        print(f"  Anchor={anc_idx}  yx={anc.tolist()}  K={K}")

        span_max = int(torch.max(torch.abs(tgts.float() - anc.float())).item())
        half = span_max + margin
        P = max(2 * half + 1, 512)
        y0, x0 = int(anc[0]) - half, int(anc[1]) - half
        y1, x1 = y0 + P, x0 + P
        yy0, xx0 = max(y0, 0), max(x0, 0)
        yy1, xx1 = min(y1, H_ds), min(x1, W_ds)
        patch = torch.zeros(P, P, device=device, dtype=road_prob.dtype)
        patch[yy0-y0:yy1-y0, xx0-x0:xx1-x0] = road_prob[yy0:yy1, xx0:xx1]

        cost = self.model._road_prob_to_cost(patch)
        sr_y = max(0, min(int(anc[0]) - y0, P - 1))
        sr_x = max(0, min(int(anc[1]) - x0, P - 1))
        smask = torch.zeros(1, P, P, dtype=torch.bool, device=device)
        smask[0, sr_y, sr_x] = True
        cfg = EikonalConfig(n_iters=max(eik_iters, P), tau_min=0.03, tau_branch=0.05,
                            tau_update=0.03, use_redblack=True, monotone=True)
        T_patch = eikonal_soft_sweeping(cost.unsqueeze(0), smask, cfg)
        if T_patch.dim() == 3:
            T_patch = T_patch[0]

        src_rel = torch.tensor([sr_y, sr_x], device=device, dtype=torch.long)
        pscale = 1.0 / scale if scale != 1.0 else 1.0
        meta_px = int(info["meta_sat_height_px"])
        nn_np = nn_idx.cpu().numpy()
        case = info["case_idx"]
        gt = np.asarray(info["undirected_dist_norm"][case][anc_idx, nn_np], dtype=np.float64)
        euc = np.asarray(info["euclidean_dist_norm"][case][anc_idx, nn_np], dtype=np.float64)

        pred_norm = np.full(K, np.nan)
        paths_global = []
        t_vals = []
        for ki in range(K):
            tr_y = int(tgts[ki, 0]) - y0
            tr_x = int(tgts[ki, 1]) - x0
            in_roi = 0 <= tr_y < P and 0 <= tr_x < P
            tval = float(T_patch[max(0, min(tr_y, P-1)), max(0, min(tr_x, P-1))].item()) if in_roi else float("inf")
            t_vals.append(tval)
            if not in_roi or not np.isfinite(tval) or tval >= 1e5:
                paths_global.append(None)
                continue
            tgt_rel = torch.tensor([max(0, min(tr_y, P-1)), max(0, min(tr_x, P-1))],
                                   device=device, dtype=torch.long)
            pp = trace_path_from_T(T_patch, src_rel, tgt_rel, device, diag=True)
            if len(pp) < 2:
                paths_global.append(None)
                continue
            pg = [(p[0] + y0, p[1] + x0) for p in pp]
            paths_global.append(pg)
            pred_norm[ki] = calc_path_len(pg) * pscale / max(meta_px, 1)

        mask = np.isfinite(pred_norm) & np.isfinite(gt) & np.isfinite(euc)
        n_valid = int(mask.sum())
        metrics = {}
        if n_valid >= 3:
            metrics["spearman"] = spearmanr(pred_norm[mask], gt[mask])
            metrics["kendall"] = kendall_tau(pred_norm[mask], gt[mask])
            metrics["pw_acc"] = pairwise_order_acc(pred_norm[mask], gt[mask])
        else:
            metrics["spearman"] = metrics["kendall"] = metrics["pw_acc"] = float("nan")
        metrics["n_valid"] = n_valid
        print(f"  Spearman={metrics['spearman']:+.4f}  Kendall={metrics['kendall']:+.4f}")

        if save_path:
            self._plot_paths(rgb_ds, road_prob, anc, tgts, T_patch,
                             y0, x0, P, paths_global, t_vals, nn_np, save_path)
            metrics["save_path"] = save_path

        return metrics

    # ------------------------------------------------------------------
    # Visualization helper
    # ------------------------------------------------------------------

    def _plot_paths(self, rgb, road_prob, src, tgts, T_patch,
                    y0, x0, P, paths_global, t_vals, nn_np, save_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe

        if rgb.dim() == 4:
            rgb = rgb[0]
        img_np = (rgb.detach().float().cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        H_img, W_img = img_np.shape[:2]
        H, W = road_prob.shape
        scale_img = H_img / max(H, 1)
        road_np = road_prob.detach().float().cpu().numpy()
        T_np = T_patch.detach().float().cpu().numpy()
        K = tgts.shape[0]
        y1, x1 = y0 + P, x0 + P

        T_full = np.full((H_img, W_img), np.nan, dtype=np.float32)
        iy0, ix0 = max(0, int(y0 * scale_img)), max(0, int(x0 * scale_img))
        iy1, ix1 = min(H_img, int(np.ceil(y1 * scale_img))), min(W_img, int(np.ceil(x1 * scale_img)))
        rh, rw = iy1 - iy0, ix1 - ix0
        if rh > 0 and rw > 0:
            T_full[iy0:iy1, ix0:ix1] = F.interpolate(
                torch.from_numpy(T_np)[None, None], size=(rh, rw),
                mode="bilinear", align_corners=False).squeeze().numpy()

        finite = np.isfinite(T_full)
        vmin = float(np.min(T_full[finite])) if finite.any() else 0.0
        vmax = float(np.percentile(T_full[finite], 99)) if finite.any() else 1.0

        colors = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(img_np)
        masked_road = np.ma.masked_where(road_np < 0.1, road_np)
        ax.imshow(masked_road, cmap="Reds", vmin=0, vmax=1, alpha=0.4)
        ax.imshow(T_full, cmap="magma", vmin=vmin, vmax=vmax, alpha=0.32)
        ax.plot([ix0, ix1, ix1, ix0, ix0], [iy0, iy0, iy1, iy1, iy0],
                color="white", lw=1.5, ls="--", alpha=0.75, zorder=8)

        for ki, path in enumerate(paths_global):
            if path is not None:
                ys = [p[0] * scale_img for p in path]
                xs = [p[1] * scale_img for p in path]
                ax.plot(xs, ys, color=colors[ki], lw=2.0, alpha=0.85, zorder=10,
                        path_effects=[pe.Stroke(linewidth=3.5, foreground="black", alpha=0.5),
                                      pe.Normal()])

        for ki in range(K):
            ty, tx = int(tgts[ki, 0].item() * scale_img), int(tgts[ki, 1].item() * scale_img)
            tval = t_vals[ki]
            lbl = f"node {nn_np[ki]}  T={tval:.0f}" if np.isfinite(tval) else f"node {nn_np[ki]}"
            ax.scatter([tx], [ty], s=220, marker="x", color=colors[ki],
                       linewidths=2.5, zorder=14, label=lbl)

        sy, sx = int(src[0].item() * scale_img), int(src[1].item() * scale_img)
        ax.scatter([sx], [sy], s=300, marker="o", color="lime",
                   edgecolors="black", linewidths=2.5, zorder=15, label="Source")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8.5, framealpha=0.9)
        ax.axis("off")
        ax.set_title("SAMRoute: shortest paths (magma=T-field, red=road_prob)", fontsize=11)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved visualization: {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAMRoute Evaluator")
    sub = parser.add_subparsers(dest="command")

    # --- iou ---
    p_iou = sub.add_parser("iou", help="Evaluate road segmentation IoU")
    p_iou.add_argument("--config", required=True)
    p_iou.add_argument("--ckpt_path", required=True)
    p_iou.add_argument("--data_root", required=True)
    p_iou.add_argument("--max_batches", type=int, default=None)
    p_iou.add_argument("--road_dilation_radius", type=int, default=0)

    # --- spearman ---
    p_sp = sub.add_parser("spearman", help="Multi-anchor Spearman evaluation")
    p_sp.add_argument("--config", required=True)
    p_sp.add_argument("--ckpt_path", required=True)
    p_sp.add_argument("--image_path", required=True)
    p_sp.add_argument("--k", type=int, default=19)
    p_sp.add_argument("--min_in_patch", type=int, default=3)
    p_sp.add_argument("--downsample", type=int, default=1024)

    # --- visualize ---
    p_vis = sub.add_parser("visualize", help="Visualize segmentation results")
    p_vis.add_argument("--config", required=True)
    p_vis.add_argument("--ckpt_path", required=True)
    p_vis.add_argument("--data_root", required=True)
    p_vis.add_argument("--n_samples", type=int, default=4)
    p_vis.add_argument("--save_path", default="images/vis_seg.png")
    p_vis.add_argument("--road_dilation_radius", type=int, default=0)

    # --- single ---
    p_single = sub.add_parser("single", help="Single-anchor eval + path visualization")
    p_single.add_argument("--config", required=True)
    p_single.add_argument("--ckpt_path", required=True)
    p_single.add_argument("--image_path", required=True)
    p_single.add_argument("--anchor_idx", type=int, default=None)
    p_single.add_argument("--k", type=int, default=8)
    p_single.add_argument("--save_path", default="images/samroute_paths.png")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    ev = SAMRouteEvaluator(args.config, args.ckpt_path)

    if args.command == "iou":
        ev.eval_iou(args.data_root, max_batches=args.max_batches,
                    road_dilation_radius=args.road_dilation_radius)
    elif args.command == "spearman":
        ev.eval_spearman(args.image_path, k=args.k, min_in_patch=args.min_in_patch,
                         downsample=args.downsample)
    elif args.command == "visualize":
        ev.visualize_segmentation(args.data_root, n_samples=args.n_samples,
                                  save_path=args.save_path,
                                  road_dilation_radius=args.road_dilation_radius)
    elif args.command == "single":
        ev.eval_single_anchor(args.image_path, anchor_idx=args.anchor_idx,
                              k=args.k, save_path=args.save_path)
