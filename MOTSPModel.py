import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
from pathlib import Path


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        encoder_mode = self.model_params.get('encoder_mode', 'graph_image_fusion')
        if encoder_mode == 'graph_image_fusion' and self.model_params.get('use_satellite', False) and reset_state.sat_img is None:
            raise ValueError("use_satellite=True but reset_state.sat_img is None. Enable dataset satellite loading.")
        distance_matrix = getattr(reset_state, 'distance_matrix', None)
        self.encoded_nodes = self.encoder(
            reset_state.problems,
            reset_state.xy_img,
            reset_state.sat_img,
            distance_matrix=distance_matrix,
        )
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if state.current_node is None:
            device = self.encoded_nodes.device
            selected = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=device)

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax':
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                    .reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None
                
        return selected, prob

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        fusion_layer_num = self.model_params['fusion_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)

        # encoder_mode controls which branch is used:
        #   'graph_image_fusion' (default): graph + xy_img + satellite fusion
        #   'graph_only': only graph encoder + optional distance matrix injection
        self.encoder_mode = self.model_params.get('encoder_mode', 'graph_image_fusion')
        if self.encoder_mode not in ('graph_image_fusion', 'graph_only'):
            raise ValueError(f"Unknown encoder_mode: '{self.encoder_mode}'. Use 'graph_image_fusion' | 'graph_only'.")

        # ================================================================
        # Scheme 1: inject GT distance matrix into node embeddings (teacher)
        # - Controlled by model_params['use_distance_in_encoder'] (default False)
        # - Default uses statistics per node: mean/min/max over distances to others
        # ================================================================
        self.use_distance_in_encoder = bool(self.model_params.get('use_distance_in_encoder', False))
        if self.use_distance_in_encoder:
            # Stats projection: (mean, min, max) -> embedding_dim
            self.distance_stat_proj = nn.Sequential(
                nn.Linear(3, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            # Optional: full-row distance projection (kept for ablation / future use)
            self.distance_proj = nn.Sequential(
                nn.Linear(1, max(1, embedding_dim // 4)),
                nn.ReLU(),
                nn.Linear(max(1, embedding_dim // 4), embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            self.distance_use_stats = bool(self.model_params.get('distance_use_stats', True))
            self.distance_alpha = nn.Parameter(torch.tensor(0.1))

        if self.encoder_mode == 'graph_only':
            # Only graph encoder: full encoder_layer_num layers, no image branch
            self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        else:
            # graph_image_fusion: two-stream (graph + image) with cross-fusion
            self.embedding_patch = PatchEmbedding(**model_params)
            self.sat_encoder = SatelliteEncoder(**model_params)
            self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num - fusion_layer_num)])
            self.layers_img = nn.ModuleList(
                [EncoderLayer(**model_params) for _ in range(encoder_layer_num - fusion_layer_num)])
            self.fusion_layers = nn.ModuleList([EncoderFusionLayer(**model_params) for _ in range(fusion_layer_num)])
            self.fcp = nn.Parameter(torch.randn(1, self.model_params['bn_num'], embedding_dim))
            self.fcp_img = nn.Parameter(torch.randn(1, self.model_params['bn_img_num'], embedding_dim))

    def forward(self, data, img, sat_img=None, distance_matrix=None):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        # ============ Scheme 1: distance matrix injection into node stream ============
        if self.use_distance_in_encoder and distance_matrix is not None:
            B, N = data.shape[:2]
            if distance_matrix.dim() == 2:
                distance_matrix = distance_matrix.unsqueeze(0).expand(B, -1, -1)
            # Align device/dtype for safe math
            distance_matrix = distance_matrix.to(device=embedded_input.device, dtype=embedded_input.dtype)

            if bool(getattr(self, 'distance_use_stats', True)):
                # Stats over each row i: exclude diagonal
                # mean/min/max of distances to other nodes
                if N <= 1:
                    dist_stats = torch.zeros((B, N, 3), device=embedded_input.device, dtype=embedded_input.dtype)
                else:
                    eye = torch.eye(N, device=embedded_input.device, dtype=torch.bool).unsqueeze(0)  # (1,N,N)
                    dist_masked = distance_matrix.masked_fill(eye, float('inf'))

                    # mean: set diagonal to 0 then divide by (N-1)
                    dist_valid = distance_matrix.masked_fill(eye, 0.0)
                    dist_sum = dist_valid.sum(dim=-1)  # (B,N)
                    dist_mean = dist_sum / float(N - 1)

                    # min/max excluding diagonal
                    dist_min = dist_masked.min(dim=-1)[0]
                    dist_max = dist_masked.masked_fill(eye, float('-inf')).max(dim=-1)[0]

                    dist_stats = torch.stack([dist_mean, dist_min, dist_max], dim=-1)  # (B,N,3)

                dist_emb = self.distance_stat_proj(dist_stats)  # (B,N,E)
            else:
                # Full row projection (B,N,N) -> (B,N,E) via per-entry MLP + mean-pool
                dist_flat = distance_matrix.unsqueeze(-1)  # (B,N,N,1)
                dist_emb = self.distance_proj(dist_flat).mean(dim=2)  # (B,N,E)

            alpha = torch.sigmoid(self.distance_alpha)
            embedded_input = embedded_input + alpha * dist_emb

        # ============ encoder_mode branch ============
        if self.encoder_mode == 'graph_only':
            out = embedded_input
            for layer in self.layers:
                out = layer(out)
            return out
            # shape: (batch, problem, embedding)

        # graph_image_fusion: two-stream encoder with cross-attention fusion
        embedded_patch = self.embedding_patch(img)
        sat_tokens = self.sat_encoder(sat_img, batch_size=data.shape[0])
        if sat_tokens is not None:
            embedded_patch = torch.cat((embedded_patch, sat_tokens), dim=1)

        out = embedded_input
        out_img = embedded_patch
        for i in range(self.model_params['encoder_layer_num'] - self.model_params['fusion_layer_num']):
            out = self.layers[i](out)
            out_img = self.layers_img[i](out_img)
        fcp = self.fcp.repeat(data.shape[0], 1, 1)
        fcp_img = self.fcp_img.repeat(img.shape[0], 1, 1)
        for layer in self.fusion_layers:
            out, out_img, fcp, fcp_img = layer(out, out_img, fcp, fcp_img)
        return torch.cat((out, out_img), dim=1)


class SatelliteEncoder(nn.Module):
    """
    Satellite image encoder with support for SAM-Road backbone.
    - Input: sat_img (1, 3, H, W) or (B, 3, H, W) float in [0,1]
    - Output: (B, sat_grid^2, embedding_dim) tokens

    Supports two modes controlled by `train_sat_encoder`:
    - False (default): Freeze SAM-Road backbone, cache features for efficiency
    - True: Allow gradient flow through backbone for LoRA/fine-tuning
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.use_satellite = bool(self.model_params.get('use_satellite', False))
        self.sat_grid = int(self.model_params.get('sat_grid', 14))
        self.embedding_dim = int(self.model_params['embedding_dim'])
        self.sat_encoder_type = self.model_params.get('sat_encoder_type', 'placeholder')
        # Whether to train SAM-Road backbone (for LoRA fine-tuning)
        self.train_sat_encoder = bool(self.model_params.get('train_sat_encoder', False))

        # ---- placeholder CNN stem (kept for fallback) ----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(64, self.embedding_dim)

        # ---- SAM-Road lazy modules ----
        self._samroad = None
        self._samroad_cfg = None
        self._samroad_feat_proj = nn.Linear(256, self.embedding_dim)  # SAM-Road image_embeddings channels=256
        # 2D position encoding for sat tokens (same style as PatchEmbedding)
        self.sat_position_proj = nn.Sequential(
            nn.Linear(2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        # cache last satellite features (same tile reused for many batches)
        # Only used when train_sat_encoder=False (frozen backbone mode)
        # We cache the pooled features BEFORE projection to avoid backward graph issues.
        self._last_sat_obj = None
        self._last_placeholder_pooled = None  # (1, 64, G, G) detached
        self._last_samroad_pooled = None  # (1, 256, G, G) detached

    def _add_sat_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, G*G, embed_dim) or (1, G*G, embed_dim)
        Add explicit 2D positional encoding for the sat_grid x sat_grid token grid.
        """
        if tokens is None:
            return tokens
        if tokens.dim() != 3:
            raise ValueError(f"sat tokens must be 3D, got shape={tuple(tokens.shape)}")

        device = tokens.device
        G = self.sat_grid
        grid_x, grid_y = torch.meshgrid(
            torch.arange(G, device=device),
            torch.arange(G, device=device),
            indexing='ij'
        )
        xy = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2).float()
        if G > 1:
            xy = xy / (G - 1)
        pos = self.sat_position_proj(xy)  # (1, G*G, embed)
        return tokens + pos

    @staticmethod
    def _find_upwards(start_dir: Path, marker: str, max_depth: int = 10) -> Path | None:
        cur = start_dir.resolve()
        for _ in range(max_depth):
            if (cur / marker).exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        return None

    def _ensure_samroad_loaded(self, device: torch.device):
        if self._samroad is not None:
            return

        # auto-detect project root containing sam_road_repo/ and checkpoints/
        here = Path(__file__).resolve()
        root = self._find_upwards(here.parent, "sam_road_repo") or self._find_upwards(here.parent, "MMDataset")
        if root is None:
            raise RuntimeError("Cannot locate project root for SAM-Road (missing sam_road_repo/ or MMDataset/).")

        repo_dir = Path(self.model_params.get('sam_road_repo_dir', str(root / "sam_road_repo"))).resolve()
        ckpt_dir = Path(self.model_params.get('sam_road_ckpt_dir', str(root / "checkpoints"))).resolve()

        ckpt_name = self.model_params.get('sam_road_ckpt_name', None)
        if ckpt_name is None:
            ckpt_name = "cityscale_vitb_512_e10.ckpt"
        ckpt_path = (ckpt_dir / ckpt_name).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAM-Road checkpoint not found: {ckpt_path}")

        cfg_name = self.model_params.get('sam_road_cfg_name', None)
        if cfg_name is None:
            # minimal heuristic based on ckpt name
            lower = ckpt_path.name.lower()
            if "spacenet" in lower or "256" in lower:
                cfg_name = "toponet_vitb_256_spacenet.yaml"
            else:
                cfg_name = "toponet_vitb_512_cityscale.yaml"
        cfg_path = (repo_dir / "config" / cfg_name).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"SAM-Road config not found: {cfg_path}")

        # make sam_road_repo importable using importlib to avoid module name conflicts
        import importlib.util

        # Load utils.py from sam_road_repo explicitly
        utils_path = repo_dir / "utils.py"
        if not utils_path.exists():
            raise FileNotFoundError(f"SAM-Road utils.py not found: {utils_path}")
        spec_utils = importlib.util.spec_from_file_location("sam_road_utils", str(utils_path))
        sam_road_utils = importlib.util.module_from_spec(spec_utils)
        spec_utils.loader.exec_module(sam_road_utils)
        load_config = sam_road_utils.load_config

        # Add paths for model.py dependencies (segment-anything, etc.)
        extra_paths = [
            repo_dir,
            repo_dir / "sam",
            repo_dir / "segment-anything-road",
        ]
        for p in extra_paths:
            p = p.resolve()
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        from model import SAMRoad  # sam_road_repo/model.py

        cfg = load_config(str(cfg_path))
        # Ensure SAM official checkpoint path is resolvable (sam_road_repo expects relative path)
        sam_ckpt_path = repo_dir / "sam_ckpts" / "sam_vit_b_01ec64.pth"
        if hasattr(cfg, "SAM_CKPT_PATH"):
            if not Path(str(cfg.SAM_CKPT_PATH)).is_absolute():
                cfg.SAM_CKPT_PATH = str(sam_ckpt_path.resolve())
        else:
            cfg.SAM_CKPT_PATH = str(sam_ckpt_path.resolve())

        net = SAMRoad(cfg)

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        net.to(device)

        self._samroad = net
        self._samroad_cfg = cfg

    def forward(self, sat_img, batch_size: int):
        if (not self.use_satellite) or sat_img is None:
            return None

        if self.sat_encoder_type == 'sam_road':
            return self._forward_samroad(sat_img, batch_size)
        return self._forward_placeholder(sat_img, batch_size)

    def _forward_placeholder(self, sat_img, batch_size: int):
        # sat_img expected: (1,3,H,W) or (B,3,H,W) float in [0,1]
        # Behavior depends on train_sat_encoder:
        # - False (default): Cache pooled features (detached) for efficiency
        # - True: No caching, allow gradient flow through stem
        x = sat_img
        if x.dim() != 4:
            raise ValueError(f"sat_img must be 4D tensor, got shape={tuple(x.shape)}")
        if x.size(0) != 1 and x.size(0) != batch_size:
            raise ValueError(f"sat_img batch dim must be 1 or batch_size, got {x.size(0)} vs {batch_size}")

        # Check cache (only when not training backbone)
        use_cache = not self.train_sat_encoder
        if use_cache and sat_img is self._last_sat_obj and self._last_placeholder_pooled is not None:
            pooled = self._last_placeholder_pooled
        else:
            feat = self.stem(x)  # (1 or B, 64, h, w)
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (self.sat_grid, self.sat_grid))  # (1/B, 64, G, G)
            # Cache pooled features only when not training backbone
            if use_cache and pooled.size(0) == 1:
                self._last_sat_obj = sat_img
                self._last_placeholder_pooled = pooled.detach()
                self._last_samroad_pooled = None  # clear other cache

        # Recompute projection to ensure fresh computation graph for backprop
        tokens = pooled.flatten(2).transpose(1, 2)  # (1/B, G*G, 64)
        tokens = self.proj(tokens)  # (1/B, G*G, embed)
        tokens = self._add_sat_positional_encoding(tokens)

        if tokens.size(0) == 1 and batch_size > 1:
            tokens = tokens.expand(batch_size, -1, -1)
        return tokens

    def _forward_samroad(self, sat_img, batch_size: int):
        # sat_img expected from loader: CPU uint8 (1,3,H,W) or GPU float.
        # Behavior depends on train_sat_encoder:
        # - False (default): Freeze backbone, cache pooled features, use no_grad
        # - True: Allow gradient flow for LoRA/fine-tuning, no caching
        device = next(self._samroad_feat_proj.parameters()).device
        self._ensure_samroad_loaded(device)

        x = sat_img
        if x.dim() != 4:
            raise ValueError(f"sat_img must be 4D tensor, got shape={tuple(x.shape)}")
        if x.size(0) != 1 and x.size(0) != batch_size:
            raise ValueError(f"sat_img batch dim must be 1 or batch_size, got {x.size(0)} vs {batch_size}")

        # Check cache (only when not training backbone)
        use_cache = not self.train_sat_encoder
        if use_cache and sat_img is self._last_sat_obj and self._last_samroad_pooled is not None:
            pooled = self._last_samroad_pooled
        else:
            # Use single shared image for the whole batch to avoid repeated SAM-Road inference.
            if x.size(0) != 1:
                x = x[:1]

            # to CPU uint8 for slicing/padding (cheaper than holding full image on GPU)
            if x.device.type != "cpu":
                x_cpu = (x.detach().clamp(0, 1) * 255.0).to(torch.uint8).cpu()
            else:
                x_cpu = x
                if x_cpu.dtype != torch.uint8:
                    # assume 0..1 float or 0..255 float (avoid uint8 overflow)
                    mx = float(x_cpu.max().item()) if x_cpu.numel() else 0.0
                    if mx <= 1.5:
                        x_cpu = (x_cpu.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
                    else:
                        x_cpu = x_cpu.float().clamp(0.0, 255.0).round().to(torch.uint8)

            _, _, H, W = x_cpu.shape
            patch = int(getattr(self._samroad_cfg, "PATCH_SIZE", 512))
            overlap = int(self.model_params.get("sam_road_overlap", 64))
            bs = int(self.model_params.get("sam_road_patch_batch", 4))
            stride = max(patch - overlap, 1)

            # feature map is downsampled by 16 in SAM ViT
            ds = 16
            Hf = int(math.ceil(H / ds))
            Wf = int(math.ceil(W / ds))
            pf = patch // ds

            feat_sum = torch.zeros((1, 256, Hf, Wf), device=device, dtype=torch.float32)
            w_sum = torch.zeros((1, 1, Hf, Wf), device=device, dtype=torch.float32)

            # hann window in feature space
            wy = torch.hann_window(pf, periodic=False, device=device, dtype=torch.float32)
            wx = torch.hann_window(pf, periodic=False, device=device, dtype=torch.float32)
            win = (wy[:, None] * wx[None, :]).clamp_min_(1e-6)  # (pf,pf)

            coords = []
            ys = list(range(0, max(H - patch, 0) + 1, stride)) or [0]
            xs = list(range(0, max(W - patch, 0) + 1, stride)) or [0]
            if ys[-1] != max(H - patch, 0):
                ys.append(max(H - patch, 0))
            if xs[-1] != max(W - patch, 0):
                xs.append(max(W - patch, 0))
            coords = [(x0, y0) for y0 in ys for x0 in xs]

            # run SAM-Road patch-wise
            for i in range(0, len(coords), bs):
                batch_coords = coords[i:i + bs]
                patches = []
                valid_hw = []
                for (x0, y0) in batch_coords:
                    x1 = min(x0 + patch, W)
                    y1 = min(y0 + patch, H)
                    ph = y1 - y0
                    pw = x1 - x0
                    crop = x_cpu[0, :, y0:y1, x0:x1]  # (3,ph,pw) uint8
                    if ph != patch or pw != patch:
                        pad = torch.zeros((3, patch, patch), dtype=torch.uint8)
                        pad[:, :ph, :pw] = crop
                        crop = pad
                    patches.append(crop)
                    valid_hw.append((ph, pw))

                # [B, H, W, C] float32 0..255
                hwc = torch.stack(patches, dim=0).permute(0, 2, 3, 1).contiguous().to(device=device, dtype=torch.float32)
                
                # When training backbone (LoRA), allow gradients to flow
                if self.train_sat_encoder:
                    mask_scores, img_embeddings = self._samroad.infer_masks_and_img_features(hwc)
                else:
                    with torch.no_grad():
                        mask_scores, img_embeddings = self._samroad.infer_masks_and_img_features(hwc)

                for (x0, y0), (ph, pw), emb in zip(batch_coords, valid_hw, img_embeddings):
                    y0f = y0 // ds
                    x0f = x0 // ds
                    phf = int(math.ceil(ph / ds))
                    pwf = int(math.ceil(pw / ds))
                    emb = emb[:, :phf, :pwf]  # (256,phf,pwf)
                    w = win[:phf, :pwf]
                    feat_sum[0, :, y0f:y0f + phf, x0f:x0f + pwf] += emb * w
                    w_sum[0, 0, y0f:y0f + phf, x0f:x0f + pwf] += w

            feat = feat_sum / (w_sum + 1e-6)
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (self.sat_grid, self.sat_grid))  # (1,256,G,G)

            # Cache pooled features only when not training backbone
            if use_cache:
                self._last_sat_obj = sat_img
                self._last_samroad_pooled = pooled.detach()
                self._last_placeholder_pooled = None  # clear other cache

        # Recompute projection to ensure fresh computation graph for backprop
        tokens = pooled.flatten(2).transpose(1, 2)  # (1,G*G,256)
        tokens = self._samroad_feat_proj(tokens)  # (1,G*G,embed)
        tokens = self._add_sat_positional_encoding(tokens)  # keep (1, G*G, embed)

        if batch_size > 1:
            tokens = tokens.expand(batch_size, -1, -1)
        return tokens


class PatchEmbedding(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.img_size = self.model_params['img_size']
        self.patch_size = self.model_params['patch_size']
        self.in_channels = self.model_params['in_channels']
        self.embed_dim = self.model_params['embedding_dim']

        # patch num
        self.patches = self.img_size // self.patch_size

        self.proj = nn.Linear(self.patch_size * self.patch_size * self.in_channels, self.embed_dim)

        # positional embedding
        self.position_proj = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.in_channels, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * self.in_channels)

        # patch embedding
        embedded_patches = self.proj(patches)

        # add positional embedding
        grid_x, grid_y = torch.meshgrid(torch.arange(self.patches, device=x.device), 
                                        torch.arange(self.patches, device=x.device), indexing='ij')
        xy = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2).float()
        xy = xy / (self.patches - 1)
        embedded_patches += self.position_proj(xy)

        return embedded_patches

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

class EncoderFusionLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.Wq_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine_img = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1_img = Add_And_Normalization_Module(**model_params)
        self.feedForward_img = Feed_Forward_Module(**model_params)
        self.addAndNormalization2_img = Add_And_Normalization_Module(**model_params)

    def forward(self, input, input_img, fcp, fcp_img):
        input1 = torch.cat((input, fcp), dim=1)
        input1_img = torch.cat((input_img, fcp_img), dim=1)

        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(torch.cat((input1, fcp_img), dim=1)), head_num=head_num)
        v = reshape_by_heads(self.Wv(torch.cat((input1, fcp_img), dim=1)), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        q_img = reshape_by_heads(self.Wq_img(input1_img), head_num=head_num)
        k_img = reshape_by_heads(self.Wk_img(torch.cat((input1_img, fcp), dim=1)), head_num=head_num)
        v_img = reshape_by_heads(self.Wv_img(torch.cat((input1_img, fcp), dim=1)), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat_img = multi_head_attention(q_img, k_img, v_img)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out_img = self.multi_head_combine_img(out_concat_img)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1_img = self.addAndNormalization1_img(input1_img, multi_head_out_img)
        out2_img = self.feedForward_img(out1_img)
        out3_img = self.addAndNormalization2_img(out1_img, out2_img)

        return out3[:, :-self.model_params['bn_num']], out3_img[:, :-self.model_params['bn_img_num']], out3[:, -self.model_params['bn_num']:], out3_img[:, -self.model_params['bn_img_num']:]        # shape: (batch, problem, EMBEDDING_DIM)

########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        hyper_input_dim = 2
        hyper_hidden_embd_dim = self.model_params['hyper_hidden_dim']
        self.embd_dim = 2
        self.hyper_output_dim = 5 * self.embd_dim
        
        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)
        
        self.hyper_Wq_first = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, head_num * qkv_dim * embedding_dim, bias=False)

        self.Wq_last_para = None
        self.multi_head_combine_para = None

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention
        
    def assign(self, pref):
        
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        
        self.Wq_first_para = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[self.embd_dim:2 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wk_para = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wv_para = self.hyper_Wv(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.multi_head_combine_para = self.hyper_multi_head_combine(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)
        
        
    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        if self.model_params.get('encoder_mode', 'graph_image_fusion') == 'graph_only':
            num_img_tokens = 0
        else:
            num_xy_patches = (self.model_params['img_size'] // self.model_params['patch_size']) * (
                        self.model_params['img_size'] // self.model_params['patch_size'])
            num_sat_tokens = 0
            if self.model_params.get('use_satellite', False):
                sat_grid = int(self.model_params.get('sat_grid', 14))
                num_sat_tokens = sat_grid * sat_grid
            num_img_tokens = num_xy_patches + num_sat_tokens

        node_size = encoded_nodes.shape[1] - num_img_tokens

        self.k = reshape_by_heads(F.linear(encoded_nodes, self.Wk_para), head_num=head_num)
        self.v = reshape_by_heads(F.linear(encoded_nodes, self.Wv_para), head_num=head_num)

        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes[:, :node_size].transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(F.linear(encoded_q1, self.Wq_first_para), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
    
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(F.linear(encoded_last_node, self.Wq_last_para), head_num = head_num)
      
        q = self.q_first + q_last 
        # shape: (batch, head_num, pomo, qkv_dim)

        if self.model_params.get('encoder_mode', 'graph_image_fusion') == 'graph_only':
            num_img_tokens = 0
        else:
            num_xy_patches = (self.model_params['img_size'] // self.model_params['patch_size']) * (
                    self.model_params['img_size'] // self.model_params['patch_size'])
            num_sat_tokens = 0
            if self.model_params.get('use_satellite', False):
                sat_grid = int(self.model_params.get('sat_grid', 14))
                num_sat_tokens = sat_grid * sat_grid
            num_img_tokens = num_xy_patches + num_sat_tokens

        if num_img_tokens > 0:
            mha_mask = torch.cat(
                (ninf_mask, torch.zeros(ninf_mask.shape[0], ninf_mask.shape[1], num_img_tokens,
                                        device=ninf_mask.device, dtype=ninf_mask.dtype)), dim=-1)
        else:
            mha_mask = ninf_mask
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=mha_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = F.linear(out_concat, self.multi_head_combine_para)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))