import torch
import torch.nn as nn
from logging import getLogger
from collections import OrderedDict
import time
import math

from MOTSPEnv import TSPEnv as Env
from MOTSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.utils import *

import numpy as np
import os
import sys


class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        
        # Initialize dataset loader if using custom dataset
        self.dataset_loader = None
        
        if trainer_params.get('use_custom_dataset', False):
            from MOTSProblemDef import MultiDatasetLoaderWithSatellite
            datasets = env_params.get('datasets', [])
            switch_interval = env_params.get('switch_interval', 50)
            load_satellite = env_params.get('load_satellite', False)
            sat_cache_size = env_params.get('sat_cache_size', 10)
            
            self.dataset_loader = MultiDatasetLoaderWithSatellite(
                datasets=datasets,
                switch_interval=switch_interval,
                load_satellite=load_satellite,
                sat_cache_size=sat_cache_size
            )
            self.logger.info(f"Dataset loader initialized with {len(datasets)} datasets, {len(self.dataset_loader)} total instances")

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        
        self.env = Env(**self.env_params)

        # --- End-to-End Eikonal: load SAMRoute and build joint optimizer ---
        self.encoder_mode = self.model_params.get('encoder_mode', 'graph_image_fusion')
        self.samroute_model = None
        self._road_prob_cache = OrderedDict()
        self._road_prob_cache_size = 10
        self._disk_cache_logged = False
        self._tif_path_map = {}
        self.e2e_grad_clip_norm = float(self.model_params.get('e2e_grad_clip_norm', 1.0))
        self._e2e_step_count = 0
        self._e2e_parallel = bool(self.model_params.get('e2e_parallel_cases', True))
        self._e2e_use_compile = bool(self.model_params.get('e2e_use_compile', False))
        self._e2e_grad_interval = int(self.model_params.get('e2e_grad_interval', 1))
        self._e2e_train_decoder = bool(self.model_params.get('e2e_train_decoder', False))
        self._e2e_decoder_lr = float(self.model_params.get('e2e_decoder_lr', 1e-4))
        self._encoder_cache_available = False
        self._encoder_feat_cache = OrderedDict()
        self._encoder_feat_cache_size = 20

        if self.encoder_mode == 'e2e_eikonal':
            self._build_tif_path_map()
            has_encoder_cache = self._check_encoder_cache_coverage()
            has_disk_cache = self._check_disk_cache_coverage()

            if self._e2e_train_decoder and has_encoder_cache:
                self._encoder_cache_available = True
                self.logger.info("[E2E] Encoder embedding cache found — "
                                 "decoder will be trainable (~177K params)")
                self.samroute_model = self._load_samroute(
                    device, skip_encoder=True, keep_decoder=True)
            elif has_disk_cache:
                self.logger.info("[E2E] Disk road_prob cache found for all datasets — "
                                 "loading SAMRoute without image_encoder.")
                self.samroute_model = self._load_samroute(device, skip_encoder=True)
            else:
                self.samroute_model = self._load_samroute(device, skip_encoder=False)

            samroute_cost_params = []
            samroute_decoder_params = []
            for name, p in self.samroute_model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith('map_decoder'):
                    samroute_decoder_params.append(p)
                else:
                    samroute_cost_params.append(p)

            samroute_lr = float(self.model_params.get('e2e_samroute_lr', 1e-3))
            opt_groups = [
                {'params': self.model.parameters()},
                {'params': samroute_cost_params, 'lr': samroute_lr},
            ]
            if samroute_decoder_params:
                opt_groups.append({'params': samroute_decoder_params, 'lr': self._e2e_decoder_lr})
            self.optimizer = Optimizer(opt_groups, **self.optimizer_params['optimizer'])

            n_cost = sum(p.numel() for p in samroute_cost_params)
            n_dec = sum(p.numel() for p in samroute_decoder_params)
            self.logger.info(f"[E2E] SAMRoute trainable: cost_params={n_cost}, "
                             f"decoder_params={n_dec}, total={n_cost + n_dec}")
            self.logger.info(f"[E2E] SAMRoute cost_lr={samroute_lr}, decoder_lr={self._e2e_decoder_lr}, "
                             f"grad_clip_norm={self.e2e_grad_clip_norm}")
            self.logger.info(f"[E2E] parallel_cases={self._e2e_parallel}, use_compile={self._e2e_use_compile}")
            self.logger.info(f"[E2E] grad_interval={self._e2e_grad_interval}, "
                             f"train_decoder={self._e2e_train_decoder}")
        else:
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])

        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            warmstart_only = model_load.get('warmstart_only', False)
            if warmstart_only:
                self.start_epoch = 1
                self.logger.info('Warmstart: loaded model weights only, optimizer/scheduler reset.')
            else:
                self.start_epoch = 1 + model_load['epoch']
                self.result_log.set_raw_data(checkpoint['result_log'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.last_epoch = model_load['epoch'] - 1
                self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        
        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.result_folder, 'tensorboard'))
        # Log encoder_mode for run identification (e2e vs graph_only etc.)
        _tb_mode = self.encoder_mode
        if self.encoder_mode == 'e2e_eikonal' and self._e2e_train_decoder:
            _tb_mode = 'e2e_decoder'
        self.tb_writer.add_text('config/encoder_mode', _tb_mode, 0)
        self.tb_writer.add_hparams(
            {'encoder_mode': _tb_mode},
            {'config/placeholder': 0.0},
            run_name=None
        )

        # Validation configuration
        self.validation_interval = trainer_params.get('validation_interval', 10)  # Validate every N epochs
        self.validation_batch_size = trainer_params.get('validation_batch_size', 64)
        
        # Early stopping and best model saving
        self.early_stop_patience = trainer_params.get('early_stop_patience', 0)  # 0 to disable
        self.save_best_model = trainer_params.get('save_best_model', True)
        self.best_val_gap = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
        # Initialize validation dataset loader
        self.val_loader = None
        validation_datasets = trainer_params.get('validation_datasets', None)
        
        if validation_datasets and len(validation_datasets) > 0:
            from MOTSProblemDef import ValidationDatasetLoader
            load_satellite = env_params.get('load_satellite', False)
            sat_cache_size = env_params.get('sat_cache_size', 10)
            
            try:
                self.val_loader = ValidationDatasetLoader(
                    datasets=validation_datasets,
                    load_satellite=load_satellite,
                    sat_cache_size=sat_cache_size
                )
                if not self.val_loader.dataset_names:
                    self.logger.warning('No optimal solution files found. Validation disabled.')
                    self.val_loader = None
                else:
                    self.logger.info(f'Validation enabled (every {self.validation_interval} epochs)')
                    self.logger.info(f'  Using {len(validation_datasets)} test datasets for validation:')
                    for ds in validation_datasets:
                        self.logger.info(f'    - {ds["name"]}')
            except Exception as e:
                self.logger.warning(f'Failed to initialize ValidationDatasetLoader: {e}')
                self.val_loader = None

        if self.encoder_mode == 'e2e_eikonal' and validation_datasets:
            for ds in (validation_datasets or []):
                name = ds.get('name', '')
                tif_path = ds.get('satellite_tif_path', None)
                if name and tif_path:
                    self._tif_path_map[f"val_{name}"] = tif_path
                    if name not in self._tif_path_map:
                        self._tif_path_map[name] = tif_path

    # ------------------------------------------------------------------
    #  End-to-End Eikonal helpers
    # ------------------------------------------------------------------

    def _build_tif_path_map(self):
        """Build dataset_name -> satellite_tif_path map from train loaders."""
        if self.dataset_loader is not None:
            for info in self.dataset_loader.get_dataset_info():
                name = info.get('name', '')
                tif_path = info.get('satellite_tif_path', None)
                if name and tif_path:
                    self._tif_path_map[name] = tif_path
                    self._tif_path_map[f"val_{name}"] = tif_path

    def _check_disk_cache_coverage(self):
        """Check whether all datasets have pre-computed road_prob_cache.npz."""
        if not self._tif_path_map:
            return False
        for name, tif_path in self._tif_path_map.items():
            cache_path = os.path.join(os.path.dirname(tif_path), 'road_prob_cache.npz')
            if not os.path.exists(cache_path):
                return False
        return True

    def _check_encoder_cache_coverage(self):
        """Check whether all datasets have pre-computed encoder_cache.pt."""
        if not self._tif_path_map:
            return False
        for name, tif_path in self._tif_path_map.items():
            cache_path = os.path.join(os.path.dirname(tif_path), 'encoder_cache.pt')
            if not os.path.exists(cache_path):
                return False
        return True

    def _load_samroute(self, device, skip_encoder=False, keep_decoder=False):
        """Load SAMRoute checkpoint, freeze encoder, enable grad only for cost params.

        Args:
            skip_encoder: offload image_encoder (and optionally map_decoder) to CPU.
            keep_decoder: when True (and skip_encoder=True), keep map_decoder on GPU
                          so it can participate in gradient updates.
        """
        eikonal_dir = os.path.join(os.path.dirname(__file__), 'eikonal_solver')
        if eikonal_dir not in sys.path:
            sys.path.insert(0, eikonal_dir)
        from model_multigrid import SAMRoute

        ckpt_path = self.model_params['e2e_eikonal_ckpt']
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)

        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = raw.get('state_dict', raw)
        sd = {(k[len('model.'):] if k.startswith('model.') else k): v for k, v in sd.items()}

        pe = sd.get('image_encoder.pos_embed')
        patch_size = int(pe.shape[1]) * 16 if pe is not None else 512

        w7 = sd.get('map_decoder.7.weight')
        use_smooth = w7 is not None and w7.shape[1] == 32

        from types import SimpleNamespace
        cfg = SimpleNamespace(
            SAM_VERSION='vit_b',
            PATCH_SIZE=patch_size,
            NO_SAM=False,
            USE_SAM_DECODER=False,
            USE_SMOOTH_DECODER=use_smooth,
            ENCODER_LORA=False,
            LORA_RANK=4,
            FREEZE_ENCODER=True,
            FOCAL_LOSS=False,
            TOPONET_VERSION='default',
            SAM_CKPT_PATH=os.path.join(
                os.path.dirname(__file__), 'sam_road_repo', 'sam_ckpts', 'sam_vit_b_01ec64.pth'),
            ROUTE_COST_MODE='add',
            ROUTE_ADD_ALPHA=20.0,
            ROUTE_ADD_GAMMA=2.0,
            ROUTE_ADD_BLOCK_ALPHA=0.0,
            ROUTE_BLOCK_TH=0.0,
            ROUTE_ROI_MARGIN=64,
            ROUTE_COST_NET=bool(self.model_params.get('e2e_eikonal_train_cost_net', False)),
            ROUTE_COST_NET_CH=8,
            ROUTE_COST_NET_USE_COORD=False,
        )

        model = SAMRoute(cfg)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            self.logger.warning(f"[SAMRoute] missing keys: {missing}")
        if unexpected:
            self.logger.warning(f"[SAMRoute] unexpected keys: {unexpected}")

        model.requires_grad_(False)

        trainable_names = ['cost_log_alpha', 'cost_log_gamma', 'eik_gate_logit']
        if cfg.ROUTE_COST_NET:
            trainable_names.append('cost_net')
        if self._e2e_train_decoder and keep_decoder:
            trainable_names.append('map_decoder')
        for name, param in model.named_parameters():
            if any(name.startswith(tn) or name == tn for tn in trainable_names):
                param.requires_grad_(True)
                self.logger.info(f"[SAMRoute] trainable: {name}  shape={param.shape}")

        model.to(device)
        model.eval()

        if skip_encoder and hasattr(model, 'image_encoder'):
            model.image_encoder.cpu()
            if hasattr(model, 'map_decoder') and not keep_decoder:
                model.map_decoder.cpu()
            torch.cuda.empty_cache()
            if keep_decoder:
                self.logger.info("[SAMRoute] Offloaded image_encoder to CPU, "
                                 "map_decoder kept on GPU (trainable decoder mode)")
            else:
                self.logger.info("[SAMRoute] Offloaded image_encoder + map_decoder to CPU "
                                 "(disk road_prob cache available)")

        return model

    def _compute_road_prob_cached(self, sat_img, cache_key, device):
        """
        Get road_prob, prioritizing: memory LRU cache > disk cache > online inference.

        Args:
            sat_img: [1, 3, H, W] uint8 tensor (CPU) from dataset loader.
            cache_key: string key for LRU cache (e.g. dataset_name).
            device: torch device.

        Returns:
            road_prob: [1, H, W] float32 tensor on device, detached.
        """
        if cache_key in self._road_prob_cache:
            self._road_prob_cache.move_to_end(cache_key)
            return self._road_prob_cache[cache_key]

        # Try disk cache (road_prob_cache.npz in same dir as the TIF)
        tif_path = self._tif_path_map.get(cache_key, None)
        if tif_path is not None:
            disk_cache = os.path.join(os.path.dirname(tif_path), 'road_prob_cache.npz')
            if os.path.exists(disk_cache):
                rp_data = np.load(disk_cache, allow_pickle=True)
                road_prob_np = rp_data['road_prob'].astype(np.float32)
                if not self._disk_cache_logged:
                    ckpt_name = str(rp_data['ckpt_name']) if 'ckpt_name' in rp_data else 'unknown'
                    ts = str(rp_data['timestamp']) if 'timestamp' in rp_data else 'unknown'
                    self.logger.info(f"[E2E] Disk road_prob cache: model={ckpt_name}, created={ts}")
                    self._disk_cache_logged = True
                road_prob = torch.from_numpy(road_prob_np).to(
                    device, torch.float32,
                ).unsqueeze(0).clamp(1e-6, 1.0 - 1e-6)
                self._road_prob_cache[cache_key] = road_prob
                if len(self._road_prob_cache) > self._road_prob_cache_size:
                    self._road_prob_cache.popitem(last=False)
                return road_prob

        # Fallback: online inference
        eikonal_dir = os.path.join(os.path.dirname(__file__), 'eikonal_solver')
        if eikonal_dir not in sys.path:
            sys.path.insert(0, eikonal_dir)
        from gradcheck_route_loss_v2_multigrid_fullmap import sliding_window_inference as swi

        img_np = sat_img.squeeze(0).permute(1, 2, 0).numpy()
        patch_size = getattr(self.samroute_model, 'patch_size', 512)

        with torch.no_grad():
            road_prob_np = swi(
                img_np, self.samroute_model, device,
                patch_size=patch_size, verbose=False,
            ).astype(np.float32)

        road_prob = torch.from_numpy(road_prob_np).to(
            device, torch.float32,
        ).unsqueeze(0).clamp(1e-6, 1.0 - 1e-6)

        self._road_prob_cache[cache_key] = road_prob
        if len(self._road_prob_cache) > self._road_prob_cache_size:
            self._road_prob_cache.popitem(last=False)

        return road_prob

    def _load_encoder_features(self, cache_key, device):
        """Load encoder features from LRU memory cache or disk encoder_cache.pt.

        Returns:
            (patches_info, features, win2d, H, W) or None if no cache found.
            features: [N, 256, 32, 32] float32 tensor on CPU (stacked).
        """
        if cache_key in self._encoder_feat_cache:
            self._encoder_feat_cache.move_to_end(cache_key)
            return self._encoder_feat_cache[cache_key]

        tif_path = self._tif_path_map.get(cache_key, None)
        if tif_path is None:
            return None

        cache_path = os.path.join(os.path.dirname(tif_path), 'encoder_cache.pt')
        if not os.path.exists(cache_path):
            return None

        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        if not self._disk_cache_logged:
            ckpt_name = data.get('ckpt_name', 'unknown')
            ts = data.get('timestamp', 'unknown')
            self.logger.info(f"[E2E] Encoder cache: model={ckpt_name}, created={ts}")
            self._disk_cache_logged = True

        features = data['features'].float()  # [N, 256, 32, 32]
        patches_info = data['patches_info']
        win2d = data['win2d']
        H, W = data['H'], data['W']

        entry = (patches_info, features, win2d, H, W)
        self._encoder_feat_cache[cache_key] = entry
        if len(self._encoder_feat_cache) > self._encoder_feat_cache_size:
            self._encoder_feat_cache.popitem(last=False)

        return entry

    _DECODER_CHUNK_SIZE = 32

    def _compute_road_prob_differentiable(self, cache_key, device):
        """Compute road_prob with gradients flowing through map_decoder.

        Uses chunked batch forward through map_decoder for efficiency:
        one CPU→GPU transfer of the full feature tensor, then N/chunk_size
        decoder calls instead of N individual calls.

        Returns:
            road_prob: [1, H, W] float32 tensor on device, with grad through decoder.
        """
        entry = self._load_encoder_features(cache_key, device)
        if entry is None:
            raise RuntimeError(
                f"Encoder cache not found for '{cache_key}'. "
                f"Run precompute_encoder_cache.py first.")

        patches_info, features, win2d, H, W = entry

        all_feat = features.to(device)  # [N, 256, 32, 32] single transfer
        chunk = self._DECODER_CHUNK_SIZE
        all_probs = []
        for i in range(0, all_feat.shape[0], chunk):
            logits = self.samroute_model.map_decoder(all_feat[i:i + chunk])
            all_probs.append(torch.sigmoid(logits[:, 1, :, :]))
        all_probs = torch.cat(all_probs, dim=0)  # [N, patch_H, patch_W]

        win_t = torch.from_numpy(win2d).to(device)
        prob_sum = torch.zeros(H, W, device=device)
        weight_sum = torch.zeros(H, W, device=device)

        for idx, (y0, x0, ph, pw) in enumerate(patches_info):
            prob_sum[y0:y0 + ph, x0:x0 + pw] += (
                all_probs[idx, :ph, :pw] * win_t[:ph, :pw])
            weight_sum[y0:y0 + ph, x0:x0 + pw] += win_t[:ph, :pw]

        road_prob = (prob_sum / weight_sum.clamp(min=1e-6)).unsqueeze(0)
        return road_prob.clamp(1e-6, 1.0 - 1e-6)

    def _compute_distance_matrix_e2e(self, road_prob, node_coords_norm, device):
        """
        Compute differentiable distance matrix via SAMRoute Eikonal solver.

        Uses ``forward_distance_matrix_batch`` which precomputes cost maps once
        for all cases in the batch (avoiding redundant ``_road_prob_to_cost``).

        Args:
            road_prob: [1, H, W] detached road probability tensor on device.
            node_coords_norm: [B, N, 2] normalized (x, y) bottom-left coords.
            device: torch device.

        Returns:
            distance_matrix: [B, N, N] with grad through cost_log_alpha/gamma.
        """
        B, N, _ = node_coords_norm.shape
        _, H, W = road_prob.shape

        ds = int(self.model_params.get('e2e_eikonal_ds', 16))
        eik_iters = int(self.model_params.get('e2e_eikonal_eik_iters', 40))
        pool_mode = self.model_params.get('e2e_eikonal_pool_mode', 'max')
        iter_floor_c = float(self.model_params.get('e2e_eikonal_iter_floor_c', 1.2))
        iter_floor_f = float(self.model_params.get('e2e_eikonal_iter_floor_f', 0.65))

        batch_nodes_yx = []
        for b in range(B):
            x_norm = node_coords_norm[b, :, 0]
            y_norm = node_coords_norm[b, :, 1]
            y_pix = torch.round((1.0 - y_norm) * (H - 1)).long()
            x_pix = torch.round(x_norm * (W - 1)).long()
            batch_nodes_yx.append(torch.stack([y_pix, x_pix], dim=-1))

        with torch.amp.autocast("cuda", enabled=False):
            D_batch = self.samroute_model.forward_distance_matrix_batch(
                batch_nodes_yx,
                road_prob=road_prob.float(),
                ds=ds,
                eik_iters=eik_iters,
                pool_mode=pool_mode,
                iter_floor_c=iter_floor_c,
                iter_floor_f=iter_floor_f,
                parallel_cases=self._e2e_parallel,
                use_compile=self._e2e_use_compile,
            )

        distance_matrix = D_batch.float() / float(H)
        return distance_matrix

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # Train
            train_score_obj1, train_score_obj2, train_loss = self._train_one_epoch(epoch)

            # LR Decay (must be called after optimizer.step())
            self.scheduler.step()
            self.result_log.append('train_score_obj1', epoch, train_score_obj1)
            self.result_log.append('train_score_obj2', epoch, train_score_obj2)
            self.result_log.append('train_loss', epoch, train_loss)
            
            # TensorBoard logging
            self.tb_writer.add_scalar('train/loss', train_loss, epoch)
            self.tb_writer.add_scalar('train/score_obj1', train_score_obj1, epoch)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            
            ############################
            # Validation
            ############################
            if self.val_loader is not None and (epoch % self.validation_interval == 0 or all_done):
                self.logger.info(f'Running validation at epoch {epoch}...')
                val_start_time = time.time()
                gap_results = self._validate_with_optimal()
                val_time = time.time() - val_start_time
                
                # Log to TensorBoard
                for dataset_name, metrics in gap_results.items():
                    safe_name = dataset_name.replace(' ', '_')
                    self.tb_writer.add_scalar(f'Validation/{safe_name}/Gap_Percent', metrics['gap_percent'], epoch)
                    self.tb_writer.add_scalar(f'Validation/{safe_name}/Model_Distance', metrics['model_distance'], epoch)
                    self.tb_writer.add_scalar(f'Validation/{safe_name}/Optimal_Distance', metrics['optimal_distance'], epoch)
                    
                    self.logger.info(f'  {dataset_name}: Gap={metrics["gap_percent"]:.2f}%, '
                                   f'Model={metrics["model_distance"]:.4f}, Optimal={metrics["optimal_distance"]:.4f}')
                
                # Log average gap across all datasets
                if gap_results:
                    avg_gap = np.mean([m['gap_percent'] for m in gap_results.values()])
                    self.tb_writer.add_scalar('Validation/Average_Gap_Percent', avg_gap, epoch)
                    self.logger.info(f'  Average Gap: {avg_gap:.2f}%')
                    self.tb_writer.add_scalar('Validation/Time_Sec', val_time, epoch)
                    
                    # Best model saving based on validation gap
                    if self.save_best_model and avg_gap < self.best_val_gap:
                        self.best_val_gap = avg_gap
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        
                        best_checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'result_log': self.result_log.get_raw_data(),
                            'best_val_gap': self.best_val_gap,
                        }
                        if self.samroute_model is not None:
                            best_checkpoint_dict['samroute_state_dict'] = self.samroute_model.state_dict()
                        torch.save(best_checkpoint_dict, '{}/checkpoint_motsp_best.pt'.format(self.result_folder))
                        self.logger.info(f'  *** New best model saved! Gap: {avg_gap:.2f}% (epoch {epoch}) ***')
                        self.tb_writer.add_scalar('Validation/Best_Gap_Percent', avg_gap, epoch)
                    else:
                        self.patience_counter += 1
                        if self.early_stop_patience > 0:
                            self.logger.info(f'  No improvement. Patience: {self.patience_counter}/{self.early_stop_patience} '
                                           f'(best: {self.best_val_gap:.2f}% at epoch {self.best_epoch})')
                        else:
                            self.logger.info(f'  No improvement. Best so far: {self.best_val_gap:.2f}% at epoch {self.best_epoch}')
                    
                    # Early stopping check (only if enabled)
                    if self.early_stop_patience > 0 and self.patience_counter >= self.early_stop_patience:
                        self.logger.info(f'  *** Early stopping triggered! No improvement for {self.early_stop_patience} validations. ***')
                        self.logger.info(f'  Best model: epoch {self.best_epoch} with gap {self.best_val_gap:.2f}%')
                        self.tb_writer.add_scalar('Training/Early_Stopped_Epoch', epoch, epoch)
                        self.tb_writer.close()
                        self.logger.info("TensorBoard writer closed (early stop).")
                        return  # Exit training loop
            
            ############################
            # Model Checkpoint
            ############################
            model_save_interval = self.trainer_params['logging']['model_save_interval']
       
            if epoch == self.start_epoch or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                if self.samroute_model is not None:
                    checkpoint_dict['samroute_state_dict'] = self.samroute_model.state_dict()
                torch.save(checkpoint_dict, '{}/checkpoint_motsp-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                
                # Report best model info
                if self.save_best_model and self.best_epoch > 0:
                    self.logger.info(f" *** Best model: epoch {self.best_epoch} with validation gap {self.best_val_gap:.2f}% ***")
                    self.logger.info(f" *** Best model saved at: {self.result_folder}/checkpoint_motsp_best.pt ***")
                
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                self.tb_writer.close()

    def _train_one_epoch(self, epoch):

        score_AM_obj1 = AverageMeter()
        score_AM_obj2 = AverageMeter()
    
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score_obj1, avg_score_obj2, avg_loss = self._train_one_batch(batch_size)
            score_AM_obj1.update(avg_score_obj1, batch_size)
            score_AM_obj2.update(avg_score_obj2, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        return score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        step_t0 = time.time()
        self.model.train()
        device = next(self.model.parameters()).device

        is_full_grad_step = True
        if self.encoder_mode == 'e2e_eikonal' and self._e2e_grad_interval > 1:
            is_full_grad_step = (self._e2e_step_count % self._e2e_grad_interval == 0)

        if self.dataset_loader is not None:
            problems, dist_matrix, sat_img, dataset_name = self.dataset_loader.sample_batch(batch_size)
            problems = problems.to(device)
            dist_matrix = dist_matrix.to(device)

            if self.encoder_mode == 'e2e_eikonal':
                if sat_img is None:
                    raise RuntimeError("e2e_eikonal requires satellite images but got None")

                if self._encoder_cache_available:
                    if is_full_grad_step:
                        road_prob = self._compute_road_prob_differentiable(
                            dataset_name, device)
                    else:
                        with torch.no_grad():
                            road_prob = self._compute_road_prob_differentiable(
                                dataset_name, device)
                else:
                    road_prob = self._compute_road_prob_cached(
                        sat_img, dataset_name, device)

                node_coords_xy = problems
                if is_full_grad_step:
                    dist_matrix_e2e = self._compute_distance_matrix_e2e(road_prob, node_coords_xy, device)
                else:
                    with torch.no_grad():
                        dist_matrix_e2e = self._compute_distance_matrix_e2e(road_prob, node_coords_xy, device)
                self.env.load_problems(batch_size, problems=problems,
                                       distance_matrix=dist_matrix_e2e, sat_images=None)
            else:
                self.env.load_problems(batch_size, problems=problems,
                                       distance_matrix=dist_matrix, sat_images=sat_img)
        else:
            self.env.load_problems(batch_size)

        pref = torch.tensor([1.0, 0.0], device=device).float()

        reset_state, _, _ = self.env.reset()
        
        self.model.decoder.assign(pref)
        self.model.pre_forward(reset_state)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0), device=device)
      
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            
        # Loss
        ###############################################
        reward = -reward
        if reward.size(-1) == 1:
            reward = torch.cat([reward, torch.zeros_like(reward)], dim=2)
        if self.trainer_params['dec_method'] == "WS":
            tch_reward = (pref * reward).sum(dim=2)
        elif self.trainer_params['dec_method'] == "TCH":
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward, _ = tch_reward.max(dim=2)
        else:
            return NotImplementedError
        
        # set back reward to negative
        reward = -reward
        tch_reward = -tch_reward

        log_prob = prob_list.log().sum(dim=2)

        # shape = (batch, group)
    
        tch_advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)
    
        tch_loss = -tch_advantage * log_prob # Minus Sign
        # shape = (batch, group)
        loss_mean = tch_loss.mean()
        
        # Score
        ###############################################
        _ , max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0], 1)
        max_reward_obj1 = reward[:, :, 0].gather(1, max_idx)
        score_mean_obj1 = -max_reward_obj1.float().mean()
        score_mean_obj2 = score_mean_obj1
    
        # Step & Return
        ################################################
        self.model.zero_grad()
        if self.samroute_model is not None:
            if is_full_grad_step:
                self.samroute_model.zero_grad()
            else:
                for p in self.samroute_model.parameters():
                    p.grad = None

        loss_mean.backward()

        if self.encoder_mode == 'e2e_eikonal':
            if is_full_grad_step:
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    [p for p in self.samroute_model.parameters() if p.requires_grad],
                    max_norm=self.e2e_grad_clip_norm,
                )
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.e2e_grad_clip_norm)

            self._e2e_step_count += 1
            if self._e2e_step_count % 20 == 1:
                if is_full_grad_step:
                    self._log_samroute_grads()
                else:
                    self.logger.info("  [SAMRoute] fast step — SAMRoute grads skipped")

        self.optimizer.step()

        step_time = time.time() - step_t0
        if self.encoder_mode == 'e2e_eikonal':
            tag = 'full' if is_full_grad_step else 'fast'
            self.tb_writer.add_scalar(f'e2e/step_wall_time_{tag}_sec', step_time, self._e2e_step_count)
            self.tb_writer.add_scalar('e2e/step_wall_time_sec', step_time, self._e2e_step_count)

        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item()

    def _log_samroute_grads(self):
        """Log SAMRoute trainable parameter values and gradient norms."""
        for name, p in self.samroute_model.named_parameters():
            if not p.requires_grad:
                continue
            grad_norm = p.grad.abs().mean().item() if p.grad is not None else 0.0
            val = p.item() if p.numel() == 1 else p.data.norm().item()
            self.logger.info(f"  [SAMRoute] {name}: value={val:.6f}, grad_norm={grad_norm:.8f}")
            self.tb_writer.add_scalar(f'e2e/grad_{name}', grad_norm, self._e2e_step_count)
            self.tb_writer.add_scalar(f'e2e/value_{name}', val, self._e2e_step_count)
            if p.grad is not None:
                assert grad_norm > 0, (
                    f"[E2E] {name}.grad is zero! Gradient flow is broken. "
                    f"Check that distance_matrix is not detached."
                )

    def _validate_with_optimal(self):
        """
        Validate model performance against optimal solutions.
        
        Returns:
            dict: {dataset_name: {'gap_percent': float, 'model_distance': float, 'optimal_distance': float}}
        """
        self.model.eval()
        results = {}
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for dataset_name in self.val_loader.dataset_names:
                num_samples = self.val_loader.get_num_samples(dataset_name)
                batch_size = min(self.validation_batch_size, num_samples)
                
                model_distances = []
                optimal_distances = []
                
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_indices = np.arange(start_idx, end_idx)
                    actual_batch_size = len(batch_indices)
                    
                    problems, dist_matrix, opt_dist, sat_img = \
                        self.val_loader.get_validation_batch(dataset_name, batch_indices)
                    
                    problems = problems.to(device)

                    if self.encoder_mode == 'e2e_eikonal' and sat_img is not None:
                        val_key = f"val_{dataset_name}"
                        if (self._encoder_cache_available
                                and self._load_encoder_features(val_key, device) is not None):
                            road_prob = self._compute_road_prob_differentiable(
                                val_key, device)
                        else:
                            road_prob = self._compute_road_prob_cached(
                                sat_img, val_key, device)
                        dist_matrix = self._compute_distance_matrix_e2e(
                            road_prob, problems, device)
                    else:
                        dist_matrix = dist_matrix.to(device)
                    
                    self.env.load_problems(
                        actual_batch_size, 
                        problems=problems, 
                        distance_matrix=dist_matrix,
                        sat_images=sat_img if self.encoder_mode != 'e2e_eikonal' else None,
                    )
                    
                    pref = torch.tensor([1.0, 0.0], device=device).float()
                    reset_state, _, _ = self.env.reset()
                    
                    self.model.decoder.assign(pref)
                    self.model.pre_forward(reset_state)
                    
                    state, reward, done = self.env.pre_step()
                    while not done:
                        selected, _ = self.model(state)
                        state, reward, done = self.env.step(selected)
                    
                    tour_distances = -reward[:, :, 0]
                    best_distances = tour_distances.min(dim=1).values
                    
                    model_distances.extend(best_distances.cpu().numpy().tolist())
                    optimal_distances.extend(opt_dist.tolist())
                
                model_distances = np.array(model_distances)
                optimal_distances = np.array(optimal_distances)
                
                gap_percent = np.mean((model_distances - optimal_distances) / optimal_distances * 100)
                
                results[dataset_name] = {
                    'gap_percent': gap_percent,
                    'model_distance': np.mean(model_distances),
                    'optimal_distance': np.mean(optimal_distances),
                }
        
        self.model.train()
        return results