import torch
from logging import getLogger
import time

from MOTSPEnv import TSPEnv as Env
from MOTSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.utils import *

import numpy as np
import os


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
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        
        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.result_folder, 'tensorboard'))
        
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
                        
                        # Save best model
                        best_checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'result_log': self.result_log.get_raw_data(),
                            'best_val_gap': self.best_val_gap,
                        }
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
        self.model.train()
        device = next(self.model.parameters()).device

        if self.dataset_loader is not None:
            # Use the dataset loader
            problems, dist_matrix, sat_img, dataset_name = self.dataset_loader.sample_batch(batch_size)
            problems = problems.to(device)
            dist_matrix = dist_matrix.to(device)
            # sat_img: (1, 3, H, W), uint8 on CPU - keep on CPU for SAM-Road patch-wise inference
            self.env.load_problems(batch_size, problems=problems, distance_matrix=dist_matrix, sat_images=sat_img)
        else:
            # No custom dataset, generate random problems
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
    
        #Step & Return
        ################################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        
        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item()

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
                
                # Process all samples in batches
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_indices = np.arange(start_idx, end_idx)
                    actual_batch_size = len(batch_indices)
                    
                    # Get validation batch
                    problems, dist_matrix, opt_dist, sat_img = \
                        self.val_loader.get_validation_batch(dataset_name, batch_indices)
                    
                    # Move to device
                    problems = problems.to(device)
                    dist_matrix = dist_matrix.to(device)
                    
                    # Load problems into environment
                    self.env.load_problems(
                        actual_batch_size, 
                        problems=problems, 
                        distance_matrix=dist_matrix,
                        sat_images=sat_img
                    )
                    
                    # Generate preference vector (single objective)
                    pref = torch.tensor([1.0, 0.0], device=device).float()
                    
                    # Reset environment
                    reset_state, _, _ = self.env.reset()
                    
                    # Model forward
                    self.model.decoder.assign(pref)
                    self.model.pre_forward(reset_state)
                    
                    # POMO Rollout (greedy)
                    state, reward, done = self.env.pre_step()
                    while not done:
                        selected, _ = self.model(state)
                        state, reward, done = self.env.step(selected)
                    
                    # reward shape: (batch, pomo, 1) - negative distance
                    # Get the best among POMO
                    tour_distances = -reward[:, :, 0]  # (batch, pomo)
                    best_distances = tour_distances.min(dim=1).values  # (batch,)
                    
                    model_distances.extend(best_distances.cpu().numpy().tolist())
                    optimal_distances.extend(opt_dist.tolist())
                
                # Calculate gap: (model - optimal) / optimal * 100%
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