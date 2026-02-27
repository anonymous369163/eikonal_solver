import torch
import torch.nn.functional as F
from torch import nn

# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy
import sys
import os
_MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
_SAM_REPO   = os.path.join(_MODEL_DIR, '../sam_road_repo')
for _sp in [
    os.path.join(_SAM_REPO, 'segment-anything-road'),
    os.path.join(_SAM_REPO, 'sam'),
    _SAM_REPO,
]:
    if _sp not in sys.path:
        sys.path.append(_sp)

from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve

import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT  # pyright: ignore[reportMissingImports]
from sam.segment_anything.modeling.mask_decoder import MaskDecoder  # pyright: ignore[reportMissingImports]
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder  # pyright: ignore[reportMissingImports]
from sam.segment_anything.modeling.transformer import TwoWayTransformer  # pyright: ignore[reportMissingImports]
from sam.segment_anything.modeling.common import LayerNorm2d  # pyright: ignore[reportMissingImports]

import wandb
import pprint
import torchvision

# Only needed for the ablation experiment of using a ViT-B model without SA-1B pre-training.
# It depends on detectron2 library. Not super important. 
# import vitdet


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points):
        """
        Args:
            feature_maps (Tensor): The input feature tensor of shape [B, D, H, W].
            sample_points (Tensor): The 2D sample points of shape [B, N_points, 2],
                                    each point in the range [-1, 1], format (x, y).
        Returns:
            Tensor: Sampled feature vectors of shape [B, N_points, D].
        """
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape

        # normalize cooridinates to (-1, 1) for grid_sample
        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        
        # sample_points from [B, N_points, 2] to [B, N_points, 1, 2] for grid_sample
        sample_points = sample_points.unsqueeze(2)
        
        # Use grid_sample for bilinear sampling. Align_corners set to False to use -1 to 1 grid space.
        # [B, D, N_points, 1]
        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        
        # sampled_features is [B, N_points, D]
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)
        return sampled_features
    

class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim)

        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True  # Input format is [batch size, sequence length, features]
        )
        
        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, pairs, pairs_valid):
        # points: [B, N_points, 2]
        # point_features: [B, N_points, D]
        # pairs: [B, N_samples, N_pairs, 2]
        # pairs_valid: [B, N_samples, N_pairs]
        
        point_features = F.relu(self.feature_proj(point_features))
        # gathers pairs
        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)
        
        batch_indices = torch.arange(batch_size, device=point_features.device).view(-1, 1).expand(-1, n_samples * n_pairs)
        # Use advanced indexing to fetch the corresponding feature vectors
        # [B, N_samples * N_pairs, D]
        src_features = point_features[batch_indices, pairs[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        # [B, N_samples * N_pairs, 2]
        src_points = points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]
        offset = tgt_points - src_points

        ## ablation study
        # [B, N_samples * N_pairs, 2D + 2]
        if self.config.TOPONET_VERSION == 'no_tgt_features':
            pair_features = torch.concat([src_features, torch.zeros_like(tgt_features), offset], dim=2)
        elif self.config.TOPONET_VERSION == 'no_offset':
            pair_features = torch.concat([src_features, tgt_features, torch.zeros_like(offset)], dim=2)
        else:
            pair_features = torch.concat([src_features, tgt_features, offset], dim=2)
        
        
        # [B, N_samples * N_pairs, D]
        pair_features = F.relu(self.pair_proj(pair_features))
        
        # attn applies within each local graph sample
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        # valid->not a padding
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        # [B * N_samples, 1]
        #### flips mask for all-invalid pairs to prevent NaN
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)

        padding_mask = ~pairs_valid
        
        ## ablation study
        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)
        
        ## Seems like at inference time, the returned n_pairs heres might be less - it's the
        # max num of valid pairs across all samples in the batch
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        # [B, N_samples, N_pairs, 1]
        logits = self.output_proj(pair_features)

        scores = torch.sigmoid(logits)

        return logits, scores



class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        # self.qkv = qkv
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        # qkv = self.qkv(x)  # B,N,N,3*org_C
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv



class SAMRoad(pl.LightningModule):
    """This is the RelationFormer module that performs object detection"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}
        if config.SAM_VERSION == 'vit_b':
            ### SAM config (B)
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]
            ###
        elif config.SAM_VERSION == 'vit_l':
            ### SAM config (L)
            encoder_embed_dim=1024
            encoder_depth=24
            encoder_num_heads=16
            encoder_global_attn_indexes=[5, 11, 17, 23]
            ###
        elif config.SAM_VERSION == 'vit_h':
            ### SAM config (H)
            encoder_embed_dim=1280
            encoder_depth=32
            encoder_num_heads=16
            encoder_global_attn_indexes=[7, 15, 23, 31]
            ###
            
        prompt_embed_dim = 256
        # SAM default is 1024
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        encoder_output_dim = prompt_embed_dim

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        if self.config.NO_SAM:
            # Only needed for the ablation experiment of using a ViT-B model without SA-1B pre-training.
            # It depends on detectron2 library. Not super important. 
            ### im1k + mae pre-trained vitb
            # self.image_encoder = vitdet.VITBEncoder(image_size=image_size, output_feature_dim=prompt_embed_dim)
            # self.matched_param_names = self.image_encoder.matched_param_names
            raise NotImplementedError((
                "This ablation experiment depends on detectron2, "
                "which is a bit messy and is not super important, "
                "so not including in the release. "
                "If you are interested, feel free to uncomment."))
        else:
            ### SAM vitb
            self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim
            )

        if self.config.USE_SAM_DECODER:
            # SAM DECODER
            # Not used, just produce null embeddings
            self.prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            self.mask_decoder=MaskDecoder(
                num_multimask_outputs=2, # keypoint, road
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        else:
            activation = nn.GELU
            use_smooth = getattr(self.config, 'USE_SMOOTH_DECODER', False)
            if use_smooth:
                # Smooth decoder: identical first 3 upsampling stages, then
                # ConvTranspose2d(32→32) followed by a 3×3 Conv2d(32→2).
                # The final Conv2d aggregates a 3×3 neighbourhood at full
                # resolution, bridging the 16-px ViT token-grid boundaries
                # and eliminating the checkerboard pattern.
                self.map_decoder = nn.Sequential(
                    nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    activation(),
                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                    activation(),
                    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                    activation(),
                    nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
                    activation(),
                    nn.Conv2d(32, 2, kernel_size=3, padding=1),
                )
            else:
                # Standard decoder: 4× ConvTranspose2d, backward-compatible
                # with all existing checkpoints.
                self.map_decoder = nn.Sequential(
                    nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    activation(),
                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                    activation(),
                    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                    activation(),
                    nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
                )

        
        #### TOPONet
        self.bilinear_sampler = BilinearSampler(config)
        self.topo_net = TopoNet(config, encoder_output_dim)


        #### LORA
        if config.ENCODER_LORA:
            r = self.config.LORA_RANK
            lora_layer_selection = None
            assert r > 0
            if lora_layer_selection:
                self.lora_layer_selection = lora_layer_selection
            else:
                self.lora_layer_selection = list(
                    range(len(self.image_encoder.blocks)))  # Only apply lora to the image encoder by default
            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # lets freeze first
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            # Here, we do the surgery
            for t_layer_i, blk in enumerate(self.image_encoder.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer_selection:
                    continue
                w_qkv_linear = blk.attn.qkv
                dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            # Init LoRA params
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)

        #### Losses
        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        #### Metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        # testing only, not used in training
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        if self.config.NO_SAM:
            return
        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)

            ## Resize pos embeddings, if needed
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
                ckpt_state_dict = new_state_dict
            
            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)

    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict

    
    def forward(self, rgb, graph_points, pairs, valid):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)
        
        ## Predicts local topology
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid)
        
        
        # [B, H, W, 2]
        mask_logits = mask_logits.permute(0, 2, 3, 1)
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_logits, mask_scores, topo_logits, topo_scores
    
    def infer_masks_and_img_features(self, rgb):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)
        
        # [B, H, W, 2]
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_scores, image_embeddings
    

    def infer_toponet(self, image_embeddings, graph_points, pairs, valid):
        # image_embeddings: [B, D, h, w]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        ## Predicts local topology
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid)
        return topo_scores


    def training_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # [B, H, W, 2]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))

        topo_loss *= topo_loss_mask.unsqueeze(-1)
        # topo_loss = torch.nansum(torch.nansum(topo_loss) / topo_loss_mask.sum())
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()

        loss = mask_loss + topo_loss
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)


        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()
        loss = mask_loss + topo_loss
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log images
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]
            viz_pred_keypoint = mask_scores[:max_viz_num, :, :, 0]
            viz_pred_road = mask_scores[:max_viz_num, :, :, 1]
            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]
            
            columns = ['rgb', 'gt_keypoint', 'gt_road', 'pred_keypoint', 'pred_road']
            data = [[wandb.Image(x.cpu().numpy()) for x in row] for row in list(zip(viz_rgb, viz_gt_keypoint, viz_gt_road, viz_pred_keypoint, viz_pred_road))]
            self.logger.log_table(key='viz_table', columns=columns, data=data)

        self.keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.road_iou.update(mask_scores[..., 1], road_mask)
        
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))
        

    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        road_iou = self.road_iou.compute()
        topo_f1 = self.topo_f1.compute()
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)
        self.log("topo_f1", topo_f1)
        self.keypoint_iou.reset()
        self.road_iou.reset()
        self.topo_f1.reset()

    def test_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)

        self.keypoint_pr_curve.update(mask_scores[..., 0], keypoint_mask.to(torch.int32))
        self.road_pr_curve.update(mask_scores[..., 1], road_mask.to(torch.int32))
        
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_pr_curve.update(topo_scores, topo_gt.unsqueeze(-1).to(torch.int32))

    def on_test_end(self):
        def find_best_threshold(pr_curve_metric, category):
            print(f'======= {category} ======')   
            precision, recall, thresholds = pr_curve_metric.compute()
            f1_scores = 2 * (precision * recall) / (precision + recall)
            best_threshold_index = torch.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_index]
            best_precision = precision[best_threshold_index]
            best_recall = recall[best_threshold_index]
            best_f1 = f1_scores[best_threshold_index]
            print(f'Best threshold {best_threshold}, P={best_precision} R={best_recall} F1={best_f1}')
        
        print('======= Finding best thresholds ======')
        find_best_threshold(self.keypoint_pr_curve, 'keypoint')
        find_best_threshold(self.road_pr_curve, 'road')
        find_best_threshold(self.topo_pr_curve, 'topo')


    def configure_optimizers(self):
        param_dicts = []

        if not self.config.FREEZE_ENCODER and not self.config.ENCODER_LORA:
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'image_encoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
            }
            param_dicts.append(encoder_params)
        if self.config.ENCODER_LORA:
            # LoRA params only
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'qkv.linear_' in k],
                'lr': self.config.BASE_LR,
            }
            param_dicts.append(encoder_params)
        
        if self.config.USE_SAM_DECODER:
            matched_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if 'mask_decoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * 0.1
            }
            fresh_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if 'mask_decoder.'+k not in self.matched_param_names],
                'lr': self.config.BASE_LR
            }
            decoder_params = [matched_decoder_params, fresh_decoder_params]
        else:
            decoder_params = [{
                'params': [p for p in self.map_decoder.parameters()],
                'lr': self.config.BASE_LR
            }]
        param_dicts += decoder_params

        topo_net_params = [{
            'params': [p for p in self.topo_net.parameters()],
            'lr': self.config.BASE_LR
        }]
        param_dicts += topo_net_params
        
        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')

        # optimizer = torch.optim.AdamW(param_dicts, lr=self.config.BASE_LR, betas=(0.9, 0.999), weight_decay=0.1)
        optimizer = torch.optim.Adam(param_dicts, lr=self.config.BASE_LR)
        # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}



# ===================== SAMRoute: differentiable end-to-end Eikonal routing =====================
import math as _math
from torch.utils.checkpoint import checkpoint as _grad_checkpoint
from eikonal import (
    EikonalConfig, prob_to_cost, make_source_mask, eikonal_soft_sweeping,
    _local_update_fast, _monotone_update, softmin_algebraic,
)


def _prob_to_cost_smooth(
    prob: torch.Tensor,
    gamma: float = 3.0,
    offroad_penalty: float = 100.0,
    block_th: float = 0.05,
    block_smooth: float = 50.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Differentiable prob → cost conversion (exponential mode).

    Replaces the hard torch.where(prob < block_th, cost * 10, cost) in the original
    prob_to_cost with a smooth sigmoid amplification, eliminating the gradient
    discontinuity at the threshold boundary.

      cost(p) = exp(log(offroad_penalty) * (1-p)^gamma)
              * (1 + 9 * sigmoid(-block_smooth * (p - block_th)))

    The sigmoid term smoothly transitions from 10x amplification (p ≪ block_th)
    to 1x (p ≫ block_th), matching the hard version's intent while being C∞.

    Issue: T values are in "cost units", not "pixel units", making direct
    supervision against GT pixel path lengths difficult.
    Consider _prob_to_cost_additive for a length-preserving alternative.
    """
    prob = prob.clamp(eps, 1.0 - eps)
    q = (1.0 - prob).clamp_min(eps)
    k = _math.log(max(float(offroad_penalty), 1.0 + 1e-6))
    cost = torch.exp(k * q.pow(gamma))
    if block_th > 0.0:
        amp = 1.0 + 9.0 * torch.sigmoid(-float(block_smooth) * (prob - float(block_th)))
        cost = cost * amp
    return cost.clamp_min(eps)


def _prob_to_cost_additive(
    prob: torch.Tensor,
    alpha: float = 20.0,
    gamma: float = 2.0,
    block_th: float = 0.05,
    block_alpha: float = 50.0,
    block_smooth: float = 50.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Length-preserving differentiable prob → cost conversion (additive mode).

    Design principle:
      - On road  (p ≈ 1): cost ≈ 1  → T accumulates like pixel count
      - Off road (p ≈ 0): cost ≈ 1 + alpha  → T ≫ GT, path avoids off-road

      cost(p) = 1 + alpha * (1-p)^gamma
              + block_alpha * sigmoid(-block_smooth * (p - block_th))

    Three terms:
      1. Constant 1      — base "one pixel = cost 1" so T ≈ geometric path length
      2. Penalty term    — smooth polynomial ramp, penalises off-road cells
      3. Hard-block term — additional steep penalty below block_th to prevent
                           paths from crossing near-zero road-probability cells

    Advantage over exponential:
      T(tgt) ≈ path_pixel_length + alpha * ∫(1-p) ds
      The first term dominates on well-segmented roads, so
        Huber(T / meta_height_px, gt_norm)
      is directly meaningful without any re-scaling.

    Args:
        alpha:        Off-road penalty scale.  alpha=20 → off-road cells are
                      ~21× more expensive than road cells.
        gamma:        Penalty sharpness.  gamma=1 is purely linear; gamma=2
                      concentrates the penalty on cells with p < 0.7.
        block_th:     Road-probability threshold below which an extra hard
                      block penalty is applied.
        block_alpha:  Magnitude of the hard-block penalty (default 50 adds
                      a steep cliff near p=block_th, effectively making
                      p < block_th cells ~71× costlier than road).
        block_smooth: Steepness of the sigmoid for the hard-block term.
        eps:          Floor to keep cost strictly positive.

    Typical cost values with defaults (alpha=20, gamma=2):
        p = 1.0  →  cost =  1.0
        p = 0.9  →  cost =  1.2
        p = 0.7  →  cost =  2.8
        p = 0.5  →  cost =  6.0
        p = 0.3  →  cost = 10.9
        p = 0.1  →  cost = 17.2  (+block≈50 if block_th=0.05)
        p = 0.0  →  cost = 21.0  (+block≈50)
    """
    # Clamp away from exact 0/1 to avoid log(0)=-inf in the backward of
    # q.pow(gamma) when gamma is a learnable tensor: d/d(gamma)[q^gamma] =
    # q^gamma * log(q), and 0*log(0) = 0*(-inf) = NaN in PyTorch.
    prob = prob.clamp(eps, 1.0 - eps)
    q = (1.0 - prob).clamp_min(eps)
    cost = 1.0 + alpha * q.pow(gamma)
    # smooth hard-block near block_th (sigmoid gives a C∞ step)
    if block_th > 0.0 and block_alpha > 0.0:
        block = float(block_alpha) * torch.sigmoid(
            -float(block_smooth) * (prob - float(block_th))
        )
        cost = cost + block
    return cost.clamp_min(eps)


def _mix_eikonal_euclid(
    T_eik: torch.Tensor,
    d_euc: torch.Tensor,
    gate_logit: torch.Tensor,
    gate_min: float = 0.3,
    gate_max: float = 0.95,
) -> torch.Tensor:
    """Blend Eikonal T-values with Euclidean distances via learnable gate.

    Uses residual form:  pred = d_euc + gate * (T_eik - d_euc)
      gate=1 → pure Eikonal,  gate=0 → pure Euclidean.
    Gate is clamped to [gate_min, gate_max] to prevent degeneration.
    """
    gate = torch.sigmoid(gate_logit).clamp(gate_min, gate_max)
    return d_euc + gate * (T_eik - d_euc)


def _eikonal_iter_chunk(
    T: torch.Tensor,
    cost: torch.Tensor,
    src_mask: torch.Tensor,
    even_mask: torch.Tensor,
    odd_mask: torch.Tensor,
    h_val: float,
    large_val: float,
    tau_min: float,
    tau_branch: float,
    tau_update: float,
    use_redblack: bool,
    monotone: bool,
    mode: str,
    n_chunk: int,
    gate_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Run n_chunk Eikonal sweep iterations with optional gated residual.

    When gate_alpha < 1.0, each update is blended with the previous value:
        T_updated = gate_alpha * T_new + (1 - gate_alpha) * T_old
    This GPPN-inspired residual connection improves gradient flow through
    the deep iterative computation graph (analogous to LSTM gating in GPPN
    replacing VIN's max-pooling).
    """
    use_gate = gate_alpha < 1.0
    for _ in range(n_chunk):
        if use_redblack:
            T_old = T
            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch, mode)
            T_e = _monotone_update(T, T_new, tau_update, mode, monotone)
            if use_gate:
                T_e = gate_alpha * T_e + (1.0 - gate_alpha) * T_old
            T = torch.where(even_mask, T_e, T)
            T = torch.where(src_mask, torch.zeros_like(T), T)

            T_old = T
            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch, mode)
            T_o = _monotone_update(T, T_new, tau_update, mode, monotone)
            if use_gate:
                T_o = gate_alpha * T_o + (1.0 - gate_alpha) * T_old
            T = torch.where(odd_mask, T_o, T)
            T = torch.where(src_mask, torch.zeros_like(T), T)
        else:
            T_old = T
            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch, mode)
            T = _monotone_update(T, T_new, tau_update, mode, monotone)
            if use_gate:
                T = gate_alpha * T + (1.0 - gate_alpha) * T_old
            T = torch.where(src_mask, torch.zeros_like(T), T)
    return T


def _eikonal_soft_sweeping_diff(
    cost: torch.Tensor,
    src_mask: torch.Tensor,
    cfg: "EikonalConfig",
    checkpoint_chunk: int = 10,
    gate_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable Eikonal soft sweeping with gradient checkpointing.

    The standard eikonal_soft_sweeping from fast_sweeping.py is also differentiable
    in principle, but running 100-200 iterations builds a computation graph of depth
    O(n_iters), which consumes O(B*H*W*n_iters) memory for backward.

    This version divides n_iters into segments of `checkpoint_chunk` iterations each
    and wraps every segment in torch.utils.checkpoint. This trades recomputation for
    memory, keeping peak memory at O(B*H*W) regardless of n_iters.

    Args:
        cost:             [B, H, W] or [H, W]
        src_mask:         [B, H, W] or [H, W]  bool
        cfg:              EikonalConfig
        checkpoint_chunk: recompute every N iterations during backward pass
        gate_alpha:       GPPN-style residual gate (1.0 = no gating, <1.0 = blend
                          with previous T for smoother gradient flow)
    """
    squeeze_back = cost.dim() == 2
    if squeeze_back:
        cost = cost.unsqueeze(0)
        if src_mask.dim() == 2:
            src_mask = src_mask.unsqueeze(0)
    elif src_mask.dim() == 2:
        src_mask = src_mask.unsqueeze(0)

    B, H, W = cost.shape
    device = cost.device

    T = torch.full((B, H, W), cfg.large_val, device=device, dtype=cost.dtype)
    T = torch.where(src_mask, torch.zeros_like(T), T)

    h_val      = float(cfg.h)
    large_val  = float(cfg.large_val)
    tau_min    = float(cfg.tau_min)
    tau_branch = float(cfg.tau_branch)
    tau_update = float(cfg.tau_update)
    use_redblack = bool(cfg.use_redblack)
    monotone     = bool(cfg.monotone)
    mode = getattr(cfg, 'mode', None)
    if mode is None:
        mode = "hard_eval" if getattr(cfg, 'monotone_hard', False) else "ste_train"
    ga           = float(gate_alpha)

    if use_redblack:
        yy = torch.arange(H, device=device)[:, None]
        xx = torch.arange(W, device=device)[None, :]
        even_mask = ((yy + xx) % 2 == 0)[None, :, :]   # [1, H, W]; broadcast over B
        odd_mask  = ~even_mask
    else:
        dummy = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
        even_mask = dummy
        odd_mask  = dummy

    n_iters = int(cfg.n_iters)
    chunk   = max(1, int(checkpoint_chunk))

    def _make_fn(nc):
        def fn(T_in, cost_in, src_m, even_m, odd_m):
            return _eikonal_iter_chunk(
                T_in, cost_in, src_m, even_m, odd_m,
                h_val, large_val, tau_min, tau_branch, tau_update,
                use_redblack, monotone, mode, nc, ga,
            )
        return fn

    for _ in range(n_iters // chunk):
        T = _grad_checkpoint(
            _make_fn(chunk), T, cost, src_mask, even_mask, odd_mask,
            use_reentrant=False,
        )

    remainder = n_iters % chunk
    if remainder > 0:
        T = _grad_checkpoint(
            _make_fn(remainder), T, cost, src_mask, even_mask, odd_mask,
            use_reentrant=False,
        )

    return T.squeeze(0) if squeeze_back else T


class SAMRoute(SAMRoad):
    """
    SAMRoute: end-to-end differentiable shortest road-network distance prediction.

    The entire forward pipeline is differentiable:
      RGB
        → SAMRoad encoder + decoder
        → road_prob [B, H, W]           (sigmoid probabilities, channel 1)
        → _prob_to_cost_smooth          (smooth, no hard threshold discontinuities)
        → cost [B, H, W]
        → _eikonal_soft_sweeping_diff   (gradient-checkpointed Eikonal solver)
        → T [B, H, W]                   (distance field from src)
        → T[tgt_y, tgt_x] = dist [B]   (predicted road distance)

    Gradients flow all the way back through the Eikonal iterations into the
    road segmentation decoder and encoder weights.

    For training, ROI-based solving is used by default: a small patch is cropped
    from road_prob centered at src (just large enough to cover tgt + margin) and
    Eikonal is solved on that patch only.  This is 10-100x more memory efficient
    than solving on the full image while preserving full gradient fidelity.

    Config attributes (all optional, with defaults):
        ROUTE_GAMMA:          float = 3.0    cost = exp(log(penalty)*(1-p)^gamma)
        ROUTE_OFFROAD_PENALTY float = 100.0  cost at p=0
        ROUTE_BLOCK_TH:       float = 0.05   smooth amplification center
        ROUTE_BLOCK_SMOOTH:   float = 50.0   sharpness of the smooth step
        ROUTE_EPS:            float = 1e-6   cost floor
        ROUTE_EIK_ITERS:      int   = 100    Eikonal iterations
        ROUTE_CKPT_CHUNK:     int   = 10     checkpoint every N iterations
        ROUTE_ROI_MARGIN:     int   = 64     ROI margin (px) beyond the src-tgt span
        ROUTE_LAMBDA_DIST:    float = 1.0    weight of distance regression loss
        ROUTE_LAMBDA_SEG:     float = 1.0    weight of road segmentation loss
    """

    @staticmethod
    def _cfg(config, key: str, default):
        """
        Safe config getter compatible with addict.Dict.
        addict.Dict returns an empty Dict (truthy = False) for missing keys
        instead of raising AttributeError, so plain getattr() defaults don't work.
        """
        val = getattr(config, key, None)
        if val is None or val == {} or (hasattr(val, '__len__') and len(val) == 0
                                        and not isinstance(val, (str, list, tuple))):
            return default
        return val

    def __init__(self, config):
        super().__init__(config)
        _cfg = self._cfg
        # cost_mode: "exp" (original exponential) or "add" (additive/length-preserving)
        self.route_cost_mode        = str(_cfg(config,   "ROUTE_COST_MODE",         "add"))
        # --- exponential-mode params ---
        self.route_gamma            = float(_cfg(config, "ROUTE_GAMMA",             3.0))
        self.route_offroad_penalty  = float(_cfg(config, "ROUTE_OFFROAD_PENALTY",   100.0))
        self.route_block_th         = float(_cfg(config, "ROUTE_BLOCK_TH",          0.05))
        self.route_block_smooth     = float(_cfg(config, "ROUTE_BLOCK_SMOOTH",      50.0))
        # --- additive-mode params (alpha/gamma are learnable nn.Parameters) ---
        _alpha_init = float(_cfg(config, "ROUTE_ADD_ALPHA", 20.0))
        _gamma_init = float(_cfg(config, "ROUTE_ADD_GAMMA", 2.0))
        self.cost_log_alpha = torch.nn.Parameter(
            torch.tensor(_math.log(max(_alpha_init, 1e-3))))
        self.cost_log_gamma = torch.nn.Parameter(
            torch.tensor(_math.log(max(_gamma_init, 1e-3))))
        self.route_add_block_alpha  = float(_cfg(config, "ROUTE_ADD_BLOCK_ALPHA",   50.0))
        self.route_add_block_smooth = float(_cfg(config, "ROUTE_ADD_BLOCK_SMOOTH",  50.0))
        self.route_eps              = float(_cfg(config, "ROUTE_EPS",               1e-6))
        self.route_eik_cfg = EikonalConfig(
            n_iters     = int(_cfg(config, "ROUTE_EIK_ITERS", 200)),
            tau_min     = 0.03,
            tau_branch  = 0.05,
            tau_update  = 0.03,
            use_redblack= True,
            monotone    = True,
            mode       = str(_cfg(config, "ROUTE_EIK_MODE", "soft_train")),
        )
        self.route_ckpt_chunk   = int(_cfg(config,   "ROUTE_CKPT_CHUNK",   10))
        self.route_roi_margin   = int(_cfg(config,   "ROUTE_ROI_MARGIN",   64))
        self.route_lambda_dist  = float(_cfg(config, "ROUTE_LAMBDA_DIST",  1.0))
        self.route_lambda_seg   = float(_cfg(config, "ROUTE_LAMBDA_SEG",   1.0))

        # --- GPPN-inspired stabilisation for Phase 2 ---
        self.route_dist_warmup_steps = int(_cfg(config,  "ROUTE_DIST_WARMUP_STEPS", 500))
        self.route_eik_warmup_epochs = int(_cfg(config,  "ROUTE_EIK_WARMUP_EPOCHS", 5))
        self.route_eik_iters_min     = int(_cfg(config,  "ROUTE_EIK_ITERS_MIN",     30))
        self.route_dist_norm_px      = float(_cfg(config,"ROUTE_DIST_NORM_PX",      512.0))
        self.route_gate_alpha        = float(_cfg(config,"ROUTE_GATE_ALPHA",        0.8))
        self.route_eik_downsample    = int(_cfg(config,  "ROUTE_EIK_DOWNSAMPLE",   4))
        self.route_k_targets         = int(_cfg(config,  "ROUTE_K_TARGETS",        4))
        self.route_cap_mode          = str(_cfg(config,  "ROUTE_CAP_MODE",        "tanh"))
        self.route_cap_mult          = float(_cfg(config,"ROUTE_CAP_MULT",        10.0))

        # Learnable Eikonal-Euclidean blending gate (residual form)
        _gate_init = float(_cfg(config, "ROUTE_EIK_GATE_INIT", 0.80))
        _gate_init = max(0.01, min(0.99, _gate_init))
        self.eik_gate_logit = torch.nn.Parameter(
            torch.tensor(_math.log(_gate_init / (1.0 - _gate_init))))

        # pos_weight compensates for class imbalance (road density ~6-7%).
        # With ~6.7% road pixels: pos_weight = 0.933/0.067 ≈ 13.9 balances the loss.
        # Without this, BCE drives the model to predict all-background, collapsing IoU.
        self._road_pos_weight  = float(_cfg(config, "ROAD_POS_WEIGHT", 13.9))
        # Dice loss weight: 0.0 = pure BCE, 0.5 = equal BCE+Dice, 1.0 = pure Dice.
        # Dice directly optimises overlap (IoU-like), preventing the BCE "confidence
        # calibration" plateau where loss keeps falling but binary IoU stays flat.
        self._road_dice_weight  = float(_cfg(config, "ROAD_DICE_WEIGHT", 0.5))
        # Dual-target: BCE against thick mask, Dice against original thin mask.
        # This gives the model wide-road recall (BCE) + sharp boundaries (Dice).
        self._road_dual_target  = bool(_cfg(config, "ROAD_DUAL_TARGET", False))
        # Extra BCE weight for thin-road pixels when dual_target. Higher = more emphasis on fine roads.
        self._road_thin_boost   = float(_cfg(config, "ROAD_THIN_BOOST", 5.0))
        self.road_criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight applied dynamically in _seg_forward
        self.dist_criterion = torch.nn.HuberLoss(delta=2.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_mask_logits_scores(
        self,
        rgb: torch.Tensor,
        encoder_feat: "torch.Tensor | None" = None,
    ):
        """
        Returns:
            mask_logits: [B, H, W, 2]  (pre-sigmoid)
            mask_scores: [B, H, W, 2]  (sigmoid)

        encoder_feat: optional pre-computed image embeddings [B, C, H_feat, W_feat].
            When supplied (and FREEZE_ENCODER=True), the SAM encoder is bypassed
            entirely, saving ~90 ms per step.
        """
        if encoder_feat is not None:
            image_embeddings = encoder_feat.to(self.pixel_mean.device, dtype=self.pixel_mean.dtype)
        else:
            x = rgb.permute(0, 3, 1, 2)
            x = (x - self.pixel_mean) / self.pixel_std
            freeze = getattr(self.config, "FREEZE_ENCODER", False)
            if freeze and not getattr(self.config, "ENCODER_LORA", False):
                with torch.no_grad():
                    image_embeddings = self.image_encoder(x)
            else:
                image_embeddings = self.image_encoder(x)

        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, _ = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear", align_corners=False,
            )
        else:
            mask_logits = self.map_decoder(image_embeddings)

        mask_scores = torch.sigmoid(mask_logits)
        return mask_logits.permute(0, 2, 3, 1), mask_scores.permute(0, 2, 3, 1)

    def _road_prob_to_cost(self, road_prob: torch.Tensor) -> torch.Tensor:
        """
        Differentiable prob → cost, dispatched by self.route_cost_mode.

        "add" (default):
            cost = 1 + alpha*(1-p)^gamma + block_alpha*sigmoid(...)
            T ≈ pixel_path_length + alpha * ∫(1-p)ds
            → T is in pixel units, directly comparable to GT path length.

        "exp" (original):
            cost = exp(log(offroad_penalty)*(1-p)^gamma) * sigmoid_amp
            → T is in cost units; steeper but harder to supervise with GT lengths.
        """
        if self.route_cost_mode == "exp":
            return _prob_to_cost_smooth(
                road_prob,
                gamma           = self.route_gamma,
                offroad_penalty = self.route_offroad_penalty,
                block_th        = self.route_block_th,
                block_smooth    = self.route_block_smooth,
                eps             = self.route_eps,
            )
        # default: "add" — alpha/gamma are learnable (log-space + clamp)
        alpha = self.cost_log_alpha.clamp(
            _math.log(5.0), _math.log(100.0)).exp()
        gamma = self.cost_log_gamma.clamp(
            _math.log(0.5), _math.log(4.0)).exp()
        return _prob_to_cost_additive(
            road_prob,
            alpha        = alpha,
            gamma        = gamma,
            block_th     = self.route_block_th,
            block_alpha  = self.route_add_block_alpha,
            block_smooth = self.route_add_block_smooth,
            eps          = self.route_eps,
        )

    def _apply_dist_cap(
        self,
        pred_dist: torch.Tensor,
        gt_dist: torch.Tensor,
    ) -> tuple:
        """Apply distance cap before loss computation.

        Returns (pred_normed, gt_normed, sat_mask).
        cap_mode controls the capping strategy:
          - "tanh":  smooth cap, always differentiable (recommended)
          - "clamp": hard cap, zero gradient in saturation zone
          - "none":  no capping
        """
        norm = self.route_dist_norm_px
        cap  = norm * self.route_cap_mult
        mode = self.route_cap_mode

        if mode == "tanh":
            pred_c = cap * torch.tanh(pred_dist / cap)
            sat = pred_dist > cap
        elif mode == "clamp":
            pred_c = pred_dist.clamp(max=cap)
            sat = pred_dist > cap
        else:
            pred_c = pred_dist
            sat = torch.zeros_like(pred_dist, dtype=torch.bool)

        return pred_c / norm, gt_dist / norm, sat

    def _roi_eikonal_solve(
        self,
        road_prob: torch.Tensor,   # [B, H, W]
        src_yx: torch.Tensor,      # [B, 2] long
        tgt_yx: torch.Tensor,      # [B, 2] long
        eik_cfg: "EikonalConfig | None" = None,
    ) -> torch.Tensor:
        """
        Fixed-ds ROI-cropped differentiable Eikonal solve.

        ds is fixed to ROUTE_EIK_DOWNSAMPLE (decoupled from n_iters).
        n_iters only controls solver convergence, not grid resolution.
        Require n_iters >= P/ds for full convergence; partial convergence
        is acceptable during early training (warmup).

        Returns:
            dist: [B]  predicted distance in original pixel units
        """
        B, H, W = road_prob.shape
        device = road_prob.device
        cfg    = eik_cfg if eik_cfg is not None else self.route_eik_cfg
        margin = self.route_roi_margin
        min_ds = max(1, int(getattr(self, 'route_eik_downsample', 4)))
        n_iters = int(cfg.n_iters)

        dists = []
        for b in range(B):
            prob_b = road_prob[b]   # [H, W]
            src    = src_yx[b].long()
            tgt    = tgt_yx[b].long()

            span = int(torch.max(torch.abs(tgt.float() - src.float())).item())
            half = span + margin
            P    = max(2 * half + 1, 64)

            y0 = int(src[0].item()) - half;  x0 = int(src[1].item()) - half
            y1 = y0 + P;                     x1 = x0 + P

            yy0 = max(y0, 0); xx0 = max(x0, 0)
            yy1 = min(y1, H); xx1 = min(x1, W)

            patch = F.pad(
                prob_b[yy0:yy1, xx0:xx1],
                (xx0 - x0, x1 - xx1, yy0 - y0, y1 - yy1),
                value=0.0,
            )  # [P, P]

            src_rel_y = max(0, min(int(src[0].item()) - y0, P - 1))
            src_rel_x = max(0, min(int(src[1].item()) - x0, P - 1))
            tgt_rel_y = max(0, min(int(tgt[0].item()) - y0, P - 1))
            tgt_rel_x = max(0, min(int(tgt[1].item()) - x0, P - 1))

            ds = min_ds

            if ds > 1:
                P_pad = int(_math.ceil(P / ds) * ds)
                if P_pad > P:
                    patch = F.pad(patch, (0, P_pad - P, 0, P_pad - P), value=0.0)

                patch_coarse = F.max_pool2d(
                    patch.unsqueeze(0).unsqueeze(0), kernel_size=ds, stride=ds
                ).squeeze(0).squeeze(0)  # [P_c, P_c]

                cost_coarse = self._road_prob_to_cost(patch_coarse).to(dtype=torch.float32)
                P_c = cost_coarse.shape[0]

                src_c_y = max(0, min(src_rel_y // ds, P_c - 1))
                src_c_x = max(0, min(src_rel_x // ds, P_c - 1))
                tgt_c_y = max(0, min(tgt_rel_y // ds, P_c - 1))
                tgt_c_x = max(0, min(tgt_rel_x // ds, P_c - 1))

                src_mask_b = torch.zeros(1, P_c, P_c, dtype=torch.bool, device=device)
                src_mask_b[0, src_c_y, src_c_x] = True

                from dataclasses import replace
                cfg_coarse = replace(cfg, h=float(ds))

                T_b = _eikonal_soft_sweeping_diff(
                    cost_coarse.unsqueeze(0),
                    src_mask_b,
                    cfg_coarse,
                    checkpoint_chunk=self.route_ckpt_chunk,
                    gate_alpha=self.route_gate_alpha,
                )  # [1, P_c, P_c]

                dists.append(T_b[0, tgt_c_y, tgt_c_x])
            else:
                cost_patch = self._road_prob_to_cost(patch).to(dtype=torch.float32)

                src_mask_b = torch.zeros(1, P, P, dtype=torch.bool, device=device)
                src_mask_b[0, src_rel_y, src_rel_x] = True

                T_b = _eikonal_soft_sweeping_diff(
                    cost_patch.unsqueeze(0),
                    src_mask_b,
                    cfg,
                    checkpoint_chunk=self.route_ckpt_chunk,
                    gate_alpha=self.route_gate_alpha,
                )  # [1, P, P]

                dists.append(T_b[0, tgt_rel_y, tgt_rel_x])

        T_eik = torch.stack(dists)  # [B]

        # Euclidean residual blending
        d_euc = ((src_yx.float() - tgt_yx.float()) ** 2).sum(-1).sqrt()  # [B]
        return _mix_eikonal_euclid(T_eik, d_euc, self.eik_gate_logit)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        rgb: torch.Tensor,
        src_yx: torch.Tensor,
        tgt_yx: torch.Tensor,
        *,
        use_roi: bool = True,
        return_road_prob: bool = False,
        return_distance_field: bool = False,
        detach_road_prob: bool = False,
        eik_cfg: "EikonalConfig | None" = None,
    ):
        """
        End-to-end differentiable forward: RGB → road distance (src → tgt).

        Args:
            rgb:                  [B, H, W, 3]  float 0..255
            src_yx:               [B, 2]         source pixel (y, x)
            tgt_yx:               [B, 2]         target pixel (y, x)
            use_roi:              True  = ROI-based solve (recommended for training,
                                          much more memory efficient).
                                  False = full-image solve (for inference/visualization).
            return_road_prob:     also return road_prob [B, H, W]
            return_distance_field: also return T [B, H, W]  (only if use_roi=False)
            detach_road_prob:     stop gradient through road_prob (debugging)
            eik_cfg:              override default EikonalConfig

        Returns:
            dist: [B]
            (optionally) road_prob: [B, H, W]
            (optionally) T:         [B, H, W]   (only when use_roi=False)
        """
        B, H, W, _ = rgb.shape
        device = rgb.device

        _, mask_scores = self._predict_mask_logits_scores(rgb)
        road_prob = mask_scores[..., 1]   # [B, H, W]

        if detach_road_prob:
            road_prob = road_prob.detach()

        if use_roi:
            dist = self._roi_eikonal_solve(road_prob, src_yx, tgt_yx, eik_cfg)
            if return_road_prob:
                return dist, road_prob
            return dist

        # Full-image solve — for inference / visualization only, expensive during training
        cost = self._road_prob_to_cost(road_prob).to(dtype=torch.float32)
        src_yx_l = src_yx.to(device).long().view(B, 1, 2)
        tgt_yx_l = tgt_yx.to(device).long().view(B, 2)
        src_mask = make_source_mask(H, W, src_yx_l)

        cfg = eik_cfg if eik_cfg is not None else self.route_eik_cfg
        T   = _eikonal_soft_sweeping_diff(cost, src_mask, cfg, self.route_ckpt_chunk)

        yy   = tgt_yx_l[:, 0].clamp(0, H - 1)
        xx   = tgt_yx_l[:, 1].clamp(0, W - 1)
        dist = T[torch.arange(B, device=device), yy, xx]

        if return_road_prob and return_distance_field:
            return dist, road_prob, T
        if return_road_prob:
            return dist, road_prob
        if return_distance_field:
            return dist, T
        return dist

    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------

    def _seg_forward(self, batch):
        """Shared forward for seg loss used by train/val steps.

        When the batch contains 'encoder_feat' (pre-computed SAM image embeddings),
        the encoder forward is skipped entirely (~83x speedup).  In this mode the
        batch may omit 'rgb'; a dummy zero tensor is used as a placeholder since
        it is never accessed.
        """
        enc_feat  = batch.get("encoder_feat", None)  # [B, C, H_f, W_f] or None
        road_mask = batch["road_mask"].to(torch.float32)

        if enc_feat is not None:
            # Bypass encoder; rgb not needed.
            dummy_rgb = torch.zeros(
                enc_feat.shape[0], 1, 1, 3,
                dtype=torch.float32, device=enc_feat.device,
            )
            mask_logits, mask_scores = self._predict_mask_logits_scores(dummy_rgb, enc_feat)
        else:
            rgb = batch["rgb"]
            mask_logits, mask_scores = self._predict_mask_logits_scores(rgb)

        road_logits = mask_logits[..., 1]   # [B, H, W]
        road_prob   = mask_scores[..., 1]   # [B, H, W]

        # --- Robust normalized spatial weighted BCE loss ---
        # 1. Per-pixel BCE (no reduction)
        bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            road_logits, road_mask, reduction='none'
        )

        # 2. Base weight map (background = 1.0)
        w = torch.ones_like(road_mask, dtype=road_logits.dtype)
        pos_w = self._road_pos_weight

        # 3. Thick road positives: weight = pos_w
        w = w + (pos_w - 1.0) * (road_mask > 0.5).to(w.dtype)

        # 4. If dual_target, extra boost for thin skeleton (thin_boost x thick weight)
        if self._road_dual_target and "road_mask_thin" in batch:
            thin = batch["road_mask_thin"].to(w.device)
            thin_boost = self._road_thin_boost
            w = w + (pos_w * thin_boost - 1.0) * (thin > 0.5).to(w.dtype)

        # 5. Normalize by weight sum (stable loss scale across batches)
        loss_bce = (bce_raw * w).sum() / (w.sum() + 1e-6)

        # --- Dice loss ---
        # When dual_target is enabled, Dice is computed against the THIN
        # (original) mask.  This teaches the model to predict sharp
        # boundaries: BCE says "find all road pixels (thick)", Dice says
        # "match the precise road centre (thin)".  The gradient pulls
        # road_prob toward high values only at true road centres,
        # producing sharp edges ideal for the Eikonal cost landscape.
        dice_weight = self._road_dice_weight
        if dice_weight > 0.0:
            smooth = 1e-6
            dice_target = batch.get("road_mask_thin", None)
            if dice_target is None or not self._road_dual_target:
                dice_target = road_mask
            else:
                dice_target = dice_target.to(road_prob.device, dtype=road_prob.dtype)
            p = road_prob.reshape(-1)
            t = dice_target.reshape(-1)
            inter = (p * t).sum()
            loss_dice = 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
            loss_seg = (1.0 - dice_weight) * loss_bce + dice_weight * loss_dice
        else:
            loss_seg = loss_bce
        return loss_seg, road_prob

    def _effective_eik_cfg(self):
        """Return EikonalConfig with iteration count adapted by warmup schedule."""
        base_iters = self.route_eik_cfg.n_iters
        min_iters  = self.route_eik_iters_min
        warmup_ep  = self.route_eik_warmup_epochs
        if warmup_ep > 0 and self.current_epoch < warmup_ep:
            frac = self.current_epoch / warmup_ep
            n = int(min_iters + (base_iters - min_iters) * frac)
        else:
            n = base_iters
        if n == base_iters:
            return self.route_eik_cfg
        from dataclasses import replace
        return replace(self.route_eik_cfg, n_iters=n)

    def _effective_lambda_dist(self):
        """Linear warmup of distance loss weight over the first N global steps."""
        warmup = self.route_dist_warmup_steps
        if warmup <= 0:
            return self.route_lambda_dist
        step = self.global_step
        return self.route_lambda_dist * min(1.0, step / warmup)

    def training_step(self, batch, batch_idx):
        """
        End-to-end training with segmentation + distance regression losses.

        Improvements over naive Phase 2 (GPPN-inspired):
          - Distance loss weight warmup (avoids overwhelming seg loss early on)
          - Eikonal iteration count warmup (shorter backprop graph initially)
          - Normalized distance (pred & gt divided by ROUTE_DIST_NORM_PX)
          - Clamped pred_dist (caps runaway eikonal values before loss)
        """
        loss_seg, road_prob = self._seg_forward(batch)
        loss = self.route_lambda_seg * loss_seg

        self.log("train_seg_loss", loss_seg, on_step=True, on_epoch=False, prog_bar=True)

        if self.route_lambda_dist > 0 and "src_yx" in batch:
            src_yx  = batch["src_yx"]
            tgt_yx  = batch["tgt_yx"]
            gt_dist = batch["gt_dist"].to(torch.float32)

            eik_cfg = self._effective_eik_cfg()

            if tgt_yx.dim() == 3:
                # New anchor-centered K-target format: tgt_yx [B, K, 2], gt_dist [B, K]
                valid = gt_dist > 0          # [B, K] mask
                pred_dist_raw = self._roi_multi_target_diff_solve(
                    road_prob, src_yx, tgt_yx, gt_dist, eik_cfg)  # [B, K]

                pred_norm, gt_norm, _sat = self._apply_dist_cap(pred_dist_raw, gt_dist)

                if valid.any():
                    loss_dist = self.dist_criterion(pred_norm[valid], gt_norm[valid])
                else:
                    loss_dist = torch.tensor(0.0, device=road_prob.device,
                                             requires_grad=False)
            else:
                # Legacy single-target format: tgt_yx [B, 2], gt_dist [B]
                pred_dist_raw = self._roi_eikonal_solve(road_prob, src_yx, tgt_yx, eik_cfg)
                pred_norm, gt_norm, _sat = self._apply_dist_cap(pred_dist_raw, gt_dist)
                loss_dist = self.dist_criterion(pred_norm, gt_norm)
                valid     = torch.ones(gt_dist.shape, dtype=torch.bool,
                                       device=gt_dist.device)

            lam  = self._effective_lambda_dist()
            dist_contrib = lam * loss_dist
            loss = loss + dist_contrib

            self.log("train_dist_loss", loss_dist, on_step=True, on_epoch=False, prog_bar=True)
            self.log("dist_loss_weighted", dist_contrib, on_step=True, on_epoch=False, prog_bar=True)
            self.log("lambda_dist_eff", lam,       on_step=True, on_epoch=False)
            self.log("eik_iters_eff", float(eik_cfg.n_iters), on_step=True, on_epoch=False)

            with torch.no_grad():
                _a = self.cost_log_alpha.clamp(
                    _math.log(5.0), _math.log(100.0)).exp()
                _g = self.cost_log_gamma.clamp(
                    _math.log(0.5), _math.log(4.0)).exp()
                _gate = torch.sigmoid(self.eik_gate_logit).clamp(0.3, 0.95)
                self.log("cost_alpha", _a, on_step=False, on_epoch=True)
                self.log("cost_gamma", _g, on_step=False, on_epoch=True)
                self.log("eik_gate", _gate, on_step=False, on_epoch=True)

            with torch.no_grad():
                valid_pred = pred_dist_raw[valid] if valid.any() else pred_dist_raw.flatten()[:1]
                valid_gt   = gt_dist[valid]        if valid.any() else gt_dist.flatten()[:1]
                self.log("pred_dist_mean", valid_pred.mean(), on_step=True, on_epoch=False)
                self.log("pred_dist_max",  valid_pred.max(),  on_step=True, on_epoch=False)
                self.log("gt_dist_mean",   valid_gt.mean(),   on_step=True, on_epoch=False)
                self.log("dist_abs_err",   (valid_pred - valid_gt).abs().mean(),
                         on_step=True, on_epoch=False)
                if loss_seg.item() > 0 and valid.any():
                    self.log("dist_seg_ratio", loss_dist.item() / loss_seg.item(),
                             on_step=True, on_epoch=False)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_after_backward(self):
        if self.route_lambda_dist > 0:
            grads = [p.grad for p in self.parameters() if p.grad is not None]
            if grads:
                has_nan = any(g.isnan().any() for g in grads)
                has_inf = any(g.isinf().any() for g in grads)

                if has_nan or has_inf:
                    for p in self.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad.zero_()
                    self._nan_grad_skipped = True
                else:
                    self._nan_grad_skipped = False

                total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
                self.log("grad_norm_total", total_norm, on_step=True, on_epoch=False)
                self.log("grad_has_nan", float(has_nan), on_step=True, on_epoch=False)
                self.log("grad_has_inf", float(has_inf), on_step=True, on_epoch=False)

            dec_grads = [p.grad for p in self.map_decoder.parameters()
                         if p.grad is not None]
            if dec_grads:
                dec_norm = torch.sqrt(sum(g.norm() ** 2 for g in dec_grads))
                self.log("grad_norm_decoder", dec_norm, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        """Validation: compute seg loss (and optionally dist loss) without gradients."""
        loss_seg, road_prob = self._seg_forward(batch)
        loss = self.route_lambda_seg * loss_seg

        road_mask = batch["road_mask"].to(torch.float32)
        road_gt = (road_mask > 0.5).to(torch.long)  # target must be int per BinaryJaccardIndex API
        self.road_iou.update(road_prob, road_gt)    # preds: float probs (thresholded at 0.5)

        self.log("val_seg_loss", loss_seg, on_step=False, on_epoch=True, prog_bar=True)

        if self.route_lambda_dist > 0 and "src_yx" in batch:
            src_yx  = batch["src_yx"]
            tgt_yx  = batch["tgt_yx"]
            gt_dist = batch["gt_dist"].to(torch.float32)
            with torch.no_grad():
                if tgt_yx.dim() == 3:
                    # New K-target format
                    valid = gt_dist > 0
                    pred_dist_raw = self._roi_multi_target_diff_solve(
                        road_prob, src_yx, tgt_yx, gt_dist)  # [B, K]
                    pred_norm, gt_norm, _sat = self._apply_dist_cap(pred_dist_raw, gt_dist)
                    if valid.any():
                        loss_dist = self.dist_criterion(pred_norm[valid], gt_norm[valid])
                    else:
                        loss_dist = torch.tensor(0.0, device=road_prob.device)
                else:
                    # Legacy single-target format
                    pred_dist_raw = self._roi_eikonal_solve(road_prob, src_yx, tgt_yx)
                    pred_norm, gt_norm, _sat = self._apply_dist_cap(pred_dist_raw, gt_dist)
                    loss_dist = self.dist_criterion(pred_norm, gt_norm)
            loss = loss + self.route_lambda_dist * loss_dist
            self.log("val_dist_loss", loss_dist, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Override parent to log only road IoU (skip keypoint/topo metrics)."""
        road_iou = self.road_iou.compute()
        self.log("val_road_iou", road_iou, prog_bar=True)
        self.road_iou.reset()

    # ------------------------------------------------------------------
    # configure_optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Same parameter groups as SAMRoad, without topo_net."""
        param_dicts = []

        if not self.config.FREEZE_ENCODER and not self.config.ENCODER_LORA:
            param_dicts.append({
                'params': [p for k, p in self.image_encoder.named_parameters()
                           if ('image_encoder.' + k) in self.matched_param_names],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
            })

        if self.config.ENCODER_LORA:
            param_dicts.append({
                'params': [p for k, p in self.image_encoder.named_parameters()
                           if 'qkv.linear_' in k],
                'lr': self.config.BASE_LR,
            })

        if self.config.USE_SAM_DECODER:
            param_dicts += [
                {
                    'params': [p for k, p in self.mask_decoder.named_parameters()
                               if ('mask_decoder.' + k) in self.matched_param_names],
                    'lr': self.config.BASE_LR * 0.1,
                },
                {
                    'params': [p for k, p in self.mask_decoder.named_parameters()
                               if ('mask_decoder.' + k) not in self.matched_param_names],
                    'lr': self.config.BASE_LR,
                },
            ]
        else:
            param_dicts.append({
                'params': list(self.map_decoder.parameters()),
                'lr': self.config.BASE_LR,
            })

        if self.route_lambda_dist > 0:
            eik_params = [self.cost_log_alpha, self.cost_log_gamma,
                          self.eik_gate_logit]
            param_dicts.append({
                'params': eik_params,
                'lr': self.config.BASE_LR,
            })

        for i, pd in enumerate(param_dicts):
            print(f'[SAMRoute] param group {i}: {sum(p.numel() for p in pd["params"])} params')

        optimizer = torch.optim.Adam(param_dicts, lr=self.config.BASE_LR)
        milestones = list(self.config.get("LR_MILESTONES", [150]))
        step_lr   = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}

    # ------------------------------------------------------------------
    # Training: batched one-src → K-targets differentiable Eikonal solve
    # ------------------------------------------------------------------

    def _roi_multi_target_diff_solve(
        self,
        road_prob: torch.Tensor,    # [B, H, W]
        src_yx: torch.Tensor,       # [B, 2] long  — patch-relative coords
        tgt_yx_k: torch.Tensor,     # [B, K, 2] long  — patch-relative coords
        gt_dist_k: torch.Tensor,    # [B, K] float  — -1.0 for padding entries
        eik_cfg: "EikonalConfig | None" = None,
    ) -> torch.Tensor:              # [B, K]
        """
        Differentiable one-src → K-targets Eikonal solve for training.

        Batched across the full batch B:
          1. Compute per-element ROI geometry (P_b, ds_b) independently.
          2. Use ds_common = max(ds_b) so all coarse grids share the same cell
             size, enabling a single _eikonal_soft_sweeping_diff call on a
             [B, P_c, P_c] tensor instead of B sequential calls.
          3. Read K target T-values per element from the batched T field.

        Elements with no valid targets are handled by a sentinel cost/src.

        Returns [B, K] tensor; padding positions hold cfg.large_val.
        """
        from dataclasses import replace as _replace

        B, H, W = road_prob.shape
        K       = tgt_yx_k.shape[1]
        device  = road_prob.device
        cfg     = eik_cfg if eik_cfg is not None else self.route_eik_cfg
        margin  = self.route_roi_margin
        min_ds  = max(1, int(getattr(self, 'route_eik_downsample', 4)))
        n_iters = int(cfg.n_iters)
        large_val = float(cfg.large_val)

        # ------------------------------------------------------------------ #
        # Pass 1: compute per-element ROI geometry                            #
        # ------------------------------------------------------------------ #
        geom = []   # list of (y0, x0, P, src_rel_y, src_rel_x, ds, valid_mask_b)
        for b in range(B):
            src        = src_yx[b].long()
            tgts       = tgt_yx_k[b]
            valid_mask = gt_dist_k[b] > 0

            if valid_mask.any():
                valid_tgts = tgts[valid_mask]
                span_max   = int(torch.max(torch.abs(
                    valid_tgts.float() - src.float())).item())
            else:
                span_max = 0

            half = span_max + margin
            P    = max(2 * half + 1, 64)
            y0   = int(src[0].item()) - half
            x0   = int(src[1].item()) - half
            sr_y = max(0, min(int(src[0].item()) - y0, P - 1))
            sr_x = max(0, min(int(src[1].item()) - x0, P - 1))
            ds   = min_ds
            geom.append((y0, x0, P, sr_y, sr_x, ds, valid_mask))

        # ------------------------------------------------------------------ #
        # Pass 2: common downsample factor and coarse-grid size               #
        # ------------------------------------------------------------------ #
        ds_common = max(g[5] for g in geom)
        # P_c = coarse grid side length for the largest ROI
        P_max = max(g[2] for g in geom)
        P_c   = int(_math.ceil(P_max / ds_common))

        # ------------------------------------------------------------------ #
        # Pass 3: build batched cost [B, P_c, P_c] and src_mask [B, P_c, P_c]#
        # ------------------------------------------------------------------ #
        costs_list = []
        src_mask   = torch.zeros(B, P_c, P_c, dtype=torch.bool, device=device)

        for b in range(B):
            y0, x0, P, sr_y, sr_x, _ds_b, valid_mask = geom[b]
            y1, x1   = y0 + P, x0 + P
            prob_b   = road_prob[b]

            yy0 = max(y0, 0); xx0 = max(x0, 0)
            yy1 = min(y1, H); xx1 = min(x1, W)

            patch = F.pad(
                prob_b[yy0:yy1, xx0:xx1],
                (xx0 - x0, x1 - xx1, yy0 - y0, y1 - yy1),
                value=0.0,
            )  # [P, P]

            # Pad P to a multiple of ds_common before pooling
            P_pad = int(_math.ceil(P / ds_common) * ds_common)
            if P_pad > P:
                patch = F.pad(patch, (0, P_pad - P, 0, P_pad - P), value=0.0)

            patch_c = F.max_pool2d(
                patch.unsqueeze(0).unsqueeze(0), kernel_size=ds_common, stride=ds_common
            ).squeeze(0).squeeze(0)  # [ceil(P/ds), ceil(P/ds)]

            # Pad to the common P_c × P_c size (zero-cost = high cost after conversion)
            ph, pw = patch_c.shape
            if ph < P_c or pw < P_c:
                patch_c = F.pad(patch_c, (0, P_c - pw, 0, P_c - ph), value=0.0)

            costs_list.append(self._road_prob_to_cost(patch_c).to(dtype=torch.float32))  # [P_c, P_c]

            sc_y = max(0, min(sr_y // ds_common, P_c - 1))
            sc_x = max(0, min(sr_x // ds_common, P_c - 1))
            src_mask[b, sc_y, sc_x] = True

        cost_batch = torch.stack(costs_list)  # [B, P_c, P_c]

        # ------------------------------------------------------------------ #
        # Pass 4: single batched differentiable Eikonal solve                 #
        # ------------------------------------------------------------------ #
        cfg_common = _replace(cfg, h=float(ds_common))
        T_all = _eikonal_soft_sweeping_diff(
            cost_batch,
            src_mask,
            cfg_common,
            checkpoint_chunk=self.route_ckpt_chunk,
            gate_alpha=self.route_gate_alpha,
        )  # [B, P_c, P_c]

        # ------------------------------------------------------------------ #
        # Pass 5: read K target T-values per element                          #
        # ------------------------------------------------------------------ #
        all_dists: list = []
        for b in range(B):
            y0, x0, P, _sr_y, _sr_x, _ds_b, valid_mask = geom[b]
            tgts    = tgt_yx_k[b]
            T_b     = T_all[b]

            dists_b = []
            for k in range(K):
                if valid_mask[k]:
                    ty = int(tgts[k, 0].item())
                    tx = int(tgts[k, 1].item())
                    tc_y = max(0, min((ty - y0) // ds_common, P_c - 1))
                    tc_x = max(0, min((tx - x0) // ds_common, P_c - 1))
                    dists_b.append(T_b[tc_y, tc_x])
                else:
                    dists_b.append(
                        torch.tensor(large_val, device=device,
                                     dtype=cost_batch.dtype))
            all_dists.append(torch.stack(dists_b))

        T_eik = torch.stack(all_dists)  # [B, K]

        # Euclidean residual blending
        d_euc = ((src_yx.unsqueeze(1).float() - tgt_yx_k.float()) ** 2
                 ).sum(-1).sqrt()  # [B, K]
        return _mix_eikonal_euclid(T_eik, d_euc, self.eik_gate_logit)

    # ------------------------------------------------------------------
    # Inference: one source → many targets (single Eikonal solve)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_knn_distances(
        self,
        road_prob: torch.Tensor,   # [H, W]
        src_yx: torch.Tensor,      # [2] long
        tgt_yx: torch.Tensor,      # [K, 2] long
        eik_cfg: "EikonalConfig | None" = None,
        margin: int | None = None,
        min_patch: int = 512,
        max_patch: int = 4096,
    ) -> torch.Tensor:
        """
        One-to-many Eikonal distance: compute distances from src to all K targets
        using a single ROI-cropped Eikonal solve.

        Unlike _roi_eikonal_solve() (which loops B times, one solve per pair),
        this method builds ONE ROI large enough to contain all K targets and runs
        the Eikonal solver once.  This is the inference-time equivalent of
        compute_distance_field_roi_for_targets() in the old integrator.

        Uses eikonal_soft_sweeping (non-differentiable, @torch.no_grad) for speed.

        Args:
            road_prob: [H, W]  road probability map (0..1)
            src_yx:    [2]     source pixel (y, x)
            tgt_yx:    [K, 2]  target pixels (y, x)
            eik_cfg:   override default EikonalConfig
            margin:    ROI margin in pixels beyond the farthest target
                       (defaults to self.route_roi_margin)
            min_patch: minimum ROI side length
            max_patch: hard cap on ROI side length

        Returns:
            dist: [K] float  Eikonal cost T(tgt_k) for each target
                  (inf = target unreachable within ROI)
        """
        cfg    = eik_cfg if eik_cfg is not None else self.route_eik_cfg
        margin = int(margin if margin is not None else self.route_roi_margin)

        H, W   = road_prob.shape
        device = road_prob.device
        src    = src_yx.to(device).long()
        tgts   = tgt_yx.to(device).long()
        K      = tgts.shape[0]

        if K == 0:
            return torch.empty(0, device=device)

        # ROI: centered at src, large enough to cover the farthest target + margin
        span_max = int(torch.max(torch.abs(tgts.float() - src.float())).item())
        half     = span_max + margin
        P        = max(2 * half + 1, int(min_patch))
        P        = min(P, int(max_patch))

        y0 = int(src[0].item()) - half;  x0 = int(src[1].item()) - half
        y1 = y0 + P;                     x1 = x0 + P

        yy0 = max(y0, 0); xx0 = max(x0, 0)
        yy1 = min(y1, H); xx1 = min(x1, W)

        # extract patch (zero-pad out-of-bounds areas → treated as off-road)
        patch = torch.zeros(P, P, device=device, dtype=road_prob.dtype)
        patch[yy0 - y0: yy1 - y0, xx0 - x0: xx1 - x0] = road_prob[yy0:yy1, xx0:xx1]

        # cost map and source mask
        cost_patch = self._road_prob_to_cost(patch)   # [P, P]

        src_rel_y = max(0, min(int(src[0].item()) - y0, P - 1))
        src_rel_x = max(0, min(int(src[1].item()) - x0, P - 1))
        src_mask  = torch.zeros(1, P, P, dtype=torch.bool, device=device)
        src_mask[0, src_rel_y, src_rel_x] = True

        # single Eikonal solve (use original fast version — no grad needed)
        n_iters_eff = max(int(cfg.n_iters), P)
        cfg_eff     = EikonalConfig(
            n_iters     = n_iters_eff,
            h           = cfg.h,
            tau_min     = cfg.tau_min,
            tau_branch  = cfg.tau_branch,
            tau_update  = cfg.tau_update,
            large_val   = cfg.large_val,
            use_redblack= cfg.use_redblack,
            monotone    = cfg.monotone,
        )
        T_patch = eikonal_soft_sweeping(
            cost_patch.unsqueeze(0),   # [1, P, P]
            src_mask,
            cfg_eff,
        )  # [1, P, P]  or [P, P] if squeezed
        if T_patch.dim() == 3:
            T_patch = T_patch[0]       # [P, P]

        # gather T values at each target
        tgt_rel_y = (tgts[:, 0] - y0).clamp(0, P - 1)
        tgt_rel_x = (tgts[:, 1] - x0).clamp(0, P - 1)
        # mark targets that fall outside the ROI as inf
        valid = (
            (tgts[:, 0] >= y0) & (tgts[:, 0] < y1) &
            (tgts[:, 1] >= x0) & (tgts[:, 1] < x1)
        )
        dist = T_patch[tgt_rel_y, tgt_rel_x]
        dist = torch.where(valid, dist, torch.full_like(dist, float('inf')))
        return dist   # [K]