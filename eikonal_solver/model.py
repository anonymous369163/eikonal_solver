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
        
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)
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
        if self.config.TOPONET_VERSION == 'no_offset':
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
            #### Naive decoder
            activation = nn.GELU
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

        #### DEBUG NAN
        for nan_index in torch.nonzero(torch.isnan(topo_loss[:, :, :, 0])):
            print('nan index: B, Sample, Pair')
            print(nan_index)
            import pdb
            pdb.set_trace()

        #### DEBUG NAN


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
from fast_sweeping import (
    EikonalConfig, prob_to_cost, make_source_mask, eikonal_soft_sweeping,
    _local_update_fast, softmin_algebraic,
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
    prob = prob.clamp(0.0, 1.0)
    k = _math.log(max(float(offroad_penalty), 1.0 + 1e-6))
    cost = torch.exp(k * (1.0 - prob).pow(gamma))
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
    prob = prob.clamp(0.0, 1.0)
    # base + polynomial penalty
    cost = 1.0 + float(alpha) * (1.0 - prob).pow(gamma)
    # smooth hard-block near block_th (sigmoid gives a C∞ step)
    if block_th > 0.0 and block_alpha > 0.0:
        block = float(block_alpha) * torch.sigmoid(
            -float(block_smooth) * (prob - float(block_th))
        )
        cost = cost + block
    return cost.clamp_min(eps)


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
            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch)
            T_e = softmin_algebraic(T, T_new, tau_update) if monotone else T_new
            if use_gate:
                T_e = gate_alpha * T_e + (1.0 - gate_alpha) * T_old
            T = torch.where(even_mask, T_e, T)
            T = torch.where(src_mask, torch.zeros_like(T), T)

            T_old = T
            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch)
            T_o = softmin_algebraic(T, T_new, tau_update) if monotone else T_new
            if use_gate:
                T_o = gate_alpha * T_o + (1.0 - gate_alpha) * T_old
            T = torch.where(odd_mask, T_o, T)
            T = torch.where(src_mask, torch.zeros_like(T), T)
        else:
            T_old = T
            T_new = _local_update_fast(T, cost, h_val, large_val, tau_min, tau_branch)
            T = softmin_algebraic(T, T_new, tau_update) if monotone else T_new
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
    ga           = float(gate_alpha)

    if use_redblack:
        yy = torch.arange(H, device=device)[:, None]
        xx = torch.arange(W, device=device)[None, :]
        even_mask = ((yy + xx) % 2 == 0)[None, :, :].expand(B, -1, -1).contiguous()
        odd_mask  = (~((yy + xx) % 2 == 0))[None, :, :].expand(B, -1, -1).contiguous()
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
                use_redblack, monotone, nc, ga,
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
        # --- additive-mode params ---
        self.route_add_alpha        = float(_cfg(config, "ROUTE_ADD_ALPHA",         20.0))
        self.route_add_gamma        = float(_cfg(config, "ROUTE_ADD_GAMMA",         2.0))
        self.route_add_block_alpha  = float(_cfg(config, "ROUTE_ADD_BLOCK_ALPHA",   50.0))
        self.route_add_block_smooth = float(_cfg(config, "ROUTE_ADD_BLOCK_SMOOTH",  50.0))
        self.route_eps              = float(_cfg(config, "ROUTE_EPS",               1e-6))
        self.route_eik_cfg = EikonalConfig(
            n_iters     = int(_cfg(config, "ROUTE_EIK_ITERS", 100)),
            tau_min     = 0.03,
            tau_branch  = 0.05,
            tau_update  = 0.03,
            use_redblack= True,
            monotone    = True,
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
        self.road_criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight applied dynamically in _seg_forward
        self.dist_criterion = torch.nn.HuberLoss(delta=50.0)

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
        # default: "add"
        return _prob_to_cost_additive(
            road_prob,
            alpha        = self.route_add_alpha,
            gamma        = self.route_add_gamma,
            block_th     = self.route_block_th,
            block_alpha  = self.route_add_block_alpha,
            block_smooth = self.route_add_block_smooth,
            eps          = self.route_eps,
        )

    def _roi_eikonal_solve(
        self,
        road_prob: torch.Tensor,   # [B, H, W]
        src_yx: torch.Tensor,      # [B, 2] long
        tgt_yx: torch.Tensor,      # [B, 2] long
        eik_cfg: "EikonalConfig | None" = None,
    ) -> torch.Tensor:
        """
        Adaptive multi-resolution ROI-cropped differentiable Eikonal solve.

        Key insight (GPPN-inspired): instead of a fixed downsample factor, we
        compute ds *per sample* so that the coarse patch size P_c <= n_iters.
        This guarantees the wavefront always reaches the target regardless of
        the source-target distance.

        ds = max(min_ds, ceil(P / n_iters))

        For example: P=1129, n_iters=100 → ds=12, P_c=95 ≤ 100.
        The Eikonal h parameter is set to ds, so T values are in original
        pixel units.

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

            # Adaptive ds: ensure P_coarse <= n_iters so wavefront always reaches target
            ds = max(min_ds, int(_math.ceil(P / max(n_iters, 1))))

            if ds > 1:
                P_pad = int(_math.ceil(P / ds) * ds)
                if P_pad > P:
                    patch = F.pad(patch, (0, P_pad - P, 0, P_pad - P), value=0.0)

                patch_coarse = F.avg_pool2d(
                    patch.unsqueeze(0).unsqueeze(0), kernel_size=ds, stride=ds
                ).squeeze(0).squeeze(0)  # [P_c, P_c]

                cost_coarse = self._road_prob_to_cost(patch_coarse)
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
                cost_patch = self._road_prob_to_cost(patch)

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

        return torch.stack(dists)  # [B]

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

        # --- BCE loss against THICK mask (class imbalance compensation) ---
        # Thick GT (r=8) matches the 16px feature stride so the encoder can
        # reliably encode road locations.
        if self._road_pos_weight != 1.0:
            pw = torch.tensor(self._road_pos_weight, device=road_logits.device,
                              dtype=road_logits.dtype)
            loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                road_logits, road_mask, pos_weight=pw)
        else:
            loss_bce = self.road_criterion(road_logits, road_mask)

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

                norm      = self.route_dist_norm_px
                pred_norm = pred_dist_raw.clamp(max=norm * 10) / norm
                gt_norm   = gt_dist / norm

                if valid.any():
                    loss_dist = self.dist_criterion(pred_norm[valid], gt_norm[valid])
                else:
                    loss_dist = torch.tensor(0.0, device=road_prob.device,
                                             requires_grad=False)
            else:
                # Legacy single-target format: tgt_yx [B, 2], gt_dist [B]
                pred_dist_raw = self._roi_eikonal_solve(road_prob, src_yx, tgt_yx, eik_cfg)
                norm      = self.route_dist_norm_px
                pred_norm = pred_dist_raw.clamp(max=norm * 10) / norm
                gt_norm   = gt_dist / norm
                loss_dist = self.dist_criterion(pred_norm, gt_norm)
                valid     = torch.ones(gt_dist.shape, dtype=torch.bool,
                                       device=gt_dist.device)

            lam  = self._effective_lambda_dist()
            loss = loss + lam * loss_dist

            self.log("train_dist_loss", loss_dist, on_step=True, on_epoch=False, prog_bar=True)
            self.log("lambda_dist_eff", lam,       on_step=True, on_epoch=False)
            self.log("eik_iters_eff", float(eik_cfg.n_iters), on_step=True, on_epoch=False)

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
                total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
                self.log("grad_norm_total", total_norm, on_step=True, on_epoch=False)

                has_nan = any(g.isnan().any() for g in grads)
                has_inf = any(g.isinf().any() for g in grads)
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
                    norm      = self.route_dist_norm_px
                    pred_norm = pred_dist_raw.clamp(max=norm * 10) / norm
                    gt_norm   = gt_dist / norm
                    if valid.any():
                        loss_dist = self.dist_criterion(pred_norm[valid], gt_norm[valid])
                    else:
                        loss_dist = torch.tensor(0.0, device=road_prob.device)
                else:
                    # Legacy single-target format
                    pred_dist_raw = self._roi_eikonal_solve(road_prob, src_yx, tgt_yx)
                    norm      = self.route_dist_norm_px
                    pred_norm = pred_dist_raw.clamp(max=norm * 10) / norm
                    gt_norm   = gt_dist / norm
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

        For each batch element b, ONE ROI solve is run (sized to cover all
        valid K targets), then T values are read at all K target locations.
        This gives K supervision signals per Eikonal computation instead of
        the previous 1-per-solve, achieving K× more efficient use of the
        expensive iterative solver.

        Padding entries (gt_dist_k[b, k] <= 0) are included in the gather
        but their T values are not used for loss (caller masks them out with
        the valid = gt_dist_k > 0 mask).

        Uses the same adaptive downsampling as _roi_eikonal_solve so the
        wave-front always reaches the farthest valid target regardless of
        distance.

        Returns [B, K] tensor where padding positions hold cfg.large_val.
        """
        B, H, W = road_prob.shape
        K       = tgt_yx_k.shape[1]
        device  = road_prob.device
        cfg     = eik_cfg if eik_cfg is not None else self.route_eik_cfg
        margin  = self.route_roi_margin
        min_ds  = max(1, int(getattr(self, 'route_eik_downsample', 4)))
        n_iters = int(cfg.n_iters)
        large_val = float(cfg.large_val)

        all_dists: list = []

        for b in range(B):
            prob_b     = road_prob[b]      # [H, W]
            src        = src_yx[b].long()  # [2]
            tgts       = tgt_yx_k[b]       # [K, 2]
            valid_mask = gt_dist_k[b] > 0  # [K] bool

            if not valid_mask.any():
                all_dists.append(
                    torch.full((K,), large_val, device=device, dtype=prob_b.dtype))
                continue

            # ROI sized to cover src + all VALID targets
            valid_tgts = tgts[valid_mask]  # [V, 2]
            span_max   = int(torch.max(torch.abs(
                valid_tgts.float() - src.float())).item())
            half = span_max + margin
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

            # Adaptive downsample: P_coarse <= n_iters guarantees wave-front
            # always reaches the farthest valid target
            ds = max(min_ds, int(_math.ceil(P / max(n_iters, 1))))

            if ds > 1:
                P_pad = int(_math.ceil(P / ds) * ds)
                if P_pad > P:
                    patch = F.pad(patch, (0, P_pad - P, 0, P_pad - P), value=0.0)

                patch_coarse = F.avg_pool2d(
                    patch.unsqueeze(0).unsqueeze(0), kernel_size=ds, stride=ds
                ).squeeze(0).squeeze(0)  # [P_c, P_c]

                cost_coarse = self._road_prob_to_cost(patch_coarse)
                P_c = cost_coarse.shape[0]

                src_c_y = max(0, min(src_rel_y // ds, P_c - 1))
                src_c_x = max(0, min(src_rel_x // ds, P_c - 1))

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
                )[0]  # [P_c, P_c]

                dists_b = []
                for k in range(K):
                    if valid_mask[k]:
                        ty = int(tgts[k, 0].item())
                        tx = int(tgts[k, 1].item())
                        tc_y = max(0, min((ty - y0) // ds, P_c - 1))
                        tc_x = max(0, min((tx - x0) // ds, P_c - 1))
                        dists_b.append(T_b[tc_y, tc_x])
                    else:
                        dists_b.append(
                            torch.tensor(large_val, device=device, dtype=T_b.dtype))
            else:
                cost_patch = self._road_prob_to_cost(patch)

                src_mask_b = torch.zeros(1, P, P, dtype=torch.bool, device=device)
                src_mask_b[0, src_rel_y, src_rel_x] = True

                T_b = _eikonal_soft_sweeping_diff(
                    cost_patch.unsqueeze(0),
                    src_mask_b,
                    cfg,
                    checkpoint_chunk=self.route_ckpt_chunk,
                    gate_alpha=self.route_gate_alpha,
                )[0]  # [P, P]

                dists_b = []
                for k in range(K):
                    if valid_mask[k]:
                        ty = int(tgts[k, 0].item())
                        tx = int(tgts[k, 1].item())
                        tr_y = max(0, min(ty - y0, P - 1))
                        tr_x = max(0, min(tx - x0, P - 1))
                        dists_b.append(T_b[tr_y, tr_x])
                    else:
                        dists_b.append(
                            torch.tensor(large_val, device=device, dtype=T_b.dtype))

            all_dists.append(torch.stack(dists_b))

        return torch.stack(all_dists)  # [B, K]

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

    # ------------------------------------------------------------------
    # Visualization: shortest paths from one source to K targets
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualize_paths(
        self,
        rgb: torch.Tensor,             # [1, 3, H, W] or [3, H, W]  float [0,1]
        road_prob: torch.Tensor,       # [H, W]  (same spatial scale as rgb)
        src_yx: torch.Tensor,          # [2]  source pixel
        tgt_yx: torch.Tensor,          # [K, 2]  target pixels
        *,
        eik_cfg: "EikonalConfig | None" = None,
        margin: int | None = None,
        min_patch: int = 512,
        labels: "list[str] | None" = None,
        save_path: str = "images/samroute_paths.png",
        show_T_field: bool = True,
        show_road_prob: bool = True,
        figsize: tuple = (14, 14),
        dpi: int = 150,
    ) -> str:
        """
        End-to-end visualization: run Eikonal, backtrace each path, draw on RGB.

        What is shown:
          - RGB satellite image (background)
          - road_prob as a red heatmap overlay (if show_road_prob=True)
          - Eikonal T-field as a magma heatmap overlay within the ROI (if show_T_field=True)
          - White dashed ROI bounding box
          - Each shortest path drawn as a colored polyline
          - Source (green circle) and each target (colored ×)
          - Legend: path pixel length + T value for each target

        Args:
            rgb:          satellite image tensor
            road_prob:    [H, W] probabilities from the segmentation head
            src_yx:       source node pixel coords (y, x)
            tgt_yx:       [K, 2] target node pixel coords
            eik_cfg:      optional override for Eikonal solver config
            margin:       ROI margin beyond the farthest target (default: self.route_roi_margin)
            min_patch:    minimum ROI side length in pixels
            labels:       optional list of K label strings for the legend
            save_path:    output image file path (parent dirs created automatically)
            show_T_field: overlay the T distance field in the ROI
            show_road_prob: overlay road_prob as red channel
            figsize:      matplotlib figure size
            dpi:          output resolution

        Returns:
            save_path (str) — where the figure was written
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        import numpy as np
        from pathlib import Path as _Path

        cfg    = eik_cfg if eik_cfg is not None else self.route_eik_cfg
        margin = int(margin if margin is not None else self.route_roi_margin)

        # ---- normalise inputs ------------------------------------------------
        if rgb.dim() == 4:
            rgb = rgb[0]           # [3, H, W]
        img_np = (rgb.detach().float().cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        H_img, W_img = img_np.shape[:2]

        road_prob = road_prob.detach().float().cpu()
        H, W = road_prob.shape

        device = road_prob.device if road_prob.device.type != "cpu" else (
            next(self.parameters()).device
        )
        road_prob = road_prob.to(device)

        src  = src_yx.to(device).long()
        tgts = tgt_yx.to(device).long()
        K    = tgts.shape[0]

        # ---- build ROI -------------------------------------------------------
        if K > 0:
            span_max = int(torch.max(torch.abs(tgts.float() - src.float())).item())
        else:
            span_max = 0
        half = span_max + margin
        P    = max(2 * half + 1, int(min_patch))

        y0 = int(src[0].item()) - half;  x0 = int(src[1].item()) - half
        y1 = y0 + P;                     x1 = x0 + P

        yy0 = max(y0, 0); xx0 = max(x0, 0)
        yy1 = min(y1, H); xx1 = min(x1, W)

        patch = torch.zeros(P, P, device=device, dtype=road_prob.dtype)
        patch[yy0 - y0: yy1 - y0, xx0 - x0: xx1 - x0] = road_prob[yy0:yy1, xx0:xx1]

        # ---- Eikonal solve ---------------------------------------------------
        cost_patch = self._road_prob_to_cost(patch)
        src_rel_y  = max(0, min(int(src[0].item()) - y0, P - 1))
        src_rel_x  = max(0, min(int(src[1].item()) - x0, P - 1))
        src_mask   = torch.zeros(1, P, P, dtype=torch.bool, device=device)
        src_mask[0, src_rel_y, src_rel_x] = True

        n_iters_eff = max(int(cfg.n_iters), P)
        cfg_eff = EikonalConfig(
            n_iters=n_iters_eff, h=cfg.h,
            tau_min=cfg.tau_min, tau_branch=cfg.tau_branch, tau_update=cfg.tau_update,
            large_val=cfg.large_val, use_redblack=cfg.use_redblack, monotone=cfg.monotone,
        )
        T_patch = eikonal_soft_sweeping(cost_patch.unsqueeze(0), src_mask, cfg_eff)
        if T_patch.dim() == 3:
            T_patch = T_patch[0]   # [P, P]

        # ---- path backtrace for each target ----------------------------------
        try:
            from utils import trace_path_from_distance_field_common, calc_path_len_vec
        except ImportError:
            trace_path_from_distance_field_common = None
            calc_path_len_vec = None

        src_rel_t = torch.tensor([src_rel_y, src_rel_x], device=device, dtype=torch.long)
        INF = float(1e5)

        paths_global = []    # list of [(y_global, x_global), ...] or None
        path_lens    = []    # pixel lengths
        t_vals       = []    # T values at target

        for k in range(K):
            tgt_rel_y = int(tgts[k, 0].item()) - y0
            tgt_rel_x = int(tgts[k, 1].item()) - x0
            in_roi = (0 <= tgt_rel_y < P) and (0 <= tgt_rel_x < P)
            tval   = float(T_patch[
                max(0, min(tgt_rel_y, P-1)),
                max(0, min(tgt_rel_x, P-1)),
            ].item()) if in_roi else float('inf')
            t_vals.append(tval)

            if (not in_roi) or (not np.isfinite(tval)) or tval >= INF or (
                    trace_path_from_distance_field_common is None):
                paths_global.append(None)
                path_lens.append(float('nan'))
                continue

            tgt_rel_t = torch.tensor(
                [max(0, min(tgt_rel_y, P-1)), max(0, min(tgt_rel_x, P-1))],
                device=device, dtype=torch.long,
            )
            path_patch = trace_path_from_distance_field_common(
                T_patch, src_rel_t, tgt_rel_t, device, diag=True, max_steps=200_000,
            )
            if len(path_patch) < 2:
                paths_global.append(None)
                path_lens.append(float('nan'))
                continue

            path_glob = [(p[0] + y0, p[1] + x0) for p in path_patch]
            paths_global.append(path_glob)
            path_lens.append(calc_path_len_vec(path_glob) if calc_path_len_vec else float('nan'))

        # ---- build T full canvas (NaN outside ROI) ---------------------------
        T_np  = T_patch.detach().float().cpu().numpy()
        scale = H_img / max(H, 1)   # map road_prob coords → image pixel coords

        T_full = np.full((H_img, W_img), np.nan, dtype=np.float32)
        # ROI in image-pixel coords
        iy0 = max(0,     int(np.floor(y0 * scale)))
        ix0 = max(0,     int(np.floor(x0 * scale)))
        iy1 = min(H_img, int(np.ceil( y1 * scale)))
        ix1 = min(W_img, int(np.ceil( x1 * scale)))
        roi_h = iy1 - iy0;  roi_w = ix1 - ix0
        if roi_h > 0 and roi_w > 0:
            T_resized = F.interpolate(
                torch.from_numpy(T_np).unsqueeze(0).unsqueeze(0),
                size=(roi_h, roi_w), mode="bilinear", align_corners=False,
            ).squeeze().numpy()
            T_full[iy0:iy1, ix0:ix1] = T_resized

        finite = np.isfinite(T_full)
        vmin   = float(np.min(T_full[finite]))           if finite.any() else 0.0
        vmax   = float(np.percentile(T_full[finite], 99)) if finite.any() else 1.0

        # road_prob in image coords
        road_np = road_prob.detach().float().cpu().numpy()
        if scale != 1.0:
            road_np = F.interpolate(
                torch.from_numpy(road_np).unsqueeze(0).unsqueeze(0),
                size=(H_img, W_img), mode="bilinear", align_corners=False,
            ).squeeze().numpy()

        # ---- plotting --------------------------------------------------------
        colors = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_np)

        if show_road_prob:
            masked_road = np.ma.masked_where(road_np < 0.1, road_np)
            ax.imshow(masked_road, cmap="Reds", vmin=0, vmax=1, alpha=0.40)

        if show_T_field:
            ax.imshow(T_full, cmap="magma", vmin=vmin, vmax=vmax, alpha=0.32)

        # ROI bounding box (in image-pixel coords)
        ax.plot([ix0, ix1, ix1, ix0, ix0],
                [iy0, iy0, iy1, iy1, iy0],
                color="white", lw=1.5, ls="--", alpha=0.75, zorder=8)

        # draw paths
        for k, path in enumerate(paths_global):
            c = colors[k]
            if path is not None:
                ys = [p[0] * scale for p in path]
                xs = [p[1] * scale for p in path]
                ax.plot(xs, ys, color=c, lw=2.0, alpha=0.85, zorder=10,
                        path_effects=[pe.Stroke(linewidth=3.5, foreground="black", alpha=0.5),
                                      pe.Normal()])

        # draw targets
        for k in range(K):
            ty_img = int(tgts[k, 0].item() * scale)
            tx_img = int(tgts[k, 1].item() * scale)
            c = colors[k]
            plen = path_lens[k]
            tval = t_vals[k]
            lbl  = labels[k] if labels and k < len(labels) else f"tgt {k}"
            plen_s = f"{plen:.1f}px" if np.isfinite(plen) else "unreachable"
            tval_s = f"T={tval:.0f}"  if np.isfinite(tval) else "T=inf"
            legend_label = f"{lbl}  len={plen_s}  {tval_s}"

            ax.scatter([tx_img], [ty_img], s=220, marker="x", color=c,
                       linewidths=2.5, zorder=14, label=legend_label)

        # draw source
        sy_img = int(src[0].item() * scale)
        sx_img = int(src[1].item() * scale)
        ax.scatter([sx_img], [sy_img], s=300, marker="o", color="lime",
                   edgecolors="black", linewidths=2.5, zorder=15, label="Source")

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                  fontsize=8.5, framealpha=0.9, borderaxespad=0)
        ax.axis("off")
        ax.set_title(
            "SAMRoute: shortest paths  (magma=T-field, red=road_prob, colored lines=paths)",
            fontsize=11,
        )

        _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAMRoute.visualize_paths] Saved → {save_path}")
        return save_path



if __name__ == "__main__":
    """
    Usage example:
        cd /home/yuepeng/code/mmdm_V3/MMDataset/eikonal_solver
        conda run -n satdino python model.py \
            --image_path "../Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/crop_34.350697_108.914569_3000.0_z16.tif" \
            --anchor_idx 3 --k 8
    """
    import argparse
    from pathlib import Path

    # ── extra paths so imports work when run directly ──────────────────────
    _here = Path(__file__).resolve().parent
    _repo = _here.parent / "sam_road_repo"
    for _p in [
        str(_repo / "segment-anything-road"),
        str(_repo / "sam"),
        str(_repo),
    ]:
        if _p not in sys.path:
            sys.path.append(_p)
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    # ── argument parsing ───────────────────────────────────────────────────
    _parser = argparse.ArgumentParser(
        description="SAMRoute: predict Eikonal road distances + visualize shortest paths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Default paths are resolved relative to the script file so the demo works
    # regardless of the current working directory.
    _default_image = str(
        _here.parent
        / "Gen_dataset_V2/Gen_dataset/34.269888_108.947180"
        / "01_34.350697_108.914569_3000.0"
        / "crop_34.350697_108.914569_3000.0_z16.tif"
    )
    _default_save = str(_here / "images" / "samroute_demo.png")

    _parser.add_argument(
        "--image_path", type=str,
        default=_default_image,
        help="Path to a satellite .tif image",
    )
    _parser.add_argument("--config_path",  type=str, default=None,
                         help="SAMRoad yaml config (auto-detected if omitted)")
    _parser.add_argument("--ckpt_path",    type=str, default=None,
                         help="SAMRoad/SAMRoute checkpoint (auto-detected if omitted)")
    _parser.add_argument("--anchor_idx",   type=int, default=None,
                         help="Anchor node index (random if omitted)")
    _parser.add_argument("--k",            type=int, default=8,
                         help="Number of KNN neighbours")
    _parser.add_argument("--downsample",   type=int, default=1024,
                         help="Downsample longer side to this resolution for Eikonal (0=off)")
    _parser.add_argument("--eik_iters",    type=int, default=700,
                         help="Eikonal solver iterations")
    _parser.add_argument("--margin",       type=int, default=96,
                         help="ROI margin beyond farthest target (pixels)")
    _parser.add_argument("--save_path",    type=str, default=_default_save,
                         help="Output visualization path")
    _args = _parser.parse_args()

    # ── auto-find config / checkpoint if not supplied ─────────────────────
    def _find_upwards(start: Path, marker: str) -> "Path | None":
        cur = start.resolve()
        for _ in range(10):
            if (cur / marker).exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        return None

    if _args.config_path is None or _args.ckpt_path is None:
        _root = _find_upwards(_here, "sam_road_repo")
        if _root:
            _args.config_path = _args.config_path or str(
                _root / "sam_road_repo" / "config" / "toponet_vitb_512_cityscale.yaml"
            )
            _args.ckpt_path = _args.ckpt_path or str(
                _root / "checkpoints" / "cityscale_vitb_512_e10.ckpt"
            )

    if not _args.config_path or not Path(_args.config_path).exists():
        raise FileNotFoundError(f"Config not found: {_args.config_path}")
    if not _args.ckpt_path or not Path(_args.ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {_args.ckpt_path}")
    if not Path(_args.image_path).exists():
        raise FileNotFoundError(f"Image not found: {_args.image_path}")

    # ── imports that need sys.path set up above ────────────────────────────
    import yaml
    import numpy as np
    from addict import Dict as AdDict
    from feature_extractor import load_tif_image, SamRoadTiledInferencer
    from load_nodes_from_npz import load_nodes_from_npz
    from utils import (
        pick_anchor_and_knn, mask_finite,
        fmt_rank_table, compute_ranking_metrics,
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {_device}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — Build SAMRoute model
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== STEP 1: Build SAMRoute ===")
    _config = AdDict(yaml.safe_load(open(_args.config_path)))
    _sam_ckpt = Path(_args.config_path).parent.parent / "sam_ckpts" / "sam_vit_b_01ec64.pth"
    if _sam_ckpt.exists():
        _config.SAM_CKPT_PATH = str(_sam_ckpt)

    _model = SAMRoute(_config)
    _ckpt_data = torch.load(_args.ckpt_path, map_location="cpu")
    _model.load_state_dict(_ckpt_data.get("state_dict", _ckpt_data), strict=False)
    _model.eval().to(_device)
    print(f"  Checkpoint loaded: {_args.ckpt_path}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — Load image and extract road_prob
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== STEP 2: Load image + extract road_prob ===")
    _rgb = load_tif_image(_args.image_path).to(_device)   # [1, 3, H, W]
    _, _, _H_orig, _W_orig = _rgb.shape
    print(f"  Image: {_H_orig} × {_W_orig}")

    _inf = SamRoadTiledInferencer(sat_grid=14,
                                  sam_road_ckpt_name=Path(_args.ckpt_path).name)
    _inf.ensure_loaded(_device)
    _road_result = _inf.infer(_rgb, _device, return_mask=True)
    _road_prob   = torch.from_numpy(_road_result["road_mask"]).to(_device).float().squeeze()

    # optional downsampling for faster Eikonal
    _scale = 1.0
    if _args.downsample > 0 and max(_road_prob.shape) > _args.downsample:
        _scale = _args.downsample / max(_road_prob.shape)
        _nH = max(1, int(_road_prob.shape[0] * _scale))
        _nW = max(1, int(_road_prob.shape[1] * _scale))
        _road_prob = F.interpolate(
            _road_prob.unsqueeze(0).unsqueeze(0),
            size=(_nH, _nW), mode="bilinear", align_corners=False,
        ).squeeze()
        print(f"  Downsampled road_prob → {_nH}×{_nW}  (scale={_scale:.4f})")
    _H_ds, _W_ds = _road_prob.shape
    print(f"  road_prob: {_H_ds}×{_W_ds}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3 — Load road-network nodes from NPZ
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== STEP 3: Load nodes from NPZ ===")
    # snap nodes to road_prob at original resolution, then scale
    _rp_for_snap = _road_prob if _scale == 1.0 else F.interpolate(
        _road_prob.unsqueeze(0).unsqueeze(0),
        size=(_H_orig, _W_orig), mode="bilinear", align_corners=False,
    ).squeeze()
    _nodes_yx_orig, _info = load_nodes_from_npz(
        _args.image_path, _rp_for_snap,
        p_count=20, snap=True, snap_threshold=0.30, snap_win=10, verbose=True,
    )
    if _nodes_yx_orig.shape[0] < 2:
        raise RuntimeError("Not enough nodes in NPZ (<2). Check the image path / dataset.")

    _nodes_yx = _nodes_yx_orig
    if _scale != 1.0:
        _nodes_yx = (_nodes_yx_orig.float() * _scale).long()
        _nodes_yx[:, 0].clamp_(0, _H_ds - 1)
        _nodes_yx[:, 1].clamp_(0, _W_ds - 1)
    print(f"  Nodes loaded: {_nodes_yx.shape[0]}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4 — Pick anchor + K nearest neighbours
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== STEP 4: Pick anchor + KNN ===")
    _anchor_idx, _nn_idx = pick_anchor_and_knn(
        _nodes_yx, k=_args.k, anchor_idx=_args.anchor_idx
    )
    _anchor  = _nodes_yx[_anchor_idx]   # [2]
    _targets = _nodes_yx[_nn_idx]       # [K, 2]
    print(f"  Anchor  idx={_anchor_idx}  yx={_anchor.tolist()}")
    print(f"  Targets idx={_nn_idx.tolist()}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5 — Compute Eikonal distances (one-to-many, single ROI solve)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n=== STEP 5: Eikonal solve (anchor → {_args.k} targets) ===")
    _eik_cfg = EikonalConfig(
        n_iters=_args.eik_iters, tau_min=0.03, tau_branch=0.05,
        tau_update=0.03, use_redblack=True, monotone=True,
    )
    import time as _time
    _t0 = _time.time()
    _dist_k = _model.compute_knn_distances(
        _road_prob, _anchor, _targets,
        eik_cfg=_eik_cfg, margin=_args.margin,
    )
    print(f"  Solved in {_time.time()-_t0:.3f}s")
    print(f"  T values: {np.round(_dist_k.cpu().numpy(), 1).tolist()}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6 — Ranking metrics vs GT road distance and Euclidean distance
    # ─────────────────────────────────────────────────────────────────────
    print("\n=== STEP 6: Ranking evaluation ===")
    from utils import trace_path_from_distance_field_common, calc_path_len_vec

    # re-run to get T_patch for backtrace (same ROI logic)
    _span_max = int(torch.max(torch.abs(_targets.float() - _anchor.float())).item())
    _half  = _span_max + _args.margin
    _P     = max(2 * _half + 1, 512)
    _y0    = int(_anchor[0].item()) - _half
    _x0    = int(_anchor[1].item()) - _half
    _y1    = _y0 + _P;  _x1 = _x0 + _P
    _yy0   = max(_y0, 0); _xx0 = max(_x0, 0)
    _yy1   = min(_y1, _H_ds); _xx1 = min(_x1, _W_ds)
    _patch = torch.zeros(_P, _P, device=_device, dtype=_road_prob.dtype)
    _patch[_yy0-_y0:_yy1-_y0, _xx0-_x0:_xx1-_x0] = _road_prob[_yy0:_yy1, _xx0:_xx1]
    _cost  = _model._road_prob_to_cost(_patch)
    _src_ry = max(0, min(int(_anchor[0].item()) - _y0, _P-1))
    _src_rx = max(0, min(int(_anchor[1].item()) - _x0, _P-1))
    _smask = torch.zeros(1, _P, _P, dtype=torch.bool, device=_device)
    _smask[0, _src_ry, _src_rx] = True
    _cfg_eff = EikonalConfig(n_iters=max(_args.eik_iters, _P),
                             tau_min=0.03, tau_branch=0.05, tau_update=0.03,
                             use_redblack=True, monotone=True)
    _T_patch = eikonal_soft_sweeping(_cost.unsqueeze(0), _smask, _cfg_eff)
    if _T_patch.dim() == 3:
        _T_patch = _T_patch[0]

    _src_rel_t = torch.tensor([_src_ry, _src_rx], device=_device, dtype=torch.long)
    _meta_px   = int(_info["meta_sat_height_px"])
    _pscale    = 1.0 / _scale if _scale != 1.0 else 1.0
    _nn_np     = _nn_idx.cpu().numpy()
    _K         = int(_targets.shape[0])
    _pred_pix  = np.full(_K, np.nan)
    _pred_norm = np.full(_K, np.nan)
    _tvals     = _dist_k.cpu().numpy()

    for _t in range(_K):
        _tval = float(_dist_k[_t].item())
        if not np.isfinite(_tval) or _tval >= 1e5:
            continue
        _try = int(_targets[_t, 0].item()) - _y0
        _trx = int(_targets[_t, 1].item()) - _x0
        if not (0 <= _try < _P and 0 <= _trx < _P):
            continue
        _tgt_rel_t = torch.tensor(
            [max(0, min(_try, _P-1)), max(0, min(_trx, _P-1))],
            device=_device, dtype=torch.long,
        )
        _pp = trace_path_from_distance_field_common(
            _T_patch, _src_rel_t, _tgt_rel_t, _device, diag=True, max_steps=200_000,
        )
        if len(_pp) < 2:
            continue
        _pg   = [(_p[0] + _y0, _p[1] + _x0) for _p in _pp]
        _plen = calc_path_len_vec(_pg) * _pscale
        _pred_pix[_t]  = _plen
        _pred_norm[_t] = _plen / max(_meta_px, 1)

    _case  = _info["case_idx"]
    _gt    = np.asarray(_info["undirected_dist_norm"][_case][_anchor_idx, _nn_np], dtype=np.float64)
    _euc   = np.asarray(_info["euclidean_dist_norm"][_case][_anchor_idx, _nn_np],  dtype=np.float64)
    _mask  = mask_finite(_pred_norm, _gt, _euc)
    _mets  = compute_ranking_metrics(_nn_idx, _pred_norm, _gt, _euc, _mask)

    print("\n" + "=" * 62)
    print(f"  Anchor={_anchor_idx}   k={_K}   scale={_scale:.4f}")
    print(f"\n  {'Metric':<18} {'SAMRoute (path len)':<22} {'Euclidean'}")
    print("  " + "-" * 54)
    print(f"  {'Spearman':<18} {_mets['s_pred']:>20.4f}   {_mets['s_euc']:.4f}")
    print(f"  {'Kendall':<18} {_mets['k_pred']:>20.4f}   {_mets['k_euc']:.4f}")
    print(f"  {'Pairwise Acc':<18} {_mets['pw_pred']:>20.4f}   {_mets['pw_euc']:.4f}")
    print(f"  {'Top-1 (GT)':<18} {'node '+str(_mets['top1_gt']):>20}   node {_mets['top1_euc']}")
    print(f"  {'Top-1 (pred)':<18} {'node '+str(_mets['top1_pred']):>20}")
    print("  " + "-" * 54)
    print("\n  [Table sorted by GT ascending]")
    print(" ", fmt_rank_table(_nn_np, _pred_norm, _gt, _euc, _tvals,
                              np.isfinite(_pred_norm)).replace("\n", "\n  "))

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7 — Visualize shortest paths on the satellite image
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n=== STEP 7: Visualize paths → {_args.save_path} ===")
    _rgb_ds = F.interpolate(_rgb, size=(_H_ds, _W_ds), mode="bilinear", align_corners=False)
    _model.visualize_paths(
        rgb=_rgb_ds,
        road_prob=_road_prob,
        src_yx=_anchor,
        tgt_yx=_targets,
        labels=[f"node {int(i)}  {_pred_pix[t]:.0f}px" if np.isfinite(_pred_pix[t])
                else f"node {int(i)}" for t, i in enumerate(_nn_np)],
        save_path=_args.save_path,
        eik_cfg=_eik_cfg,
        margin=_args.margin,
        show_T_field=True,
        show_road_prob=True,
        figsize=(16, 16),
        dpi=180,
    )
    print("\nDone.")
