#!/usr/bin/env bash
# ===========================================================================
#  SAMRoute Full-Dataset Fine-tuning Script
#  Dataset: Gen_dataset_V2/Gen_dataset (31 cities, 433 sub-regions)
#  Hardware: 2× RTX 4090, conda env: satdino
# ===========================================================================
set -euo pipefail

# ---------- Environment ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source activate satdino 2>/dev/null || conda activate satdino 2>/dev/null || true

DATA_ROOT="Gen_dataset_V2/Gen_dataset"
PRETRAINED="checkpoints/cityscale_vitb_512_e10.ckpt"
NGPUS=1
WORKERS=4
BATCH=32          # per-GPU (effective = BATCH * NGPUS)
VAL_FRAC=0.1
SAMPLES=50        # samples per region per epoch (~21650 total)
EPOCHS=50
DILATION=3        # roadnet_normalized_r3.png

# Select mode: seg_only | seg_dist_npz | seg_dist_gtmask
MODE="${1:-seg_only}"
RUN_NAME="${2:-}"

echo "============================================="
echo " Mode: ${MODE}"
echo " GPUs: ${NGPUS}  Batch/GPU: ${BATCH}  Workers: ${WORKERS}"
echo " Epochs: ${EPOCHS}  Samples/region: ${SAMPLES}"
echo "============================================="

# ---------- Common args ----------
COMMON_ARGS=(
    --data_root              "$DATA_ROOT"
    --pretrained_ckpt        "$PRETRAINED"
    --epochs                 "$EPOCHS"
    --batch_size             "$BATCH"
    --devices                "$NGPUS"
    --workers                "$WORKERS"
    --val_fraction           "$VAL_FRAC"
    --samples_per_region     "$SAMPLES"
    --road_dilation_radius   "$DILATION"
    --use_cached_features
)

case "$MODE" in
# ------------------------------------------------------------------
# 1) Seg-only baseline (no distance loss)
# ------------------------------------------------------------------
seg_only)
    OUTPUT_DIR="training_outputs/fulldataset_seg_only"
    RUN_NAME="${RUN_NAME:-seg_only_full}"
    python finetune_demo_updated.py \
        "${COMMON_ARGS[@]}" \
        --output_dir   "$OUTPUT_DIR" \
        --run_name     "$RUN_NAME" \
        --lr           5e-4
    ;;

# ------------------------------------------------------------------
# 2) Seg + Dist joint (NPZ supervision — uses pre-computed distance files)
# ------------------------------------------------------------------
seg_dist_npz)
    OUTPUT_DIR="training_outputs/fulldataset_seg_dist_npz"
    RUN_NAME="${RUN_NAME:-seg_dist_npz_full}"
    python finetune_demo_updated.py \
        "${COMMON_ARGS[@]}" \
        --output_dir       "$OUTPUT_DIR" \
        --run_name         "$RUN_NAME" \
        --lr               5e-4 \
        --lambda_dist      0.2 \
        --dist_supervision npz \
        --multigrid \
        --eik_iters        40 \
        --eik_downsample   16 \
        --mg_factor        4
    ;;

# ------------------------------------------------------------------
# 3) Seg + Dist joint (GT-mask random teacher — online Eikonal on GT road mask)
# ------------------------------------------------------------------
seg_dist_gtmask)
    OUTPUT_DIR="training_outputs/fulldataset_seg_dist_gtmask"
    RUN_NAME="${RUN_NAME:-seg_dist_gtmask_full}"
    python finetune_demo_updated.py \
        "${COMMON_ARGS[@]}" \
        --output_dir           "$OUTPUT_DIR" \
        --run_name             "$RUN_NAME" \
        --lr                   5e-4 \
        --lambda_dist          0.2 \
        --dist_supervision     gtmask_random \
        --dist_k_targets       8 \
        --dist_src_onroad_p    0.9 \
        --dist_tgt_onroad_p    0.9 \
        --dist_min_euclid_px   64 \
        --dist_teacher_alpha   20.0 \
        --dist_teacher_gamma   2.0 \
        --dist_teacher_mask    thick \
        --dist_gt_cap_factor   0.85 \
        --dist_detour_cap      12.0 \
        --multigrid \
        --eik_iters            40 \
        --eik_downsample       16 \
        --mg_factor            4
    ;;

# ------------------------------------------------------------------
# 4) LoRA encoder fine-tuning + seg (encoder adapts, stronger but slower)
# ------------------------------------------------------------------
seg_lora)
    OUTPUT_DIR="training_outputs/fulldataset_seg_lora"
    RUN_NAME="${RUN_NAME:-seg_lora_full}"
    python finetune_demo_updated.py \
        "${COMMON_ARGS[@]}" \
        --output_dir   "$OUTPUT_DIR" \
        --run_name     "$RUN_NAME" \
        --lr           2e-4 \
        --encoder_lora \
        --lora_rank    4
    ;;

# ------------------------------------------------------------------
# 5) Seg + Dist joint + LoRA encoder (stabilised)
#    Grad clip + cosine LR + frozen cost params
#    LR: cosine annealing with 3-epoch warmup, 2e-4 -> 1e-6
# ------------------------------------------------------------------
seg_dist_lora)
    V2_CKPT="training_outputs/fulldataset_seg_dist_lora_v2/checkpoints/best_seg_dist_lora_v2.ckpt"
    OUTPUT_DIR="training_outputs/fulldataset_seg_dist_lora_v3"
    RUN_NAME="${RUN_NAME:-seg_dist_lora_v3}"
    python finetune_demo_updated.py \
        --data_root              "$DATA_ROOT" \
        --pretrained_ckpt        "$V2_CKPT" \
        --epochs                 20 \
        --batch_size             16 \
        --devices                "$NGPUS" \
        --workers                "$WORKERS" \
        --val_fraction           "$VAL_FRAC" \
        --samples_per_region     "$SAMPLES" \
        --road_dilation_radius   "$DILATION" \
        --output_dir             "$OUTPUT_DIR" \
        --run_name               "$RUN_NAME" \
        --lr                     2e-4 \
        --lr_scheduler           cosine \
        --lr_warmup_epochs       3 \
        --lr_min                 1e-6 \
        --encoder_lora \
        --lora_rank              4 \
        --lambda_dist            0.05 \
        --dist_supervision       gtmask_random \
        --dist_k_targets         4 \
        --dist_src_onroad_p      0.9 \
        --dist_tgt_onroad_p      0.9 \
        --dist_min_euclid_px     64 \
        --dist_max_euclid_px     192 \
        --dist_teacher_alpha     20.0 \
        --dist_teacher_gamma     2.0 \
        --dist_teacher_mask      thick \
        --dist_gt_cap_factor     0.85 \
        --dist_detour_cap        12.0 \
        --multigrid \
        --eik_iters              40 \
        --eik_downsample         16 \
        --mg_factor              4 \
        --ckpt_chunk             20 \
        --grad_clip              1.0 \
        --freeze_cost_params \
        --road_pos_weight        15 \
        --road_dice_weight       0.5 \
        --road_thin_boost        4
    ;;

# ------------------------------------------------------------------
# 6) Focal Loss ablation — single-image quick test on GPU0
#    Base: seg_dist_lora config, warm-start from v2 best checkpoint
#    Usage: CUDA_VISIBLE_DEVICES=0 ./run_fulldataset_train.sh ablation_focal [run_name]
# ------------------------------------------------------------------
ablation_focal)
    HAINAN_REGION="Gen_dataset_V2/Gen_dataset/19.940688_110.276704/00_20.021516_110.190699_3000.0"
    V2_CKPT="training_outputs/fulldataset_seg_dist_lora_v2/checkpoints/best_seg_dist_lora_v2.ckpt"
    OUTPUT_DIR="training_outputs/ablation_focal"
    RUN_NAME="${RUN_NAME:-ablation_focal}"
    python finetune_demo_updated.py \
        --single_region_dir      "$HAINAN_REGION" \
        --pretrained_ckpt        "$V2_CKPT" \
        --epochs                 10 \
        --batch_size             16 \
        --devices                1 \
        --workers                "$WORKERS" \
        --val_fraction           "$VAL_FRAC" \
        --road_dilation_radius   "$DILATION" \
        --output_dir             "$OUTPUT_DIR" \
        --run_name               "$RUN_NAME" \
        --lr                     2e-4 \
        --lr_scheduler           cosine \
        --lr_warmup_epochs       1 \
        --lr_min                 1e-6 \
        --encoder_lora \
        --lora_rank              4 \
        --lambda_dist            0.05 \
        --dist_supervision       gtmask_random \
        --dist_k_targets         4 \
        --dist_src_onroad_p      0.9 \
        --dist_tgt_onroad_p      0.9 \
        --dist_min_euclid_px     64 \
        --dist_max_euclid_px     192 \
        --dist_teacher_alpha     20.0 \
        --dist_teacher_gamma     2.0 \
        --dist_teacher_mask      thick \
        --dist_gt_cap_factor     0.85 \
        --dist_detour_cap        12.0 \
        --multigrid \
        --eik_iters              40 \
        --eik_downsample         16 \
        --mg_factor              4 \
        --ckpt_chunk             20 \
        --grad_clip              1.0 \
        --freeze_cost_params \
        --road_focal_loss \
        --road_focal_alpha       0.75 \
        --road_focal_gamma       2.0
    ;;

# ------------------------------------------------------------------
# 7) v4: same as v2, but with ResidualCostNet CNN enabled
#    Pretrained: original SAM-Road checkpoint (same starting point as v2)
#    cost_net: 3-layer CNN predicting bounded log-cost residual
#    freeze_cost_params: still frozen (alpha/gamma/gate fixed, same as v2)
#    cost_net is independently trainable (zero-init last layer => starts as identity)
# ------------------------------------------------------------------
seg_dist_lora_v4)
    OUTPUT_DIR="training_outputs/fulldataset_seg_dist_lora_v4"
    RUN_NAME="${RUN_NAME:-seg_dist_lora_v4}"
    python finetune_demo_updated.py \
        --data_root              "$DATA_ROOT" \
        --pretrained_ckpt        "$PRETRAINED" \
        --epochs                 20 \
        --batch_size             16 \
        --devices                "$NGPUS" \
        --workers                "$WORKERS" \
        --val_fraction           "$VAL_FRAC" \
        --samples_per_region     "$SAMPLES" \
        --road_dilation_radius   "$DILATION" \
        --output_dir             "$OUTPUT_DIR" \
        --run_name               "$RUN_NAME" \
        --lr                     2e-4 \
        --lr_scheduler           cosine \
        --lr_warmup_epochs       3 \
        --lr_min                 1e-6 \
        --encoder_lora \
        --lora_rank              4 \
        --road_pos_weight        5.0 \
        --road_dice_weight       0.8 \
        --road_dual_target \
        --road_thin_boost        10.0 \
        --lambda_dist            0.05 \
        --dist_supervision       gtmask_random \
        --multigrid \
        --eik_iters              40 \
        --eik_downsample         16 \
        --mg_factor              4 \
        --grad_clip              1.0 \
        --freeze_cost_params \
        --cost_net \
        --cost_net_ch            8 \
        --cost_net_delta_scale   0.75 \
        --lambda_cost_reg        0.01 \
        --lambda_cost_tv         0.005
    ;;

# ------------------------------------------------------------------
# 8) v5: same as v4, but with U-Net-like multiscale ResidualCostNet
#    cost_net_arch=multiscale: 2-stage down/up FPN (20360 params vs basic 672)
#    Captures multi-scale spatial cost patterns (road width, intersections)
# ------------------------------------------------------------------
seg_dist_lora_v5)
    OUTPUT_DIR="training_outputs/fulldataset_seg_dist_lora_v5"
    RUN_NAME="${RUN_NAME:-seg_dist_lora_v5}"
    python finetune_demo_updated.py \
        --data_root              "$DATA_ROOT" \
        --pretrained_ckpt        "$PRETRAINED" \
        --epochs                 20 \
        --batch_size             16 \
        --devices                "$NGPUS" \
        --workers                "$WORKERS" \
        --val_fraction           "$VAL_FRAC" \
        --samples_per_region     "$SAMPLES" \
        --road_dilation_radius   "$DILATION" \
        --output_dir             "$OUTPUT_DIR" \
        --run_name               "$RUN_NAME" \
        --lr                     2e-4 \
        --lr_scheduler           cosine \
        --lr_warmup_epochs       3 \
        --lr_min                 1e-6 \
        --encoder_lora \
        --lora_rank              4 \
        --road_pos_weight        5.0 \
        --road_dice_weight       0.8 \
        --road_dual_target \
        --road_thin_boost        10.0 \
        --lambda_dist            0.05 \
        --dist_supervision       gtmask_random \
        --multigrid \
        --eik_iters              40 \
        --eik_downsample         16 \
        --mg_factor              4 \
        --grad_clip              1.0 \
        --freeze_cost_params \
        --cost_net \
        --cost_net_ch            8 \
        --cost_net_arch          multiscale \
        --cost_net_delta_scale   0.75 \
        --lambda_cost_reg        0.01 \
        --lambda_cost_tv         0.005
    ;;

*)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {seg_only|seg_dist_npz|seg_dist_gtmask|seg_lora|seg_dist_lora|seg_dist_lora_v4|seg_dist_lora_v5|ablation_focal} [run_name]"
    exit 1
    ;;
esac

echo ""
echo "Training complete. Check outputs in: $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/tensorboard --port 6006"
