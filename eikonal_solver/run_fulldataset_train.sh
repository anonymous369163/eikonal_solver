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
# 5) Seg + Dist joint + LoRA encoder (both losses + encoder adapts)
# ------------------------------------------------------------------
seg_dist_lora)
    OUTPUT_DIR="training_outputs/fulldataset_seg_dist_lora"
    RUN_NAME="${RUN_NAME:-seg_dist_lora_full}"
    python finetune_demo_updated.py \
        --data_root              "$DATA_ROOT" \
        --pretrained_ckpt        "$PRETRAINED" \
        --epochs                 "$EPOCHS" \
        --batch_size             16 \
        --devices                "$NGPUS" \
        --workers                "$WORKERS" \
        --val_fraction           "$VAL_FRAC" \
        --samples_per_region     "$SAMPLES" \
        --road_dilation_radius   "$DILATION" \
        --output_dir             "$OUTPUT_DIR" \
        --run_name               "$RUN_NAME" \
        --lr                     2e-4 \
        --encoder_lora \
        --lora_rank              4 \
        --lambda_dist            0.2 \
        --dist_supervision       gtmask_random \
        --dist_k_targets         8 \
        --dist_src_onroad_p      0.9 \
        --dist_tgt_onroad_p      0.9 \
        --dist_min_euclid_px     64 \
        --dist_teacher_alpha     20.0 \
        --dist_teacher_gamma     2.0 \
        --dist_teacher_mask      thick \
        --dist_gt_cap_factor     0.85 \
        --dist_detour_cap        12.0 \
        --multigrid \
        --eik_iters              40 \
        --eik_downsample         16 \
        --mg_factor              4
    ;;

*)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {seg_only|seg_dist_npz|seg_dist_gtmask|seg_lora|seg_dist_lora} [run_name]"
    exit 1
    ;;
esac

echo ""
echo "Training complete. Check outputs in: $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/tensorboard --port 6006"
