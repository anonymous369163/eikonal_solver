#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 1 Training — Road Segmentation Only (no LoRA, frozen encoder)
#
# Usage:
#   bash eikonal_solver/run_train_phase1.sh              # default 200 epochs
#   bash eikonal_solver/run_train_phase1.sh --epochs 50  # quick smoke-test
#   bash eikonal_solver/run_train_phase1.sh --resume training_outputs/phase1_seg/checkpoints/last.ckpt
#
# Spearman evaluation (optional, logged to TensorBoard):
#   bash eikonal_solver/run_train_phase1.sh \
#       --eval_every_n_epochs 10 \
#       --eval_image_path Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/crop_34.350697_108.914569_3000.0_z16.tif
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# ---- default arguments (can be overridden on the command line) ----
CONFIG="${SCRIPT_DIR}/config_sam_route_demo.yaml"
DATA_ROOT="Gen_dataset_V2/Gen_dataset"
OUTPUT_DIR="training_outputs/phase1_seg"
EPOCHS=200
BATCH_SIZE=64
WORKERS=24
SAMPLES_PER_REGION=50
PRECISION="16-mixed"
GPUS=1
RESUME=""
ROAD_DILATION_RADIUS=0

# ---- Spearman evaluation (0 = disabled) ----
EVAL_EVERY_N_EPOCHS=0
# Default: first region in dataset (edit to a region that has a distance npz)
EVAL_IMAGE_PATH="Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/crop_34.350697_108.914569_3000.0_z16.tif"
EVAL_K=19
EVAL_MIN_IN_PATCH=3

# ---- parse optional overrides ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)                CONFIG="$2";                shift 2 ;;
        --data_root)             DATA_ROOT="$2";             shift 2 ;;
        --output_dir)            OUTPUT_DIR="$2";            shift 2 ;;
        --epochs)                EPOCHS="$2";                shift 2 ;;
        --batch_size)            BATCH_SIZE="$2";            shift 2 ;;
        --workers)               WORKERS="$2";               shift 2 ;;
        --samples_per_region)    SAMPLES_PER_REGION="$2";    shift 2 ;;
        --precision)             PRECISION="$2";             shift 2 ;;
        --gpus)                  GPUS="$2";                  shift 2 ;;
        --resume)                RESUME="$2";                shift 2 ;;
        --road_dilation_radius)  ROAD_DILATION_RADIUS="$2";  shift 2 ;;
        --eval_every_n_epochs)   EVAL_EVERY_N_EPOCHS="$2";   shift 2 ;;
        --eval_image_path)       EVAL_IMAGE_PATH="$2";       shift 2 ;;
        --eval_k)                EVAL_K="$2";                shift 2 ;;
        --eval_min_in_patch)     EVAL_MIN_IN_PATCH="$2";     shift 2 ;;
        *) echo "[run_train_phase1] Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- print config ----
echo "======================================================"
echo "  SAMRoute Phase-1 Training (segmentation only, no LoRA)"
echo "======================================================"
echo "  Config            : ${CONFIG}"
echo "  Data root         : ${DATA_ROOT}"
echo "  Output dir        : ${OUTPUT_DIR}"
echo "  Epochs            : ${EPOCHS}"
echo "  Batch size        : ${BATCH_SIZE}"
echo "  Workers           : ${WORKERS}"
echo "  Samples/region    : ${SAMPLES_PER_REGION}"
echo "  Precision         : ${PRECISION}"
echo "  GPUs              : ${GPUS}"
echo "  Resume ckpt       : ${RESUME:-<none>}"
echo "  Road dilation r   : ${ROAD_DILATION_RADIUS} (0=off)"
echo "  Spearman eval     : every ${EVAL_EVERY_N_EPOCHS} epochs (0=off)"
if [[ "${EVAL_EVERY_N_EPOCHS}" -gt 0 ]]; then
echo "  Eval image        : ${EVAL_IMAGE_PATH}"
fi
echo "======================================================"

mkdir -p "${OUTPUT_DIR}"

# ---- build optional flags ----
RESUME_FLAG=""
if [[ -n "${RESUME}" ]]; then
    RESUME_FLAG="--resume ${RESUME}"
fi

EVAL_FLAGS=""
if [[ "${EVAL_EVERY_N_EPOCHS}" -gt 0 ]]; then
    EVAL_FLAGS="--eval_every_n_epochs ${EVAL_EVERY_N_EPOCHS} \
    --eval_image_path ${EVAL_IMAGE_PATH} \
    --eval_k ${EVAL_K} \
    --eval_min_in_patch ${EVAL_MIN_IN_PATCH}"
fi

# ---- launch training ----
python eikonal_solver/train_sam_route.py \
    --config                "${CONFIG}" \
    --data_root             "${DATA_ROOT}" \
    --output_dir            "${OUTPUT_DIR}" \
    --gpus                  "${GPUS}" \
    --epochs                "${EPOCHS}" \
    --batch_size            "${BATCH_SIZE}" \
    --workers               "${WORKERS}" \
    --samples_per_region    "${SAMPLES_PER_REGION}" \
    --precision             "${PRECISION}" \
    --road_dilation_radius  "${ROAD_DILATION_RADIUS}" \
    ${RESUME_FLAG} \
    ${EVAL_FLAGS}

echo ""
echo "[run_train_phase1] Done. Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
echo "  TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/tensorboard --port 6006"
