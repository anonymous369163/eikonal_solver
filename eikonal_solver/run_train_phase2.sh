#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 2 Training — Road Segmentation + Distance Supervision
#
# Fine-tunes from the best Phase 1 dual-target checkpoint, adding Eikonal
# distance loss to improve Spearman ranking.
#
# Usage:
#   bash eikonal_solver/run_train_phase2.sh
#   bash eikonal_solver/run_train_phase2.sh --epochs 50
#   bash eikonal_solver/run_train_phase2.sh --ckpt_path path/to/other.ckpt
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# ---- default arguments ----
CONFIG="${SCRIPT_DIR}/config_phase2_dual_target.yaml"
DATA_ROOT="Gen_dataset_V2/Gen_dataset"
OUTPUT_DIR="training_outputs/phase2_r8_dual_target"
EPOCHS=100
BATCH_SIZE=""
WORKERS=""
SAMPLES_PER_REGION=50
PRECISION="16-mixed"
GPUS=1
RESUME=""
CKPT_PATH=""
ROAD_DILATION_RADIUS=8

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
        --ckpt_path)             CKPT_PATH="$2";             shift 2 ;;
        --road_dilation_radius)  ROAD_DILATION_RADIUS="$2";  shift 2 ;;
        *) echo "[run_train_phase2] Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- print config ----
echo "======================================================"
echo "  SAMRoute Phase-2 Training (seg + distance supervision)"
echo "======================================================"
echo "  Config            : ${CONFIG}"
echo "  Data root         : ${DATA_ROOT}"
echo "  Output dir        : ${OUTPUT_DIR}"
echo "  Epochs            : ${EPOCHS}"
echo "  Batch size        : ${BATCH_SIZE:-<from config>}"
echo "  Workers           : ${WORKERS:-<from config>}"
echo "  Samples/region    : ${SAMPLES_PER_REGION}"
echo "  Precision         : ${PRECISION}"
echo "  GPUs              : ${GPUS}"
echo "  Resume ckpt       : ${RESUME:-<none>}"
echo "  Ckpt path (init)  : ${CKPT_PATH:-<from config PRETRAINED_CKPT>}"
echo "  Road dilation r   : ${ROAD_DILATION_RADIUS}"
echo "======================================================"

mkdir -p "${OUTPUT_DIR}"

# ---- build optional flags ----
RESUME_FLAG=""
if [[ -n "${RESUME}" ]]; then
    RESUME_FLAG="--resume ${RESUME}"
fi

CKPT_FLAG=""
if [[ -n "${CKPT_PATH}" ]]; then
    CKPT_FLAG="--ckpt_path ${CKPT_PATH}"
fi

BATCH_FLAG=""
if [[ -n "${BATCH_SIZE}" ]]; then
    BATCH_FLAG="--batch_size ${BATCH_SIZE}"
fi

WORKERS_FLAG=""
if [[ -n "${WORKERS}" ]]; then
    WORKERS_FLAG="--workers ${WORKERS}"
fi

# ---- launch training ----
python eikonal_solver/train_sam_route.py \
    --config                "${CONFIG}" \
    --data_root             "${DATA_ROOT}" \
    --output_dir            "${OUTPUT_DIR}" \
    --gpus                  "${GPUS}" \
    --epochs                "${EPOCHS}" \
    --samples_per_region    "${SAMPLES_PER_REGION}" \
    --precision             "${PRECISION}" \
    --road_dilation_radius  "${ROAD_DILATION_RADIUS}" \
    --include_dist \
    ${BATCH_FLAG} \
    ${WORKERS_FLAG} \
    ${CKPT_FLAG} \
    ${RESUME_FLAG}

echo ""
echo "[run_train_phase2] Done. Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
echo "  TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/tensorboard --port 6006"
