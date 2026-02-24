#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Unified SAMRoute Training Script
#
# Usage:
#   bash eikonal_solver/run_train.sh --phase 1 --epochs 200
#   bash eikonal_solver/run_train.sh --phase 2 --epochs 100
#   bash eikonal_solver/run_train.sh --phase 1 --eval_every_n_epochs 10
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- defaults ----
PHASE=1
EPOCHS=""
BATCH_SIZE=""
WORKERS=""
CONFIG=""
DATA_ROOT="Gen_dataset_V2/Gen_dataset"
OUTPUT_DIR=""
SAMPLES_PER_REGION=50
PRECISION="16-mixed"
GPUS=1
RESUME=""
CKPT_PATH=""
ROAD_DILATION_RADIUS=5

# Spearman eval (phase 1 only, 0 = disabled)
EVAL_EVERY_N_EPOCHS=0
EVAL_IMAGE_PATH="Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/crop_34.350697_108.914569_3000.0_z16.tif"
EVAL_K=10
EVAL_MIN_IN_PATCH=3

# ---- parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)                 PHASE="$2";                 shift 2 ;;
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
        --eval_every_n_epochs)   EVAL_EVERY_N_EPOCHS="$2";   shift 2 ;;
        --eval_image_path)       EVAL_IMAGE_PATH="$2";       shift 2 ;;
        --eval_k)                EVAL_K="$2";                shift 2 ;;
        --eval_min_in_patch)     EVAL_MIN_IN_PATCH="$2";     shift 2 ;;
        *) echo "[run_train] Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- phase-specific defaults ----
if [[ "${PHASE}" == "1" ]]; then
    CONFIG="${CONFIG:-${SCRIPT_DIR}/config_phase1.yaml}"
    OUTPUT_DIR="${OUTPUT_DIR:-training_outputs/phase1_r${ROAD_DILATION_RADIUS}_seg}"
    EPOCHS="${EPOCHS:-200}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    WORKERS="${WORKERS:-24}"
elif [[ "${PHASE}" == "2" ]]; then
    CONFIG="${CONFIG:-${SCRIPT_DIR}/config_phase2.yaml}"
    OUTPUT_DIR="${OUTPUT_DIR:-training_outputs/phase2_r${ROAD_DILATION_RADIUS}_dual_target}"
    EPOCHS="${EPOCHS:-100}"
else
    echo "[run_train] ERROR: --phase must be 1 or 2 (got: ${PHASE})"
    exit 1
fi

# ---- print config ----
echo "======================================================"
echo "  SAMRoute Training — Phase ${PHASE}"
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
echo "  Ckpt path (init)  : ${CKPT_PATH:-<from config>}"
echo "  Road dilation r   : ${ROAD_DILATION_RADIUS}"
if [[ "${PHASE}" == "1" && "${EVAL_EVERY_N_EPOCHS}" -gt 0 ]]; then
echo "  Spearman eval     : every ${EVAL_EVERY_N_EPOCHS} epochs"
echo "  Eval image        : ${EVAL_IMAGE_PATH}"
fi
echo "======================================================"

mkdir -p "${OUTPUT_DIR}"

# ---- build flags ----
OPTIONAL_FLAGS=""
[[ -n "${RESUME}" ]]     && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --resume ${RESUME}"
[[ -n "${CKPT_PATH}" ]]  && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --ckpt_path ${CKPT_PATH}"
[[ -n "${BATCH_SIZE}" ]]  && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --batch_size ${BATCH_SIZE}"
[[ -n "${WORKERS}" ]]     && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --workers ${WORKERS}"

if [[ "${PHASE}" == "2" ]]; then
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --include_dist"
fi

if [[ "${PHASE}" == "1" && "${EVAL_EVERY_N_EPOCHS}" -gt 0 ]]; then
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --eval_every_n_epochs ${EVAL_EVERY_N_EPOCHS}"
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --eval_image_path ${EVAL_IMAGE_PATH}"
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --eval_k ${EVAL_K}"
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --eval_min_in_patch ${EVAL_MIN_IN_PATCH}"
fi

# ---- launch training ----
python eikonal_solver/trainer.py \
    --config                "${CONFIG}" \
    --data_root             "${DATA_ROOT}" \
    --output_dir            "${OUTPUT_DIR}" \
    --gpus                  "${GPUS}" \
    --epochs                "${EPOCHS}" \
    --samples_per_region    "${SAMPLES_PER_REGION}" \
    --precision             "${PRECISION}" \
    --road_dilation_radius  "${ROAD_DILATION_RADIUS}" \
    ${OPTIONAL_FLAGS}

echo ""
echo "[run_train] Phase ${PHASE} done. Checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "  TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/tensorboard --port 6006"
