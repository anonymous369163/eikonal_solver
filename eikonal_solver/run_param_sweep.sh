#!/usr/bin/env bash
#
# 单轮训练：使用默认 ROAD_POS_WEIGHT / ROAD_DICE_WEIGHT 训练一轮，并运行推理。
# 可修改顶部参数或启用 ENCODER_LORA_MODE=1 进行 LoRA 微调。
#
# Usage:
#   cd /path/to/MMDataset
#   ./eikonal_solver/run_param_sweep.sh              # 使用默认 GPU
#   ./eikonal_solver/run_param_sweep.sh 2            # 指定 GPU 2
#   GPU_ID=3 ./eikonal_solver/run_param_sweep.sh    # 或通过环境变量
#

set -e

# Project root (parent of eikonal_solver)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# GPU 选择：留空则自动使用默认 GPU；传入数字则用该 GPU（如 2 或 0,1）
# 可通过命令行参数或环境变量 GPU_ID 指定
if [[ -n "$1" && "$1" =~ ^[0-9,]+$ ]]; then
  GPU_ID="$1"
elif [[ -z "${GPU_ID}" ]]; then
  GPU_ID=""
fi
if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  echo "使用 GPU: ${GPU_ID}"
fi

# 单轮默认参数（跑一轮即可）
ROAD_POS_WEIGHT=15.0
ROAD_DICE_WEIGHT=0.5
ROAD_THIN_BOOST=6.0

# LoRA 模式：设为 1 启用 LoRA encoder 微调，0 则只训练 Decoder
ENCODER_LORA_MODE=0
LORA_RANK=4
# LoRA 时 batch 较小（需在线跑 Encoder），非 LoRA 可用较大 batch
BATCH_SIZE=32

SINGLE_REGION_DIR="Gen_dataset_V2/Gen_dataset/19.940688_110.276704/00_20.021516_110.190699_3000.0"
CKPT_DIR="training_outputs/finetune_demo/checkpoints"
TARGET_TIF="${SINGLE_REGION_DIR}/crop_20.021516_110.190699_3000.0_z16.tif"

if [[ "${ENCODER_LORA_MODE}" -eq 1 ]]; then
  LORA_FLAG="--encoder_lora --lora_rank ${LORA_RANK}"
  BATCH_SIZE=8
  RUN_PREFIX="lora_"
  echo "LoRA 模式: BATCH_SIZE=${BATCH_SIZE}, rank=${LORA_RANK}"
else
  LORA_FLAG=""
  RUN_PREFIX=""
fi

echo "=============================================="
echo "单轮训练: ROAD_POS_WEIGHT=${ROAD_POS_WEIGHT} ROAD_DICE_WEIGHT=${ROAD_DICE_WEIGHT} ROAD_THIN_BOOST=${ROAD_THIN_BOOST} [LoRA=${ENCODER_LORA_MODE}]"
echo "=============================================="

RUN_NAME_BASE="${RUN_PREFIX}pos${ROAD_POS_WEIGHT}_dice${ROAD_DICE_WEIGHT}_thin${ROAD_THIN_BOOST}"
# 若已存在同名 checkpoint，则使用 v2、v3... 避免覆盖
RUN_NAME="${RUN_NAME_BASE}"
v=2
while [[ -f "${CKPT_DIR}/best_${RUN_NAME}.ckpt" ]]; do
  RUN_NAME="${RUN_NAME_BASE}_v${v}"
  echo "已存在 checkpoint，本次使用 v${v} 命名: best_${RUN_NAME}.ckpt"
  v=$((v + 1))
done

python eikonal_solver/finetune_demo.py \
  --single_region_dir "${SINGLE_REGION_DIR}" \
  --epochs 100 \
  --batch_size "${BATCH_SIZE}" \
  --workers 4 \
  --road_pos_weight "${ROAD_POS_WEIGHT}" \
  --road_dice_weight "${ROAD_DICE_WEIGHT}" \
  --road_thin_boost "${ROAD_THIN_BOOST}" \
  --run_name "${RUN_NAME}" \
  ${LORA_FLAG}

CKPT_PATH="${CKPT_DIR}/best_${RUN_NAME}.ckpt"
OUTPUT_PNG="tif_inference_result_${RUN_NAME}.png"

if [[ -f "${CKPT_PATH}" ]]; then
  echo ""
  echo ">>> Inference: ${CKPT_PATH} -> ${OUTPUT_PNG}"
  python eikonal_solver/test_inference.py \
    --ckpt "${CKPT_PATH}" \
    --output "${OUTPUT_PNG}" \
    --tif "${TARGET_TIF}" \
    ${LORA_FLAG}
else
  echo "WARNING: Checkpoint not found: ${CKPT_PATH}, skipping inference."
fi

echo ""
echo "=============================================="
echo "训练完成。Checkpoint: ${CKPT_PATH}"
echo "结果图: ${OUTPUT_PNG}"
echo "=============================================="
