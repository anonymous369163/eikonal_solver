#!/usr/bin/env bash
# ===========================================================================
#  seg_dist_lora Ablation Study — Single-Region Fast Verification
#  GPU: configurable via GPU_ID (default 2)
#  Region: 19.940688_110.276704/00_20.021516_110.190699_3000.0
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source activate satdino 2>/dev/null || conda activate satdino 2>/dev/null || true

# ---------- GPU ----------
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU: ${GPU_ID}"

# ---------- Paths (absolute to avoid CWD confusion) ----------
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REGION_DIR="${PROJECT_ROOT}/Gen_dataset_V2/Gen_dataset/19.940688_110.276704/00_20.021516_110.190699_3000.0"
REGION_TIF="${REGION_DIR}/crop_20.021516_110.190699_3000.0_z16.tif"
PRETRAINED="${PROJECT_ROOT}/checkpoints/cityscale_vitb_512_e10.ckpt"
OUTPUT_ROOT="${PROJECT_ROOT}/training_outputs/ablation_seg_dist_lora"
SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

# ---------- Baseline defaults (matching seg_dist_lora) ----------
EPOCHS=30
BATCH=8
WORKERS=4
LR="2e-4"
LORA_RANK=4
LAMBDA_DIST=0.2
K_TARGETS=8
SRC_ONROAD=0.9
TGT_ONROAD=0.9
MIN_EUCLID=64
TEACHER_ALPHA=20.0
TEACHER_GAMMA=2.0
TEACHER_MASK="thick"
GT_CAP_FACTOR=0.85
DETOUR_CAP=12.0
EIK_ITERS=40
EIK_DS=16
MG_FACTOR=4
DILATION=3

mkdir -p "${OUTPUT_ROOT}"

# ---------- Header ----------
if [[ ! -f "${SUMMARY_FILE}" ]]; then
    printf "%-30s | %10s | %10s | %10s | %10s | %10s | %10s | %10s\n" \
        "experiment" "pw_acc" "kendall" "top1" "top3" "top5" "rel_err" "val_loss" \
        > "${SUMMARY_FILE}"
    printf "%s\n" "$(printf '=%.0s' {1..120})" >> "${SUMMARY_FILE}"
fi

# ===========================================================================
#  run_one <exp_name> [extra training args...]
# ===========================================================================
run_one() {
    local EXP_NAME="$1"; shift
    local EXTRA_ARGS=("$@")

    local EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
    local CKPT_DIR="${EXP_DIR}/checkpoints"
    local CKPT_BEST="${CKPT_DIR}/best_${EXP_NAME}.ckpt"
    local CKPT_LAST="${CKPT_DIR}/last.ckpt"
    local EVAL_LOG="${EXP_DIR}/eval_log.txt"

    mkdir -p "${CKPT_DIR}"

    echo ""
    echo "============================================="
    echo " Experiment: ${EXP_NAME}"
    echo " Extra args: ${EXTRA_ARGS[*]:-<none>}"
    echo "============================================="

    # --- Train (skip if best checkpoint already exists) ---
    if [[ -f "${CKPT_BEST}" ]]; then
        echo "[SKIP] Checkpoint exists: ${CKPT_BEST}"
    else
        echo "[TRAIN] Starting..."
        python finetune_demo_updated.py \
            --single_region_dir  "${REGION_DIR}" \
            --pretrained_ckpt    "${PRETRAINED}" \
            --output_dir         "${EXP_DIR}" \
            --run_name           "${EXP_NAME}" \
            --epochs             ${EPOCHS} \
            --batch_size         ${BATCH} \
            --workers            ${WORKERS} \
            --lr                 ${LR} \
            --encoder_lora \
            --lora_rank          ${LORA_RANK} \
            --lambda_dist        ${LAMBDA_DIST} \
            --dist_supervision   gtmask_random \
            --dist_k_targets     ${K_TARGETS} \
            --dist_src_onroad_p  ${SRC_ONROAD} \
            --dist_tgt_onroad_p  ${TGT_ONROAD} \
            --dist_min_euclid_px ${MIN_EUCLID} \
            --dist_teacher_alpha ${TEACHER_ALPHA} \
            --dist_teacher_gamma ${TEACHER_GAMMA} \
            --dist_teacher_mask  ${TEACHER_MASK} \
            --dist_gt_cap_factor ${GT_CAP_FACTOR} \
            --dist_detour_cap    ${DETOUR_CAP} \
            --multigrid \
            --eik_iters          ${EIK_ITERS} \
            --eik_downsample     ${EIK_DS} \
            --mg_factor          ${MG_FACTOR} \
            --road_dilation_radius ${DILATION} \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${EXP_DIR}/train_log.txt"
        echo "[TRAIN] Done."
    fi

    # --- Find best checkpoint ---
    local CKPT=""
    if [[ -f "${CKPT_BEST}" ]]; then
        CKPT="${CKPT_BEST}"
    elif [[ -f "${CKPT_LAST}" ]]; then
        CKPT="${CKPT_LAST}"
    else
        echo "[ERROR] No checkpoint found for ${EXP_NAME}, skipping eval."
        printf "%-30s | %10s | %10s | %10s | %10s | %10s | %10s | %10s\n" \
            "${EXP_NAME}" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" \
            >> "${SUMMARY_FILE}"
        return
    fi

    # --- Eval (skip if eval log already exists) ---
    if [[ -f "${EVAL_LOG}" ]] && grep -q "Pairwise Ordering" "${EVAL_LOG}" 2>/dev/null; then
        echo "[SKIP] Eval log exists: ${EVAL_LOG}"
    else
        echo "[EVAL] Running eval_ranking_accuracy.py on ${CKPT}..."
        rm -f "${EVAL_LOG}"
        if python eval_ranking_accuracy.py \
            --tif      "${REGION_TIF}" \
            --ckpt     "${CKPT}" \
            --n_cases  200 \
            --p_count  20 \
            --downsample 16 \
            --eik_iters  80 \
            --gate_override  -1 \
            --alpha_override -1 \
            --gamma_override -1 \
            2>&1 | tee "${EVAL_LOG}"; then
            echo "[EVAL] Done."
        else
            echo "[EVAL] FAILED for ${EXP_NAME}, continuing..."
        fi
    fi

    # --- Parse metrics from eval log ---
    local PW_ACC="N/A"
    local KENDALL="N/A"
    local TOP1="N/A"
    local TOP3="N/A"
    local TOP5="N/A"
    local REL_ERR="N/A"
    local VAL_LOSS="N/A"

    if [[ -f "${EVAL_LOG}" ]]; then
        PW_ACC=$(grep -A1 "Pairwise Ordering" "${EVAL_LOG}" | grep "Eikonal:" | head -1 | awk '{print $2}') || true
        KENDALL=$(grep -A2 "Kendall" "${EVAL_LOG}" | grep "Eikonal:" | head -1 | awk '{print $2}') || true
        TOP1=$(grep -A2 "Top-1 Recall" "${EVAL_LOG}" | grep "Eikonal:" | head -1 | awk '{print $2}') || true
        TOP3=$(grep -A2 "Top-3 Recall" "${EVAL_LOG}" | grep "Eikonal:" | head -1 | awk '{print $2}') || true
        TOP5=$(grep -A2 "Top-5 Recall" "${EVAL_LOG}" | grep "Eikonal:" | head -1 | awk '{print $2}') || true
        REL_ERR=$(grep -A1 "Distance Relative Error" "${EVAL_LOG}" | grep "Eikonal:" | head -1 | awk '{print $2}' | sed 's/mean=//') || true
    fi

    # Try to extract val_loss from train log
    if [[ -f "${EXP_DIR}/train_log.txt" ]]; then
        VAL_LOSS=$(grep "val_loss" "${EXP_DIR}/train_log.txt" | tail -1 | grep -oP "val_loss[=:]\K[0-9.]+" || echo "N/A")
    fi

    printf "%-30s | %10s | %10s | %10s | %10s | %10s | %10s | %10s\n" \
        "${EXP_NAME}" "${PW_ACC:-N/A}" "${KENDALL:-N/A}" "${TOP1:-N/A}" \
        "${TOP3:-N/A}" "${TOP5:-N/A}" "${REL_ERR:-N/A}" "${VAL_LOSS:-N/A}" \
        >> "${SUMMARY_FILE}"

    echo "[RESULT] ${EXP_NAME}: pw=${PW_ACC:-N/A} tau=${KENDALL:-N/A} top1=${TOP1:-N/A} rel_err=${REL_ERR:-N/A}"
}

# ===========================================================================
#  Experiment Groups
# ===========================================================================

echo ""
echo "########################################"
echo "#  seg_dist_lora Ablation Study        #"
echo "#  GPU=${GPU_ID}  Epochs=${EPOCHS}     #"
echo "########################################"
echo ""

# --- Group 0: Baseline ---
run_one "baseline"

# --- Group 1: LoRA Rank ---
LORA_RANK=2
run_one "lora_rank2"
LORA_RANK=8
run_one "lora_rank8"
LORA_RANK=4  # reset

# --- Group 2: dist_k_targets ---
K_TARGETS=4
run_one "k_targets4"
K_TARGETS=16 MIN_EUCLID=32
run_one "k_targets16"
K_TARGETS=8 MIN_EUCLID=64  # reset

# --- Group 3: lambda_dist ---
LAMBDA_DIST=0.1
run_one "lambda_dist0.1"
LAMBDA_DIST=0.5
run_one "lambda_dist0.5"
LAMBDA_DIST=1.0
run_one "lambda_dist1.0"
LAMBDA_DIST=0.2  # reset

# --- Group 4: Learning Rate ---
LR="1e-4"
run_one "lr1e-4"
LR="5e-4"
run_one "lr5e-4"
LR="2e-4"  # reset

# --- Group 5: Eikonal Downsample ---
EIK_DS=8
run_one "eik_ds8"
EIK_DS=16  # reset

# --- Group 6: Teacher Mask ---
TEACHER_MASK="thin"
run_one "teacher_thin"
TEACHER_MASK="thick"  # reset

# ===========================================================================
#  Summary
# ===========================================================================
echo ""
echo "========================================================"
echo "  ABLATION COMPLETE — Results Summary"
echo "========================================================"
cat "${SUMMARY_FILE}"
echo ""
echo "Full summary: ${SUMMARY_FILE}"
echo "Per-experiment logs: ${OUTPUT_ROOT}/<exp_name>/"
