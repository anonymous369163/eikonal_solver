#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_eval_spearman.sh — Compare Spearman coefficient before/after fine-tuning
#
# Usage examples:
#
#   # Quick test: 3 images, default checkpoints (pretrained vs phase1_r5_seg vs new)
#   bash eikonal_solver/run_eval_spearman.sh --n_images 3
#
#   # Evaluate all images, save CSV:
#   bash eikonal_solver/run_eval_spearman.sh --n_images 10 --save_csv results/compare.csv
#
#   # Compare only specific checkpoints:
#   bash eikonal_solver/run_eval_spearman.sh \
#       --ckpt_before "Pretrained:checkpoints/cityscale_vitb_512_e10.ckpt" \
#       --ckpt_after  "NewModel:training_outputs/phase1_spearman_track/checkpoints/samroute-epoch=89-val_seg_loss=0.8235.ckpt" \
#       --n_images 5
#
#   # Single specific image:
#   bash eikonal_solver/run_eval_spearman.sh \
#       --images "Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/crop_34.350697_108.914569_3000.0_z16.tif"
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- defaults ----
CONFIG="${SCRIPT_DIR}/config_phase1.yaml"
DATA_ROOT="Gen_dataset_V2/Gen_dataset"
N_IMAGES=""
IMAGES=""
K=15
MIN_IN_PATCH=3
DOWNSAMPLE=1024
SAVE_CSV=""
DEVICE="cuda"

# Default checkpoint pairs (Label:path)
# The first entry is the "baseline"; all others show Δ relative to it.
CKPT_BEFORE="Pretrained:checkpoints/cityscale_vitb_512_e10.ckpt"
CKPT_MIDDLE="phase1_r5_seg(ep73):training_outputs/phase1_r5_seg/checkpoints/samroute-epoch=73-val_seg_loss=0.6838.ckpt"
CKPT_AFTER="phase1_spearman_track(ep89):training_outputs/phase1_spearman_track/checkpoints/samroute-epoch=89-val_seg_loss=0.8235.ckpt"
EXTRA_CKPTS=""

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG="$2";       shift 2 ;;
        --data_root)        DATA_ROOT="$2";    shift 2 ;;
        --n_images)         N_IMAGES="$2";     shift 2 ;;
        --images)           IMAGES="$2";       shift 2 ;;
        --k)                K="$2";            shift 2 ;;
        --min_in_patch)     MIN_IN_PATCH="$2"; shift 2 ;;
        --downsample)       DOWNSAMPLE="$2";   shift 2 ;;
        --save_csv)         SAVE_CSV="$2";     shift 2 ;;
        --device)           DEVICE="$2";       shift 2 ;;
        --ckpt_before)      CKPT_BEFORE="$2";  shift 2 ;;
        --ckpt_middle)      CKPT_MIDDLE="$2";  shift 2 ;;
        --ckpt_after)       CKPT_AFTER="$2";   shift 2 ;;
        --extra_ckpts)      EXTRA_CKPTS="$2";  shift 2 ;;
        *) echo "[run_eval_spearman] Unknown argument: $1"; exit 1 ;;
    esac
done

# ---- collect checkpoint flags ----
CKPT_FLAGS=""
for ckpt in "${CKPT_BEFORE}" "${CKPT_MIDDLE}" "${CKPT_AFTER}"; do
    label="${ckpt%%:*}"
    path="${ckpt#*:}"
    if [[ -f "${path}" ]]; then
        CKPT_FLAGS="${CKPT_FLAGS} ${ckpt}"
    else
        echo "[WARN] Checkpoint not found, skipping: ${path}"
    fi
done
for ckpt in ${EXTRA_CKPTS}; do
    CKPT_FLAGS="${CKPT_FLAGS} ${ckpt}"
done

if [[ -z "${CKPT_FLAGS}" ]]; then
    echo "[ERROR] No valid checkpoints found."
    exit 1
fi

# ---- optional flags ----
OPTIONAL_FLAGS=""
[[ -n "${N_IMAGES}" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --n_images ${N_IMAGES}"
[[ -n "${SAVE_CSV}" ]] && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --save_csv ${SAVE_CSV}"
[[ -n "${IMAGES}" ]]   && OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --images ${IMAGES}"

echo "======================================================"
echo "  Spearman Comparison"
echo "======================================================"
echo "  Config     : ${CONFIG}"
echo "  Data root  : ${DATA_ROOT}"
echo "  k          : ${K}  min_in_patch: ${MIN_IN_PATCH}  downsample: ${DOWNSAMPLE}"
[[ -n "${N_IMAGES}" ]] && echo "  N images   : ${N_IMAGES}"
[[ -n "${SAVE_CSV}" ]] && echo "  CSV output : ${SAVE_CSV}"
echo "======================================================"

python eikonal_solver/eval_compare_spearman.py \
    --config      "${CONFIG}" \
    --data_root   "${DATA_ROOT}" \
    --k           "${K}" \
    --min_in_patch "${MIN_IN_PATCH}" \
    --downsample  "${DOWNSAMPLE}" \
    --device      "${DEVICE}" \
    --ckpts       ${CKPT_FLAGS} \
    ${OPTIONAL_FLAGS}
