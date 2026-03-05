#!/usr/bin/env bash
# ===========================================================================
#  CostNet Ablation Validation Script
#  Single-region quick training (Xi'an) to validate cost_net improvements
#  Usage:
#    CUDA_VISIBLE_DEVICES=1 bash eikonal_solver/run_costnet_ablation.sh A
#    CUDA_VISIBLE_DEVICES=1 bash eikonal_solver/run_costnet_ablation.sh all
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

source activate satdino 2>/dev/null || conda activate satdino 2>/dev/null || true

# ---------- Fixed paths (relative to PROJECT_ROOT) ----------
REGION="${PROJECT_ROOT}/Gen_dataset_V2/Gen_dataset/34.269888_108.947180/00_34.350697_108.849347_3000.0"
ORIG_CKPT="${PROJECT_ROOT}/checkpoints/cityscale_vitb_512_e10.ckpt"
V2_CKPT="${PROJECT_ROOT}/training_outputs/fulldataset_seg_dist_lora_v2/checkpoints/best_seg_dist_lora_v2.ckpt"
OUT_BASE="${PROJECT_ROOT}/training_outputs/ablation_costnet"

MODE="${1:-all}"

run_experiment() {
    local EXP_ID="$1"
    local CKPT="$2"
    local LAMBDA_DIST="$3"
    local TEACHER_ALPHA="$4"
    local COST_REG="$5"
    local COST_NET_ARCH="${6:-basic}"

    local OUTPUT_DIR="${OUT_BASE}/exp_${EXP_ID}"
    local RUN_NAME="ablation_${EXP_ID}"

    echo ""
    echo "============================================="
    echo " Exp ${EXP_ID}: ckpt=$(basename $CKPT) lambda_dist=${LAMBDA_DIST}"
    echo "   teacher_alpha=${TEACHER_ALPHA} cost_reg=${COST_REG} arch=${COST_NET_ARCH}"
    echo "============================================="

    local ARCH_ARGS=""
    if [ "$COST_NET_ARCH" = "multiscale" ]; then
        ARCH_ARGS="--cost_net_arch multiscale"
    fi

    python finetune_demo_updated.py \
        --single_region_dir      "$REGION" \
        --pretrained_ckpt        "$CKPT" \
        --epochs                 50 \
        --batch_size             16 \
        --devices                1 \
        --workers                4 \
        --val_fraction           0.1 \
        --road_dilation_radius   3 \
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
        --lambda_dist            "$LAMBDA_DIST" \
        --dist_supervision       gtmask_random \
        --dist_teacher_alpha     "$TEACHER_ALPHA" \
        --dist_teacher_gamma     2.0 \
        --dist_teacher_mask      thick \
        --dist_gt_cap_factor     0.85 \
        --dist_detour_cap        12.0 \
        --multigrid \
        --eik_iters              40 \
        --eik_downsample         16 \
        --mg_factor              4 \
        --grad_clip              1.0 \
        --freeze_cost_params \
        --cost_net \
        --cost_net_ch            8 \
        --cost_net_delta_scale   0.75 \
        --lambda_cost_reg        "$COST_REG" \
        --lambda_cost_tv         0.005 \
        $ARCH_ARGS

    echo ">>> Exp ${EXP_ID} done. Output: ${OUTPUT_DIR}"
}

# ---------- Experiment definitions ----------
# run_experiment  ID  CKPT  LAMBDA_DIST  TEACHER_ALPHA  COST_REG  [ARCH]

run_A() { run_experiment A "$ORIG_CKPT" 0.05 20.0 0.01 basic; }
run_B() { run_experiment B "$V2_CKPT"   0.05 20.0 0.01 basic; }
run_C() { run_experiment C "$V2_CKPT"   0.3  20.0 0.01 basic; }
run_D() { run_experiment D "$V2_CKPT"   0.3  50.0 0.01 basic; }
run_E() { run_experiment E "$V2_CKPT"   0.3  50.0 0.1  basic; }
run_F() { run_experiment F "$V2_CKPT"   0.3  50.0 0.1  multiscale; }

case "$MODE" in
    A) run_A ;;
    B) run_B ;;
    C) run_C ;;
    D) run_D ;;
    E) run_E ;;
    F) run_F ;;
    all)
        run_A
        run_B
        run_C
        run_D
        run_E
        run_F
        echo ""
        echo "============================================="
        echo " All 6 ablation experiments complete!"
        echo " Results in: ${OUT_BASE}/exp_{A,B,C,D,E,F}/"
        echo "============================================="
        ;;
    *)
        echo "Usage: $0 {A|B|C|D|E|F|all}"
        exit 1
        ;;
esac
