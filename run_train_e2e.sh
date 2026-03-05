#!/usr/bin/env bash
# ===========================================================================
#  E2E Eikonal TSP Training Script
#
#  Trains the NCO model with differentiable distance matrix generation
#  via SAMRoute Eikonal solver (encoder_mode='e2e_eikonal').
#
#  Recommended hardware: 1x RTX 4090 / A100 (24GB+ VRAM)
#  Estimated time:  ~6-8h (with grad_interval=5, torch.compile, warmstart)
#
#  Pre-requisites:
#    1. SAMRoute checkpoint at training_outputs/fulldataset_seg_dist_lora_v2/
#    2. Dataset split manifest (run once with SPLIT_MANIFEST_MODE='create')
#    3. (Optional) Pre-computed road_prob caches:
#         python precompute_road_prob.py
#       Saves ~350MB GPU memory + eliminates 4s/image first-call latency
#
#  Usage:
#    bash run_train_e2e.sh                        # default: e2e_eikonal
#    bash run_train_e2e.sh e2e_decoder            # e2e + trainable decoder (~177K extra params)
#    bash run_train_e2e.sh graph_only             # graph-only baseline
#    bash run_train_e2e.sh graph_image_fusion     # full multi-modal
#    bash run_train_e2e.sh e2e_warmstart          # graph_only pretrain -> e2e fine-tune
#    CUDA_VISIBLE_DEVICES=1 bash run_train_e2e.sh # use GPU 1
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- Environment ----------
source activate satdino 2>/dev/null || conda activate satdino 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Mode selection ----------
MODE="${1:-e2e_eikonal}"   # e2e_eikonal | e2e_decoder | graph_only | graph_image_fusion | e2e_warmstart

echo "============================================="
echo " NCO TSP Training"
echo " Mode:  ${MODE}"
echo " GPU:   ${CUDA_VISIBLE_DEVICES}"
echo " Start: $(date)"
echo "============================================="

# ---------- Pre-compute caches (if needed) ----------
if [ "$MODE" = "e2e_eikonal" ]; then
    CACHE_COUNT=$(find Gen_dataset_V2/Gen_dataset -name 'road_prob_cache.npz' 2>/dev/null | wc -l)
    TIF_COUNT=$(find Gen_dataset_V2/Gen_dataset -name 'crop_*.tif' 2>/dev/null | wc -l)
    echo "[Cache] road_prob caches: ${CACHE_COUNT}/${TIF_COUNT}"
    if [ "$CACHE_COUNT" -lt "$TIF_COUNT" ]; then
        echo "[Cache] Pre-computing missing road_prob caches..."
        python precompute_road_prob.py --device cuda
        echo "[Cache] Pre-computation done."
    else
        echo "[Cache] All road_prob caches available."
    fi
fi

if [ "$MODE" = "e2e_decoder" ]; then
    # Encoder embedding cache required for trainable decoder mode
    ENC_CACHE_COUNT=$(find Gen_dataset_V2/Gen_dataset -name 'encoder_cache.pt' 2>/dev/null | wc -l)
    TIF_COUNT=$(find Gen_dataset_V2/Gen_dataset -name 'crop_*.tif' 2>/dev/null | wc -l)
    echo "[Cache] encoder embedding caches: ${ENC_CACHE_COUNT}/${TIF_COUNT}"
    if [ "$ENC_CACHE_COUNT" -lt "$TIF_COUNT" ]; then
        echo "[Cache] Pre-computing missing encoder embedding caches..."
        python precompute_encoder_cache.py --device cuda
        echo "[Cache] Pre-computation done."
    else
        echo "[Cache] All encoder embedding caches available."
    fi
    # Also ensure road_prob caches for fast-step fallback
    CACHE_COUNT=$(find Gen_dataset_V2/Gen_dataset -name 'road_prob_cache.npz' 2>/dev/null | wc -l)
    if [ "$CACHE_COUNT" -lt "$TIF_COUNT" ]; then
        echo "[Cache] Pre-computing missing road_prob caches (for fast-step fallback)..."
        python precompute_road_prob.py --device cuda
    fi
fi

# ---------- Build arguments per mode ----------
case "$MODE" in
    e2e_eikonal)
        # Accelerated configuration:
        #   ds=32, parallel, torch.compile, grad_interval=5
        #   B=16, ~0.2s avg/batch (80% fast batches)
        ARGS=(
            --encoder_mode e2e_eikonal
            --batch_size 16
            --episodes 4000
            --epochs 200
            --e2e_ds 32
            --e2e_parallel 1
            --e2e_grad_clip 1.0
            --e2e_samroute_lr 1e-3
            --e2e_use_compile 1
            --e2e_grad_interval 5
            --use_distance 1
        )
        ;;
    e2e_decoder)
        # E2E Eikonal with trainable map_decoder (~177K extra params)
        # Requires pre-computed encoder_cache.pt (SAM encoder embeddings)
        # Uses grad_interval=10 (decoder makes all steps use fresh road_prob
        # via no_grad, so fewer full grad steps still maintain training quality)
        # B=64, episodes=16000 → 250 batches/epoch (same update count as B=16/4000)
        ARGS=(
            --encoder_mode e2e_eikonal
            --batch_size 64
            --episodes 16000
            --epochs 200
            --e2e_ds 32
            --e2e_parallel 1
            --e2e_grad_clip 1.0
            --e2e_samroute_lr 1e-3
            --e2e_decoder_lr 1e-4
            --e2e_use_compile 1
            --e2e_grad_interval 10
            --e2e_train_decoder 1
            --use_distance 1
        )
        ;;
    graph_only)
        # Graph encoder + pre-computed NPZ distance matrix
        ARGS=(
            --encoder_mode graph_only
            --batch_size 256
            --episodes 100000
            --epochs 200
            --use_distance 1
        )
        ;;
    graph_image_fusion)
        # Full multi-modal: graph + satellite image fusion
        ARGS=(
            --encoder_mode graph_image_fusion
            --batch_size 256
            --episodes 100000
            --epochs 200
        )
        ;;
    e2e_warmstart)
        # Two-phase: graph_only pretrain (200ep) -> e2e_eikonal fine-tune (150ep)
        PRETRAIN_EPOCHS=200
        FINETUNE_EPOCHS=150

        # Phase 1: graph_only (~2.65h)
        echo "[Warmstart] Phase 1: graph_only pre-training (${PRETRAIN_EPOCHS} epochs)..."
        python train_motsp_n20.py \
            --encoder_mode graph_only \
            --batch_size 256 \
            --episodes 100000 \
            --epochs ${PRETRAIN_EPOCHS} \
            --use_distance 1 \
            2>&1 | tee "train_warmstart_phase1_$(date +%Y%m%d_%H%M%S).log"

        # Find latest graph_only checkpoint
        GRAPH_CKPT_FILE=$(ls -t result/*/checkpoint_motsp-${PRETRAIN_EPOCHS}.pt 2>/dev/null | head -1)
        if [ -z "$GRAPH_CKPT_FILE" ]; then
            echo "[Warmstart] ERROR: No graph_only checkpoint found (epoch ${PRETRAIN_EPOCHS})"
            exit 1
        fi
        GRAPH_CKPT_PATH=$(dirname "$GRAPH_CKPT_FILE")
        echo "[Warmstart] Phase 1 done. Checkpoint: ${GRAPH_CKPT_PATH}"

        # Phase 2: e2e_eikonal fine-tune (~5.3h with grad_interval + compile)
        echo "[Warmstart] Phase 2: e2e_eikonal fine-tuning (${FINETUNE_EPOCHS} epochs)..."
        ARGS=(
            --encoder_mode e2e_eikonal
            --batch_size 16
            --episodes 4000
            --epochs ${FINETUNE_EPOCHS}
            --e2e_ds 32
            --e2e_parallel 1
            --e2e_grad_clip 1.0
            --e2e_samroute_lr 1e-3
            --e2e_use_compile 1
            --e2e_grad_interval 5
            --use_distance 1
            --warmstart_path "${GRAPH_CKPT_PATH}"
            --warmstart_epoch ${PRETRAIN_EPOCHS}
        )
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [e2e_eikonal|graph_only|graph_image_fusion|e2e_warmstart]"
        exit 1
        ;;
esac

# ---------- Log file ----------
LOGFILE="train_${MODE}_$(date +%Y%m%d_%H%M%S).log"

# ---------- Run ----------
echo "[Run] python train_motsp_n20.py ${ARGS[*]}"
echo ""

python train_motsp_n20.py "${ARGS[@]}" 2>&1 | tee "$LOGFILE"

echo ""
echo "============================================="
echo " Training complete: ${MODE}"
echo " Log:  ${LOGFILE}"
echo " End:  $(date)"
echo "============================================="
