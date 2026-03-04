#!/bin/bash
set -e

TIF="/home/yuepeng/code/mmdm_V3/MMDataset/Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/crop_34.350697_108.914569_3000.0_z16.tif"
CKPT="/tmp/distance_training/phase1/train/34.269888_108.947180__01_34.350697_108.914569_3000.0/routing.pt"
GT_SKEL="/home/yuepeng/code/mmdm_V3/MMDataset/Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/roadnet_skeleton_r1.png"
GT_R3="/home/yuepeng/code/mmdm_V3/MMDataset/Gen_dataset_V2/Gen_dataset/34.269888_108.947180/01_34.350697_108.914569_3000.0/roadnet_normalized_r3.png"
SCRIPT="/home/yuepeng/code/mmdm_V3/MMDataset/eikonal_solver/eval_ranking_accuracy.py"
NCASES=50
SUMMARY="/tmp/sweep_eikonal_summary.txt"

run_one() {
    local label="$1"; shift
    echo ">>> Running: $label"
    local output
    output=$(PYTHONUNBUFFERED=1 python "$SCRIPT" \
        --tif "$TIF" --tsp_load_ckpt "$CKPT" \
        --n_cases $NCASES --cost_net --device cuda --seed 42 "$@" 2>&1)

    local pw_eik pw_euc tau_eik tau_euc t1_eik t1_euc rel_eik rel_euc gate alpha gamma ds_val pool_val elapsed
    pw_eik=$(echo "$output" | grep -A2 "Pairwise Ordering" | grep "Eikonal:" | head -1 | awk '{print $2}')
    pw_euc=$(echo "$output" | grep -A3 "Pairwise Ordering" | grep "Euclidean:" | head -1 | awk '{print $2}')
    tau_eik=$(echo "$output" | grep -A2 "Kendall" | grep "Eikonal:" | head -1 | awk '{print $2}')
    tau_euc=$(echo "$output" | grep -A3 "Kendall" | grep "Euclidean:" | head -1 | awk '{print $2}')
    t1_eik=$(echo "$output" | grep -A2 "Top-1" | grep "Eikonal:" | head -1 | awk '{print $2}')
    t1_euc=$(echo "$output" | grep -A3 "Top-1" | grep "Euclidean:" | head -1 | awk '{print $2}')
    rel_eik=$(echo "$output" | grep -A1 "Relative Error" | grep "Eikonal:" | head -1 | sed 's/.*mean=\([0-9.]*\).*/\1/')
    rel_euc=$(echo "$output" | grep -A2 "Relative Error" | grep "Euclidean:" | head -1 | sed 's/.*mean=\([0-9.]*\).*/\1/')
    elapsed=$(echo "$output" | grep "^Time:" | awk '{print $2}')

    local pw_adv tau_adv t1_adv
    pw_adv=$(echo "$output" | grep -A4 "Pairwise" | grep "advantage" | head -1 | awk '{print $NF}')
    tau_adv=$(echo "$output" | grep -A4 "Kendall" | grep "advantage" | head -1 | awk '{print $NF}')
    t1_adv=$(echo "$output" | grep -A4 "Top-1" | grep "advantage" | head -1 | awk '{print $NF}')

    printf "%-38s | PW_adv=%-7s Tau_adv=%-7s T1_adv=%-7s | EikRel=%-6s EucRel=%-6s | %s\n" \
        "$label" "$pw_adv" "$tau_adv" "$t1_adv" "$rel_eik" "$rel_euc" "$elapsed" \
        | tee -a "$SUMMARY"
}

echo "" > "$SUMMARY"
echo "==================================================================================" | tee -a "$SUMMARY"
echo "EIKONAL RANKING PARAMETER SWEEP  (GT skeleton r=1, n_cases=$NCASES, p20)" | tee -a "$SUMMARY"
echo "==================================================================================" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

echo "--- SWEEP 1: Gate Mixing (ds=12) ---" | tee -a "$SUMMARY"
run_one "S1: gate=0.90(trained) ds=12"   --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200
run_one "S1: gate=0.95 ds=12"            --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 0.95
run_one "S1: gate=0.99 ds=12"            --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 0.99
run_one "S1: gate=1.00(pure_eik) ds=12"  --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0
echo "" | tee -a "$SUMMARY"

echo "--- SWEEP 2: Cost Contrast (ds=12, gate=1.0) ---" | tee -a "$SUMMARY"
run_one "S2: a=9.2/g=1.08(trained)"      --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0
run_one "S2: a=30/g=2.0"                 --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0 --alpha_override 30 --gamma_override 2.0
run_one "S2: a=50/g=2.0"                 --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S2: a=100/g=3.0"                --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0 --alpha_override 100 --gamma_override 3.0
run_one "S2: a=50/g=1.0"                 --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 1.0
echo "" | tee -a "$SUMMARY"

echo "--- SWEEP 3: Downsample (gate=1.0, best cost) ---" | tee -a "$SUMMARY"
run_one "S3: ds=48 (1x cost)"            --gt_mask "$GT_SKEL" --downsample 48 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S3: ds=24 (4x cost)"            --gt_mask "$GT_SKEL" --downsample 24 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S3: ds=16 (8x cost)"            --gt_mask "$GT_SKEL" --downsample 16 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S3: ds=12 (16x cost)"           --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
echo "" | tee -a "$SUMMARY"

echo "--- SWEEP 4: Mask width at best config (ds=12, gate=1.0, a=50/g=2) ---" | tee -a "$SUMMARY"
run_one "S4: skeleton_r1"                --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S4: normalized_r3"              --gt_mask "$GT_R3"   --downsample 12 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
echo "" | tee -a "$SUMMARY"

echo "--- SWEEP 5: Iteration Ablation (ds=16, gate=1.0, a=50/g=2) ---" | tee -a "$SUMMARY"
run_one "S5: ds=16 iters=200 (baseline)" --gt_mask "$GT_SKEL" --downsample 16 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S5: ds=16 iters=100"            --gt_mask "$GT_SKEL" --downsample 16 --eik_iters 100 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S5: ds=16 iters=80"             --gt_mask "$GT_SKEL" --downsample 16 --eik_iters 80  --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S5: ds=16 iters=50"             --gt_mask "$GT_SKEL" --downsample 16 --eik_iters 50  --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S5: ds=16 iters=40"             --gt_mask "$GT_SKEL" --downsample 16 --eik_iters 40  --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
echo "" | tee -a "$SUMMARY"

echo "--- SWEEP 5b: Iteration Ablation (ds=12, gate=1.0, a=50/g=2) ---" | tee -a "$SUMMARY"
run_one "S5b: ds=12 iters=200 (baseline)" --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 200 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S5b: ds=12 iters=100"            --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 100 --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S5b: ds=12 iters=80"             --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 80  --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S5b: ds=12 iters=50"             --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 50  --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
run_one "S5b: ds=12 iters=40"             --gt_mask "$GT_SKEL" --downsample 12 --eik_iters 40  --gate_override 1.0 --alpha_override 50 --gamma_override 2.0
echo "" | tee -a "$SUMMARY"

echo "==================================================================================" | tee -a "$SUMMARY"
echo "SWEEP COMPLETE" | tee -a "$SUMMARY"
echo "==================================================================================" | tee -a "$SUMMARY"
echo ""
echo "Summary: $SUMMARY"
