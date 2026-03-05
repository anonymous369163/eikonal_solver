#!/usr/bin/env python3
"""Batch evaluation for CostNet ablation experiments + delta_log diagnostics."""
import subprocess, sys, os, json, re, textwrap
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.abspath(__file__))
EIK_DIR = os.path.join(PROJECT, "eikonal_solver")
TIF = os.path.join(
    PROJECT,
    "Gen_dataset_V2/Gen_dataset/34.269888_108.947180/"
    "00_34.350697_108.849347_3000.0/crop_34.350697_108.849347_3000.0_z16.tif",
)

EXPERIMENTS = {
    "v2_baseline": {
        "ckpt": os.path.join(PROJECT, "training_outputs/fulldataset_seg_dist_lora_v2/checkpoints/best_seg_dist_lora_v2.ckpt"),
        "cost_net": False,
        "arch": "basic",
    },
    "A_reproduce_bug": {
        "ckpt": os.path.join(PROJECT, "training_outputs/ablation_costnet/exp_A/checkpoints/best_ablation_A.ckpt"),
        "cost_net": True,
        "arch": "basic",
    },
    "B_warmstart_v2": {
        "ckpt": os.path.join(PROJECT, "training_outputs/ablation_costnet/exp_B/checkpoints/best_ablation_B.ckpt"),
        "cost_net": True,
        "arch": "basic",
    },
    "C_higher_lambda": {
        "ckpt": os.path.join(PROJECT, "training_outputs/ablation_costnet/exp_C/checkpoints/best_ablation_C.ckpt"),
        "cost_net": True,
        "arch": "basic",
    },
    "D_match_alpha": {
        "ckpt": os.path.join(PROJECT, "training_outputs/ablation_costnet/exp_D/checkpoints/best_ablation_D.ckpt"),
        "cost_net": True,
        "arch": "basic",
    },
    "E_strong_reg": {
        "ckpt": os.path.join(PROJECT, "training_outputs/ablation_costnet/exp_E/checkpoints/best_ablation_E.ckpt"),
        "cost_net": True,
        "arch": "basic",
    },
    "F_multiscale": {
        "ckpt": os.path.join(PROJECT, "training_outputs/ablation_costnet/exp_F/checkpoints/best_ablation_F.ckpt"),
        "cost_net": True,
        "arch": "multiscale",
    },
}


def run_eval(name, cfg):
    """Run eval_ranking_accuracy.py and parse output."""
    cmd = [
        sys.executable, os.path.join(EIK_DIR, "eval_ranking_accuracy.py"),
        "--ckpt", cfg["ckpt"],
        "--tif", TIF,
        "--n_cases", "200",
        "--gate_override", "-1",
        "--alpha_override", "-1",
        "--gamma_override", "-1",
    ]
    if cfg["cost_net"]:
        cmd += ["--cost_net", "--cost_net_ch", "8", "--cost_net_delta_scale", "0.75"]
        if cfg["arch"] == "multiscale":
            cmd += ["--cost_net_arch", "multiscale"]

    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    metrics = {}
    lines = output.split("\n")
    for i, line in enumerate(lines):
        # "  Pairwise Ordering Accuracy (higher=better):"
        #   next line: "    Eikonal:   0.8482  (median=0.8713)"
        if "Pairwise Ordering Accuracy" in line:
            for j in range(i+1, min(i+4, len(lines))):
                if "Eikonal:" in lines[j] and "advantage" not in lines[j]:
                    m = re.search(r"Eikonal:\s+([\d.]+)", lines[j])
                    if m: metrics["PW_Acc"] = float(m.group(1))
                    break
        elif "Kendall" in line and "Tau" in line and "higher" in line:
            for j in range(i+1, min(i+4, len(lines))):
                if "Eikonal:" in lines[j] and "advantage" not in lines[j]:
                    m = re.search(r"Eikonal:\s+([\d.]+)", lines[j])
                    if m: metrics["Kendall_Tau"] = float(m.group(1))
                    break
        elif "Top-3 Recall" in line:
            for j in range(i+1, min(i+4, len(lines))):
                if "Eikonal:" in lines[j] and "advantage" not in lines[j]:
                    m = re.search(r"Eikonal:\s+([\d.]+)", lines[j])
                    if m: metrics["Top3_Recall"] = float(m.group(1))
                    break
        elif "Distance Relative Error" in line:
            for j in range(i+1, min(i+4, len(lines))):
                if "Eikonal:" in lines[j]:
                    m_mean = re.search(r"mean=([\d.]+)", lines[j])
                    m_med = re.search(r"median=([\d.]+)", lines[j])
                    if m_mean: metrics["RelErr_Mean"] = float(m_mean.group(1))
                    if m_med: metrics["RelErr_Median"] = float(m_med.group(1))
                    break

    if not metrics:
        print(f"  WARNING: Could not parse metrics. Raw output (last 2000 chars):")
        print(output[-2000:])

    return metrics, output


def diagnose_delta_log(name, cfg):
    """Load checkpoint and compute delta_log statistics on the eval TIF."""
    if not cfg["cost_net"]:
        return {"delta_log_mean": "N/A", "delta_log_std": "N/A", "exp_delta_mean": "N/A"}

    sys.path.insert(0, EIK_DIR)
    from model_multigrid import SAMRoute
    from gradcheck_route_loss_v2_multigrid_fullmap import (
        GradcheckConfig, sliding_window_inference, _load_lightning_ckpt,
        _detect_smooth_decoder, _detect_patch_size, _load_rgb_from_tif,
    )

    sd = _load_lightning_ckpt(cfg["ckpt"])
    ecfg = GradcheckConfig()
    has_lora = any("linear_a_q" in k for k in sd)
    if has_lora:
        ecfg.ENCODER_LORA = True
        ecfg.FREEZE_ENCODER = False
        for k, v in sd.items():
            if "linear_a_q.weight" in k:
                ecfg.LORA_RANK = v.shape[0]
                break
    ecfg.USE_SMOOTH_DECODER = _detect_smooth_decoder(sd)
    ps = _detect_patch_size(sd)
    if ps: ecfg.PATCH_SIZE = ps
    ecfg.ROUTE_COST_NET = True
    ecfg.ROUTE_COST_NET_CH = 8
    ecfg.ROUTE_COST_NET_DELTA_SCALE = 0.75
    ecfg.ROUTE_COST_NET_ARCH = cfg["arch"]

    device = torch.device("cuda")
    model = SAMRoute(ecfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    rgb = _load_rgb_from_tif(TIF)
    road_prob_np = sliding_window_inference(rgb, model, device)
    road_prob_t = torch.from_numpy(road_prob_np).unsqueeze(0).float().to(device)
    with torch.no_grad():
        delta_log = model._cost_net_delta_log(road_prob_t)

    if delta_log is None:
        return {"delta_log_mean": "N/A", "delta_log_std": "N/A", "exp_delta_mean": "N/A"}

    dl = delta_log.cpu().numpy().flatten()
    exp_dl = np.exp(dl)
    stats = {
        "delta_log_mean": f"{dl.mean():.4f}",
        "delta_log_std": f"{dl.std():.4f}",
        "exp_delta_mean": f"{exp_dl.mean():.4f}",
        "exp_delta_min": f"{exp_dl.min():.4f}",
        "exp_delta_max": f"{exp_dl.max():.4f}",
    }
    del model
    torch.cuda.empty_cache()
    return stats


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    all_results = {}
    all_delta = {}

    for name, cfg in EXPERIMENTS.items():
        metrics, raw = run_eval(name, cfg)
        all_results[name] = metrics
        print(f"  Metrics: {metrics}")

        delta = diagnose_delta_log(name, cfg)
        all_delta[name] = delta
        print(f"  delta_log: {delta}")

    header = f"{'Experiment':<22} {'PW_Acc':>8} {'Tau':>8} {'Top3':>8} {'RelErr_M':>9} {'RelErr_m':>9} {'dlog_mu':>9} {'dlog_sd':>9} {'exp_mu':>9}"
    sep = "-" * len(header)
    lines = [header, sep]
    for name in EXPERIMENTS:
        m = all_results.get(name, {})
        d = all_delta.get(name, {})
        lines.append(
            f"{name:<22} "
            f"{m.get('PW_Acc', '-'):>8} "
            f"{m.get('Kendall_Tau', '-'):>8} "
            f"{m.get('Top3_Recall', '-'):>8} "
            f"{m.get('RelErr_Mean', '-'):>9} "
            f"{m.get('RelErr_Median', '-'):>9} "
            f"{str(d.get('delta_log_mean', '-')):>9} "
            f"{str(d.get('delta_log_std', '-')):>9} "
            f"{str(d.get('exp_delta_mean', '-')):>9}"
        )

    print(f"\n\n{'='*80}")
    print("  ABLATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    for l in lines:
        print(l)

    out_path = os.path.join(PROJECT, "result", "ablation_costnet_comparison.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("CostNet Ablation Experiment Results\n")
        f.write(f"TIF: {TIF}\n")
        f.write(f"n_cases: 200\n\n")
        for l in lines:
            f.write(l + "\n")
        f.write(f"\n\nDetailed delta_log diagnostics:\n")
        for name, d in all_delta.items():
            f.write(f"  {name}: {d}\n")
    print(f"\nResults saved to: {out_path}")
