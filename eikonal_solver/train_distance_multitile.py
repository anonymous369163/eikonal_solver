#!/usr/bin/env python3
"""Multi-tile sequential training orchestrator for distance prediction.

Trains the routing cost-mapping parameters (cost_log_alpha, cost_log_gamma,
eik_gate_logit, and optional CostNet) across multiple tiles sequentially,
carrying forward parameters between tiles.

Usage examples:

  # Phase 1: single-tile intensive training
  python train_distance_multitile.py \
    --city 34.269888_108.947180 --tiles 01 \
    --tsp_n_train 300 --tsp_epochs 15 --cost_net --cost_net_ch 16 \
    --out_dir /tmp/distance_training/phase1

  # Phase 2: all tiles in one city
  python train_distance_multitile.py \
    --city 34.269888_108.947180 --n_train_tiles 14 --n_eval_tiles 2 \
    --tsp_n_train 100 --tsp_epochs 3 --cost_net --cost_net_ch 16 \
    --out_dir /tmp/distance_training/phase2

  # Phase 3: multi-city
  python train_distance_multitile.py \
    --cities 34.269888_108.947180 31.240186_121.496062 23.203150_113.352068 \
             39.906357_116.391299 30.596069_114.297691 \
    --n_train_tiles 4 --n_eval_tiles 1 \
    --tsp_n_train 50 --tsp_epochs 2 --cost_net --cost_net_ch 16 \
    --out_dir /tmp/distance_training/phase3
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_DATA_ROOT = os.path.join(_PROJECT_ROOT, "Gen_dataset_V2", "Gen_dataset")
_TRAIN_SCRIPT = os.path.join(_HERE, "gradcheck_route_loss_v2_multigrid_fullmap.py")


def _find_tiles(city: str):
    """Return sorted list of tile dirs for a city."""
    city_dir = os.path.join(_DATA_ROOT, city)
    if not os.path.isdir(city_dir):
        raise FileNotFoundError(f"City dir not found: {city_dir}")
    tiles = []
    for d in sorted(os.listdir(city_dir)):
        tile_dir = os.path.join(city_dir, d)
        if not os.path.isdir(tile_dir):
            continue
        tifs = glob.glob(os.path.join(tile_dir, "crop_*.tif"))
        npzs = glob.glob(os.path.join(tile_dir, "*_p20.npz"))
        if tifs and npzs:
            tiles.append({
                "name": d,
                "tif": tifs[0],
                "npz": npzs[0],
                "dir": tile_dir,
            })
    return tiles


def _build_train_cmd(tile_info, args, ckpt_load_path, ckpt_save_path,
                     out_dir, eval_only=False):
    """Build the command to run the training script for one tile."""
    cache_prob = os.path.join(out_dir, "road_prob.npy")

    cmd = [
        sys.executable, _TRAIN_SCRIPT,
        "--tif", tile_info["tif"],
        "--cache_prob", cache_prob,
        "--save_debug", out_dir,
        "--downsample", str(args.downsample),
        "--lr", str(args.lr),
        "--tsp_n_train", str(args.tsp_n_train),
        "--tsp_epochs", str(args.tsp_epochs if not eval_only else 0),
        "--tsp_n_eval", str(args.tsp_n_eval),
        "--tsp_k_neighbors", str(args.tsp_k_neighbors),
        "--tsp_log_interval", str(args.tsp_log_interval),
        "--gate_alpha", str(args.gate_alpha),
        "--eik_iters", str(args.eik_iters),
    ]

    if args.cost_net:
        cmd += ["--cost_net", "--cost_net_ch", str(args.cost_net_ch)]
    if args.cost_net_use_coord:
        cmd += ["--cost_net_use_coord"]
    if args.cost_net_delta_scale != 0.75:
        cmd += ["--cost_net_delta_scale", str(args.cost_net_delta_scale)]
    if args.lambda_cost_reg != 1e-3:
        cmd += ["--lambda_cost_reg", str(args.lambda_cost_reg)]

    cmd += ["--tsp_train", "--tsp_eval_before", "--freeze_encoder",
            "--multigrid", "--tube_roi", "--sample_from_npz"]

    if getattr(args, "online_decoder", False) and not eval_only:
        cmd += ["--online_decoder",
                "--decoder_grad_every", str(args.decoder_grad_every)]

    if ckpt_load_path and os.path.isfile(ckpt_load_path):
        cmd += ["--tsp_load_ckpt", ckpt_load_path]
    if ckpt_save_path:
        cmd += ["--tsp_save_ckpt", ckpt_save_path]

    return cmd


def _run_cmd(cmd, log_path):
    """Run a command, stream output to console and log file."""
    print(f"\n{'='*70}")
    print(f"  CMD: {' '.join(cmd[:6])} ...")
    print(f"  LOG: {log_path}")
    print(f"{'='*70}\n")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"[ERROR] Command failed with exit code {proc.returncode}")
        print(f"  See log: {log_path}")
    return proc.returncode


def _parse_metrics_file(path):
    """Parse tsp_metrics.txt into dict."""
    result = {}
    current_section = None
    if not os.path.isfile(path):
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== "):
                current_section = line.strip("= ")
                result[current_section] = {}
            elif ":" in line and current_section:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    v = float(v)
                except ValueError:
                    pass
                result[current_section][k] = v
    return result


def _write_summary(all_results, summary_path):
    """Write a consolidated summary of all tile results."""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-TILE TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for entry in all_results:
            tag = entry.get("tag", "unknown")
            tile = entry.get("tile", "?")
            city = entry.get("city", "?")
            f.write(f"--- {tag}: {city} / {tile} ---\n")
            metrics = entry.get("metrics", {})
            for section, vals in metrics.items():
                f.write(f"  [{section}]\n")
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        if isinstance(v, float):
                            f.write(f"    {k}: {v:.4f}\n")
                        else:
                            f.write(f"    {k}: {v}\n")
            f.write("\n")

        # aggregate
        train_after = [e["metrics"].get("AFTER TRAINING", {})
                       for e in all_results if e.get("tag") == "train"]
        eval_after = [e["metrics"].get("AFTER TRAINING", {})
                      for e in all_results if e.get("tag") == "eval_heldout"]
        eval_base = [e["metrics"].get("BASELINE", {})
                     for e in all_results if e.get("tag") == "eval_heldout"]

        for label, data_list in [("TRAIN TILES (after)", train_after),
                                 ("HELD-OUT TILES (baseline)", eval_base),
                                 ("HELD-OUT TILES (after)", eval_after)]:
            if not data_list:
                continue
            f.write(f"\n{'='*60}\n")
            f.write(f"AGGREGATE: {label}  (n={len(data_list)} tiles)\n")
            f.write(f"{'='*60}\n")
            for key in ["rel_mean", "rel_median", "mae_mean", "mae_median",
                         "rel_medium(500-2000px)", "rel_short(<500px)",
                         "rel_long(>2000px)"]:
                vals = [d.get(key) for d in data_list if key in d]
                if vals:
                    avg = sum(vals) / len(vals)
                    f.write(f"  avg {key}: {avg:.4f}\n")

    print(f"[saved] {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="Multi-tile distance training orchestrator")

    # tile selection
    ap.add_argument("--city", type=str, default="",
                    help="Single city name (e.g. 34.269888_108.947180)")
    ap.add_argument("--cities", nargs="+", default=[],
                    help="Multiple city names for cross-city training")
    ap.add_argument("--tiles", nargs="+", default=[],
                    help="Specific tile prefixes (e.g. 01 02 03). If empty, use all tiles.")
    ap.add_argument("--n_train_tiles", type=int, default=0,
                    help="Max tiles per city for training (0=all except eval)")
    ap.add_argument("--n_eval_tiles", type=int, default=2,
                    help="Number of held-out tiles per city for evaluation")

    # training params (passed to underlying script)
    ap.add_argument("--tsp_n_train", type=int, default=50)
    ap.add_argument("--tsp_epochs", type=int, default=3)
    ap.add_argument("--tsp_n_eval", type=int, default=200)
    ap.add_argument("--tsp_k_neighbors", type=int, default=4)
    ap.add_argument("--tsp_log_interval", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--downsample", type=int, default=48)
    ap.add_argument("--gate_alpha", type=float, default=0.8)
    ap.add_argument("--eik_iters", type=int, default=200)

    # cost net
    ap.add_argument("--cost_net", action="store_true")
    ap.add_argument("--cost_net_ch", type=int, default=8)
    ap.add_argument("--cost_net_use_coord", action="store_true")
    ap.add_argument("--cost_net_delta_scale", type=float, default=0.75)
    ap.add_argument("--lambda_cost_reg", type=float, default=1e-3)

    # online decoder
    ap.add_argument("--online_decoder", action="store_true",
                    help="Recompute road_prob through decoder (enables decoder gradients)")
    ap.add_argument("--decoder_grad_every", type=int, default=5,
                    help="Recompute road_prob every N cases")

    # output
    ap.add_argument("--out_dir", type=str, default="/tmp/distance_training")
    ap.add_argument("--resume_ckpt", type=str, default="",
                    help="Path to routing ckpt to resume from (warm start)")

    args = ap.parse_args()

    # resolve cities
    cities = args.cities if args.cities else ([args.city] if args.city else [])
    if not cities:
        ap.error("Specify --city or --cities")

    # gather all tiles per city
    city_tiles = {}
    for city in cities:
        all_tiles = _find_tiles(city)
        if args.tiles:
            all_tiles = [t for t in all_tiles
                         if any(t["name"].startswith(p) for p in args.tiles)]
        if not all_tiles:
            print(f"[WARN] No matching tiles for city {city}, skipping")
            continue
        city_tiles[city] = all_tiles

    if not city_tiles:
        ap.error("No tiles found for any city")

    # split train / eval per city
    train_plan = []
    eval_plan = []
    for city, tiles in city_tiles.items():
        n_eval = min(args.n_eval_tiles, len(tiles))
        eval_tiles = tiles[-n_eval:] if n_eval > 0 else []
        train_tiles = tiles[:-n_eval] if n_eval > 0 else tiles
        if args.n_train_tiles > 0:
            train_tiles = train_tiles[:args.n_train_tiles]

        for t in train_tiles:
            train_plan.append({"city": city, **t})
        for t in eval_tiles:
            eval_plan.append({"city": city, **t})

    print(f"\n{'='*60}")
    print(f"MULTI-TILE TRAINING PLAN")
    print(f"{'='*60}")
    print(f"  Cities: {len(city_tiles)}")
    print(f"  Train tiles: {len(train_plan)}")
    print(f"  Eval tiles: {len(eval_plan)}")
    print(f"  Per-tile config: n_train={args.tsp_n_train}, epochs={args.tsp_epochs}")
    print(f"  CostNet: {'ON (ch=' + str(args.cost_net_ch) + ')' if args.cost_net else 'OFF'}")
    print(f"  Output: {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # save plan
    plan_info = {
        "cities": cities,
        "train_tiles": [{"city": t["city"], "name": t["name"]} for t in train_plan],
        "eval_tiles": [{"city": t["city"], "name": t["name"]} for t in eval_plan],
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "plan.json"), "w") as f:
        json.dump(plan_info, f, indent=2)

    all_results = []
    ckpt_path = args.resume_ckpt or ""
    total_train_time = 0.0

    # === TRAINING PHASE ===
    for ti, tile_info in enumerate(train_plan):
        city = tile_info["city"]
        tile_name = tile_info["name"]
        tile_out = os.path.join(args.out_dir, "train", f"{city}__{tile_name}")
        ckpt_save = os.path.join(tile_out, "routing.pt")
        os.makedirs(tile_out, exist_ok=True)

        print(f"\n[TRAIN {ti+1}/{len(train_plan)}] {city} / {tile_name}")

        cmd = _build_train_cmd(
            tile_info, args,
            ckpt_load_path=ckpt_path,
            ckpt_save_path=ckpt_save,
            out_dir=tile_out,
        )
        log_path = os.path.join(tile_out, "train.log")

        t0 = time.time()
        rc = _run_cmd(cmd, log_path)
        dt = time.time() - t0
        total_train_time += dt

        if rc != 0:
            print(f"[ERROR] Training failed on tile {tile_name}. Stopping.")
            break

        ckpt_path = ckpt_save

        metrics_path = os.path.join(tile_out, "tsp_metrics.txt")
        metrics = _parse_metrics_file(metrics_path)
        all_results.append({
            "tag": "train",
            "city": city,
            "tile": tile_name,
            "time_s": dt,
            "metrics": metrics,
        })
        print(f"  Training time: {dt:.1f}s")
        if "AFTER TRAINING" in metrics:
            m = metrics["AFTER TRAINING"]
            print(f"  RelErr: mean={m.get('rel_mean', '?'):.4f}  "
                  f"median={m.get('rel_median', '?'):.4f}  "
                  f"MAE={m.get('mae_mean', '?'):.1f}px")

    # === EVALUATION PHASE (held-out tiles) ===
    if eval_plan and ckpt_path:
        print(f"\n{'='*60}")
        print(f"HELD-OUT EVALUATION ({len(eval_plan)} tiles)")
        print(f"{'='*60}")

        for ei, tile_info in enumerate(eval_plan):
            city = tile_info["city"]
            tile_name = tile_info["name"]
            tile_out = os.path.join(args.out_dir, "eval", f"{city}__{tile_name}")
            os.makedirs(tile_out, exist_ok=True)

            print(f"\n[EVAL {ei+1}/{len(eval_plan)}] {city} / {tile_name}")

            cmd = _build_train_cmd(
                tile_info, args,
                ckpt_load_path=ckpt_path,
                ckpt_save_path="",
                out_dir=tile_out,
                eval_only=True,
            )
            log_path = os.path.join(tile_out, "eval.log")

            t0 = time.time()
            rc = _run_cmd(cmd, log_path)
            dt = time.time() - t0

            metrics_path = os.path.join(tile_out, "tsp_metrics.txt")
            metrics = _parse_metrics_file(metrics_path)
            all_results.append({
                "tag": "eval_heldout",
                "city": city,
                "tile": tile_name,
                "time_s": dt,
                "metrics": metrics,
            })
            if "AFTER TRAINING" in metrics:
                m = metrics["AFTER TRAINING"]
                print(f"  Held-out RelErr: mean={m.get('rel_mean', '?'):.4f}  "
                      f"median={m.get('rel_median', '?'):.4f}")

    # === SUMMARY ===
    summary_path = os.path.join(args.out_dir, "summary.txt")
    _write_summary(all_results, summary_path)

    # also save raw results as JSON
    json_path = os.path.join(args.out_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[saved] {json_path}")

    print(f"\nTotal training time: {total_train_time:.0f}s ({total_train_time/60:.1f}min)")
    print(f"Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
