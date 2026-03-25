"""
IDW benchmark for the Australian rain gauge dataset.

Mirrors the exact same 5-fold spatial CV split used in train_australia.py
so results are directly comparable to the GNN experiments.

Usage:
    python run_idw_australia.py                        # full dataset, 5 folds
    python run_idw_australia.py --data_fraction 0.01   # 1% smoke-test
    python run_idw_australia.py --fold_count 1         # single fold
    python run_idw_australia.py --n_nearest 10         # tune IDW neighbours
"""

import argparse
import os
import csv
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

from src.sampling.main import stratified_spatial_kfold_dual  # must be first
from src.raingauge.australia_utils import load_australia_raingauge_dataset
from benchmarks.models.idw import run_IDW_benchmark


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="IDW benchmark on Australian rain gauge data")
parser.add_argument("--data_fraction", type=float, default=1.0,
                    help="Fraction of timesteps to use (e.g. 0.01 for smoke-test)")
parser.add_argument("--fold_count",   type=int,   default=5,
                    help="Number of spatial CV folds (default: 5)")
parser.add_argument("--n_nearest",    type=int,   default=10,
                    help="IDW: number of nearest training stations to interpolate from")
parser.add_argument("--power",        type=float, default=2.0,
                    help="IDW: distance-weighting power (default: 2)")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

au_cfg            = config["australia"]
uptime_threshold  = config["filters"]["uptime_threshold"]

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"idw_results/{timestamp}_australia_nn{args.n_nearest}_p{args.power}"
os.makedirs(out_dir, exist_ok=True)

print(f"IDW benchmark — Australia")
print(f"  folds      : {args.fold_count}")
print(f"  n_nearest  : {args.n_nearest}")
print(f"  power      : {args.power}")
print(f"  output     : {out_dir}")
print()

# ── Load data (same as train_australia.py) ────────────────────────────────────
raingauge_df, station_metadata_df = load_australia_raingauge_dataset(
    csv_path=au_cfg["dataset_path"],
    metadata_path=au_cfg["station_metadata_path"],
    uptime_threshold=uptime_threshold,
)

# Optional: smoke-test with a fraction of timesteps
if args.data_fraction < 1.0:
    n_steps = max(1, int(len(raingauge_df) * args.data_fraction))
    raingauge_df = raingauge_df.iloc[:n_steps]
    print(f"Trimmed to {n_steps:,} timesteps ({raingauge_df.index[0]} → {raingauge_df.index[-1]})")

print(f"Raingauge df shape : {raingauge_df.shape}")

# Build coordinate mapping  {station_id -> (lat, lon)}
raingauge_mappings = {
    row["id"]: (row["latitude"], row["longitude"])
    for _, row in station_metadata_df.iterrows()
}

# ── Same spatial CV split as GNN training ────────────────────────────────────
split_info = stratified_spatial_kfold_dual(
    station_metadata_df, seed=123, plot=False, n_splits=args.fold_count
)

# ── Run IDW for each fold ─────────────────────────────────────────────────────
results = []

for fold in range(args.fold_count):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")

    training_gauges = split_info[fold]["ml"]["train"]
    test_gauges     = split_info[fold]["ml"]["test"]

    rmse, pearson_r = run_IDW_benchmark(
        raingauge_data=raingauge_df,
        coordinates=raingauge_mappings,
        training_stations=training_gauges,
        test_stations=test_gauges,
        power=args.power,
        fold=fold,
        n_nearest=args.n_nearest,
        regression_plot=True,
    )

    # Move plot into our output directory
    src_plot = f"idw_results/plot{fold}.png"
    if os.path.exists(src_plot):
        os.rename(src_plot, f"{out_dir}/plot_fold{fold}.png")

    results.append({"fold": fold, "rmse_mm_hr": rmse, "pearson_r": pearson_r})
    print(f"Fold {fold} RMSE: {rmse:.4f} mm/hr  |  Pearson r: {pearson_r:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
rmse_vals      = [r["rmse_mm_hr"] for r in results]
pearson_vals   = [r["pearson_r"]  for r in results]
mean_rmse      = np.mean(rmse_vals)
std_rmse       = np.std(rmse_vals)
mean_pearson   = np.mean(pearson_vals)
std_pearson    = np.std(pearson_vals)

print(f"\n{'='*60}")
print(f"IDW BENCHMARK SUMMARY")
print(f"  n_nearest  : {args.n_nearest}")
print(f"  power      : {args.power}")
print(f"  folds      : {args.fold_count}")
for r in results:
    print(f"  Fold {r['fold']}  RMSE: {r['rmse_mm_hr']:.4f} mm/hr  |  Pearson r: {r['pearson_r']:.4f}")
print(f"  Mean RMSE  : {mean_rmse:.4f} mm/hr  |  Mean Pearson r: {mean_pearson:.4f}")
print(f"  Std  RMSE  : {std_rmse:.4f} mm/hr  |  Std  Pearson r: {std_pearson:.4f}")
print(f"{'='*60}")

# Save to CSV
results_path = f"{out_dir}/results.csv"
with open(results_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fold", "rmse_mm_hr", "pearson_r"])
    writer.writeheader()
    writer.writerows(results)
    writer.writerow({"fold": "mean", "rmse_mm_hr": mean_rmse, "pearson_r": mean_pearson})
    writer.writerow({"fold": "std",  "rmse_mm_hr": std_rmse,  "pearson_r": std_pearson})

print(f"\nResults saved to {results_path}")
