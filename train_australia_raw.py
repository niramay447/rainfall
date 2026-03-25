from src.sampling.main import stratified_spatial_kfold_dual  # must be first import

import argparse
import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.transforms import ToUndirected

from src.performance_logger import PerformanceLogger
from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.australia_utils import load_australia_raingauge_dataset
from training.logic_hetero_raw import train_epoch, validate, test_model
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive


# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train GNN on Australian rain gauge data")
parser.add_argument("--hidden_channels", type=int,   default=32)
parser.add_argument("--num_layers",      type=int,   default=4)
parser.add_argument("--knn",             type=int,   default=5)
parser.add_argument("--lr",              type=float, default=0.001)
parser.add_argument("--epochs",          type=int,   default=100)
parser.add_argument("--patience",        type=int,   default=20,
                    help="Early stopping patience (epochs without val improvement)")
parser.add_argument("--fold_count",      type=int,   default=None,
                    help="Override fold_count from config (e.g. 1 for quick test)")
parser.add_argument("--data_fraction",   type=float, default=1.0,
                    help="Fraction of timesteps to use (0.01 = 1%% for smoke test)")
parser.add_argument("--experiment_name", type=str,   default=None,
                    help="Custom experiment name (default: auto-generated timestamp)")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = read_config('config.yaml')

batch_size  = config['training_params']['batch_size']
fold_count  = args.fold_count if args.fold_count is not None \
              else config['training_params']['fold_count']

# Build a descriptive experiment name that encodes the hyperparameters
if args.experiment_name:
    experiment_name = args.experiment_name
else:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = (
        f"{ts}_australia_RAW"
        f"_h{args.hidden_channels}"
        f"_l{args.num_layers}"
        f"_k{args.knn}"
        f"_lr{args.lr}"
    )
    if args.data_fraction < 1.0:
        experiment_name += f"_frac{args.data_fraction}"

os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
perf = PerformanceLogger(f"experiments/{experiment_name}/training_log.jsonl")

print(f"Experiment : {experiment_name}")
print(f"Device     : {device}")
print(f"hidden     : {args.hidden_channels}  layers: {args.num_layers}  "
      f"knn: {args.knn}  lr: {args.lr}  epochs: {args.epochs}  "
      f"patience: {args.patience}  folds: {fold_count}")
if args.data_fraction < 1.0:
    print(f"Data fraction: {args.data_fraction:.1%}  (smoke-test mode)")
print()

# ── Load Australian raingauge data ────────────────────────────────────────────
au_cfg = config['australia']
raingauge_df, raingauge_station_mappings_df = load_australia_raingauge_dataset(
    csv_path=au_cfg['dataset_path'],
    metadata_path=au_cfg['station_metadata_path'],
    uptime_threshold=config['filters']['uptime_threshold'],
)

# NOTE: do NOT fillna(0) here — NaN values carry sensor-malfunction information
# that fill_heterodata() uses to set validity=0 for bad readings.
# The fillna(0) is handled inside GaugeGraphNew.fill_heterodata().

# ── Optional: trim to a fraction of timesteps (smoke-test mode) ───────────────
if args.data_fraction < 1.0:
    n_steps = max(1, int(len(raingauge_df) * args.data_fraction))
    raingauge_df = raingauge_df.iloc[:n_steps]
    print(f"Trimmed to {n_steps:,} timesteps ({raingauge_df.index[0]} → {raingauge_df.index[-1]})")

# ── Spatial K-fold split ──────────────────────────────────────────────────────
split_info = stratified_spatial_kfold_dual(
    raingauge_station_mappings_df, seed=123, plot=False, n_splits=fold_count
)

# ── Build gauge graphs for each fold ─────────────────────────────────────────
gauge_graph_arr = []
for i in range(fold_count):
    gauge_graph = GaugeGraphNew(
        raingauge_df, raingauge_station_mappings_df,
        split_info=split_info[i], knn=args.knn,
    )
    gauge_graph_arr.append(gauge_graph)

# ── Build models ──────────────────────────────────────────────────────────────
model_arr = []
for i in range(fold_count):
    model_arr.append(
        GNNInductiveHetero(
            in_channels_dict={"raingauge": 4},  # rainfall | validity | month_sin | month_cos
            hidden_channels=args.hidden_channels,
            out_channels=1,
            num_layers=args.num_layers,
            edge_types=gauge_graph_arr[i].get_train_heterodata().edge_types,
        ).to(device=device)
    )

# ── Build data loaders ────────────────────────────────────────────────────────
train_loader_arr, val_loader_arr, test_loader_arr = [], [], []
for i in range(fold_count):
    train_loader_arr.append(GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(gauge_graph_arr[i].get_train_heterodata()),
        batch_size=batch_size, shuffle=False,
    ))
    val_loader_arr.append(GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(gauge_graph_arr[i].get_validation_heterodata()),
        batch_size=batch_size, shuffle=False,
    ))
    test_loader_arr.append(GeometricDataLoader(
        HeterogeneousWeatherGraphDatasetInductive(gauge_graph_arr[i].get_test_heterodata()),
        batch_size=batch_size, shuffle=False,
    ))


def train_fold(model, train_loader, val_loader, fold, device="cpu"):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_loss_arr, validation_loss_arr = [], []
    best_val = float("inf")
    patience_counter = 0
    training_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            verbose=False, random_noise_masking=False,
        )
        val_loss = validate(model, val_loader, device)

        training_loss_arr.append(train_loss)
        validation_loss_arr.append(val_loss)
        perf.log_epoch(epoch, train_loss, val_loss)

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                f"experiments/{experiment_name}/best_model_fold{fold}.pth",
            )
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:>3}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"patience={patience_counter}/{args.patience}  "
              f"({epoch_time:.1f}s)")

        if patience_counter >= args.patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    total_time = time.time() - training_start
    perf.finalise(total_time)
    print(f"\nFold {fold} done in {total_time:.1f}s  |  best val loss: {best_val:.4f}")

    plt.figure()
    plt.plot(training_loss_arr,   label="train",      color="blue")
    plt.plot(validation_loss_arr, label="validation", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} — {experiment_name}")
    plt.legend()
    plt.savefig(f"experiments/{experiment_name}/loss_fold{fold}.png", dpi=150)
    plt.close()


# ── Run all folds ─────────────────────────────────────────────────────────────
for i in range(fold_count):
    train_fold(model_arr[i], train_loader_arr[i], val_loader_arr[i], fold=i, device=device)
    RMSE = test_model(
        model_arr[i],
        raingauge_station_mappings_df,
        test_loader_arr[i],
        device,
        fold=i,
        experiment_name=experiment_name,
    )
