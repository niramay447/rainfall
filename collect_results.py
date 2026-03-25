"""
collect_results.py — Re-evaluate saved GNN checkpoints and write metrics to CSV.

Run from the project root:
    python collect_results.py

Outputs:
    all_results.csv   — one row per experiment with per-fold + mean RMSE & Pearson R
"""

import os
import re
import csv
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader as GeometricDataLoader

from src.sampling.main import stratified_spatial_kfold_dual
from models.gnn import GNNInductiveHetero
from src.utils import read_config
from src.raingauge.australia_utils import load_australia_raingauge_dataset
from src.graph.gaugegraphnew import GaugeGraphNew, HeterogeneousWeatherGraphDatasetInductive

FOLD_COUNT = 5
EXPERIMENTS_DIR = "experiments"

# Matches: 20260322_184511_australia_h32_l4_k5_lr0.001
#      or: 20260323_162833_australia_RAW_h32_l4_k5_lr0.001
EXP_RE = re.compile(
    r'^\d{8}_\d{6}_australia(_RAW)?_h(\d+)_l(\d+)_k(\d+)_lr([\d.e+-]+)$'
)


def evaluate_fold(model, test_loader, device, is_raw):
    """Lightweight evaluation — returns (rmse_mm, pearson_r) without creating any plots."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x = batch['raingauge'].x.clone()
            y = batch['raingauge'].y
            mask = batch['raingauge'].mask

            if not is_raw:
                y = torch.log1p(y)
                x[:, 0] = torch.log1p(x[:, 0])

            x[mask, :2] = 0.0  # mask test nodes' rainfall + validity

            edge_index_dict = batch.edge_index_dict
            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if hasattr(batch[et], 'edge_attr')
            }
            x_dict = {nt: batch[nt].x for nt in batch.node_types}
            x_dict['raingauge'] = x

            out = model(x_dict, edge_index_dict, edge_attr_dict)

            preds   = out['raingauge'][mask].detach().cpu()
            targets = y[mask].detach().cpu()

            if not is_raw:
                preds   = torch.expm1(preds).clamp(min=0)
                targets = torch.expm1(targets).clamp(min=0)
            else:
                preds   = preds.clamp(min=0)
                targets = targets.clamp(min=0)

            all_preds.append(preds)
            all_targets.append(targets)

    preds_np   = torch.cat(all_preds).numpy().flatten()
    targets_np = torch.cat(all_targets).numpy().flatten()

    valid = (~np.isnan(preds_np)) & (~np.isnan(targets_np))
    rmse = float(np.sqrt(np.mean((preds_np[valid] - targets_np[valid]) ** 2)))
    r, _ = pearsonr(targets_np[valid], preds_np[valid])

    return rmse, float(r)


def main():
    device = torch.device('cpu')
    config = read_config('config.yaml')
    au_cfg  = config['australia']
    batch_size = config['training_params']['batch_size']

    print("Loading dataset...")
    raingauge_df, station_meta_df = load_australia_raingauge_dataset(
        csv_path=au_cfg['dataset_path'],
        metadata_path=au_cfg['station_metadata_path'],
        uptime_threshold=config['filters']['uptime_threshold'],
    )

    print("Building fold splits...")
    split_info = stratified_spatial_kfold_dual(
        station_meta_df, seed=123, plot=False, n_splits=FOLD_COUNT
    )

    # Cache graphs per knn value to avoid rebuilding the same graph twice
    graph_cache = {}  # knn -> list of GaugeGraphNew

    results = []

    folders = sorted([
        f for f in os.listdir(EXPERIMENTS_DIR)
        if EXP_RE.match(f)
    ])

    print(f"\nFound {len(folders)} experiments to evaluate:\n")
    for f in folders:
        print(f"  {f}")

    for folder in folders:
        m = EXP_RE.match(folder)
        is_raw  = m.group(1) is not None
        hidden  = int(m.group(2))
        layers  = int(m.group(3))
        knn     = int(m.group(4))
        lr      = float(m.group(5))

        ckpt_paths = [
            f"{EXPERIMENTS_DIR}/{folder}/best_model_fold{i}.pth"
            for i in range(FOLD_COUNT)
        ]
        missing = [p for p in ckpt_paths if not os.path.exists(p)]
        if missing:
            print(f"\nSKIPPING {folder} — missing checkpoints: {missing}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {folder}")
        print(f"  RAW={is_raw}  h={hidden}  l={layers}  k={knn}  lr={lr}")
        print(f"{'='*60}")

        # Build or reuse graphs for this knn
        if knn not in graph_cache:
            graphs = []
            for i in range(FOLD_COUNT):
                g = GaugeGraphNew(
                    raingauge_df, station_meta_df,
                    split_info=split_info[i], knn=knn,
                )
                graphs.append(g)
            graph_cache[knn] = graphs
        graphs = graph_cache[knn]

        fold_rmse, fold_pearson = [], []

        for i in range(FOLD_COUNT):
            test_loader = GeometricDataLoader(
                HeterogeneousWeatherGraphDatasetInductive(
                    graphs[i].get_test_heterodata()
                ),
                batch_size=batch_size, shuffle=False,
            )

            model = GNNInductiveHetero(
                in_channels_dict={"raingauge": 4},
                hidden_channels=hidden,
                out_channels=1,
                num_layers=layers,
                edge_types=graphs[i].get_train_heterodata().edge_types,
            ).to(device)

            model.load_state_dict(
                torch.load(ckpt_paths[i], map_location=device)
            )

            rmse, r = evaluate_fold(model, test_loader, device, is_raw)
            fold_rmse.append(rmse)
            fold_pearson.append(r)
            print(f"  Fold {i}: RMSE={rmse:.4f}  Pearson r={r:.4f}")

        row = {
            'experiment':    folder,
            'transform':     'RAW' if is_raw else 'log1p',
            'hidden':        hidden,
            'layers':        layers,
            'knn':           knn,
            'lr':            lr,
            'rmse_fold0':    fold_rmse[0],
            'rmse_fold1':    fold_rmse[1],
            'rmse_fold2':    fold_rmse[2],
            'rmse_fold3':    fold_rmse[3],
            'rmse_fold4':    fold_rmse[4],
            'mean_rmse':     float(np.mean(fold_rmse)),
            'std_rmse':      float(np.std(fold_rmse)),
            'pearson_fold0': fold_pearson[0],
            'pearson_fold1': fold_pearson[1],
            'pearson_fold2': fold_pearson[2],
            'pearson_fold3': fold_pearson[3],
            'pearson_fold4': fold_pearson[4],
            'mean_pearson':  float(np.mean(fold_pearson)),
            'std_pearson':   float(np.std(fold_pearson)),
        }
        results.append(row)
        print(f"  → mean RMSE={row['mean_rmse']:.4f}  mean Pearson r={row['mean_pearson']:.4f}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = 'all_results.csv'
    if results:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} experiments → {out_path}")
    else:
        print("\nNo complete experiments found.")


if __name__ == '__main__':
    main()
