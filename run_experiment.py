"""
run_experiment.py
===================================================================================
Runs the seven optimizer protocols (Algorithm 1 / Section 3.3 of the paper)
across KAN and MLP architectures on the California Housing regression
dataset -- a well-known scikit-learn / Google Colab regression benchmark
(20,640 samples, 8 features, predicting median house value). This dataset
is unrelated to the datasets used in the paper and is fetched automatically
by scikit-learn; no external files are required.

Prints a results table in the same format used in the paper's Tables 3-5:
architecture, protocol, R^2, MAE, wall-clock time, and the MAE x Time
efficiency ratio, ranked within each architecture block.

USAGE:
    python run_experiment.py --seeds 5 --time-budget 60
"""

import argparse
import copy
import pickle
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from optimizer_staging import (
    Config, ARCHITECTURES, PROTOCOLS, MLP_ALLOWED_PROTOCOLS, train_staged,
)


def load_data(cfg: Config):
    data = fetch_california_housing()
    X, y = data.data, data.target.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    x_scaler = MinMaxScaler(feature_range=cfg.grid_range)
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)

    tensors = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
    }
    return tensors, X.shape[1], y_scaler


def evaluate(model, tensors, y_scaler) -> Dict[str, float]:
    with torch.no_grad():
        pred = model(tensors["X_val"]).cpu().numpy()
        true = tensors["y_val"].cpu().numpy()
    pred = y_scaler.inverse_transform(pred)
    true = y_scaler.inverse_transform(true)
    return {
        "mae": mean_absolute_error(true, pred),
        "r2": r2_score(true, pred),
    }


def run_all(seeds: List[int], time_budget: float, out_dir: str = "results") -> pd.DataFrame:
    cfg = Config(time_budget_seconds=time_budget)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensors, n_features, y_scaler = load_data(cfg)
    tensors = {k: v.to(device) for k, v in tensors.items()}

    # Parameter-matched MLP hidden width, mirroring the paper's approach:
    # match the parameter count of Efficient-KAN at this input dimension.
    ref = ARCHITECTURES["Efficient-KAN"](n_features, cfg)
    target_params = ref.n_params()
    del ref

    def solve_hidden(target: int, lo=4, hi=2048) -> int:
        def count(h):
            return (n_features * h + h) + (h * h + h) + (h * 1 + 1)
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if count(mid) <= target:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return max(best, 1)

    matched_hidden = solve_hidden(target_params)

    archs = dict(ARCHITECTURES)
    archs["MLP"] = lambda in_dim, c: ARCHITECTURES["MLP"](in_dim, c, hidden_dim=matched_hidden)

    print(f"California Housing: {tensors['X_train'].shape[0]} train / "
          f"{tensors['X_val'].shape[0]} val, {n_features} features")
    print(f"Time budget: {cfg.time_budget_seconds:.0f}s/run | seeds: {seeds}\n")

    records = []
    for seed in seeds:
        for arch_name, arch_ctor in archs.items():
            for protocol in PROTOCOLS:
                if arch_name == "MLP" and protocol.name not in MLP_ALLOWED_PROTOCOLS:
                    continue

                torch.manual_seed(seed)
                np.random.seed(seed)
                model = arch_ctor(n_features, cfg).to(device)

                info = train_staged(
                    model, tensors["X_train"], tensors["y_train"],
                    tensors["X_val"], tensors["y_val"], protocol, cfg,
                )
                metrics = evaluate(model, tensors, y_scaler)
                mae_x_time = metrics["mae"] * info["wall_time"]

                print(f"  [{arch_name:<14}] {protocol.name:<22} | seed={seed} | "
                      f"epochs={info['epochs']:>5} | t={info['wall_time']:>6.1f}s | "
                      f"R2={metrics['r2']:.4f} | stop={info['stop_reason']}")

                records.append({
                    "architecture": arch_name,
                    "protocol": protocol.name,
                    "seed": seed,
                    "epochs": info["epochs"],
                    "wall_time": info["wall_time"],
                    "stop_reason": info["stop_reason"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "mae_x_time": mae_x_time,
                })

    df = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "california_housing_results.csv"), index=False)
    with open(os.path.join(out_dir, "california_housing_results.pkl"), "wb") as f:
        pickle.dump(df, f)
    return df


def print_results_table(df: pd.DataFrame):
    """Prints a table in the same format as the paper's Tables 3-5:
    ranked by MAE x Time within each architecture block."""
    agg = (
        df.groupby(["architecture", "protocol"])[["r2", "mae", "wall_time", "mae_x_time"]]
          .agg(["mean", "std"])
    )

    print("\n" + "=" * 100)
    print("RESULTS -- California Housing (mean +/- std across seeds)")
    print("=" * 100)
    print(f"{'Architecture':<14} {'Rank':>4} {'Protocol':<22} {'R2':>16} "
          f"{'MAE':>14} {'WallTime(s)':>14} {'MAExTime':>10}")
    print("-" * 100)

    for arch in agg.index.get_level_values(0).unique():
        sub = agg.loc[arch]
        ranks = sub[("mae_x_time", "mean")].rank(method="min").astype(int)
        order = sub[("mae_x_time", "mean")].sort_values().index
        for proto in order:
            row = sub.loc[proto]
            rank = ranks[proto]
            print(f"{arch:<14} {rank:>4} {proto:<22} "
                  f"{row[('r2','mean')]:.4f}+/-{row[('r2','std')]:.4f} "
                  f"{row[('mae','mean')]:>6.3f}+/-{row[('mae','std')]:.3f} "
                  f"{row[('wall_time','mean')]:>7.1f}+/-{row[('wall_time','std')]:<5.1f} "
                  f"{row[('mae_x_time','mean')]:>9.2f}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimizer staging on California Housing")
    parser.add_argument("--seeds", type=int, default=5, help="number of random seeds to run (1..N)")
    parser.add_argument("--time-budget", type=float, default=60.0, help="per-run wall-clock budget (s)")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    seed_list = list(range(1, args.seeds + 1))
    results_df = run_all(seed_list, args.time_budget, args.out_dir)
    print_results_table(results_df)
    print(f"\nSaved: {args.out_dir}/california_housing_results.csv")
    print(f"Saved: {args.out_dir}/california_housing_results.pkl")
