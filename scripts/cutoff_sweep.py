"""Sweep training date cutoffs to find the optimal early-season exclusion point.

For each cutoff (None, Dec 1, Dec 15, Dec 20, Jan 1, Jan 15):
- Load training features from 2015-2024 with date filter applied
- Train MLPRegressor on filtered data
- Evaluate MAE on holdout season 2025 (all games, no date filter)
- Report results

Uses existing Optuna best hyperparameters from artifacts/best_hparams.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import load_multi_season_features, load_season_features
from src.features import get_feature_matrix, get_targets
from src.trainer import fit_scaler, train_regressor

TRAIN_SEASONS = list(range(2015, 2025))  # 2015-2024
HOLDOUT_SEASON = 2025
NO_GARBAGE = True

# Cutoffs to test: None = no filter, then various MM-DD strings
CUTOFFS = [None, "12-01", "12-15", "12-20", "01-01", "01-15"]


def load_best_hparams() -> dict:
    """Load Optuna best hyperparameters."""
    path = config.ARTIFACTS_DIR / "best_hparams.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def evaluate_mae(model: MLPRegressor, X_test: np.ndarray, y_test: np.ndarray,
                 scaler) -> float:
    """Compute MAE on test set."""
    model.eval()
    X_scaled = scaler.transform(X_test)
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        mu, _ = model(X_t)
        preds = mu.numpy()
    return float(np.mean(np.abs(preds - y_test)))


def main():
    best_hp = load_best_hparams()
    reg_hp = best_hp.get("regressor", {})
    print(f"Using regressor hparams: {reg_hp}")

    # Load holdout (2025) — always unfiltered
    print(f"\nLoading holdout season {HOLDOUT_SEASON} (no date filter)...")
    holdout_df = load_season_features(HOLDOUT_SEASON, no_garbage=NO_GARBAGE)
    holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])
    X_test = get_feature_matrix(holdout_df).values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = get_targets(holdout_df)["spread_home"].values.astype(np.float32)
    print(f"  Holdout games: {len(holdout_df)}")

    results = []

    for cutoff in CUTOFFS:
        label = cutoff if cutoff else "None (all games)"
        print(f"\n{'='*60}")
        print(f"Cutoff: {label}")
        print(f"{'='*60}")

        # Load training data with filter
        train_df = load_multi_season_features(
            TRAIN_SEASONS, no_garbage=NO_GARBAGE, min_month_day=cutoff
        )
        train_df = train_df.dropna(subset=["homeScore", "awayScore"])
        n_train = len(train_df)
        print(f"  Training samples: {n_train}")

        X_train = get_feature_matrix(train_df).values.astype(np.float32)
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = get_targets(train_df)["spread_home"].values.astype(np.float32)

        # Fit scaler on training data
        scaler = fit_scaler(X_train)
        X_train_scaled = scaler.transform(X_train)

        # Train regressor with best hparams
        hp = {**reg_hp, "epochs": 100}
        model = train_regressor(X_train_scaled, y_train, hparams=hp)

        # Evaluate on holdout
        mae = evaluate_mae(model, X_test, y_test, scaler)
        print(f"\n  >>> Holdout MAE: {mae:.4f}")

        results.append({
            "cutoff": label,
            "train_rows": n_train,
            "holdout_mae": mae,
        })

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Cutoff':<25} {'Train Rows':>12} {'Holdout MAE':>12}")
    print("-" * 52)
    for r in results:
        print(f"{r['cutoff']:<25} {r['train_rows']:>12,} {r['holdout_mae']:>12.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    out_path = Path(__file__).resolve().parent.parent / "reports" / "cutoff_sweep_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
