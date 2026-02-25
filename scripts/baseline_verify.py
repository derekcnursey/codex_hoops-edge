"""Verify the 9.87 baseline MAE and diagnose the 10.14 discrepancy.

Tests:
1. Current features (train sos=1.0 / test sos=0.85) + default hparams
2. Current features (train sos=1.0 / test sos=0.85) + Optuna hparams
3. Current features (train sos=1.0 / test sos=1.0 from gold) + default hparams  [baseline repro]
4. Current features (train sos=1.0 / test sos=1.0 from gold) + Optuna hparams
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset import load_multi_season_features, load_season_features
from src.features import get_feature_matrix, get_targets, build_features
from src.trainer import fit_scaler, train_regressor, train_classifier
from src.architecture import MLPRegressor, MLPClassifier, gaussian_nll_loss

TRAIN_SEASONS = list(range(2015, 2025))
HOLDOUT_SEASON = 2025


def load_best_hparams() -> dict:
    path = config.ARTIFACTS_DIR / "best_hparams.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@torch.no_grad()
def evaluate(model, X_test, y_test, scaler):
    """Compute MAE for a trained model."""
    model.eval()
    X_scaled = scaler.transform(X_test)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    mu, log_sigma = model(X_t)
    sigma = torch.nn.functional.softplus(log_sigma) + 1e-3
    sigma = sigma.clamp(min=0.5, max=30.0)
    preds = mu.numpy()
    mae = float(np.mean(np.abs(preds - y_test)))
    return mae, preds, sigma.numpy()


@torch.no_grad()
def evaluate_monthly(model, holdout_df, X_test, y_test, scaler):
    model.eval()
    X_scaled = scaler.transform(X_test)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    mu, _ = model(X_t)
    preds = mu.numpy()

    dates = pd.to_datetime(holdout_df["startDate"], errors="coerce", utc=True)
    months = dates.dt.tz_localize(None).dt.to_period("M")
    errors = np.abs(preds - y_test)

    result = {}
    df_tmp = pd.DataFrame({"month": months.values, "abs_error": errors})
    for month, group in df_tmp.groupby("month"):
        result[str(month)] = (float(group["abs_error"].mean()), len(group))
    return result


def run_experiment(label, train_df, holdout_df, hparams_reg, epochs=100):
    """Train model and evaluate."""
    print(f"\n--- {label} ---")

    X_train = get_feature_matrix(train_df).values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0)
    y_train = get_targets(train_df)["spread_home"].values.astype(np.float32)

    X_test = get_feature_matrix(holdout_df).values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = get_targets(holdout_df)["spread_home"].values.astype(np.float32)

    print(f"  Train: {len(train_df)}, Test: {len(holdout_df)}")
    print(f"  Train adj_oe mean: {train_df['home_team_adj_oe'].mean():.2f}")
    print(f"  Test adj_oe mean: {holdout_df['home_team_adj_oe'].mean():.2f}")

    # Fit scaler on training data (don't save to disk)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Train regressor
    hp = {**hparams_reg, "epochs": epochs}
    print(f"  Hparams: hidden1={hp.get('hidden1', 256)}, hidden2={hp.get('hidden2', 128)}, "
          f"dropout={hp.get('dropout', 0.3):.3f}, lr={hp.get('lr', 1e-3):.6f}, "
          f"batch_size={hp.get('batch_size', 256)}")

    model = train_regressor(X_train_scaled, y_train, hparams=hp)

    # Evaluate
    mae, preds, sigmas = evaluate(model, X_test, y_test, scaler)
    monthly = evaluate_monthly(model, holdout_df, X_test, y_test, scaler)

    print(f"  Overall MAE: {mae:.4f}")
    print(f"  Sigma mean: {sigmas.mean():.2f}, median: {np.median(sigmas):.2f}")
    for month in sorted(monthly.keys()):
        m_mae, n = monthly[month]
        print(f"    {month}: MAE={m_mae:.2f} (n={n})")

    return mae, monthly, preds, sigmas


def main():
    best_hp = load_best_hparams()
    optuna_reg_hp = best_hp.get("regressor", {})
    default_hp = {}  # Will use trainer defaults

    print("=" * 70)
    print("BASELINE VERIFICATION")
    print("=" * 70)

    print(f"\nOptuna best regressor hparams: {optuna_reg_hp}")
    print(f"Default hparams: hidden1=256, hidden2=128, dropout=0.3, lr=0.001, wd=0.0001, bs=256")

    # Load training data
    print("\nLoading training features (2015-2024, no-garbage)...")
    train_df = load_multi_season_features(TRAIN_SEASONS, no_garbage=True)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Training samples: {len(train_df)}")

    # Check training feature stats
    for s in [2015, 2020, 2024]:
        try:
            sdf = load_season_features(s, no_garbage=True)
            print(f"  Season {s}: {len(sdf)} rows, adj_oe mean={sdf['home_team_adj_oe'].mean():.2f}")
        except FileNotFoundError:
            print(f"  Season {s}: NOT FOUND")

    # ── Experiment 1: Current test features (sos=0.85) + default hparams ──
    print("\n\nLoading 2025 test features (current, sos=0.85)...")
    holdout_085 = load_season_features(HOLDOUT_SEASON, no_garbage=True)
    holdout_085 = holdout_085.dropna(subset=["homeScore", "awayScore"])
    print(f"  Test samples: {len(holdout_085)}")

    mae_1, _, _, _ = run_experiment(
        "Exp 1: train(sos=1.0) + test(sos=0.85) + DEFAULT hparams",
        train_df, holdout_085, default_hp,
    )

    # ── Experiment 2: Current test features (sos=0.85) + Optuna hparams ──
    mae_2, _, _, _ = run_experiment(
        "Exp 2: train(sos=1.0) + test(sos=0.85) + OPTUNA hparams",
        train_df, holdout_085, optuna_reg_hp,
    )

    # ── Experiment 3: Rebuild 2025 test with sos=1.0 from gold, default hparams ──
    # Read old gold layer data (if both asof partitions exist, this matches old state)
    print("\n\nRebuilding 2025 features from current gold layer (to get sos=1.0 baseline)...")
    print("  Building features from S3 gold layer...")
    holdout_gold = build_features(HOLDOUT_SEASON, no_garbage=True)
    holdout_gold = holdout_gold.dropna(subset=["homeScore", "awayScore"])
    print(f"  Rebuilt test samples: {len(holdout_gold)}")
    print(f"  adj_oe mean: {holdout_gold['home_team_adj_oe'].mean():.2f}")

    # Check if gold layer was rebuilt with sos=0.85 (only 1 asof partition from Feb 25)
    # If adj_oe mean is ~109 it's sos=0.85, if ~145+ it's sos=1.0
    gold_oe_mean = holdout_gold['home_team_adj_oe'].mean()
    if gold_oe_mean < 120:
        print(f"  NOTE: Gold layer appears to be sos=0.85 (adj_oe mean={gold_oe_mean:.1f})")
        print("  To get the true sos=1.0 baseline, would need to rebuild gold with sos=1.0")
        # Still run the experiment for comparison
        mae_3, _, _, _ = run_experiment(
            "Exp 3: train(sos=1.0) + test(sos=0.85 from gold) + DEFAULT hparams",
            train_df, holdout_gold, default_hp,
        )
    else:
        print(f"  Gold layer appears to be sos=1.0 (adj_oe mean={gold_oe_mean:.1f})")
        mae_3, _, _, _ = run_experiment(
            "Exp 3: train(sos=1.0) + test(sos=1.0 from gold) + DEFAULT hparams",
            train_df, holdout_gold, default_hp,
        )

    mae_4, _, _, _ = run_experiment(
        "Exp 4: train(sos=1.0) + test(from gold) + OPTUNA hparams",
        train_df, holdout_gold, optuna_reg_hp,
    )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Exp 1 (sos mismatch + default hp):  MAE = {mae_1:.4f}")
    print(f"  Exp 2 (sos mismatch + Optuna hp):   MAE = {mae_2:.4f}")
    print(f"  Exp 3 (gold test + default hp):      MAE = {mae_3:.4f}")
    print(f"  Exp 4 (gold test + Optuna hp):       MAE = {mae_4:.4f}")
    print()
    print(f"  Clean holdout target:                MAE = 9.87")
    print(f"  Hparams impact: {abs(mae_1 - mae_2):.4f} pts")
    print(f"  Gold layer mismatch impact: {abs(mae_1 - mae_3):.4f} pts")


if __name__ == "__main__":
    main()
