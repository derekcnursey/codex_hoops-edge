"""Sweep SOS adjustment parameters via full pipeline rebuild.

For each variant:
1. Update ETL config.yaml with sos_exponent/shrinkage
2. Rebuild gold layer for season 2025
3. Rebuild features in predictor
4. Evaluate model MAE on holdout

Trains model once (training data unchanged), then swaps test features.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
import torch
import yaml

PREDICTOR_ROOT = Path(__file__).resolve().parent.parent
ETL_ROOT = PREDICTOR_ROOT.parent / "hoops_edge_database_etl"
ETL_CONFIG = ETL_ROOT / "config.yaml"

sys.path.insert(0, str(PREDICTOR_ROOT))

from src import config
from src.dataset import load_multi_season_features, load_season_features
from src.features import get_feature_matrix, get_targets
from src.trainer import fit_scaler, train_regressor

TRAIN_SEASONS = list(range(2015, 2025))
HOLDOUT_SEASON = 2025

# Parameter grid (reduced for speed)
VARIANTS = [
    {"sos_exponent": 1.0,  "shrinkage": 0.0,  "label": "baseline"},
    {"sos_exponent": 0.85, "shrinkage": 0.0,  "label": "sos=0.85"},
    {"sos_exponent": 0.7,  "shrinkage": 0.0,  "label": "sos=0.7"},
    {"sos_exponent": 0.5,  "shrinkage": 0.0,  "label": "sos=0.5"},
    {"sos_exponent": 1.0,  "shrinkage": 0.05, "label": "shrink=0.05"},
    {"sos_exponent": 0.85, "shrinkage": 0.05, "label": "sos=0.85+shrink"},
]


def load_best_hparams() -> dict:
    path = config.ARTIFACTS_DIR / "best_hparams.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def update_etl_config(sos_exponent: float, shrinkage: float):
    """Update ETL config.yaml with new SOS parameters."""
    with open(ETL_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["gold"]["adjusted_efficiencies"]["sos_exponent"] = sos_exponent
    cfg["gold"]["adjusted_efficiencies"]["shrinkage"] = shrinkage
    with open(ETL_CONFIG, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def rebuild_gold():
    """Rebuild gold layer for season 2025."""
    # Delete existing
    subprocess.run(
        ["aws", "s3", "rm",
         f"s3://hoops-edge/gold/team_adjusted_efficiencies_no_garbage/season={HOLDOUT_SEASON}/",
         "--recursive"],
        capture_output=True,
    )
    # Rebuild
    result = subprocess.run(
        ["poetry", "run", "python", "-m", "cbbd_etl.gold.runner",
         "--season", str(HOLDOUT_SEASON),
         "--table", "team_adjusted_efficiencies_no_garbage"],
        cwd=str(ETL_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR rebuilding gold: {result.stderr[:200]}")
        return False
    return True


def rebuild_features():
    """Rebuild features for season 2025."""
    result = subprocess.run(
        ["poetry", "run", "hoops", "build-features",
         "--season", str(HOLDOUT_SEASON), "--no-garbage"],
        cwd=str(PREDICTOR_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR rebuilding features: {result.stderr[:200]}")
        return False
    return True


def load_torvik_monthly():
    """Load Torvik ratings for comparison."""
    conn = pymysql.connect(
        host="localhost", user="derek", password="jake3241",
        database="sports", charset="utf8mb4",
    )
    query = """
    SELECT team_name, date, adj_oe, adj_de
    FROM daily_data
    WHERE date >= '2024-11-01' AND date <= '2025-03-31'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def compute_torvik_correlation(gold_table_name="team_adjusted_efficiencies_no_garbage"):
    """Compute November correlation with current gold layer data."""
    from scripts.prior_eval import GOLD_TO_TORVIK
    from src.s3_reader import read_gold_table

    tbl = read_gold_table(gold_table_name, season=HOLDOUT_SEASON)
    if hasattr(tbl, "to_pandas"):
        gold = tbl.to_pandas()
    else:
        gold = tbl

    gold["date"] = pd.to_datetime(gold["rating_date"]).dt.date
    gold["torvik_name"] = gold["team"].map(lambda x: GOLD_TO_TORVIK.get(x, x))

    torvik = load_torvik_monthly()

    results = {}
    for month_name, start, end in [
        ("nov", "2024-11-01", "2024-11-30"),
        ("dec", "2024-12-01", "2024-12-31"),
    ]:
        s = pd.Timestamp(start).date()
        e = pd.Timestamp(end).date()
        g = gold[(gold["date"] >= s) & (gold["date"] <= e)]
        t = torvik[(torvik["date"] >= s) & (torvik["date"] <= e)]

        merged = g.merge(t, left_on=["torvik_name", "date"],
                         right_on=["team_name", "date"], suffixes=("_g", "_t"))

        for metric in ["adj_oe", "adj_de"]:
            valid = merged[[f"{metric}_g", f"{metric}_t"]].dropna()
            if len(valid) > 0:
                results[f"{month_name}_{metric}_r"] = valid[f"{metric}_g"].corr(valid[f"{metric}_t"])
            else:
                results[f"{month_name}_{metric}_r"] = None

    return results


def evaluate_mae(model, X_test, y_test, scaler):
    model.eval()
    X_scaled = scaler.transform(X_test)
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        mu, _ = model(X_t)
        preds = mu.numpy()
    return float(np.mean(np.abs(preds - y_test)))


def evaluate_monthly_mae(model, holdout_df, X_test, y_test, scaler):
    model.eval()
    X_scaled = scaler.transform(X_test)
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        mu, _ = model(X_t)
        preds = mu.numpy()

    dates = pd.to_datetime(holdout_df["startDate"], errors="coerce", utc=True)
    months = dates.dt.tz_localize(None).dt.to_period("M")
    errors = np.abs(preds - y_test)

    result = {}
    df_tmp = pd.DataFrame({"month": months.values, "abs_error": errors})
    for month, group in df_tmp.groupby("month"):
        result[str(month)] = float(group["abs_error"].mean())
    return result


def main():
    best_hp = load_best_hparams()
    reg_hp = best_hp.get("regressor", {})
    print(f"Regressor hparams: {reg_hp}")

    # Train model once on training data
    print("\nLoading training data...")
    train_df = load_multi_season_features(TRAIN_SEASONS, no_garbage=True)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    X_train = get_feature_matrix(train_df).values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0)
    y_train = get_targets(train_df)["spread_home"].values.astype(np.float32)
    print(f"  Training samples: {len(train_df)}")

    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)

    print("Training model (one-time)...")
    hp = {**reg_hp, "epochs": 100}
    model = train_regressor(X_train_scaled, y_train, hparams=hp)

    results = []

    for i, variant in enumerate(VARIANTS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(VARIANTS)}] {variant['label']}")
        print(f"  sos_exponent={variant['sos_exponent']}, shrinkage={variant['shrinkage']}")
        print(f"{'='*60}")

        t0 = time.time()

        # Update ETL config
        update_etl_config(variant["sos_exponent"], variant["shrinkage"])

        # Rebuild gold layer
        print("  Rebuilding gold layer...")
        if not rebuild_gold():
            continue

        # Rebuild features
        print("  Rebuilding features...")
        if not rebuild_features():
            continue

        # Compute Torvik correlation
        print("  Computing Torvik correlations...")
        torvik_corr = compute_torvik_correlation()

        # Load holdout features
        holdout_df = load_season_features(HOLDOUT_SEASON, no_garbage=True)
        holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])
        X_test = get_feature_matrix(holdout_df).values.astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = get_targets(holdout_df)["spread_home"].values.astype(np.float32)

        # Evaluate
        mae = evaluate_mae(model, X_test, y_test, scaler)
        monthly = evaluate_monthly_mae(model, holdout_df, X_test, y_test, scaler)

        elapsed = time.time() - t0

        print(f"  MAE={mae:.4f}, Nov MAE={monthly.get('2024-11', 'N/A'):.2f}")
        print(f"  Nov adj_oe r={torvik_corr.get('nov_adj_oe_r', 'N/A')}")
        print(f"  Time: {elapsed:.0f}s")

        result_row = {
            "label": variant["label"],
            "sos_exponent": variant["sos_exponent"],
            "shrinkage": variant["shrinkage"],
            "holdout_mae": mae,
            "nov_mae": monthly.get("2024-11"),
            "dec_mae": monthly.get("2024-12"),
            "jan_mae": monthly.get("2025-01"),
            **torvik_corr,
            "elapsed_s": elapsed,
        }
        results.append(result_row)

    # Restore baseline config
    update_etl_config(1.0, 0.0)

    # Summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("holdout_mae")

    print(f"\n{'='*80}")
    print("SOS SWEEP RESULTS (sorted by holdout MAE)")
    print(f"{'='*80}")
    print(f"{'Label':<20} {'MAE':>8} {'Nov MAE':>8} {'Nov OE r':>9} {'Dec OE r':>9}")
    print("-" * 60)
    for _, r in results_df.iterrows():
        nov_r = r.get("nov_adj_oe_r")
        dec_r = r.get("dec_adj_oe_r")
        print(f"{r['label']:<20} {r['holdout_mae']:>8.4f} "
              f"{r.get('nov_mae', 0):>8.2f} "
              f"{nov_r:>9.4f}" if nov_r else f"{'N/A':>9}" +
              f" {dec_r:>9.4f}" if dec_r else f" {'N/A':>9}")

    out_path = PREDICTOR_ROOT / "reports" / "sos_sweep_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
