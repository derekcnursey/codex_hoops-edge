"""Evaluate the impact of preseason prior on ratings and model predictions.

Compares:
1. November correlation with Torvik (before vs after prior)
2. Model MAE on season 2025 holdout (retrain and evaluate)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor
from src.dataset import load_multi_season_features, load_season_features
from src.features import get_feature_matrix, get_targets
from src.trainer import fit_scaler, train_regressor

# ──────────────────────────────────────────────────────────────
# 1. Compare November ratings with Torvik
# ──────────────────────────────────────────────────────────────

def load_gold_no_garbage_2025():
    """Load the newly-rebuilt gold layer (with preseason prior)."""
    from src.s3_reader import read_gold_table
    tbl = read_gold_table("team_adjusted_efficiencies_no_garbage", season=2025)
    if hasattr(tbl, "to_pandas"):
        return tbl.to_pandas()
    return tbl


def load_torvik_2025():
    """Load Torvik daily ratings from MySQL."""
    conn = pymysql.connect(
        host="localhost", user="derek", password="jake3241",
        database="sports", charset="utf8mb4",
    )
    query = """
    SELECT team_name, date, adj_oe, adj_de, BARTHAG, adj_pace
    FROM daily_data
    WHERE date >= '2024-08-01' AND date <= '2025-07-31'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Name mapping (abbreviated version for most common mismatches)
GOLD_TO_TORVIK = {
    "Alabama State": "Alabama St.", "Alcorn State": "Alcorn St.",
    "Appalachian State": "Appalachian St.", "Arizona State": "Arizona St.",
    "Arkansas State": "Arkansas St.", "Ball State": "Ball St.",
    "Bethune-Cookman": "Bethune Cookman", "Boise State": "Boise St.",
    "Central Connecticut": "Central Connecticut St.",
    "Chicago State": "Chicago St.", "Cleveland State": "Cleveland St.",
    "Colorado State": "Colorado St.", "Coppin State": "Coppin St.",
    "Delaware State": "Delaware St.", "Eastern Illinois": "Eastern Illinois",
    "Fairleigh Dickinson": "FDU", "Fresno State": "Fresno St.",
    "Georgia State": "Georgia St.", "Grambling": "Grambling St.",
    "Idaho State": "Idaho St.", "Illinois State": "Illinois St.",
    "Indiana State": "Indiana St.", "Iowa State": "Iowa St.",
    "Jackson State": "Jackson St.", "Jacksonville State": "Jacksonville St.",
    "Kansas State": "Kansas St.", "Kennesaw State": "Kennesaw St.",
    "Kent State": "Kent St.", "Little Rock": "Little Rock",
    "LIU": "Long Island University",
    "McNeese": "McNeese St.", "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.", "Mississippi Valley State": "Mississippi Valley St.",
    "Missouri State": "Missouri St.", "Montana State": "Montana St.",
    "Morehead State": "Morehead St.", "Morgan State": "Morgan St.",
    "Murray State": "Murray St.", "New Mexico State": "New Mexico St.",
    "Norfolk State": "Norfolk St.", "North Carolina A&T": "North Carolina A&T",
    "North Carolina Central": "North Carolina Central",
    "North Dakota State": "North Dakota St.", "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.", "Ole Miss": "Mississippi",
    "Oregon State": "Oregon St.", "Penn State": "Penn St.",
    "Portland State": "Portland St.", "Prairie View A&M": "Prairie View A&M",
    "Sacramento State": "Sacramento St.", "Sam Houston State": "Sam Houston St.",
    "San Diego State": "San Diego St.", "San José State": "San Jose St.",
    "Savannah State": "Savannah St.", "South Carolina State": "South Carolina St.",
    "South Dakota State": "South Dakota St.",
    "Southeast Missouri State": "Southeast Missouri St.",
    "Southern Illinois": "Southern Illinois",
    "Southern Mississippi": "Southern Miss",
    "Stephen F. Austin": "Stephen F. Austin",
    "Tennessee State": "Tennessee St.",
    "Texas A&M-Commerce": "Texas A&M Commerce",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Christi",
    "Texas State": "Texas St.", "Troy": "Troy",
    "UConn": "Connecticut", "UMass": "Massachusetts",
    "UMass Lowell": "UMass Lowell", "UMBC": "UMBC",
    "UNC Asheville": "UNC Asheville", "UNC Greensboro": "UNC Greensboro",
    "UNC Wilmington": "UNC Wilmington",
    "UT Arlington": "UT Arlington",
    "UT Martin": "UT Martin",
    "Utah State": "Utah St.", "Utah Valley": "Utah Valley",
    "Virginia Tech": "Virginia Tech",
    "Weber State": "Weber St.", "Wichita State": "Wichita St.",
    "Wright State": "Wright St.", "Youngstown State": "Youngstown St.",
    "East Tennessee State": "East Tennessee St.",
    "Sam Houston": "Sam Houston St.",
    "Tarleton State": "Tarleton St.", "Texas Southern": "Texas Southern",
    "Abilene Christian": "Abilene Christian",
    "North Alabama": "North Alabama",
    "Purdue Fort Wayne": "Purdue Fort Wayne",
    "SIU Edwardsville": "SIU Edwardsville",
    "St. Francis (PA)": "St. Francis PA",
    "Saint Francis": "St. Francis PA",
    "Saint Peter's": "Saint Peter's",
    "St. John's": "St. John's",
    "Queens University": "Queens",
    "Lindenwood": "Lindenwood",
    "Stonehill": "Stonehill",
    "Le Moyne": "Le Moyne",
    "West Georgia": "West Georgia",
    "Illinois Chicago": "Illinois Chicago",
}


def compare_november_correlations():
    """Compare gold layer vs Torvik for November (early season)."""
    print("=" * 60)
    print("NOVEMBER CORRELATION WITH TORVIK (Season 2025)")
    print("=" * 60)

    gold = load_gold_no_garbage_2025()
    torvik = load_torvik_2025()

    # Standardize gold columns
    gold = gold.rename(columns={"team": "team_name"})
    gold["date"] = pd.to_datetime(gold["rating_date"]).dt.date

    # Map gold names to Torvik names
    gold["torvik_name"] = gold["team_name"].map(
        lambda x: GOLD_TO_TORVIK.get(x, x)
    )

    torvik["date"] = pd.to_datetime(torvik["date"]).dt.date

    # Filter November only
    nov_start = pd.Timestamp("2024-11-01").date()
    nov_end = pd.Timestamp("2024-11-30").date()

    gold_nov = gold[(gold["date"] >= nov_start) & (gold["date"] <= nov_end)]
    torvik_nov = torvik[(torvik["date"] >= nov_start) & (torvik["date"] <= nov_end)]

    # Join
    merged = gold_nov.merge(
        torvik_nov,
        left_on=["torvik_name", "date"],
        right_on=["team_name", "date"],
        suffixes=("_gold", "_torvik"),
    )

    print(f"  November joined rows: {len(merged)}")

    if len(merged) == 0:
        print("  No joined rows! Check name mapping.")
        return

    for metric in ["adj_oe", "adj_de"]:
        g_col = f"{metric}_gold"
        t_col = f"{metric}_torvik"
        valid = merged[[g_col, t_col]].dropna()
        if len(valid) > 0:
            corr = valid[g_col].corr(valid[t_col])
            mae = (valid[g_col] - valid[t_col]).abs().mean()
            bias = (valid[g_col] - valid[t_col]).mean()
            print(f"  {metric}: r={corr:.4f}, MAE={mae:.3f}, bias={bias:+.3f} (N={len(valid)})")

    # Also check December
    dec_start = pd.Timestamp("2024-12-01").date()
    dec_end = pd.Timestamp("2024-12-31").date()

    gold_dec = gold[(gold["date"] >= dec_start) & (gold["date"] <= dec_end)]
    torvik_dec = torvik[(torvik["date"] >= dec_start) & (torvik["date"] <= dec_end)]

    merged_dec = gold_dec.merge(
        torvik_dec,
        left_on=["torvik_name", "date"],
        right_on=["team_name", "date"],
        suffixes=("_gold", "_torvik"),
    )
    print(f"\n  December joined rows: {len(merged_dec)}")
    for metric in ["adj_oe", "adj_de"]:
        g_col = f"{metric}_gold"
        t_col = f"{metric}_torvik"
        valid = merged_dec[[g_col, t_col]].dropna()
        if len(valid) > 0:
            corr = valid[g_col].corr(valid[t_col])
            mae = (valid[g_col] - valid[t_col]).abs().mean()
            print(f"  {metric}: r={corr:.4f}, MAE={mae:.3f} (N={len(valid)})")


# ──────────────────────────────────────────────────────────────
# 2. Model MAE on holdout
# ──────────────────────────────────────────────────────────────

def load_best_hparams() -> dict:
    path = config.ARTIFACTS_DIR / "best_hparams.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def evaluate_mae(model, X_test, y_test, scaler):
    model.eval()
    X_scaled = scaler.transform(X_test)
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        mu, _ = model(X_t)
        preds = mu.numpy()
    return float(np.mean(np.abs(preds - y_test)))


def evaluate_monthly_mae(model, holdout_df, X_test, y_test, scaler):
    """Break down MAE by month."""
    model.eval()
    X_scaled = scaler.transform(X_test)
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        mu, _ = model(X_t)
        preds = mu.numpy()

    dates = pd.to_datetime(holdout_df["startDate"], errors="coerce", utc=True)
    months = dates.dt.tz_localize(None).dt.to_period("M")

    errors = np.abs(preds - y_test)
    holdout_df = holdout_df.copy()
    holdout_df["month"] = months.values
    holdout_df["abs_error"] = errors

    print("\n  Monthly MAE breakdown:")
    for month, group in holdout_df.groupby("month"):
        mae = group["abs_error"].mean()
        print(f"    {month}: MAE={mae:.2f} (n={len(group)})")


def evaluate_model():
    print("\n" + "=" * 60)
    print("MODEL MAE EVALUATION (Train 2015-2024, Test 2025)")
    print("=" * 60)

    best_hp = load_best_hparams()
    reg_hp = best_hp.get("regressor", {})
    print(f"  Using regressor hparams: {reg_hp}")

    # Load holdout 2025 with new features (preseason prior)
    holdout_df = load_season_features(2025, no_garbage=True)
    holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])
    X_test = get_feature_matrix(holdout_df).values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = get_targets(holdout_df)["spread_home"].values.astype(np.float32)
    print(f"  Holdout games: {len(holdout_df)}")

    # Training data (unchanged — no prior applied to 2015-2024 gold layer)
    train_seasons = list(range(2015, 2025))
    train_df = load_multi_season_features(train_seasons, no_garbage=True)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    X_train = get_feature_matrix(train_df).values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0)
    y_train = get_targets(train_df)["spread_home"].values.astype(np.float32)
    print(f"  Training samples: {len(train_df)}")

    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)

    hp = {**reg_hp, "epochs": 100}
    model = train_regressor(X_train_scaled, y_train, hparams=hp)

    mae = evaluate_mae(model, X_test, y_test, scaler)
    print(f"\n  >>> Overall Holdout MAE: {mae:.4f}")
    print(f"  >>> Baseline (no prior): 10.14")
    print(f"  >>> Clean holdout baseline: 9.87")

    evaluate_monthly_mae(model, holdout_df, X_test, y_test, scaler)


if __name__ == "__main__":
    compare_november_correlations()
    evaluate_model()
