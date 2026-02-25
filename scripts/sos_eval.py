"""Evaluate SOS variants: compare Torvik correlation + model MAE.

Reads pre-computed ratings from features/sos_sweep/ and builds features
for each variant to evaluate against the holdout.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset import load_multi_season_features
from src.features import (
    get_feature_matrix,
    get_targets,
    load_games,
    load_boxscores,
    compute_game_four_factors,
    compute_rolling_averages,
    _get_asof_rating,
    AWAY_ROLLING_MAP,
    HOME_ROLLING_MAP,
)
from src.trainer import fit_scaler, train_regressor

SWEEP_DIR = Path(__file__).resolve().parent.parent / "features" / "sos_sweep"
TRAIN_SEASONS = list(range(2015, 2025))
HOLDOUT_SEASON = 2025


def load_best_hparams() -> dict:
    path = config.ARTIFACTS_DIR / "best_hparams.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_torvik():
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


GOLD_TO_TORVIK = {
    "Alabama State": "Alabama St.", "Alcorn State": "Alcorn St.",
    "Appalachian State": "Appalachian St.", "Arizona State": "Arizona St.",
    "Arkansas State": "Arkansas St.", "Ball State": "Ball St.",
    "Bethune-Cookman": "Bethune Cookman", "Boise State": "Boise St.",
    "Central Connecticut": "Central Connecticut St.",
    "Chicago State": "Chicago St.", "Cleveland State": "Cleveland St.",
    "Colorado State": "Colorado St.", "Coppin State": "Coppin St.",
    "Delaware State": "Delaware St.",
    "Fairleigh Dickinson": "FDU", "Fresno State": "Fresno St.",
    "Georgia State": "Georgia St.", "Grambling": "Grambling St.",
    "Idaho State": "Idaho St.", "Illinois State": "Illinois St.",
    "Indiana State": "Indiana St.", "Iowa State": "Iowa St.",
    "Jackson State": "Jackson St.", "Jacksonville State": "Jacksonville St.",
    "Kansas State": "Kansas St.", "Kennesaw State": "Kennesaw St.",
    "Kent State": "Kent St.",
    "LIU": "Long Island University",
    "McNeese": "McNeese St.", "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.", "Mississippi Valley State": "Mississippi Valley St.",
    "Missouri State": "Missouri St.", "Montana State": "Montana St.",
    "Morehead State": "Morehead St.", "Morgan State": "Morgan St.",
    "Murray State": "Murray St.", "New Mexico State": "New Mexico St.",
    "Norfolk State": "Norfolk St.",
    "North Dakota State": "North Dakota St.", "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.", "Ole Miss": "Mississippi",
    "Oregon State": "Oregon St.", "Penn State": "Penn St.",
    "Portland State": "Portland St.", "Prairie View A&M": "Prairie View A&M",
    "Sacramento State": "Sacramento St.", "Sam Houston": "Sam Houston St.",
    "San Diego State": "San Diego St.", "San José State": "San Jose St.",
    "South Carolina State": "South Carolina St.",
    "South Dakota State": "South Dakota St.",
    "Southeast Missouri State": "Southeast Missouri St.",
    "Southern Mississippi": "Southern Miss",
    "Tennessee State": "Tennessee St.",
    "Texas A&M-Commerce": "Texas A&M Commerce",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Christi",
    "Texas State": "Texas St.",
    "UConn": "Connecticut", "UMass": "Massachusetts",
    "Utah State": "Utah St.",
    "Weber State": "Weber St.", "Wichita State": "Wichita St.",
    "Wright State": "Wright St.", "Youngstown State": "Youngstown St.",
    "East Tennessee State": "East Tennessee St.",
    "Tarleton State": "Tarleton St.",
    "Queens University": "Queens",
    "Illinois Chicago": "Illinois Chicago",
    "St. John's": "St. John's",
    "North Alabama": "North Alabama",
    "Purdue Fort Wayne": "Purdue Fort Wayne",
    "SIU Edwardsville": "SIU Edwardsville",
    "Saint Francis": "St. Francis PA",
    "Saint Peter's": "Saint Peter's",
    "Stonehill": "Stonehill",
    "Le Moyne": "Le Moyne",
    "West Georgia": "West Georgia",
    "Lindenwood": "Lindenwood",
}


def compute_torvik_correlations(ratings_df, torvik_df):
    """Compute monthly correlations with Torvik."""
    gold = ratings_df.copy()
    gold["date"] = pd.to_datetime(gold["rating_date"]).dt.date
    gold["torvik_name"] = gold["team"].map(lambda x: GOLD_TO_TORVIK.get(x, x))

    results = {}
    for month_name, start, end in [
        ("nov", "2024-11-01", "2024-11-30"),
        ("dec", "2024-12-01", "2024-12-31"),
        ("jan", "2025-01-01", "2025-01-31"),
        ("mar", "2025-03-01", "2025-03-31"),
    ]:
        s = pd.Timestamp(start).date()
        e = pd.Timestamp(end).date()
        g = gold[(gold["date"] >= s) & (gold["date"] <= e)]
        t = torvik_df[(torvik_df["date"] >= s) & (torvik_df["date"] <= e)]

        merged = g.merge(t, left_on=["torvik_name", "date"],
                         right_on=["team_name", "date"], suffixes=("_g", "_t"))

        for metric in ["adj_oe", "adj_de"]:
            valid = merged[[f"{metric}_g", f"{metric}_t"]].dropna()
            if len(valid) > 0:
                results[f"{month_name}_{metric}_r"] = valid[f"{metric}_g"].corr(valid[f"{metric}_t"])
                results[f"{month_name}_{metric}_mae"] = (valid[f"{metric}_g"] - valid[f"{metric}_t"]).abs().mean()

    return results


def build_features_from_ratings(ratings_df, games_df, rolling_lookup):
    """Build feature DataFrame using custom efficiency ratings."""
    # Build efficiency lookup
    ratings_df = ratings_df.copy()
    ratings_df["rating_date"] = pd.to_datetime(ratings_df["rating_date"])

    eff_lookup = {}
    for tid, group in ratings_df.groupby("teamId"):
        eff_lookup[int(tid)] = group[["rating_date", "adj_oe", "adj_de", "adj_tempo", "barthag"]].copy()

    games_copy = games_df.copy()
    games_copy["_game_dt"] = pd.to_datetime(games_copy["startDate"], errors="coerce")

    feat_records = []
    for _, game in games_copy.iterrows():
        gid = int(game["gameId"])
        home_tid = int(game["homeTeamId"])
        away_tid = int(game["awayTeamId"])
        game_dt = game["_game_dt"]

        home_eff = _get_asof_rating(eff_lookup, home_tid, game_dt)
        away_eff = _get_asof_rating(eff_lookup, away_tid, game_dt)

        home_roll = rolling_lookup.get((gid, home_tid), {})
        away_roll = rolling_lookup.get((gid, away_tid), {})

        neutral = bool(game.get("neutralSite", False))

        rec = {
            "gameId": gid,
            "homeTeamId": home_tid,
            "awayTeamId": away_tid,
            "startDate": game.get("startDate"),
            "homeScore": game.get("homeScore"),
            "awayScore": game.get("awayScore"),
            # Indicator features
            "neutral_site": int(neutral),
            "home_team_home": int(not neutral),
            "away_team_home": 0,
            # Group 1: efficiency features
            "home_team_adj_oe": home_eff.get("adj_oe"),
            "home_team_adj_de": home_eff.get("adj_de"),
            "home_team_adj_pace": home_eff.get("adj_tempo"),
            "home_team_BARTHAG": home_eff.get("barthag"),
            "away_team_adj_oe": away_eff.get("adj_oe"),
            "away_team_adj_de": away_eff.get("adj_de"),
            "away_team_adj_pace": away_eff.get("adj_tempo"),
            "away_team_BARTHAG": away_eff.get("barthag"),
        }

        # Group 2: rolling four-factor features (use correct column mappings)
        away_roll = rolling_lookup.get((gid, away_tid), {})
        for feat_name, rolling_col in AWAY_ROLLING_MAP.items():
            rec[feat_name] = away_roll.get(rolling_col)

        home_roll = rolling_lookup.get((gid, home_tid), {})
        for feat_name, rolling_col in HOME_ROLLING_MAP.items():
            rec[feat_name] = home_roll.get(rolling_col)

        feat_records.append(rec)

    feat_df = pd.DataFrame(feat_records)
    return feat_df


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
    # Find all variant parquets
    variant_files = sorted(SWEEP_DIR.glob("ratings_*.parquet"))
    if not variant_files:
        print(f"No rating files found in {SWEEP_DIR}")
        return
    print(f"Found {len(variant_files)} variants:")
    for f in variant_files:
        print(f"  {f.name}")

    # Load model training data
    best_hp = load_best_hparams()
    reg_hp = best_hp.get("regressor", {})

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

    # Load holdout game data for feature building
    print("\nLoading holdout game data...")
    games = load_games(HOLDOUT_SEASON)
    boxscores = load_boxscores(HOLDOUT_SEASON)
    rolling_df = pd.DataFrame()
    if not boxscores.empty:
        ff = compute_game_four_factors(boxscores)
        rolling_df = compute_rolling_averages(ff)
    rolling_lookup = {}
    if not rolling_df.empty:
        for _, row in rolling_df.iterrows():
            key = (int(row["gameid"]), int(row["teamid"]))
            rolling_lookup[key] = row.to_dict()
    print(f"  Games: {len(games)}, Rolling entries: {len(rolling_lookup)}")

    # Load Torvik for correlation
    print("Loading Torvik data...")
    torvik_df = load_torvik()

    results = []

    for f in variant_files:
        label = f.stem.replace("ratings_", "")
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        # Load variant ratings
        ratings_df = pd.read_parquet(f)

        # Torvik correlations
        torvik_corr = compute_torvik_correlations(ratings_df, torvik_df)
        nov_oe_r = torvik_corr.get("nov_adj_oe_r", None)
        dec_oe_r = torvik_corr.get("dec_adj_oe_r", None)
        mar_oe_r = torvik_corr.get("mar_adj_oe_r", None)
        print(f"  Torvik r: Nov={nov_oe_r:.4f}" if nov_oe_r else "  Torvik r: Nov=N/A",
              f"Dec={dec_oe_r:.4f}" if dec_oe_r else "Dec=N/A",
              f"Mar={mar_oe_r:.4f}" if mar_oe_r else "Mar=N/A")

        # Build features
        feat_df = build_features_from_ratings(ratings_df, games, rolling_lookup)
        feat_df = feat_df.dropna(subset=["homeScore", "awayScore"])

        X_test = get_feature_matrix(feat_df).values.astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = get_targets(feat_df)["spread_home"].values.astype(np.float32)

        # Model evaluation
        mae = evaluate_mae(model, X_test, y_test, scaler)
        monthly = evaluate_monthly_mae(model, feat_df, X_test, y_test, scaler)

        print(f"  Holdout MAE: {mae:.4f}")
        print(f"  Monthly: Nov={monthly.get('2024-11', 'N/A'):.2f}",
              f"Dec={monthly.get('2024-12', 'N/A'):.2f}",
              f"Jan={monthly.get('2025-01', 'N/A'):.2f}")

        result_row = {
            "label": label,
            "holdout_mae": mae,
            "nov_mae": monthly.get("2024-11"),
            "dec_mae": monthly.get("2024-12"),
            "jan_mae": monthly.get("2025-01"),
            "feb_mae": monthly.get("2025-02"),
            "mar_mae": monthly.get("2025-03"),
            **torvik_corr,
        }
        results.append(result_row)

    # Summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("holdout_mae")

    print(f"\n{'='*80}")
    print("SOS SWEEP RESULTS (sorted by holdout MAE)")
    print(f"{'='*80}")
    cols = ["label", "holdout_mae", "nov_mae", "dec_mae", "nov_adj_oe_r", "dec_adj_oe_r", "mar_adj_oe_r"]
    print(f"{'Label':<25} {'MAE':>7} {'NovMAE':>7} {'DecMAE':>7} {'NovOEr':>7} {'DecOEr':>7} {'MarOEr':>7}")
    print("-" * 80)
    for _, r in results_df.iterrows():
        print(f"{r['label']:<25} {r['holdout_mae']:>7.4f} "
              f"{r.get('nov_mae', 0):>7.2f} {r.get('dec_mae', 0):>7.2f} "
              f"{r.get('nov_adj_oe_r', 0):>7.4f} {r.get('dec_adj_oe_r', 0):>7.4f} "
              f"{r.get('mar_adj_oe_r', 0):>7.4f}")

    out_path = Path(__file__).resolve().parent.parent / "reports" / "sos_sweep_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
