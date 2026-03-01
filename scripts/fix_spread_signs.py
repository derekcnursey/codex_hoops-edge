#!/usr/bin/env python3
"""Fix flipped spread signs in existing prediction CSVs.

Uses cross-provider majority vote: when multiple providers report spreads
for the same game and the selected provider's sign disagrees with the
majority, flip it. Falls back to moneyline cross-check for single-provider
games.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import load_lines  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "predictions" / "csv"
CSV_TO_JSON = PROJECT_ROOT / "scripts" / "csv_to_json.py"


def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def prob_to_american(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    out = np.full_like(p, np.nan, dtype=float)
    fav = p >= 0.5
    dog = ~fav
    out[fav] = -100.0 * (p[fav] / (1.0 - p[fav]))
    out[dog] = 100.0 * ((1.0 - p[dog]) / p[dog])
    return out


def build_correct_signs() -> dict[int, float]:
    """Build gameId → correct spread sign from multi-provider majority vote."""
    correct_sign: dict[int, float] = {}

    for season in range(2016, 2027):
        lines_df = load_lines(season)
        if lines_df.empty:
            continue

        lines_df = lines_df.copy()
        lines_df["spread"] = pd.to_numeric(lines_df["spread"], errors="coerce")
        lines_df["homeMoneyline"] = pd.to_numeric(
            lines_df["homeMoneyline"], errors="coerce"
        )

        # Majority vote from providers with non-zero spread
        has_spread = lines_df["spread"].notna() & (lines_df["spread"] != 0)
        if not has_spread.any():
            continue

        spread_sign = np.sign(lines_df.loc[has_spread, "spread"])
        majority = (
            spread_sign.groupby(lines_df.loc[has_spread, "gameId"]).sum()
        )

        # Only store where majority is decisive (net sign != 0)
        for gid, net in majority.items():
            if net != 0:
                correct_sign[int(gid)] = float(np.sign(net))

        # For single-provider games, use moneyline as cross-check
        provider_count = (
            lines_df.loc[has_spread]
            .groupby("gameId")["provider"]
            .nunique()
        )
        single_provider_games = set(provider_count[provider_count == 1].index)

        for gid in single_provider_games:
            if gid in correct_sign:
                continue  # already have majority
            row = lines_df[lines_df["gameId"] == gid].iloc[0]
            sp = row["spread"]
            ml = row["homeMoneyline"]
            if pd.notna(sp) and pd.notna(ml) and abs(sp) > 3:
                if (sp > 0 and ml < -150) or (sp < 0 and ml > 150):
                    correct_sign[int(gid)] = float(-np.sign(sp))

    return correct_sign


def recalc_edges(df: pd.DataFrame) -> None:
    """Recalculate all edge metrics in-place after spread fix."""
    df["model_spread"] = -df["predicted_spread"]
    df["spread_diff"] = df["model_spread"] - df["book_spread"]
    df["edge_home_points"] = df["predicted_spread"] + df["book_spread"]

    sigma_safe = df["spread_sigma"].clip(lower=0.5)
    edge_z = df["edge_home_points"] / sigma_safe
    home_cover_prob = normal_cdf(edge_z.values)
    away_cover_prob = 1.0 - home_cover_prob

    df["pick_side"] = np.where(df["edge_home_points"] >= 0, "HOME", "AWAY")
    df["pick_cover_prob"] = np.where(
        df["edge_home_points"] >= 0, home_cover_prob, away_cover_prob
    )

    pick_breakeven = 110 / 210  # -110 odds breakeven
    pick_profit = 100 / 110  # -110 odds profit per $1

    df["pick_prob_edge"] = df["pick_cover_prob"] - pick_breakeven
    df["pick_ev_per_1"] = (
        df["pick_cover_prob"] * pick_profit - (1.0 - df["pick_cover_prob"])
    )
    df["pick_fair_odds"] = prob_to_american(df["pick_cover_prob"].values)


def main():
    print("Loading multi-provider lines from S3...")
    correct_signs = build_correct_signs()
    print(f"Built correct signs for {len(correct_signs)} games\n")

    csv_files = sorted(CSV_DIR.glob("preds_*_edge.csv"))
    total_fixed = 0
    files_changed = 0

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "book_spread" not in df.columns or "gameId" not in df.columns:
            continue

        sp = pd.to_numeric(df["book_spread"], errors="coerce")
        n_fix = 0

        for idx, row in df.iterrows():
            gid = int(row["gameId"]) if pd.notna(row["gameId"]) else None
            spread = sp[idx]
            if gid is None or pd.isna(spread) or spread == 0:
                continue

            target_sign = correct_signs.get(gid)
            if target_sign is not None and np.sign(spread) != target_sign:
                df.at[idx, "book_spread"] = -spread
                n_fix += 1

        if n_fix == 0:
            continue

        recalc_edges(df)
        df.to_csv(csv_path, index=False)
        total_fixed += n_fix
        files_changed += 1

        # Regenerate site JSON
        parts = csv_path.stem.replace("preds_", "").replace("_edge", "").split("_")
        if len(parts) >= 3:
            date_str = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            subprocess.run(
                [sys.executable, str(CSV_TO_JSON), str(csv_path), date_str],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
            )
            print(f"  Fixed {n_fix} rows in {csv_path.name} → {date_str}")

    print(f"\nDone: {total_fixed} total rows fixed across {files_changed} files")


if __name__ == "__main__":
    main()
