#!/usr/bin/env python3
"""Fix flipped spread signs in existing prediction CSVs.

Some games in the CBBD API have spread signs that disagree with moneyline
(e.g. spread says home is 23-pt underdog while moneyline says massive
home favorite). This script detects and fixes those, then recalculates
all edge metrics and regenerates site JSONs.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def fix_csv(csv_path: Path) -> int:
    """Fix spread signs in a single CSV. Returns number of rows fixed."""
    df = pd.read_csv(csv_path)

    if "book_spread" not in df.columns or "home_moneyline" not in df.columns:
        return 0

    sp = pd.to_numeric(df["book_spread"], errors="coerce")
    ml = pd.to_numeric(df["home_moneyline"], errors="coerce")

    mask = (
        sp.notna() & ml.notna()
        & (
            ((sp > 3) & (ml < -150))
            | ((sp < -3) & (ml > 150))
        )
    )

    n_fix = mask.sum()
    if n_fix == 0:
        return 0

    # Flip the spread sign
    df.loc[mask, "book_spread"] = -sp[mask]

    # Recalculate edge metrics
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
    df["pick_ev_per_1"] = df["pick_cover_prob"] * pick_profit - (1.0 - df["pick_cover_prob"])
    df["pick_fair_odds"] = prob_to_american(df["pick_cover_prob"].values)

    df.to_csv(csv_path, index=False)
    return n_fix


def main():
    csv_files = sorted(CSV_DIR.glob("preds_*_edge.csv"))
    total_fixed = 0
    files_changed = 0

    for csv_path in csv_files:
        n = fix_csv(csv_path)
        if n > 0:
            total_fixed += n
            files_changed += 1

            # Extract date from filename and regenerate JSON
            parts = csv_path.stem.replace("preds_", "").replace("_edge", "").split("_")
            if len(parts) >= 3:
                date_str = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                subprocess.run(
                    [sys.executable, str(CSV_TO_JSON), str(csv_path), date_str],
                    check=True,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                )

            print(f"  Fixed {n} rows in {csv_path.name} → {date_str}")

    print(f"\nDone: {total_fixed} total rows fixed across {files_changed} files")


if __name__ == "__main__":
    main()
