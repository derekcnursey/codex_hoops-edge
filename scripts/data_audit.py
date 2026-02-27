"""Pre-training data audit for session 13.

Loads the full training dataset (seasons 2015-2025) using the same code path
as `cli.py train` and runs 8 comprehensive checks. Outputs a markdown report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── project imports ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config
from src.features import get_feature_matrix, get_targets

FEATURE_ORDER = config.FEATURE_ORDER
SEASONS = list(range(2015, 2026))
ADJ_SUFFIX = "no_garbage_adj_a0.85_p10"

# ── Load data (same path as cli.py train) ────────────────────────

def load_all() -> pd.DataFrame:
    dfs = []
    for s in SEASONS:
        path = config.FEATURES_DIR / f"season_{s}_{ADJ_SUFFIX}_features.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
        else:
            print(f"WARNING: missing {path}")
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_lines_all() -> pd.DataFrame:
    dfs = []
    for s in SEASONS:
        path = config.FEATURES_DIR / f"lines_{s}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ── Report builder ───────────────────────────────────────────────

lines: list[str] = []

def h1(text: str):
    lines.append(f"\n# {text}\n")

def h2(text: str):
    lines.append(f"\n## {text}\n")

def h3(text: str):
    lines.append(f"\n### {text}\n")

def p(text: str):
    lines.append(text)

def table(headers: list[str], rows: list[list], align: list[str] | None = None):
    """Emit a markdown table."""
    if align is None:
        align = ["---"] * len(headers)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(align) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")


def main():
    h1("Data Audit — Pre-Session 13 Training")
    p(f"**Seasons**: {SEASONS[0]}–{SEASONS[-1]}")
    p(f"**Feature set**: {len(FEATURE_ORDER)} features ({ADJ_SUFFIX})")
    p(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # Load
    print("Loading all seasons...")
    df_raw = load_all()
    p(f"\n**Raw rows loaded**: {len(df_raw):,}")

    # Drop unplayed games (same as training)
    df = df_raw.dropna(subset=["homeScore", "awayScore"]).copy()
    p(f"**After dropping unplayed**: {len(df):,}")

    feat_df = get_feature_matrix(df, feature_order=FEATURE_ORDER)
    targets = get_targets(df)
    y = targets["spread_home"].values.astype(np.float32)

    # ================================================================
    # AUDIT 1 — FEATURE COMPLETENESS
    # ================================================================
    h2("Audit 1 — Feature Completeness")

    flagged_nan = []
    flagged_zero = []
    audit1_rows = []
    for col in FEATURE_ORDER:
        vals = feat_df[col]
        n = len(vals)
        n_nan = int(vals.isna().sum())
        nan_pct = 100 * n_nan / n
        non_na = vals.dropna()
        n_zero = int((non_na == 0).sum())
        zero_pct = 100 * n_zero / n if n > 0 else 0
        if len(non_na) > 0:
            mean_v = f"{non_na.mean():.4f}"
            std_v = f"{non_na.std():.4f}"
            min_v = f"{non_na.min():.4f}"
            max_v = f"{non_na.max():.4f}"
        else:
            mean_v = std_v = min_v = max_v = "N/A"

        flag = ""
        if nan_pct > 5:
            flag += " NaN!"
            flagged_nan.append((col, nan_pct))
        if zero_pct > 50:
            flag += " ZERO!"
            flagged_zero.append((col, zero_pct))

        audit1_rows.append([
            col, f"{n_nan:,}", f"{nan_pct:.1f}%",
            f"{n_zero:,}", f"{zero_pct:.1f}%",
            mean_v, std_v, min_v, max_v, flag.strip(),
        ])

    table(
        ["Feature", "NaN", "NaN%", "Zero", "Zero%", "Mean", "Std", "Min", "Max", "Flag"],
        audit1_rows,
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
    )

    if flagged_nan:
        h3("Features with >5% NaN")
        for col, pct in flagged_nan:
            p(f"- **{col}**: {pct:.1f}% NaN")
    else:
        p("\nAll features have ≤5% NaN.")

    if flagged_zero:
        h3("Features with >50% Zero")
        for col, pct in flagged_zero:
            p(f"- **{col}**: {pct:.1f}% zero")
    else:
        p("No features have >50% zero (excluding neutral_site).")

    # ================================================================
    # AUDIT 2 — TARGET VARIABLE
    # ================================================================
    h2("Audit 2 — Target Variable (Home Margin)")

    p(f"- **Total games**: {len(y):,}")
    p(f"- **Mean**: {y.mean():.2f}")
    p(f"- **Median**: {np.median(y):.2f}")
    p(f"- **Std**: {y.std():.2f}")
    p(f"- **Min**: {y.min():.0f}")
    p(f"- **Max**: {y.max():.0f}")

    p("")
    for band in [5, 10, 20, 30]:
        pct = 100 * np.mean(np.abs(y) <= band)
        p(f"- Within ±{band}: {pct:.1f}%")

    extreme = df[np.abs(y) > 60].copy()
    extreme["margin"] = y[np.abs(y) > 60]
    p(f"\n**Games with |margin| > 60**: {len(extreme)}")
    if len(extreme) > 0:
        extreme_sorted = extreme.sort_values("margin", key=abs, ascending=False)
        ext_rows = []
        for _, r in extreme_sorted.head(20).iterrows():
            ext_rows.append([
                int(r["gameId"]),
                r.get("startDate", "")[:10] if pd.notna(r.get("startDate")) else "",
                f"{int(r.get('awayTeamId', 0))}",
                f"{int(r.get('homeTeamId', 0))}",
                f"{int(r.get('awayScore', 0))}",
                f"{int(r.get('homeScore', 0))}",
                f"{r['margin']:+.0f}",
            ])
        table(
            ["gameId", "Date", "Away", "Home", "AwayPts", "HomePts", "Margin"],
            ext_rows,
        )

    n_zero_margin = int(np.sum(y == 0))
    p(f"\n**Games with margin = 0**: {n_zero_margin}")

    # ================================================================
    # AUDIT 3 — DUPLICATE / CORRUPT GAMES
    # ================================================================
    h2("Audit 3 — Duplicate / Corrupt Games")

    n_dup = int(df["gameId"].duplicated().sum())
    p(f"- **Duplicate gameIds**: {n_dup}")

    same_team = df[df["homeTeamId"] == df["awayTeamId"]]
    p(f"- **Home == Away team**: {len(same_team)}")

    missing_tid = df[df["homeTeamId"].isna() | df["awayTeamId"].isna()]
    p(f"- **Missing team IDs**: {len(missing_tid)}")

    zero_zero = df[(df["homeScore"] == 0) & (df["awayScore"] == 0)]
    p(f"- **0-0 games**: {len(zero_zero)}")
    if len(zero_zero) > 0:
        zz_rows = []
        for _, r in zero_zero.head(10).iterrows():
            zz_rows.append([
                int(r["gameId"]),
                str(r.get("startDate", ""))[:10],
                int(r.get("awayTeamId", 0)),
                int(r.get("homeTeamId", 0)),
            ])
        table(["gameId", "Date", "Away", "Home"], zz_rows)

    issues = n_dup + len(same_team) + len(missing_tid) + len(zero_zero)
    if issues == 0:
        p("\nNo data integrity issues found.")
    else:
        p(f"\n**Total issues**: {issues}")

    # ================================================================
    # AUDIT 4 — SEASON DISTRIBUTION
    # ================================================================
    h2("Audit 4 — Season Distribution")

    # Determine season from startDate
    dates = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    game_month = dates.dt.month
    game_year = dates.dt.year
    # Season convention: Aug-Dec = season year+1, Jan-Jul = season year
    df["_season"] = game_year.where(game_month <= 7, game_year + 1)

    season_counts = df.groupby("_season").size().sort_index()
    s_rows = []
    flagged_seasons = []
    for s, cnt in season_counts.items():
        flag = ""
        if cnt < 1000:
            flag = "LOW!"
            flagged_seasons.append((int(s), cnt))
        s_rows.append([int(s), f"{cnt:,}", flag])

    table(["Season", "Games", "Flag"], s_rows)

    if flagged_seasons:
        h3("Flagged Seasons (<1000 games)")
        for s, cnt in flagged_seasons:
            p(f"- Season {s}: only {cnt:,} games")
    else:
        p("\nAll seasons have ≥1000 games.")

    # ================================================================
    # AUDIT 5 — FEATURE CORRELATION WITH TARGET
    # ================================================================
    h2("Audit 5 — Feature Correlation with Target")

    X_raw = feat_df.values.astype(np.float64)
    # Fill NaN with column mean for correlation
    col_means = np.nanmean(X_raw, axis=0)
    nan_mask = np.isnan(X_raw)
    for j in range(X_raw.shape[1]):
        X_raw[nan_mask[:, j], j] = col_means[j]

    corr_rows = []
    noise_features = []
    for i, col in enumerate(FEATURE_ORDER):
        r_val = np.corrcoef(X_raw[:, i], y)[0, 1]
        flag = ""
        if abs(r_val) < 0.01:
            flag = "NOISE"
            noise_features.append((col, r_val))
        corr_rows.append([col, f"{r_val:+.4f}", flag])

    # Sort by abs correlation descending
    corr_rows.sort(key=lambda r: abs(float(r[1])), reverse=True)
    table(["Feature", "Pearson r", "Flag"], corr_rows)

    if noise_features:
        h3("Near-zero correlation features (|r| < 0.01)")
        for col, r_val in noise_features:
            p(f"- **{col}**: r = {r_val:+.4f}")
    else:
        p("\nAll features have |r| ≥ 0.01 with target.")

    # ================================================================
    # AUDIT 6 — FEATURE-TO-FEATURE CORRELATION
    # ================================================================
    h2("Audit 6 — Feature-to-Feature Correlation")

    corr_matrix = np.corrcoef(X_raw.T)
    high_corr_pairs = []
    for i in range(len(FEATURE_ORDER)):
        for j in range(i + 1, len(FEATURE_ORDER)):
            r = corr_matrix[i, j]
            if abs(r) > 0.95:
                high_corr_pairs.append((FEATURE_ORDER[i], FEATURE_ORDER[j], r))

    if high_corr_pairs:
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        hc_rows = [[a, b, f"{r:.4f}"] for a, b, r in high_corr_pairs]
        table(["Feature A", "Feature B", "r"], hc_rows)
    else:
        p("No feature pairs with |r| > 0.95.")

    # Also report pairs > 0.90 for awareness
    moderate_pairs = []
    for i in range(len(FEATURE_ORDER)):
        for j in range(i + 1, len(FEATURE_ORDER)):
            r = corr_matrix[i, j]
            if 0.90 < abs(r) <= 0.95:
                moderate_pairs.append((FEATURE_ORDER[i], FEATURE_ORDER[j], r))
    if moderate_pairs:
        h3("Pairs with 0.90 < |r| ≤ 0.95 (for awareness)")
        moderate_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        mc_rows = [[a, b, f"{r:.4f}"] for a, b, r in moderate_pairs]
        table(["Feature A", "Feature B", "r"], mc_rows)

    # ================================================================
    # AUDIT 7 — BOOK SPREAD AVAILABILITY
    # ================================================================
    h2("Audit 7 — Book Spread Availability")

    lines_df = load_lines_all()
    if lines_df.empty:
        p("No lines data found in features/ directory.")
    else:
        # Normalize column name
        if "gameId" not in lines_df.columns and "gameid" in lines_df.columns:
            lines_df = lines_df.rename(columns={"gameid": "gameId"})
        if "spread" not in lines_df.columns and "Spread" in lines_df.columns:
            lines_df = lines_df.rename(columns={"Spread": "spread"})

        # One row per game (first provider)
        if "provider" in lines_df.columns:
            lines_dedup = lines_df.sort_values("provider").drop_duplicates(subset=["gameId"], keep="first")
        else:
            lines_dedup = lines_df.drop_duplicates(subset=["gameId"], keep="first")

        has_spread = lines_dedup[lines_dedup["spread"].notna()]["gameId"]
        spread_set = set(has_spread)

        book_rows = []
        for s_val in sorted(df["_season"].dropna().unique()):
            s_int = int(s_val)
            s_games = df[df["_season"] == s_val]
            n_s = len(s_games)
            n_with = s_games["gameId"].isin(spread_set).sum()
            pct = 100 * n_with / n_s if n_s > 0 else 0
            flag = ""
            if pct < 50:
                flag = "LOW"
            book_rows.append([s_int, f"{n_s:,}", f"{n_with:,}", f"{pct:.1f}%", flag])

        table(["Season", "Games", "With Spread", "%", "Flag"], book_rows)

    # ================================================================
    # AUDIT 8 — SCALER SANITY
    # ================================================================
    h2("Audit 8 — Scaler Sanity")

    X_for_scaler = feat_df.values.astype(np.float32)
    X_for_scaler = np.nan_to_num(X_for_scaler, nan=0.0)

    scaler = StandardScaler()
    scaler.fit(X_for_scaler)

    sc_rows = []
    flagged_scaler = []
    for i, col in enumerate(FEATURE_ORDER):
        mean_v = scaler.mean_[i]
        std_v = scaler.scale_[i]
        flag = ""
        if std_v < 1e-6:
            flag = "ZERO STD!"
            flagged_scaler.append((col, "near-zero std", std_v))
        if abs(mean_v) > 1000:
            flag += " LARGE MEAN!"
            flagged_scaler.append((col, "absurdly large mean", mean_v))
        sc_rows.append([col, f"{mean_v:.6f}", f"{std_v:.6f}", flag])

    table(["Feature", "Scaler Mean", "Scaler Std", "Flag"], sc_rows)

    if flagged_scaler:
        h3("Scaler Issues")
        for col, issue, val in flagged_scaler:
            p(f"- **{col}**: {issue} ({val:.6f})")
    else:
        p("\nScaler looks healthy — no near-zero stds or extreme means.")

    # ================================================================
    # SUMMARY
    # ================================================================
    h2("Summary — Go / No-Go")

    all_issues = []
    if flagged_nan:
        all_issues.append(f"{len(flagged_nan)} features with >5% NaN")
    if flagged_zero:
        all_issues.append(f"{len(flagged_zero)} features with >50% zero")
    if n_dup > 0:
        all_issues.append(f"{n_dup} duplicate gameIds")
    if len(same_team) > 0:
        all_issues.append(f"{len(same_team)} games with home==away")
    if len(zero_zero) > 0:
        all_issues.append(f"{len(zero_zero)} games scored 0-0")
    if flagged_seasons:
        all_issues.append(f"{len(flagged_seasons)} seasons with <1000 games")
    if noise_features:
        all_issues.append(f"{len(noise_features)} features with |r|<0.01")
    if high_corr_pairs:
        all_issues.append(f"{len(high_corr_pairs)} feature pairs with |r|>0.95")
    if flagged_scaler:
        all_issues.append(f"{len(flagged_scaler)} scaler issues")

    if all_issues:
        p("**Issues found**:")
        for issue in all_issues:
            p(f"- {issue}")
    else:
        p("**All checks passed. Ready for GPU training.**")

    # ── Write report ─────────────────────────────────────────────
    report = "\n".join(lines)
    out_path = ROOT / "reports" / "data_audit_pre_session13.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport saved to: {out_path}")
    print(report)


if __name__ == "__main__":
    main()
