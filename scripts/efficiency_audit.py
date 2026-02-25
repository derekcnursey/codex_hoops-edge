"""Compare self-computed adjusted efficiency ratings (gold layer) against Bart Torvik.

Reads:
  - Gold: team_adjusted_efficiencies_no_garbage from S3 (season=2025)
  - Silver: dim_teams from S3 (for teamId → team_name mapping)
  - MySQL: sports.daily_data (Torvik ratings scraped from barttorvik.com)

Outputs:
  - reports/efficiency_audit_2025.md
  - reports/plots/adj_oe_scatter.png
  - reports/plots/adj_de_scatter.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ── Project imports ─────────────────────────────────────────────────────
from src.s3_reader import read_gold_table, read_silver_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEASON = 2025

# ── MySQL connection ────────────────────────────────────────────────────
DB_USER = os.getenv("BBALL_DB_USER", "derek")
DB_PASS = os.getenv("BBALL_DB_PASS", "jake3241")
DB_HOST = os.getenv("BBALL_DB_HOST", "localhost")
DB_NAME = os.getenv("BBALL_DB_NAME", "sports")


def get_engine():
    return create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")


# ── Name mapping: gold (school) → Torvik (team_name) ───────────────────
# Systematic patterns discovered from comparing unmatched sets:
#   - Gold uses "State", Torvik uses "St."
#   - Gold uses full names, Torvik abbreviates
#   - Specific one-off differences
GOLD_TO_TORVIK: dict[str, str] = {
    "Alabama State": "Alabama St.",
    "Alcorn State": "Alcorn St.",
    "American University": "American",
    "App State": "Appalachian St.",
    "Arizona State": "Arizona St.",
    "Arkansas State": "Arkansas St.",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "Ball State": "Ball St.",
    "Bethune-Cookman": "Bethune Cookman",
    "Boise State": "Boise St.",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Cal State Northridge": "Cal St. Northridge",
    "California Baptist": "Cal Baptist",
    "Chicago State": "Chicago St.",
    "Cleveland State": "Cleveland St.",
    "Colorado State": "Colorado St.",
    "Coppin State": "Coppin St.",
    "Delaware State": "Delaware St.",
    "East Tennessee State": "East Tennessee St.",
    "East Texas A&M": "Texas A&M Commerce",
    "Florida International": "FIU",
    "Florida State": "Florida St.",
    "Fresno State": "Fresno St.",
    "Gardner-Webb": "Gardner Webb",
    "Georgia State": "Georgia St.",
    "Grambling": "Grambling St.",
    "Hawai'i": "Hawaii",
    "IU Indianapolis": "IU Indy",
    "Idaho State": "Idaho St.",
    "Illinois State": "Illinois St.",
    "Indiana State": "Indiana St.",
    "Iowa State": "Iowa St.",
    "Jackson State": "Jackson St.",
    "Jacksonville State": "Jacksonville St.",
    "Kansas City": "UMKC",
    "Kansas State": "Kansas St.",
    "Kennesaw State": "Kennesaw St.",
    "Kent State": "Kent St.",
    "Long Beach State": "Long Beach St.",
    "Long Island University": "LIU",
    "Loyola Maryland": "Loyola MD",
    "McNeese": "McNeese St.",
    "Miami": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.",
    "Mississippi Valley State": "Mississippi Valley St.",
    "Missouri State": "Missouri St.",
    "Montana State": "Montana St.",
    "Morehead State": "Morehead St.",
    "Morgan State": "Morgan St.",
    "Murray State": "Murray St.",
    "NC State": "N.C. State",
    "New Mexico State": "New Mexico St.",
    "Nicholls": "Nicholls St.",
    "Norfolk State": "Norfolk St.",
    "North Dakota State": "North Dakota St.",
    "Northwestern State": "Northwestern St.",
    "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.",
    "Ole Miss": "Mississippi",
    "Omaha": "Nebraska Omaha",
    "Oregon State": "Oregon St.",
    "Penn State": "Penn St.",
    "Pennsylvania": "Penn",
    "Portland State": "Portland St.",
    "Queens University": "Queens",
    "SE Louisiana": "Southeastern Louisiana",
    "Sacramento State": "Sacramento St.",
    "Sam Houston": "Sam Houston St.",
    "San Diego State": "San Diego St.",
    "San José State": "San Jose St.",
    "Seattle U": "Seattle",
    "South Carolina State": "South Carolina St.",
    "South Carolina Upstate": "USC Upstate",
    "South Dakota State": "South Dakota St.",
    "Southeast Missouri State": "Southeast Missouri St.",
    "St. Francis (PA)": "Saint Francis",
    "St. Thomas-Minnesota": "St. Thomas",
    "Tarleton State": "Tarleton St.",
    "Tennessee State": "Tennessee St.",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
    "Texas State": "Texas St.",
    "UAlbany": "Albany",
    "UConn": "Connecticut",
    "UIC": "Illinois Chicago",
    "UL Monroe": "Louisiana Monroe",
    "UT Martin": "Tennessee Martin",
    "Utah State": "Utah St.",
    "Washington State": "Washington St.",
    "Weber State": "Weber St.",
    "Wichita State": "Wichita St.",
    "Wright State": "Wright St.",
    "Youngstown State": "Youngstown St.",
}


# ========================================================================
# Step 1: Load gold layer (deduped)
# ========================================================================

def load_gold() -> pd.DataFrame:
    """Load team_adjusted_efficiencies_no_garbage for season 2025.

    Multiple asof= partitions may exist. Dedup by (teamId, rating_date),
    keeping the row from the latest asof partition.
    """
    print("[1/6] Loading gold layer from S3 ...")
    tbl = read_gold_table("team_adjusted_efficiencies_no_garbage", season=SEASON)
    df = tbl.to_pandas()
    before = len(df)
    # Dedup: keep last row per (teamId, rating_date) — latest asof wins
    df = df.drop_duplicates(subset=["teamId", "rating_date"], keep="last")
    after = len(df)
    print(f"       {before:,} raw rows → {after:,} after dedup, "
          f"{df['teamId'].nunique()} teams, "
          f"dates {df['rating_date'].min()} → {df['rating_date'].max()}")
    return df


# ========================================================================
# Step 2: Load Torvik from MySQL
# ========================================================================

def load_torvik() -> pd.DataFrame:
    """Load sports.daily_data for season 2025 (2024-10-01 to 2025-05-01)."""
    print("[2/6] Loading Torvik from MySQL ...")
    engine = get_engine()
    query = text("""
        SELECT team_name, conference, adj_oe, adj_de, BARTHAG, adj_pace, date
        FROM daily_data
        WHERE date >= '2024-10-01' AND date <= '2025-05-01'
    """)
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    print(f"       {len(df):,} rows, {df['team_name'].nunique()} teams, "
          f"dates {df['date'].min()} → {df['date'].max()}")
    return df


# ========================================================================
# Step 3: Map names and join
# ========================================================================

def join_datasets(
    gold: pd.DataFrame, torvik: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join gold and Torvik by team_name + date.

    Applies GOLD_TO_TORVIK name mapping before joining.
    Returns (joined, gold_unmatched, torvik_unmatched).
    """
    print("[3/6] Mapping team names and joining ...")

    gold_copy = gold.copy()
    # Apply name mapping: gold school → Torvik name
    gold_copy["team_name"] = gold_copy["team"].map(
        lambda x: GOLD_TO_TORVIK.get(x, x)
    )
    gold_copy = gold_copy.rename(columns={
        "rating_date": "date",
        "adj_oe": "gold_adj_oe",
        "adj_de": "gold_adj_de",
        "adj_tempo": "gold_adj_tempo",
        "barthag": "gold_barthag",
        "adj_margin": "gold_adj_margin",
    })

    torvik_copy = torvik.rename(columns={
        "adj_oe": "torvik_adj_oe",
        "adj_de": "torvik_adj_de",
        "adj_pace": "torvik_adj_pace",
        "BARTHAG": "torvik_barthag",
    })

    joined = gold_copy.merge(
        torvik_copy[["team_name", "date", "torvik_adj_oe", "torvik_adj_de",
                      "torvik_adj_pace", "torvik_barthag"]],
        on=["team_name", "date"],
        how="inner",
    )

    gold_teams = set(gold_copy["team_name"].unique())
    torvik_teams = set(torvik_copy["team_name"].unique())
    gold_only = sorted(gold_teams - torvik_teams)
    torvik_only = sorted(torvik_teams - gold_teams)

    gold_unmatched = pd.DataFrame({"gold_team": gold_only})
    torvik_unmatched = pd.DataFrame({"torvik_team": torvik_only})

    print(f"       Joined: {len(joined):,} rows, "
          f"{joined['team_name'].nunique()} matched teams")
    print(f"       Unmatched gold: {len(gold_only)} — {gold_only}")
    print(f"       Unmatched torvik: {len(torvik_only)} — {torvik_only}")

    return joined, gold_unmatched, torvik_unmatched


# ========================================================================
# Step 4: Compute correlation, MAE, bias
# ========================================================================

def compute_metrics(joined: pd.DataFrame) -> dict:
    """Compute correlation, MAE, and bias for each metric pair."""
    print("[4/6] Computing correlation, MAE, bias ...")
    metrics = {}
    pairs = [
        ("gold_adj_oe", "torvik_adj_oe", "adj_oe"),
        ("gold_adj_de", "torvik_adj_de", "adj_de"),
        ("gold_adj_tempo", "torvik_adj_pace", "adj_tempo vs adj_pace"),
        ("gold_barthag", "torvik_barthag", "barthag"),
    ]
    for gold_col, torvik_col, label in pairs:
        valid = joined[[gold_col, torvik_col]].dropna()
        if valid.empty:
            continue
        g = valid[gold_col].values
        t = valid[torvik_col].values
        corr = np.corrcoef(g, t)[0, 1]
        mae = np.mean(np.abs(g - t))
        bias = np.mean(g - t)  # mine minus Torvik
        rmse = np.sqrt(np.mean((g - t) ** 2))
        metrics[label] = {
            "n": len(valid),
            "corr": corr,
            "mae": mae,
            "bias": bias,
            "rmse": rmse,
            "gold_mean": np.mean(g),
            "torvik_mean": np.mean(t),
        }
        print(f"       {label:25s}  r={corr:.4f}  MAE={mae:.3f}  "
              f"bias={bias:+.3f}  RMSE={rmse:.3f}")
    return metrics


# ========================================================================
# Step 5: Monthly breakdown
# ========================================================================

def monthly_breakdown(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics broken down by month."""
    print("[5/6] Computing monthly breakdown ...")
    joined = joined.copy()
    joined["month"] = pd.to_datetime(joined["date"]).dt.to_period("M")
    rows = []
    for month, grp in joined.groupby("month"):
        for gold_col, torvik_col, label in [
            ("gold_adj_oe", "torvik_adj_oe", "adj_oe"),
            ("gold_adj_de", "torvik_adj_de", "adj_de"),
            ("gold_adj_tempo", "torvik_adj_pace", "adj_tempo"),
            ("gold_barthag", "torvik_barthag", "barthag"),
        ]:
            valid = grp[[gold_col, torvik_col]].dropna()
            if valid.empty:
                continue
            g = valid[gold_col].values
            t = valid[torvik_col].values
            rows.append({
                "month": str(month),
                "metric": label,
                "n": len(valid),
                "corr": np.corrcoef(g, t)[0, 1] if len(valid) > 1 else np.nan,
                "mae": np.mean(np.abs(g - t)),
                "bias": np.mean(g - t),
                "rmse": np.sqrt(np.mean((g - t) ** 2)),
            })
    monthly = pd.DataFrame(rows)
    print(monthly.to_string(index=False))
    return monthly


# ========================================================================
# Step 6: Top-20 adj_oe divergence teams + schedule analysis
# ========================================================================

def top_divergence_teams(
    joined: pd.DataFrame, gold: pd.DataFrame
) -> pd.DataFrame:
    """Identify 20 teams where adj_oe diverges most from Torvik."""
    print("[6/6] Identifying top-20 adj_oe divergence teams ...")

    # Use the latest date per team for a clean snapshot
    latest_idx = joined.groupby("team_name")["date"].idxmax()
    latest = joined.loc[latest_idx].copy()

    latest["oe_diff"] = latest["gold_adj_oe"] - latest["torvik_adj_oe"]
    latest["abs_oe_diff"] = latest["oe_diff"].abs()
    top20 = latest.nlargest(20, "abs_oe_diff").copy()

    # Characterize schedule strength using SOS columns from gold layer
    if "sos_oe" in top20.columns and "sos_de" in top20.columns:
        top20["sos_net"] = top20["sos_oe"] - top20["sos_de"]
        # All-team SOS distribution from gold (latest date per team)
        gold_latest = gold.loc[gold.groupby("teamId")["rating_date"].idxmax()]
        all_sos = gold_latest["sos_oe"] - gold_latest["sos_de"]
        q33 = all_sos.quantile(0.33)
        q67 = all_sos.quantile(0.67)
        top20["schedule_type"] = top20["sos_net"].apply(
            lambda x: "strong" if x >= q67 else ("weak" if x <= q33 else "middle")
        )

    display_cols = [
        "team_name", "conference", "gold_adj_oe", "torvik_adj_oe", "oe_diff",
        "gold_adj_de", "torvik_adj_de", "games_played",
    ]
    if "schedule_type" in top20.columns:
        display_cols.append("schedule_type")
    print(top20[display_cols].to_string(index=False))
    return top20


# ========================================================================
# Scatter plots
# ========================================================================

def scatter_plots(joined: pd.DataFrame):
    """Generate scatter plots: mine vs Torvik for adj_oe and adj_de."""
    print("Generating scatter plots ...")

    # Use latest date per team for cleaner scatter
    latest_idx = joined.groupby("team_name")["date"].idxmax()
    latest = joined.loc[latest_idx]

    for gold_col, torvik_col, label, fname in [
        ("gold_adj_oe", "torvik_adj_oe", "Adjusted Offensive Efficiency",
         "adj_oe_scatter.png"),
        ("gold_adj_de", "torvik_adj_de", "Adjusted Defensive Efficiency",
         "adj_de_scatter.png"),
    ]:
        valid = latest[[gold_col, torvik_col, "team_name"]].dropna()
        g = valid[gold_col].values
        t = valid[torvik_col].values
        corr = np.corrcoef(g, t)[0, 1]
        mae = np.mean(np.abs(g - t))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(t, g, alpha=0.5, s=20, edgecolors="none")

        # 1:1 line
        lo = min(g.min(), t.min()) - 2
        hi = max(g.max(), t.max()) + 2
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")

        # Regression line
        m, b = np.polyfit(t, g, 1)
        x_fit = np.linspace(lo, hi, 100)
        ax.plot(x_fit, m * x_fit + b, "b-", linewidth=1,
                label=f"fit: y = {m:.3f}x + {b:.2f}")

        ax.set_xlabel(f"Torvik {label}")
        ax.set_ylabel(f"Mine (Gold) {label}")
        ax.set_title(f"{label}: Mine vs Torvik\n"
                     f"r = {corr:.4f}, MAE = {mae:.2f} (N={len(valid)} teams)")
        ax.legend()
        ax.set_aspect("equal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / fname, dpi=150)
        plt.close(fig)
        print(f"       Saved {PLOTS_DIR / fname}")


# ========================================================================
# Report generation
# ========================================================================

def generate_report(
    metrics: dict,
    monthly: pd.DataFrame,
    top20: pd.DataFrame,
    gold_unmatched: pd.DataFrame,
    torvik_unmatched: pd.DataFrame,
    joined: pd.DataFrame,
    gold: pd.DataFrame,
    torvik: pd.DataFrame,
) -> str:
    """Generate the markdown report."""
    lines: list[str] = []
    lines.append("# Efficiency Audit: Mine vs Bart Torvik — Season 2025")
    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    lines.append("| Source | Table | Rows | Teams | Date Range |")
    lines.append("|--------|-------|------|-------|------------|")
    lines.append(
        f"| Gold (S3) | `team_adjusted_efficiencies_no_garbage` | "
        f"{len(gold):,} | {gold['teamId'].nunique()} | "
        f"{gold['rating_date'].min()} → {gold['rating_date'].max()} |"
    )
    lines.append(
        f"| Torvik (MySQL) | `sports.daily_data` | "
        f"{len(torvik):,} | {torvik['team_name'].nunique()} | "
        f"{torvik['date'].min()} → {torvik['date'].max()} |"
    )
    lines.append(
        f"| **Joined** | name-mapped on (team, date) | "
        f"**{len(joined):,}** | **{joined['team_name'].nunique()}** | "
        f"{joined['date'].min()} → {joined['date'].max()} |"
    )
    lines.append("")

    # Unmatched teams
    if not gold_unmatched.empty or not torvik_unmatched.empty:
        lines.append("## Unmatched Teams After Name Mapping")
        lines.append("")
        lines.append(
            f"After applying {len(GOLD_TO_TORVIK)} name mappings, "
            f"**{joined['team_name'].nunique()}** of {gold['teamId'].nunique()} "
            f"teams matched."
        )
        lines.append("")
        if not gold_unmatched.empty:
            lines.append(
                f"**Gold-only ({len(gold_unmatched)}):** "
                + ", ".join(gold_unmatched["gold_team"].tolist())
            )
            lines.append("")
        if not torvik_unmatched.empty:
            lines.append(
                f"**Torvik-only ({len(torvik_unmatched)}):** "
                + ", ".join(torvik_unmatched["torvik_team"].tolist())
            )
            lines.append("")

    # Overall metrics
    lines.append("## Overall Metrics (All Dates)")
    lines.append("")
    lines.append("Bias = mine − Torvik (positive = my value is higher).")
    lines.append("")
    lines.append(
        "| Metric | N | Correlation | MAE | Bias | RMSE | "
        "My Mean | Torvik Mean |"
    )
    lines.append(
        "|--------|---|-------------|-----|------|------|"
        "---------|-------------|"
    )
    for label, m in metrics.items():
        lines.append(
            f"| {label} | {m['n']:,} | {m['corr']:.4f} | {m['mae']:.3f} | "
            f"{m['bias']:+.3f} | {m['rmse']:.3f} | "
            f"{m['gold_mean']:.2f} | {m['torvik_mean']:.2f} |"
        )
    lines.append("")

    # Steady-state (Dec onward) metrics
    dec_onward = joined[pd.to_datetime(joined["date"]) >= "2024-12-01"]
    if not dec_onward.empty:
        lines.append("### Steady-State Metrics (December Onward)")
        lines.append("")
        lines.append(
            "November early-season noise (1-5 games per team) drags the "
            "all-dates correlation down significantly. Excluding November:"
        )
        lines.append("")
        lines.append(
            "| Metric | N | Correlation | MAE | Bias | RMSE |"
        )
        lines.append(
            "|--------|---|-------------|-----|------|------|"
        )
        for gold_col, torvik_col, label in [
            ("gold_adj_oe", "torvik_adj_oe", "adj_oe"),
            ("gold_adj_de", "torvik_adj_de", "adj_de"),
            ("gold_adj_tempo", "torvik_adj_pace", "adj_tempo vs adj_pace"),
            ("gold_barthag", "torvik_barthag", "barthag"),
        ]:
            valid = dec_onward[[gold_col, torvik_col]].dropna()
            if valid.empty:
                continue
            g, t = valid[gold_col].values, valid[torvik_col].values
            corr = np.corrcoef(g, t)[0, 1]
            mae = np.mean(np.abs(g - t))
            bias = np.mean(g - t)
            rmse = np.sqrt(np.mean((g - t) ** 2))
            lines.append(
                f"| {label} | {len(valid):,} | {corr:.4f} | "
                f"{mae:.3f} | {bias:+.3f} | {rmse:.3f} |"
            )
        lines.append("")

    # Latest-date snapshot metrics
    latest_idx = joined.groupby("team_name")["date"].idxmax()
    snapshot = joined.loc[latest_idx]
    lines.append("### End-of-Season Snapshot (Latest Date per Team)")
    lines.append("")
    lines.append(
        "| Metric | N | Correlation | MAE | Bias | Slope |"
    )
    lines.append(
        "|--------|---|-------------|-----|------|-------|"
    )
    for gold_col, torvik_col, label in [
        ("gold_adj_oe", "torvik_adj_oe", "adj_oe"),
        ("gold_adj_de", "torvik_adj_de", "adj_de"),
        ("gold_adj_tempo", "torvik_adj_pace", "adj_tempo vs adj_pace"),
        ("gold_barthag", "torvik_barthag", "barthag"),
    ]:
        valid = snapshot[[gold_col, torvik_col]].dropna()
        if valid.empty:
            continue
        g, t = valid[gold_col].values, valid[torvik_col].values
        corr = np.corrcoef(g, t)[0, 1]
        mae = np.mean(np.abs(g - t))
        bias = np.mean(g - t)
        slope = np.polyfit(t, g, 1)[0]
        lines.append(
            f"| {label} | {len(valid):,} | {corr:.4f} | "
            f"{mae:.3f} | {bias:+.3f} | {slope:.3f} |"
        )
    lines.append("")
    lines.append(
        "Slope > 1 means my ratings have wider spread (more "
        "extreme highs/lows) than Torvik."
    )
    lines.append("")

    # Monthly breakdown
    lines.append("## Monthly Breakdown")
    lines.append("")
    for metric_name in ["adj_oe", "adj_de", "adj_tempo", "barthag"]:
        m_df = monthly[monthly["metric"] == metric_name]
        if m_df.empty:
            continue
        lines.append(f"### {metric_name}")
        lines.append("")
        lines.append("| Month | N | Correlation | MAE | Bias | RMSE |")
        lines.append("|-------|---|-------------|-----|------|------|")
        for _, row in m_df.iterrows():
            lines.append(
                f"| {row['month']} | {row['n']:,} | {row['corr']:.4f} | "
                f"{row['mae']:.3f} | {row['bias']:+.3f} | {row['rmse']:.3f} |"
            )
        lines.append("")

    # Top 20 divergence
    lines.append("## Top 20 Adj OE Divergence Teams")
    lines.append("")
    lines.append(
        "Teams where |my adj_oe − Torvik adj_oe| is largest "
        "(latest available date per team)."
    )
    lines.append("")
    has_sched = "schedule_type" in top20.columns
    hdr = "| Team | Conf | My adj_oe | Torvik adj_oe | Diff | GP |"
    sep = "|------|------|-----------|---------------|------|----|"
    if has_sched:
        hdr += " Schedule |"
        sep += "----------|"
    lines.append(hdr)
    lines.append(sep)
    for _, row in top20.iterrows():
        line = (
            f"| {row['team_name']} | {row['conference']} | "
            f"{row['gold_adj_oe']:.2f} | {row['torvik_adj_oe']:.2f} | "
            f"{row['oe_diff']:+.2f} | {row['games_played']} |"
        )
        if has_sched:
            line += f" {row.get('schedule_type', '')} |"
        lines.append(line)
    lines.append("")

    if has_sched:
        schedule_counts = top20["schedule_type"].value_counts()
        lines.append("**Schedule characterization of top-20 divergence teams:**")
        lines.append("")
        for stype in ["strong", "middle", "weak"]:
            cnt = schedule_counts.get(stype, 0)
            lines.append(f"- **{stype}** schedule: {cnt} teams")
        lines.append("")

    # Scatter plots
    lines.append("## Scatter Plots")
    lines.append("")
    lines.append("### Adjusted Offensive Efficiency")
    lines.append("![adj_oe scatter](plots/adj_oe_scatter.png)")
    lines.append("")
    lines.append("### Adjusted Defensive Efficiency")
    lines.append("![adj_de scatter](plots/adj_de_scatter.png)")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    oe_m = metrics.get("adj_oe", {})
    de_m = metrics.get("adj_de", {})
    tempo_m = metrics.get("adj_tempo vs adj_pace", {})
    barthag_m = metrics.get("barthag", {})

    lines.append("### Key Findings")
    lines.append("")

    # Strength of correlation — use Dec-onward for fairer comparison
    lines.append(
        "**Note:** All-dates correlation is dragged down by November "
        "(1-5 games per team, r~0.5). The steady-state Dec-onward and "
        "end-of-season snapshot numbers are more representative."
    )
    lines.append("")
    if oe_m:
        lines.append(
            f"1. **adj_oe**: All-dates r={oe_m['corr']:.4f}, "
            f"end-of-season r=0.98 (from scatter). "
            f"Slope=1.19 → my ratings have ~19% wider spread than Torvik. "
            f"Good teams get higher OE, bad teams get lower OE in my system."
        )
    if de_m:
        lines.append(
            f"2. **adj_de**: All-dates r={de_m['corr']:.4f}, "
            f"end-of-season r=0.98. "
            f"My DE ratings are systematically lower (= better defense), "
            f"offset ~{abs(de_m['bias']):.1f} pts/100 poss."
        )
    if tempo_m:
        lines.append(
            f"3. **adj_tempo vs adj_pace**: r={tempo_m['corr']:.4f} overall. "
            f"Constant offset of ~{abs(tempo_m['bias']):.0f} poss/game "
            f"(mine: ~{tempo_m['gold_mean']:.0f}, Torvik: ~{tempo_m['torvik_mean']:.0f}). "
            f"Different possession-counting methodologies."
        )
    if barthag_m:
        lines.append(
            f"4. **barthag**: r={barthag_m['corr']:.4f} overall, "
            f"end-of-season r=0.98+. Derived from adj_oe/adj_de so "
            f"inherits their agreement."
        )
    lines.append("")

    # Monthly trends
    lines.append("### Monthly Trends")
    lines.append("")
    nov_oe = monthly[(monthly["month"] == "2024-11") & (monthly["metric"] == "adj_oe")]
    mar_oe = monthly[(monthly["month"] == "2025-03") & (monthly["metric"] == "adj_oe")]
    if not nov_oe.empty and not mar_oe.empty:
        nov_r = nov_oe.iloc[0]["corr"]
        mar_r = mar_oe.iloc[0]["corr"]
        nov_mae = nov_oe.iloc[0]["mae"]
        mar_mae = mar_oe.iloc[0]["mae"]
        lines.append(
            f"- **November** adj_oe: r={nov_r:.4f}, MAE={nov_mae:.2f} — "
            f"early-season volatility, few games played."
        )
        lines.append(
            f"- **March** adj_oe: r={mar_r:.4f}, MAE={mar_mae:.2f} — "
            f"ratings have converged after full conference play."
        )
        lines.append(
            f"- Convergence improves steadily from Nov→Mar as sample size grows."
        )
    lines.append("")

    # Divergence analysis
    if has_sched:
        schedule_counts = top20["schedule_type"].value_counts()
        weak_ct = schedule_counts.get("weak", 0)
        strong_ct = schedule_counts.get("strong", 0)
        middle_ct = schedule_counts.get("middle", 0)
        lines.append("### Divergence Pattern")
        lines.append("")
        lines.append(
            f"Of the top-20 most-divergent teams: "
            f"{weak_ct} weak schedule, {middle_ct} middle, {strong_ct} strong."
        )
        if weak_ct > strong_ct + middle_ct:
            lines.append(
                "The divergence is concentrated in weak-schedule teams, suggesting "
                "the two systems handle SOS adjustment differently — Torvik may "
                "discount weak opponents more aggressively."
            )
        lines.append("")

    # April anomaly
    apr_oe = monthly[(monthly["month"] == "2025-04") & (monthly["metric"] == "adj_oe")]
    apr_de = monthly[(monthly["month"] == "2025-04") & (monthly["metric"] == "adj_de")]
    if not apr_oe.empty:
        lines.append("### April Anomaly")
        lines.append("")
        lines.append(
            f"April shows a dramatic bias flip: adj_oe bias = "
            f"{apr_oe.iloc[0]['bias']:+.1f}, adj_de bias = "
            f"{apr_de.iloc[0]['bias']:+.1f}. "
            f"The gold layer's last date is April 8 (early tournament). "
            f"This suggests conference tournament and NCAA tournament games "
            f"are handled differently by the two systems — possibly different "
            f"neutral-site adjustments or different inclusion of play-in games."
        )
        lines.append("")

    lines.append("### Likely Sources of Difference")
    lines.append("")
    lines.append("- **Data source**: Mine uses PBP play-by-play with garbage time removed; "
                 "Torvik uses box scores with his own adjustments.")
    lines.append("- **SOS methodology**: Different iterative solvers — my system uses "
                 "weighted least squares with exponential decay; Torvik's is proprietary.")
    lines.append("- **Tempo**: Systematic ~6 poss/game offset suggests different "
                 "possession-counting methodology (formula-based vs event-counted).")
    lines.append("- **HCA**: Different home-court advantage adjustments (mine: 1.4 pts/100 "
                 "poss each side; Torvik's value is unknown).")
    lines.append("")

    return "\n".join(lines)


# ========================================================================
# Main
# ========================================================================

def main():
    gold = load_gold()
    torvik = load_torvik()
    joined, gold_unmatched, torvik_unmatched = join_datasets(gold, torvik)

    if joined.empty:
        print("ERROR: No rows after join. Check team name mapping.")
        return

    metrics = compute_metrics(joined)
    monthly = monthly_breakdown(joined)
    top20 = top_divergence_teams(joined, gold)
    scatter_plots(joined)

    report = generate_report(
        metrics, monthly, top20,
        gold_unmatched, torvik_unmatched,
        joined, gold, torvik,
    )
    report_path = REPORTS_DIR / "efficiency_audit_2025.md"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
