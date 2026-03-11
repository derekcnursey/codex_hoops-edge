"""Build power rankings JSON from S3 efficiency ratings and game results.

Reads:
  - preferred: gold/team_adjusted_efficiencies_no_garbage_priorreg_k5_v1
  - fallback:  gold/team_adjusted_efficiencies_no_garbage
  - silver/fct_games (season 2026) → win-loss records
  - silver/fct_games team names → display names + conference fallback

Outputs:
  - predictions/json/rankings.json
  - site/public/data/rankings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Allow running as standalone script
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config, s3_reader


CURRENT_SEASON = 2026
PRIMARY_RATINGS_TABLE = "team_adjusted_efficiencies_no_garbage_priorreg_k5_v1"
FALLBACK_RATINGS_TABLE = "team_adjusted_efficiencies_no_garbage"
PRIMARY_SOURCE_LABEL = "Hoops Edge Ratings"
PRIMARY_SOURCE_DESCRIPTION = (
    "Best internal efficiency model. Strongest in post-Dec-15 validation."
)
PRIMARY_SOURCE_NOTE = (
    "Internal ratings source only. Torvik remains stronger on full-season pooled validation."
)


def _load_latest_ratings(season: int) -> tuple[pd.DataFrame, str]:
    """Load the most recent efficiency rating for each team."""
    table_name = PRIMARY_RATINGS_TABLE
    tbl = s3_reader.read_gold_table(table_name, season=season)
    if tbl.num_rows == 0:
        print(
            f"WARNING: rankings season {season} missing preferred ratings table "
            f"{PRIMARY_RATINGS_TABLE}; falling back to {FALLBACK_RATINGS_TABLE}"
        )
        table_name = FALLBACK_RATINGS_TABLE
        tbl = s3_reader.read_gold_table(table_name, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame(), table_name
    df = tbl.to_pandas()

    needed = ["teamId", "rating_date", "adj_oe", "adj_de", "adj_tempo", "barthag"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in efficiency ratings: {missing}")

    df["rating_date"] = pd.to_datetime(df["rating_date"], errors="coerce")
    # Keep only the latest rating_date row per team
    idx = df.groupby("teamId")["rating_date"].idxmax()
    latest = df.loc[idx].copy()
    return latest, table_name


def _load_records(season: int) -> pd.DataFrame:
    """Compute W-L and conference W-L from fct_games."""
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    games = tbl.to_pandas()

    # Normalize columns
    rename = {}
    for target, candidates in [
        ("gameId", ["gameId"]),
        ("homeTeamId", ["homeTeamId"]),
        ("awayTeamId", ["awayTeamId"]),
        ("homeScore", ["homeScore", "homePoints"]),
        ("awayScore", ["awayScore", "awayPoints"]),
        ("homeConference", ["homeConference", "homeConferenceName"]),
        ("awayConference", ["awayConference", "awayConferenceName"]),
        ("homeTeam", ["homeTeam"]),
        ("awayTeam", ["awayTeam"]),
    ]:
        for cand in candidates:
            if cand in games.columns:
                rename[cand] = target
                break
    games = games.rename(columns=rename)

    # Only completed games (drop missing scores and 0-0 placeholders for future games)
    games = games.dropna(subset=["homeScore", "awayScore"])
    games = games[~((games["homeScore"] == 0) & (games["awayScore"] == 0))]
    games = games.drop_duplicates(subset=["gameId"], keep="last")

    has_conf = "homeConference" in games.columns and "awayConference" in games.columns

    records: dict[int, dict] = {}

    for _, g in games.iterrows():
        h_id = int(g["homeTeamId"])
        a_id = int(g["awayTeamId"])
        h_score = float(g["homeScore"])
        a_score = float(g["awayScore"])
        home_won = h_score > a_score

        is_conf_game = False
        if has_conf:
            h_conf = g.get("homeConference")
            a_conf = g.get("awayConference")
            if pd.notna(h_conf) and pd.notna(a_conf) and h_conf == a_conf:
                is_conf_game = True

        for tid, won in [(h_id, home_won), (a_id, not home_won)]:
            if tid not in records:
                records[tid] = {"W": 0, "L": 0, "conf_W": 0, "conf_L": 0}
            if won:
                records[tid]["W"] += 1
            else:
                records[tid]["L"] += 1
            if is_conf_game:
                if won:
                    records[tid]["conf_W"] += 1
                else:
                    records[tid]["conf_L"] += 1

    rec_df = pd.DataFrame.from_dict(records, orient="index")
    rec_df.index.name = "teamId"
    rec_df = rec_df.reset_index()
    return rec_df


def _load_team_info(season: int) -> pd.DataFrame:
    """Load team display names and conferences from fct_games."""
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if tbl.num_rows == 0:
        return pd.DataFrame()
    games = tbl.to_pandas()

    rename = {}
    for target, candidates in [
        ("homeTeamId", ["homeTeamId"]),
        ("awayTeamId", ["awayTeamId"]),
        ("homeTeam", ["homeTeam"]),
        ("awayTeam", ["awayTeam"]),
        ("homeConference", ["homeConference", "homeConferenceName"]),
        ("awayConference", ["awayConference", "awayConferenceName"]),
    ]:
        for cand in candidates:
            if cand in games.columns:
                rename[cand] = target
                break
    games = games.rename(columns=rename)

    teams: dict[int, dict] = {}

    # Collect from home side
    if "homeTeamId" in games.columns and "homeTeam" in games.columns:
        for _, row in games.iterrows():
            tid = int(row["homeTeamId"])
            name = str(row["homeTeam"])
            conf = str(row.get("homeConference", "")) if pd.notna(row.get("homeConference")) else ""
            if tid not in teams or not teams[tid].get("conference"):
                teams[tid] = {"team": name, "conference": conf}

    # Collect from away side
    if "awayTeamId" in games.columns and "awayTeam" in games.columns:
        for _, row in games.iterrows():
            tid = int(row["awayTeamId"])
            name = str(row["awayTeam"])
            conf = str(row.get("awayConference", "")) if pd.notna(row.get("awayConference")) else ""
            if tid not in teams or not teams[tid].get("conference"):
                teams[tid] = {"team": name, "conference": conf}

    info_df = pd.DataFrame.from_dict(teams, orient="index")
    info_df.index.name = "teamId"
    info_df = info_df.reset_index()
    return info_df


def build_rankings(season: int = CURRENT_SEASON) -> dict:
    """Build the full rankings payload."""
    print(f"Loading efficiency ratings for season {season}...")
    ratings, table_name = _load_latest_ratings(season)
    if ratings.empty:
        raise RuntimeError("No efficiency ratings found.")

    print(f"  {len(ratings)} teams with ratings")

    print("Loading game records...")
    records = _load_records(season)
    print(f"  {len(records)} teams with records")

    print("Loading team info...")
    team_info = _load_team_info(season)
    print(f"  {len(team_info)} teams with info")

    # Merge ratings + records + team info
    df = ratings[["teamId", "rating_date", "adj_oe", "adj_de", "adj_tempo", "barthag"]].copy()
    df["adj_margin"] = df["adj_oe"] - df["adj_de"]

    if not records.empty:
        df = df.merge(records, on="teamId", how="left")
    else:
        df["W"] = 0
        df["L"] = 0
        df["conf_W"] = 0
        df["conf_L"] = 0

    if not team_info.empty:
        df = df.merge(team_info, on="teamId", how="left")
    else:
        df["team"] = df["teamId"].astype(str)
        df["conference"] = ""

    # Fill NaN records
    for col in ["W", "L", "conf_W", "conf_L"]:
        df[col] = df[col].fillna(0).astype(int)
    df["team"] = df["team"].fillna(df["teamId"].astype(str))
    df["conference"] = df["conference"].fillna("")

    # Sort by adj_margin descending
    df = df.sort_values("adj_margin", ascending=False).reset_index(drop=True)

    # Determine as_of_date from the latest rating_date
    as_of = df["rating_date"].max()
    as_of_str = as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else ""

    # Build output
    teams = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        adj_oe = round(float(row["adj_oe"]), 1)
        adj_de = round(float(row["adj_de"]), 1)
        adj_margin = round(float(row["adj_margin"]), 1)
        adj_tempo = round(float(row["adj_tempo"]), 1)
        edge_index = round(float(row["barthag"]), 4) if pd.notna(row["barthag"]) else None

        record = f"{row['W']}-{row['L']}"
        conf_record = f"{row['conf_W']}-{row['conf_L']}"

        teams.append({
            "rank": rank,
            "team": str(row["team"]),
            "team_id": int(row["teamId"]),
            "conference": str(row["conference"]),
            "record": record,
            "conf_record": conf_record,
            "adj_oe": adj_oe,
            "adj_de": adj_de,
            "adj_margin": adj_margin,
            "adj_tempo": adj_tempo,
            "edge_index": edge_index,
        })

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "as_of_date": as_of_str,
        "season": season,
        "source_id": PRIMARY_SOURCE_LABEL.lower().replace(" ", "_"),
        "source_label": PRIMARY_SOURCE_LABEL,
        "source_table": table_name,
        "source_description": PRIMARY_SOURCE_DESCRIPTION,
        "source_note": PRIMARY_SOURCE_NOTE,
        "teams": teams,
    }
    return payload


def save_rankings(payload: dict, season: int = CURRENT_SEASON) -> list[Path]:
    """Write rankings JSON to both output locations.

    Writes rankings_{season}.json. When season == CURRENT_SEASON also
    writes rankings.json as a backward-compatible alias.
    """
    json_dir = config.PREDICTIONS_DIR / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    site_dir = config.PROJECT_ROOT / "site" / "public" / "data"
    site_dir.mkdir(parents=True, exist_ok=True)

    blob = json.dumps(payload, indent=2)
    written: list[Path] = []

    # Always write the season-specific file
    for base_dir in (json_dir, site_dir):
        p = base_dir / f"rankings_{season}.json"
        p.write_text(blob)
        written.append(p)

    # Also write the generic alias for the current season
    if season == CURRENT_SEASON:
        for base_dir in (json_dir, site_dir):
            p = base_dir / "rankings.json"
            p.write_text(blob)
            written.append(p)

    return written


def main():
    parser = argparse.ArgumentParser(description="Build power rankings JSON")
    parser.add_argument(
        "--season",
        type=int,
        default=CURRENT_SEASON,
        help=f"Season year (default: {CURRENT_SEASON})",
    )
    args = parser.parse_args()

    payload = build_rankings(season=args.season)
    written = save_rankings(payload, season=args.season)
    print(f"\nRankings built: {len(payload['teams'])} teams (season {args.season})")
    for p in written:
        print(f"  {p}")

    # Print top 10
    print(f"\nTop 10 (as of {payload['as_of_date']}):")
    for t in payload["teams"][:10]:
        print(
            f"  {t['rank']:>3}. {t['team']:<22} "
            f"{t['record']:>6}  "
            f"Net: {t['adj_margin']:+.1f}  "
            f"O: {t['adj_oe']:.1f}  D: {t['adj_de']:.1f}  "
            f"Edge: {t['edge_index']:.3f}"
        )


if __name__ == "__main__":
    main()
