"""Load KenPom archive and teams data from the S3 lakehouse."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, s3_reader

_archive_cache: dict[int, pd.DataFrame] = {}
_teams_cache: dict[int, pd.DataFrame] = {}


def _season_from_snapshot_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return pd.Series(
        np.where(dt.dt.month >= 8, dt.dt.year + 1, dt.dt.year),
        index=series.index,
    )


def _normalize_team_name(name: object) -> str:
    value = str(name or "").lower().replace("&", " and ")
    return "".join(ch for ch in value if ch.isalnum())


def _compute_barthag(adj_oe: float | None, adj_de: float | None) -> float | None:
    if adj_oe is None or adj_de is None:
        return None
    exp = config.BARTHAG_EXPONENT
    oe_pow = adj_oe ** exp
    de_pow = adj_de ** exp
    denom = oe_pow + de_pow
    if denom == 0:
        return None
    return oe_pow / denom


def _latest_partition_prefix(base_prefix: str) -> str | None:
    return s3_reader._get_latest_asof_prefix(base_prefix)  # type: ignore[attr-defined]


def load_kenpom_teams(season: int) -> pd.DataFrame:
    if season in _teams_cache:
        return _teams_cache[season]

    base_prefix = f"{config.SILVER_PREFIX}/external/kenpom/teams/season={season}/"
    latest_prefix = _latest_partition_prefix(base_prefix)
    if latest_prefix is None:
        _teams_cache[season] = pd.DataFrame()
        return _teams_cache[season]

    keys = s3_reader.list_parquet_keys(latest_prefix)
    if not keys:
        _teams_cache[season] = pd.DataFrame()
        return _teams_cache[season]

    df = s3_reader.read_parquet_table(keys).to_pandas()
    if df.empty:
        _teams_cache[season] = df
        return df
    df["team_name_key"] = df["team_name"].map(_normalize_team_name)
    _teams_cache[season] = df
    return df


def load_kenpom_archive_season(season: int) -> pd.DataFrame:
    if season in _archive_cache:
        return _archive_cache[season]

    base_prefix = f"{config.SILVER_PREFIX}/external/kenpom/archive/"
    keys = []
    for key in s3_reader.list_parquet_keys(base_prefix):
        marker = "date="
        if marker not in key:
            continue
        date_part = key.split(marker, 1)[1][:10]
        try:
            season_for_key = int(date_part[:4]) + 1 if int(date_part[5:7]) >= 8 else int(date_part[:4])
        except ValueError:
            continue
        if season_for_key == season:
            keys.append(key)
    if not keys:
        _archive_cache[season] = pd.DataFrame()
        return _archive_cache[season]

    df = s3_reader.read_parquet_table(keys).to_pandas()
    if df.empty:
        _archive_cache[season] = df
        return df

    teams = load_kenpom_teams(season)
    if teams.empty:
        _archive_cache[season] = pd.DataFrame()
        return _archive_cache[season]

    team_cols = [c for c in ["team_id", "team_name", "team_name_key", "conf_short"] if c in teams.columns]
    team_map = teams[team_cols].drop_duplicates(subset=["team_name"], keep="last")
    df = df.merge(team_map, on="team_name", how="left", suffixes=("", "_team"))
    if "conf_short_team" in df.columns:
        df["conf_short"] = df["conf_short"].fillna(df["conf_short_team"])
        df = df.drop(columns=["conf_short_team"])
    df = df.dropna(subset=["team_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df["rating_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["barthag"] = [
        _compute_barthag(oe, de)
        for oe, de in zip(
            pd.to_numeric(df["adj_oe"], errors="coerce"),
            pd.to_numeric(df["adj_de"], errors="coerce"),
        )
    ]
    df["barthag_rank"] = pd.to_numeric(df.get("rank_adj_em"), errors="coerce")
    df = df.sort_values(["team_id", "rating_date"]).reset_index(drop=True)
    _archive_cache[season] = df
    return df


def build_kenpom_efficiency_lookup(season: int) -> dict[int, pd.DataFrame]:
    df = load_kenpom_archive_season(season)
    if df.empty:
        return {}
    lookup: dict[int, pd.DataFrame] = {}
    keep_cols = [
        "rating_date",
        "adj_oe",
        "adj_de",
        "adj_tempo",
        "barthag",
        "barthag_rank",
    ]
    for team_id, group in df.groupby("team_id"):
        available = [c for c in keep_cols if c in group.columns]
        lookup[int(team_id)] = group[available].copy()
    return lookup


def get_kenpom_asof_rating(
    team_lookup: dict[int, pd.DataFrame],
    team_id: int,
    game_date: pd.Timestamp,
) -> dict:
    team_df = team_lookup.get(team_id)
    if team_df is None or team_df.empty:
        return {}
    if hasattr(game_date, "tz") and game_date.tz is not None:
        cutoff = game_date.tz_convert("America/New_York").normalize().tz_localize(None)
    else:
        cutoff = game_date.normalize()
    eligible = team_df[team_df["rating_date"] <= cutoff]
    if eligible.empty:
        return {}
    row = eligible.iloc[-1]
    return {
        "adj_oe": row.get("adj_oe"),
        "adj_de": row.get("adj_de"),
        "adj_tempo": row.get("adj_tempo"),
        "barthag": row.get("barthag"),
        "barthag_rank": row.get("barthag_rank"),
    }


def build_kenpom_conf_strength_lookup(
    season: int,
    game_dates: list[pd.Timestamp],
) -> dict[tuple[str, str], float]:
    df = load_kenpom_archive_season(season)
    if df.empty or "conf_short" not in df.columns:
        return {}

    work = df.copy()
    work["adj_net"] = pd.to_numeric(work["adj_oe"], errors="coerce") - pd.to_numeric(
        work["adj_de"], errors="coerce"
    )
    lookup: dict[tuple[str, str], float] = {}
    for game_dt in game_dates:
        if pd.isna(game_dt):
            continue
        dt = pd.Timestamp(game_dt)
        if hasattr(dt, "tz") and dt.tz is not None:
            cutoff = dt.tz_convert("America/New_York").normalize().tz_localize(None)
        else:
            cutoff = dt.normalize()
        eligible = work[work["rating_date"] <= cutoff]
        if eligible.empty:
            continue
        latest = eligible.sort_values("rating_date").groupby("team_id").last()
        conf_means = latest.groupby("conf_short")["adj_net"].mean()
        date_str = cutoff.strftime("%Y-%m-%d")
        for conf, value in conf_means.items():
            lookup[(date_str, str(conf))] = float(value)
    return lookup
