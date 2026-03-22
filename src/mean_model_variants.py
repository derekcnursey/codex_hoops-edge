"""Mean-model variant helpers for live mu prediction and training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, s3_reader, torvik_loader
from .dataset import load_season_features
from .features import load_efficiency_ratings


LEGACY_HOME_SLOT = "legacy_home_slot"
TEAM_AB_ELITE_TAIL_ROUND64_V1 = "team_ab_elite_tail_round64_v1"
SUPPORTED_MEAN_MODEL_VARIANTS = {
    LEGACY_HOME_SLOT,
    TEAM_AB_ELITE_TAIL_ROUND64_V1,
}

TEAM_AB_CONTEXT_FEATURES = [
    "neutral_site",
    "team_a_is_home_non_neutral",
    "team_a_hca",
    "is_ncaa_tournament",
    "is_conference_tournament_neutral",
    "is_early_mte_neutral",
    "is_other_neutral",
]
TEAM_AB_LEVEL_FEATURES = [
    "team_a_adj_net",
    "team_b_adj_net",
    "team_a_barthag",
    "team_b_barthag",
    "team_a_conf_strength",
    "team_b_conf_strength",
    "team_a_sos_net",
    "team_b_sos_net",
    "team_a_form_delta",
    "team_b_form_delta",
    "team_a_rest_days",
    "team_b_rest_days",
]
TEAM_AB_DIFF_FEATURES = [
    "adj_oe_diff",
    "adj_de_edge",
    "adj_net_diff",
    "barthag_diff",
    "conf_strength_diff",
    "sos_net_diff",
    "form_delta_diff",
    "rest_advantage_ab",
]
TEAM_AB_SUM_FEATURES = [
    "adj_net_sum",
    "barthag_sum",
    "conf_strength_sum",
    "sos_net_sum",
    "form_delta_sum",
]
TEAM_AB_STYLE_FEATURES = [
    "efg_diff",
    "ft_rate_diff",
    "off_rebound_diff",
    "tov_edge",
    "margin_std_diff",
    "efg_sum",
    "ft_rate_sum",
    "off_rebound_sum",
    "tov_total",
    "margin_std_sum",
]
TEAM_AB_ELITE_CORE_FEATURES = [
    "abs_adj_net_gap",
    "abs_barthag_gap",
    "best_team_top10",
    "best_team_top15",
    "best_team_top4",
    "best_team_seed1",
    "both_teams_top25",
]
TEAM_AB_INTERACTION_FEATURES = [
    "neutral_adj_net_diff",
    "neutral_barthag_diff",
    "neutral_abs_adj_net_gap",
    "ncaa_adj_net_diff",
    "ncaa_barthag_diff",
    "ncaa_abs_adj_net_gap",
    "neutral_elite_gap_top10",
    "neutral_elite_gap_top15",
    "ncaa_elite_gap_top10",
    "ncaa_elite_gap_top15",
]
TEAM_AB_ELITE_TAIL_FEATURES = [
    "best_team_top5",
    "best_team_top2",
    "worst_team_outside_top50",
    "worst_team_outside_top100",
    "top5_adj_net_diff",
    "top2_adj_net_diff",
    "seed1_adj_net_diff",
    "seed1_barthag_diff",
    "elite_gap_power",
    "weak_opp_adj_net_diff",
    "seed1_large_gap",
    "top5_large_gap",
]
TEAM_AB_ROUND64_FEATURES = [
    "is_round_of_64",
    "round64_adj_net_diff",
    "round64_top5_adj_net_diff",
    "round64_seed1_adj_net_diff",
    "round64_seed1_gap_power",
    "round64_seed1_large_gap",
]
TEAM_AB_ELITE_TAIL_ROUND64_V1_FEATURE_ORDER = (
    TEAM_AB_CONTEXT_FEATURES
    + TEAM_AB_LEVEL_FEATURES
    + TEAM_AB_DIFF_FEATURES
    + TEAM_AB_SUM_FEATURES
    + TEAM_AB_STYLE_FEATURES
    + TEAM_AB_ELITE_CORE_FEATURES
    + TEAM_AB_INTERACTION_FEATURES
    + TEAM_AB_ELITE_TAIL_FEATURES
    + TEAM_AB_ROUND64_FEATURES
)


@dataclass(frozen=True)
class MeanModelVariantSpec:
    name: str
    checkpoint_path: Path
    torvik_checkpoint_path: Path | None
    feature_order: list[str] | None
    use_legacy_full_sym: bool
    use_mu_blend: bool


def normalize_mean_model_variant(value: str | None) -> str:
    variant = (value or LEGACY_HOME_SLOT).strip().lower()
    if variant not in SUPPORTED_MEAN_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported mean model variant {variant!r}. "
            f"Expected one of {sorted(SUPPORTED_MEAN_MODEL_VARIANTS)}."
        )
    return variant


def active_mean_model_variant() -> str:
    return normalize_mean_model_variant(config.MEAN_MODEL_VARIANT)


def variant_spec(variant: str) -> MeanModelVariantSpec:
    normalized = normalize_mean_model_variant(variant)
    if normalized == LEGACY_HOME_SLOT:
        return MeanModelVariantSpec(
            name=normalized,
            checkpoint_path=config.TREE_REGRESSOR_PATH,
            torvik_checkpoint_path=config.TORVIK_TREE_REGRESSOR_PATH,
            feature_order=list(config.FEATURE_ORDER),
            use_legacy_full_sym=True,
            use_mu_blend=True,
        )
    return MeanModelVariantSpec(
        name=normalized,
        checkpoint_path=config.TREE_REGRESSOR_TEAM_AB_ELITE_TAIL_ROUND64_V1_PATH,
        torvik_checkpoint_path=None,
        feature_order=list(TEAM_AB_ELITE_TAIL_ROUND64_V1_FEATURE_ORDER),
        use_legacy_full_sym=False,
        use_mu_blend=False,
    )


def legacy_variant_field() -> str:
    return "predicted_spread_legacy"


def variant_prediction_field(variant: str) -> str:
    return f"predicted_spread_{normalize_mean_model_variant(variant)}"


def _round_from_note(note: object) -> str | None:
    value = str(note or "").upper()
    mapping = [
        ("FIRST FOUR", "First Four"),
        ("1ST ROUND", "Round of 64"),
        ("2ND ROUND", "Round of 32"),
        ("SWEET 16", "Sweet 16"),
        ("ELITE 8", "Elite 8"),
        ("FINAL FOUR", "Final Four"),
        ("NATIONAL CHAMPIONSHIP", "Championship"),
    ]
    for needle, label in mapping:
        if needle in value:
            return label
    return None


def _neutral_subtype(frame: pd.DataFrame) -> pd.Series:
    neutral = pd.to_numeric(frame.get("neutral_site", frame.get("neutralSite", 0.0)), errors="coerce").fillna(0.0).eq(1.0)
    tournament = frame.get("tournament", pd.Series(index=frame.index, dtype=object))
    game_type = frame.get("gameType", pd.Series(index=frame.index, dtype=object))
    conference_game = frame.get("conferenceGame", pd.Series(False, index=frame.index))
    dt = pd.to_datetime(frame.get("startDate"), errors="coerce", utc=True)
    month = dt.dt.tz_convert("America/New_York").dt.month
    missing_tournament = tournament.isna() | tournament.astype(str).isin({"", "None", "nan"})

    out = pd.Series("non_neutral", index=frame.index, dtype=object)
    out.loc[neutral & tournament.eq("NCAA")] = "ncaa_neutral"
    out.loc[
        neutral
        & game_type.eq("TRNMNT")
        & missing_tournament
        & month.eq(3)
    ] = "conference_tournament_neutral"
    out.loc[
        neutral
        & game_type.eq("STD")
        & month.isin([11, 12])
        & ~conference_game.fillna(False).astype(bool)
    ] = "early_season_mte_neutral"
    out.loc[neutral & game_type.eq("STD") & missing_tournament] = "regular_season_neutral"
    out.loc[neutral & out.eq("non_neutral")] = "other_neutral"
    return out


def _season_from_start_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    return np.where(dt.dt.month >= 8, dt.dt.year + 1, dt.dt.year)


def _prepare_rank_frame_gold(season: int, *, table_name: str | None = None) -> pd.DataFrame:
    ratings = load_efficiency_ratings(
        season,
        no_garbage=config.NO_GARBAGE,
        table_name=table_name or config.PRODUCTION_GOLD_RATINGS_TABLE,
    )
    if ratings.empty:
        return pd.DataFrame(columns=["teamId", "rating_date", "barthag_rank"])
    frame = ratings[["teamId", "rating_date", "barthag"]].copy()
    frame["rating_date"] = pd.to_datetime(frame["rating_date"], errors="coerce")
    frame["barthag_rank"] = frame.groupby("rating_date")["barthag"].rank(
        ascending=False,
        method="first",
    )
    return frame.sort_values(["rating_date", "teamId"]).reset_index(drop=True)


def _prepare_rank_frame_torvik(season: int) -> pd.DataFrame:
    torvik = torvik_loader.load_torvik_season(season)
    if torvik.empty:
        return pd.DataFrame(columns=["teamId", "rating_date", "barthag_rank"])
    torvik = torvik.copy()
    torvik["rating_date"] = pd.to_datetime(torvik["date"], errors="coerce")
    torvik["barthag_rank"] = torvik.groupby("rating_date")["BARTHAG"].rank(
        ascending=False,
        method="first",
    )

    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    mapping: dict[str, int] = {}
    if tbl.num_rows:
        games = tbl.to_pandas()
        for id_col, name_col in [("homeTeamId", "homeTeam"), ("awayTeamId", "awayTeam")]:
            if id_col in games.columns and name_col in games.columns:
                pairs = games[[id_col, name_col]].dropna()
                for _, row in pairs.iterrows():
                    mapping[str(row[name_col])] = int(row[id_col])
    if not mapping:
        return pd.DataFrame(columns=["teamId", "rating_date", "barthag_rank"])
    torvik["teamId"] = torvik["team_name"].map(mapping)
    torvik = torvik.dropna(subset=["teamId"]).copy()
    torvik["teamId"] = torvik["teamId"].astype(int)
    return torvik[["teamId", "rating_date", "barthag_rank"]].sort_values(
        ["rating_date", "teamId"]
    ).reset_index(drop=True)


def _attach_barthag_ranks(
    frame: pd.DataFrame,
    *,
    efficiency_source: str,
    gold_table_name: str | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for season, season_df in frame.groupby("season", sort=True):
        season_part = season_df.copy()
        rank_frame = (
            _prepare_rank_frame_torvik(int(season))
            if efficiency_source == "torvik"
            else _prepare_rank_frame_gold(int(season), table_name=gold_table_name)
        )
        if rank_frame.empty:
            season_part["home_barthag_rank"] = np.nan
            season_part["away_barthag_rank"] = np.nan
            parts.append(season_part)
            continue
        rank_frame = rank_frame.copy()
        rank_frame["rating_date"] = (
            pd.to_datetime(rank_frame["rating_date"], errors="coerce")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )

        game_dt = pd.to_datetime(season_part["startDate"], errors="coerce", utc=True)
        season_part["feature_cutoff_date"] = (
            game_dt.dt.tz_convert(None).dt.normalize() - pd.Timedelta(days=1)
        )
        season_part["feature_cutoff_date"] = season_part["feature_cutoff_date"].astype("datetime64[ns]")

        for side in ["home", "away"]:
            left = (
                season_part[[f"{side}TeamId", "feature_cutoff_date"]]
                .rename(columns={f"{side}TeamId": "teamId"})
                .copy()
            )
            left["__row_id"] = np.arange(len(left), dtype=np.int64)
            left = left.sort_values(["feature_cutoff_date", "teamId"]).reset_index(drop=True)
            merged = pd.merge_asof(
                left,
                rank_frame,
                by="teamId",
                left_on="feature_cutoff_date",
                right_on="rating_date",
                direction="backward",
            )
            merged = merged.sort_values("__row_id").reset_index(drop=True)
            season_part[f"{side}_barthag_rank"] = merged["barthag_rank"].to_numpy(dtype=float)

        season_part = season_part.drop(columns=["feature_cutoff_date"])
        parts.append(season_part)
    return pd.concat(parts, ignore_index=True)


def _load_games_meta(season: int) -> pd.DataFrame:
    table = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["gameId"])
    games = table.to_pandas()
    keep = [
        c
        for c in [
            "gameId",
            "startDate",
            "gameType",
            "tournament",
            "neutralSite",
            "conferenceGame",
            "gameNotes",
            "homeSeed",
            "awaySeed",
        ]
        if c in games.columns
    ]
    meta = games[keep].drop_duplicates("gameId").copy()
    meta["season"] = season
    return meta


def prepare_team_ab_training_frame(
    seasons: list[int],
    *,
    no_garbage: bool,
    adj_suffix: str | None,
    efficiency_source: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    meta_parts: list[pd.DataFrame] = []
    for season in seasons:
        season_df = load_season_features(
            season,
            no_garbage=no_garbage,
            adj_suffix=adj_suffix,
            efficiency_source=efficiency_source,
        ).copy()
        season_df["season"] = season
        season_df = season_df.dropna(subset=["homeScore", "awayScore"])
        season_df = season_df[(season_df["homeScore"] != 0) | (season_df["awayScore"] != 0)].copy()
        parts.append(season_df)
        meta_parts.append(_load_games_meta(season))

    if not parts:
        return pd.DataFrame()

    frame = pd.concat(parts, ignore_index=True)
    meta = pd.concat(meta_parts, ignore_index=True) if meta_parts else pd.DataFrame(columns=["gameId"])
    if not meta.empty:
        meta = meta.drop_duplicates(subset=["gameId"], keep="last")
        missing_meta_cols = [col for col in meta.columns if col not in frame.columns]
        merge_cols = ["gameId"] + missing_meta_cols
        frame = frame.merge(meta[merge_cols], on="gameId", how="left")

    if "season" not in frame.columns:
        frame["season"] = _season_from_start_date(frame["startDate"]).astype(int)
    if "neutral_site" not in frame.columns and "neutralSite" in frame.columns:
        frame["neutral_site"] = pd.to_numeric(frame["neutralSite"], errors="coerce").fillna(0.0)
    if "neutralSite" not in frame.columns and "neutral_site" in frame.columns:
        frame["neutralSite"] = pd.to_numeric(frame["neutral_site"], errors="coerce").fillna(0.0)

    required_game_meta = {
        "gameType": None,
        "tournament": None,
        "conferenceGame": False,
        "gameNotes": None,
        "homeSeed": np.nan,
        "awaySeed": np.nan,
    }
    for col, default in required_game_meta.items():
        if col not in frame.columns:
            frame[col] = default

    if "home_barthag_rank" not in frame.columns or "away_barthag_rank" not in frame.columns:
        frame = _attach_barthag_ranks(frame, efficiency_source=efficiency_source)

    frame["startDate"] = pd.to_datetime(frame["startDate"], errors="coerce", utc=True)
    frame = frame.sort_values(["startDate", "gameId"], kind="mergesort").reset_index(drop=True)
    return frame


def ensure_team_ab_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "season" not in out.columns:
        seasons = pd.Series(_season_from_start_date(out["startDate"]), index=out.index)
        out["season"] = seasons.astype("Int64")
    if "neutral_site" not in out.columns and "neutralSite" in out.columns:
        out["neutral_site"] = pd.to_numeric(out["neutralSite"], errors="coerce").fillna(0.0)
    if "neutralSite" not in out.columns and "neutral_site" in out.columns:
        out["neutralSite"] = pd.to_numeric(out["neutral_site"], errors="coerce").fillna(0.0)
    for col, default in {
        "gameType": None,
        "tournament": None,
        "conferenceGame": False,
        "gameNotes": None,
        "homeSeed": np.nan,
        "awaySeed": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default
    return out


def build_team_ab_source(frame: pd.DataFrame) -> pd.DataFrame:
    df = ensure_team_ab_metadata(frame)
    out = pd.DataFrame(index=df.index)
    non_neutral = (
        pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0.0) == 0.0
    ).astype(float)

    pairs = [
        ("homeTeamId", "awayTeamId", "team_a_team_id", "team_b_team_id"),
        ("homeTeam", "awayTeam", "team_a_name", "team_b_name"),
        ("home_team_adj_oe", "away_team_adj_oe", "team_a_adj_oe", "team_b_adj_oe"),
        ("home_team_adj_de", "away_team_adj_de", "team_a_adj_de", "team_b_adj_de"),
        ("home_team_BARTHAG", "away_team_BARTHAG", "team_a_barthag", "team_b_barthag"),
        ("home_conf_strength", "away_conf_strength", "team_a_conf_strength", "team_b_conf_strength"),
        ("home_sos_oe", "away_sos_oe", "team_a_sos_oe", "team_b_sos_oe"),
        ("home_sos_de", "away_sos_de", "team_a_sos_de", "team_b_sos_de"),
        ("home_form_delta", "away_form_delta", "team_a_form_delta", "team_b_form_delta"),
        ("home_rest_days", "away_rest_days", "team_a_rest_days", "team_b_rest_days"),
        ("home_eff_fg_pct", "away_eff_fg_pct", "team_a_eff_fg_pct", "team_b_eff_fg_pct"),
        ("home_ft_rate", "away_ft_rate", "team_a_ft_rate", "team_b_ft_rate"),
        ("home_off_rebound_pct", "away_off_rebound_pct", "team_a_off_rebound_pct", "team_b_off_rebound_pct"),
        ("home_tov_rate", "away_tov_rate", "team_a_tov_rate", "team_b_tov_rate"),
        ("home_margin_std", "away_margin_std", "team_a_margin_std", "team_b_margin_std"),
        ("home_barthag_rank", "away_barthag_rank", "team_a_barthag_rank", "team_b_barthag_rank"),
        ("homeSeed", "awaySeed", "team_a_seed", "team_b_seed"),
    ]
    for left, right, out_left, out_right in pairs:
        out[out_left] = df[left] if left in df.columns else np.nan
        out[out_right] = df[right] if right in df.columns else np.nan

    out["season"] = df["season"]
    out["gameId"] = df["gameId"]
    out["startDate"] = df["startDate"]
    if "homeScore" in df.columns and "awayScore" in df.columns:
        actual_margin = pd.to_numeric(df["homeScore"], errors="coerce") - pd.to_numeric(df["awayScore"], errors="coerce")
    else:
        actual_margin = np.nan
    out["actual_margin"] = actual_margin
    out["target_margin_ab"] = actual_margin
    out["neutral_site"] = pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0.0)
    out["team_a_is_home_non_neutral"] = non_neutral
    out["team_a_hca"] = np.where(
        non_neutral == 1.0,
        pd.to_numeric(df.get("home_team_hca"), errors="coerce").fillna(0.0),
        0.0,
    )
    out["tournament"] = df["tournament"]
    out["gameType"] = df["gameType"]
    out["conferenceGame"] = df["conferenceGame"]
    out["gameNotes"] = df["gameNotes"]
    out["neutral_subtype"] = _neutral_subtype(df)
    out["round_label"] = df["gameNotes"].map(_round_from_note)
    out["pair_augmented"] = 0
    return out


def swap_team_ab_source(source: pd.DataFrame) -> pd.DataFrame:
    out = source.copy()
    paired_cols = [
        ("team_a_team_id", "team_b_team_id"),
        ("team_a_name", "team_b_name"),
        ("team_a_adj_oe", "team_b_adj_oe"),
        ("team_a_adj_de", "team_b_adj_de"),
        ("team_a_barthag", "team_b_barthag"),
        ("team_a_conf_strength", "team_b_conf_strength"),
        ("team_a_sos_oe", "team_b_sos_oe"),
        ("team_a_sos_de", "team_b_sos_de"),
        ("team_a_form_delta", "team_b_form_delta"),
        ("team_a_rest_days", "team_b_rest_days"),
        ("team_a_eff_fg_pct", "team_b_eff_fg_pct"),
        ("team_a_ft_rate", "team_b_ft_rate"),
        ("team_a_off_rebound_pct", "team_b_off_rebound_pct"),
        ("team_a_tov_rate", "team_b_tov_rate"),
        ("team_a_margin_std", "team_b_margin_std"),
        ("team_a_barthag_rank", "team_b_barthag_rank"),
        ("team_a_seed", "team_b_seed"),
    ]
    for left, right in paired_cols:
        tmp = out[left].copy()
        out[left] = out[right]
        out[right] = tmp
    out["target_margin_ab"] = -pd.to_numeric(out["target_margin_ab"], errors="coerce")
    out["pair_augmented"] = 1
    return out


def build_team_ab_elite_tail_round64_contract(source: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=source.index)
    team_a_adj_net = pd.to_numeric(source["team_a_adj_oe"], errors="coerce") - pd.to_numeric(source["team_a_adj_de"], errors="coerce")
    team_b_adj_net = pd.to_numeric(source["team_b_adj_oe"], errors="coerce") - pd.to_numeric(source["team_b_adj_de"], errors="coerce")
    team_a_sos_net = pd.to_numeric(source["team_a_sos_oe"], errors="coerce") - pd.to_numeric(source["team_a_sos_de"], errors="coerce")
    team_b_sos_net = pd.to_numeric(source["team_b_sos_oe"], errors="coerce") - pd.to_numeric(source["team_b_sos_de"], errors="coerce")
    team_a_rank = pd.to_numeric(source["team_a_barthag_rank"], errors="coerce")
    team_b_rank = pd.to_numeric(source["team_b_barthag_rank"], errors="coerce")
    best_rank = pd.concat([team_a_rank, team_b_rank], axis=1).min(axis=1)
    worst_rank = pd.concat([team_a_rank, team_b_rank], axis=1).max(axis=1)
    team_a_seed = pd.to_numeric(source["team_a_seed"], errors="coerce")
    team_b_seed = pd.to_numeric(source["team_b_seed"], errors="coerce")
    best_seed = pd.concat([team_a_seed, team_b_seed], axis=1).min(axis=1)

    neutral_site = pd.to_numeric(source["neutral_site"], errors="coerce").fillna(0.0)
    is_ncaa = source["tournament"].eq("NCAA").astype(float)
    is_conf_tourney_neutral = source["neutral_subtype"].eq("conference_tournament_neutral").astype(float)
    is_early_mte_neutral = source["neutral_subtype"].eq("early_season_mte_neutral").astype(float)
    is_other_neutral = (
        neutral_site.eq(1.0)
        & ~is_ncaa.astype(bool)
        & ~is_conf_tourney_neutral.astype(bool)
        & ~is_early_mte_neutral.astype(bool)
    ).astype(float)
    is_round64 = (
        source["tournament"].eq("NCAA")
        & source["round_label"].eq("Round of 64")
    ).astype(float)

    out["neutral_site"] = neutral_site
    out["team_a_is_home_non_neutral"] = pd.to_numeric(source["team_a_is_home_non_neutral"], errors="coerce").fillna(0.0)
    out["team_a_hca"] = pd.to_numeric(source["team_a_hca"], errors="coerce").fillna(0.0)
    out["is_ncaa_tournament"] = is_ncaa
    out["is_conference_tournament_neutral"] = is_conf_tourney_neutral
    out["is_early_mte_neutral"] = is_early_mte_neutral
    out["is_other_neutral"] = is_other_neutral

    out["team_a_adj_net"] = team_a_adj_net
    out["team_b_adj_net"] = team_b_adj_net
    out["team_a_barthag"] = pd.to_numeric(source["team_a_barthag"], errors="coerce")
    out["team_b_barthag"] = pd.to_numeric(source["team_b_barthag"], errors="coerce")
    out["team_a_conf_strength"] = pd.to_numeric(source["team_a_conf_strength"], errors="coerce")
    out["team_b_conf_strength"] = pd.to_numeric(source["team_b_conf_strength"], errors="coerce")
    out["team_a_sos_net"] = team_a_sos_net
    out["team_b_sos_net"] = team_b_sos_net
    out["team_a_form_delta"] = pd.to_numeric(source["team_a_form_delta"], errors="coerce")
    out["team_b_form_delta"] = pd.to_numeric(source["team_b_form_delta"], errors="coerce")
    out["team_a_rest_days"] = pd.to_numeric(source["team_a_rest_days"], errors="coerce")
    out["team_b_rest_days"] = pd.to_numeric(source["team_b_rest_days"], errors="coerce")

    out["adj_oe_diff"] = pd.to_numeric(source["team_a_adj_oe"], errors="coerce") - pd.to_numeric(source["team_b_adj_oe"], errors="coerce")
    out["adj_de_edge"] = pd.to_numeric(source["team_b_adj_de"], errors="coerce") - pd.to_numeric(source["team_a_adj_de"], errors="coerce")
    out["adj_net_diff"] = team_a_adj_net - team_b_adj_net
    out["barthag_diff"] = out["team_a_barthag"] - out["team_b_barthag"]
    out["conf_strength_diff"] = out["team_a_conf_strength"] - out["team_b_conf_strength"]
    out["sos_net_diff"] = team_a_sos_net - team_b_sos_net
    out["form_delta_diff"] = out["team_a_form_delta"] - out["team_b_form_delta"]
    out["rest_advantage_ab"] = out["team_a_rest_days"] - out["team_b_rest_days"]

    out["adj_net_sum"] = team_a_adj_net + team_b_adj_net
    out["barthag_sum"] = out["team_a_barthag"] + out["team_b_barthag"]
    out["conf_strength_sum"] = out["team_a_conf_strength"] + out["team_b_conf_strength"]
    out["sos_net_sum"] = team_a_sos_net + team_b_sos_net
    out["form_delta_sum"] = out["team_a_form_delta"] + out["team_b_form_delta"]

    out["efg_diff"] = pd.to_numeric(source["team_a_eff_fg_pct"], errors="coerce") - pd.to_numeric(source["team_b_eff_fg_pct"], errors="coerce")
    out["ft_rate_diff"] = pd.to_numeric(source["team_a_ft_rate"], errors="coerce") - pd.to_numeric(source["team_b_ft_rate"], errors="coerce")
    out["off_rebound_diff"] = pd.to_numeric(source["team_a_off_rebound_pct"], errors="coerce") - pd.to_numeric(source["team_b_off_rebound_pct"], errors="coerce")
    out["tov_edge"] = pd.to_numeric(source["team_b_tov_rate"], errors="coerce") - pd.to_numeric(source["team_a_tov_rate"], errors="coerce")
    out["margin_std_diff"] = pd.to_numeric(source["team_a_margin_std"], errors="coerce") - pd.to_numeric(source["team_b_margin_std"], errors="coerce")
    out["efg_sum"] = pd.to_numeric(source["team_a_eff_fg_pct"], errors="coerce") + pd.to_numeric(source["team_b_eff_fg_pct"], errors="coerce")
    out["ft_rate_sum"] = pd.to_numeric(source["team_a_ft_rate"], errors="coerce") + pd.to_numeric(source["team_b_ft_rate"], errors="coerce")
    out["off_rebound_sum"] = pd.to_numeric(source["team_a_off_rebound_pct"], errors="coerce") + pd.to_numeric(source["team_b_off_rebound_pct"], errors="coerce")
    out["tov_total"] = pd.to_numeric(source["team_a_tov_rate"], errors="coerce") + pd.to_numeric(source["team_b_tov_rate"], errors="coerce")
    out["margin_std_sum"] = pd.to_numeric(source["team_a_margin_std"], errors="coerce") + pd.to_numeric(source["team_b_margin_std"], errors="coerce")

    out["abs_adj_net_gap"] = out["adj_net_diff"].abs()
    out["abs_barthag_gap"] = out["barthag_diff"].abs()
    out["best_team_top10"] = best_rank.le(10).astype(float)
    out["best_team_top15"] = best_rank.le(15).astype(float)
    out["best_team_top4"] = best_rank.le(4).astype(float)
    out["best_team_seed1"] = (is_ncaa.eq(1.0) & best_seed.eq(1)).astype(float)
    out["both_teams_top25"] = worst_rank.le(25).astype(float)

    out["neutral_adj_net_diff"] = out["neutral_site"] * out["adj_net_diff"]
    out["neutral_barthag_diff"] = out["neutral_site"] * out["barthag_diff"]
    out["neutral_abs_adj_net_gap"] = out["neutral_site"] * out["abs_adj_net_gap"]
    out["ncaa_adj_net_diff"] = out["is_ncaa_tournament"] * out["adj_net_diff"]
    out["ncaa_barthag_diff"] = out["is_ncaa_tournament"] * out["barthag_diff"]
    out["ncaa_abs_adj_net_gap"] = out["is_ncaa_tournament"] * out["abs_adj_net_gap"]
    out["neutral_elite_gap_top10"] = out["neutral_site"] * out["best_team_top10"] * out["abs_adj_net_gap"]
    out["neutral_elite_gap_top15"] = out["neutral_site"] * out["best_team_top15"] * out["abs_adj_net_gap"]
    out["ncaa_elite_gap_top10"] = out["is_ncaa_tournament"] * out["best_team_top10"] * out["abs_adj_net_gap"]
    out["ncaa_elite_gap_top15"] = out["is_ncaa_tournament"] * out["best_team_top15"] * out["abs_adj_net_gap"]

    out["best_team_top5"] = best_rank.le(5).astype(float)
    out["best_team_top2"] = best_rank.le(2).astype(float)
    out["worst_team_outside_top50"] = worst_rank.gt(50).astype(float)
    out["worst_team_outside_top100"] = worst_rank.gt(100).astype(float)
    out["top5_adj_net_diff"] = out["best_team_top5"] * out["adj_net_diff"]
    out["top2_adj_net_diff"] = out["best_team_top2"] * out["adj_net_diff"]
    out["seed1_adj_net_diff"] = out["best_team_seed1"] * out["adj_net_diff"]
    out["seed1_barthag_diff"] = out["best_team_seed1"] * out["barthag_diff"]
    out["elite_gap_power"] = out["adj_net_diff"] * out["abs_adj_net_gap"]
    out["weak_opp_adj_net_diff"] = out["worst_team_outside_top100"] * out["adj_net_diff"]
    out["seed1_large_gap"] = out["best_team_seed1"] * out["abs_adj_net_gap"]
    out["top5_large_gap"] = out["best_team_top5"] * out["abs_adj_net_gap"]

    out["is_round_of_64"] = is_round64
    out["round64_adj_net_diff"] = is_round64 * out["adj_net_diff"]
    out["round64_top5_adj_net_diff"] = is_round64 * out["best_team_top5"] * out["adj_net_diff"]
    out["round64_seed1_adj_net_diff"] = is_round64 * out["best_team_seed1"] * out["adj_net_diff"]
    out["round64_seed1_gap_power"] = is_round64 * out["best_team_seed1"] * out["adj_net_diff"] * out["abs_adj_net_gap"]
    out["round64_seed1_large_gap"] = is_round64 * out["best_team_seed1"] * out["abs_adj_net_gap"]

    missing = [feature for feature in TEAM_AB_ELITE_TAIL_ROUND64_V1_FEATURE_ORDER if feature not in out.columns]
    if missing:
        raise KeyError(f"Team A/B elite-tail contract missing features: {missing}")
    return out[TEAM_AB_ELITE_TAIL_ROUND64_V1_FEATURE_ORDER].copy()


def prepare_team_ab_training_matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    source = build_team_ab_source(frame)
    feature_df = build_team_ab_elite_tail_round64_contract(source)
    neutral_source = source.loc[source["neutral_site"] == 1.0].copy().reset_index(drop=True)
    neutral_swap = swap_team_ab_source(neutral_source)
    feature_swap = build_team_ab_elite_tail_round64_contract(neutral_swap)

    aug_source = pd.concat([source, neutral_swap], ignore_index=True)
    aug_feature = pd.concat([feature_df, feature_swap], ignore_index=True)
    aug_target = aug_source["target_margin_ab"].to_numpy(dtype=np.float32)

    ordered = (
        aug_source.assign(_sort_date=pd.to_datetime(aug_source["startDate"], errors="coerce", utc=True))
        .sort_values(["_sort_date", "gameId", "pair_augmented"], kind="mergesort")
        .index.to_numpy()
    )
    aug_source = aug_source.loc[ordered].reset_index(drop=True)
    aug_feature = aug_feature.loc[ordered].reset_index(drop=True)
    aug_target = aug_target[ordered]
    return aug_source, aug_feature, aug_target


def build_mean_model_feature_frame(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    normalized = normalize_mean_model_variant(variant)
    if normalized == LEGACY_HOME_SLOT:
        return frame[list(config.FEATURE_ORDER)].copy()
    source = build_team_ab_source(frame)
    return build_team_ab_elite_tail_round64_contract(source)
