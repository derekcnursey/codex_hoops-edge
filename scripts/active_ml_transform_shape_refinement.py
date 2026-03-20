#!/usr/bin/env python3
"""Constrained refinement study for the live active-stack moneyline transform.

Goal:
  keep active mu + current sigma fixed while testing smaller, less aggressive
  NCAA-neutral probability transforms for bracket-facing ML odds.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src import config
from src.ml_odds import ACTIVE_META_MARKET_V1, logistic, mu_sigma_home_win_prob


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_CACHE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "research"
    / "neutral_probability_stack_study_v1"
    / "cache"
    / "scored_master.parquet"
)
CURRENT_NCAA_CACHE = PROJECT_ROOT / "site" / "public" / "data" / "ncaa_matchup_predictions_2026.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "research" / "active_ml_transform_shape_refinement_v1"
ALL_SEASONS = [season for season in range(2015, 2026) if season not in config.EXCLUDE_SEASONS]
EVAL_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025]
SLICE_ORDER = [
    "ncaa_tournament_neutral",
    "conference_tournament_neutral",
    "all_neutral",
    "all_games",
]
REPRESENTATIVE_CASES = [
    {"case": "neutral_mu0_sigma12", "mu": 0.0, "sigma": 12.0, "month": 3, "day": 20, "neutral": 1.0, "is_ncaa": 1.0, "is_conf": 0.0},
    {"case": "close_game_mu1_sigma12", "mu": 1.0, "sigma": 12.0, "month": 3, "day": 20, "neutral": 1.0, "is_ncaa": 1.0, "is_conf": 0.0},
    {"case": "moderate_favorite_mu3p5_sigma12p5", "mu": 3.5, "sigma": 12.5, "month": 3, "day": 20, "neutral": 1.0, "is_ncaa": 1.0, "is_conf": 0.0},
    {"case": "strong_favorite_mu7_sigma12p5", "mu": 7.0, "sigma": 12.5, "month": 3, "day": 20, "neutral": 1.0, "is_ncaa": 1.0, "is_conf": 0.0},
]
BRACKET_EXAMPLES = [
    {"label": "Louisville vs South Florida", "key": "150::271", "display_team_id": 150},
    {"label": "UCLA vs UCF", "key": "312::313", "display_team_id": 313},
    {"label": "UConn vs UCF", "key": "312::314", "display_team_id": 314},
    {"label": "Michigan State vs UCF", "key": "169::312", "display_team_id": 169},
]
CURRENT_COEFFS = {
    "intercept": float(ACTIVE_META_MARKET_V1["intercept"]),
    **{k: float(v) for k, v in ACTIVE_META_MARKET_V1["coefficients"].items()},
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    description: str
    live_safety: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for outputs (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


def _safe_prob(values: np.ndarray | pd.Series | float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1.0e-6)


def _american_to_implied_prob(odds: np.ndarray) -> np.ndarray:
    arr = np.asarray(odds, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    pos = arr > 0
    neg = arr < 0
    out[pos] = 100.0 / (arr[pos] + 100.0)
    out[neg] = (-arr[neg]) / ((-arr[neg]) + 100.0)
    return out


def _vig_free_home_prob(home_ml: pd.Series, away_ml: pd.Series) -> np.ndarray:
    home_raw = _american_to_implied_prob(pd.to_numeric(home_ml, errors="coerce").to_numpy(dtype=float))
    away_raw = _american_to_implied_prob(pd.to_numeric(away_ml, errors="coerce").to_numpy(dtype=float))
    denom = home_raw + away_raw
    out = np.full_like(home_raw, np.nan, dtype=float)
    valid = np.isfinite(home_raw) & np.isfinite(away_raw) & np.isfinite(denom) & (denom > 0.0)
    out[valid] = home_raw[valid] / denom[valid]
    return out


def _is_post_dec15(start_dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(start_dates, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    month = dt.dt.month.fillna(0).astype(int)
    day = dt.dt.day.fillna(0).astype(int)
    return (((month == 12) & (day >= 15)) | month.isin([1, 2, 3])).to_numpy(dtype=float)


def _feature_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    sigma_cap14 = np.clip(df["pred_sigma_current"].to_numpy(dtype=float), 0.5, 14.0)
    mu = df["model_mu_home"].to_numpy(dtype=float)
    z14 = mu / sigma_cap14
    post_dec15 = _is_post_dec15(df["startDate"])
    neutral = df["neutral_site"].to_numpy(dtype=float)
    is_ncaa = ((df["tournament"] == "NCAA") & df["neutral_site"].eq(1.0)).to_numpy(dtype=float)
    is_conf = ((df["regime"] == "conference_tournament") & df["neutral_site"].eq(1.0)).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "mu": mu,
            "sigma_cap14": sigma_cap14,
            "z14": z14,
            "post_dec15": post_dec15,
            "abs_mu": np.abs(mu),
        },
        index=df.index,
    )
    if feature_set in {"full", "bounded", "shape_penalized"}:
        out["neutral_site"] = neutral
        out["is_ncaa_neutral"] = is_ncaa
        out["is_conf_tourney_neutral"] = is_conf
        out["z14_x_neutral"] = z14 * neutral
        out["abs_mu_x_neutral"] = np.abs(mu) * neutral
    elif feature_set == "reduced":
        out["neutral_site"] = neutral
        out["z14_x_neutral"] = z14 * neutral
    else:
        raise ValueError(feature_set)
    return out


def _score_from_params(features: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    score = np.full(len(features), float(params["intercept"]), dtype=float)
    for key, values in features.items():
        score += float(params.get(key, 0.0)) * values.to_numpy(dtype=float)
    return score


def _prob_from_params(features: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    return _safe_prob(logistic(_score_from_params(features, params)))


def _params_from_theta(names: list[str], theta: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, theta)}


def _anchor_features(params: dict[str, float], *, mu: float, sigma: float, neutral: float, is_ncaa: float, is_conf: float) -> dict[str, float]:
    sigma_cap14 = min(max(float(sigma), 0.5), 14.0)
    z14 = float(mu) / sigma_cap14
    return {
        "mu": float(mu),
        "sigma_cap14": sigma_cap14,
        "z14": z14,
        "post_dec15": 1.0,
        "abs_mu": abs(float(mu)),
        "neutral_site": neutral,
        "is_ncaa_neutral": is_ncaa,
        "is_conf_tourney_neutral": is_conf,
        "z14_x_neutral": z14 * neutral,
        "abs_mu_x_neutral": abs(float(mu)) * neutral,
    }


def _evaluate_anchor_probability(params: dict[str, float], feature_set: str, *, mu: float, sigma: float, neutral: float, is_ncaa: float, is_conf: float) -> float:
    feature_map = _anchor_features(params, mu=mu, sigma=sigma, neutral=neutral, is_ncaa=is_ncaa, is_conf=is_conf)
    columns = ["mu", "sigma_cap14", "z14", "post_dec15", "abs_mu"]
    if feature_set in {"full", "bounded", "shape_penalized"}:
        columns += ["neutral_site", "is_ncaa_neutral", "is_conf_tourney_neutral", "z14_x_neutral", "abs_mu_x_neutral"]
    elif feature_set == "reduced":
        columns += ["neutral_site", "z14_x_neutral"]
    features = pd.DataFrame([{col: feature_map.get(col, 0.0) for col in columns}])
    return float(_prob_from_params(features, params)[0])


def _fit_transform(
    train_df: pd.DataFrame,
    features: pd.DataFrame,
    target_prob: np.ndarray,
    *,
    init_params: dict[str, float],
    feature_set: str,
    l2: float = 5e-4,
    bounds: dict[str, tuple[float, float]] | None = None,
    shape_penalty: float = 0.0,
    anchor_penalty: float = 0.0,
    moderate_penalty: float = 0.0,
    heavy_penalty: float = 0.0,
) -> dict[str, float]:
    names = ["intercept"] + list(features.columns)
    init = np.array([init_params.get(name, 0.0) for name in names], dtype=float)
    X = features.to_numpy(dtype=float)
    y = _safe_prob(target_prob)
    neutral_mask = train_df["neutral_site"].eq(1.0).to_numpy(dtype=bool)
    ncaa_mask = (train_df["tournament"] == "NCAA").to_numpy(dtype=bool) & neutral_mask
    actual_home = train_df["actual_home_win"].to_numpy(dtype=float)
    cap14 = np.asarray(train_df["cap14_prob"], dtype=float)
    moderate_mask = ncaa_mask & (cap14 >= 0.60) & (cap14 <= 0.75)
    heavy_base_mask = ncaa_mask & (cap14 >= 0.85)

    scipy_bounds = None
    if bounds:
        scipy_bounds = [bounds.get(name, (-np.inf, np.inf)) for name in names]

    def objective(theta: np.ndarray) -> float:
        intercept = theta[0]
        coef = theta[1:]
        pred = _safe_prob(logistic(intercept + X @ coef))
        mse = np.mean((pred - y) ** 2)
        reg = l2 * np.mean((theta - init) ** 2)
        loss = float(mse + reg)
        params = _params_from_theta(names, theta)
        if anchor_penalty > 0.0:
            p_zero = _evaluate_anchor_probability(params, feature_set, mu=0.0, sigma=12.0, neutral=1.0, is_ncaa=1.0, is_conf=0.0)
            loss += anchor_penalty * max(0.0, p_zero - 0.52) ** 2
        if moderate_penalty > 0.0 and moderate_mask.any():
            lift = pred[moderate_mask] - cap14[moderate_mask]
            loss += moderate_penalty * max(0.0, float(lift.mean()) - 0.045) ** 2
        if heavy_penalty > 0.0 and heavy_base_mask.any():
            favored_prob = np.maximum(pred[heavy_base_mask], 1.0 - pred[heavy_base_mask])
            favored_won = np.where(pred[heavy_base_mask] >= 0.5, actual_home[heavy_base_mask], 1.0 - actual_home[heavy_base_mask])
            overshoot = float(favored_prob.mean() - favored_won.mean())
            loss += heavy_penalty * max(0.0, overshoot - 0.015) ** 2
        if shape_penalty > 0.0:
            p_close = _evaluate_anchor_probability(params, feature_set, mu=1.0, sigma=12.0, neutral=1.0, is_ncaa=1.0, is_conf=0.0)
            p_moderate = _evaluate_anchor_probability(params, feature_set, mu=3.5, sigma=12.5, neutral=1.0, is_ncaa=1.0, is_conf=0.0)
            p_strong = _evaluate_anchor_probability(params, feature_set, mu=7.0, sigma=12.5, neutral=1.0, is_ncaa=1.0, is_conf=0.0)
            loss += shape_penalty * max(0.0, p_close - 0.56) ** 2
            loss += shape_penalty * max(0.0, p_moderate - 0.66) ** 2
            loss += shape_penalty * max(0.0, p_strong - 0.80) ** 2
        return loss

    result = minimize(objective, init, method="L-BFGS-B", bounds=scipy_bounds)
    theta = result.x if result.success else init
    return _params_from_theta(names, theta)


def _slice_mask(df: pd.DataFrame, slice_name: str) -> pd.Series:
    if slice_name == "ncaa_tournament_neutral":
        return df["tournament"].eq("NCAA") & df["neutral_site"].eq(1.0)
    if slice_name == "conference_tournament_neutral":
        return df["regime"].eq("conference_tournament") & df["neutral_site"].eq(1.0)
    if slice_name == "all_neutral":
        return df["neutral_site"].eq(1.0)
    if slice_name == "all_games":
        return pd.Series(True, index=df.index)
    raise ValueError(slice_name)


def _calibration_regression(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    prob = _safe_prob(prob)
    logit = np.log(prob / (1.0 - prob)).reshape(-1, 1)
    model = LogisticRegression(fit_intercept=True, solver="lbfgs")
    model.fit(logit, y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def _bucket_rows(frame: pd.DataFrame, candidate: str, slice_name: str) -> list[dict[str, Any]]:
    prob_col = f"{candidate}__prob"
    work = frame.loc[_slice_mask(frame, slice_name)].dropna(subset=["actual_home_win", prob_col]).copy()
    if work.empty:
        return []
    prob = _safe_prob(work[prob_col])
    y = work["actual_home_win"].to_numpy(dtype=int)
    edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    rows: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "candidate": candidate,
                "slice": slice_name,
                "bucket_lo": lo,
                "bucket_hi": hi,
                "rows": int(mask.sum()),
                "avg_pred_home_prob": float(prob[mask].mean()),
                "actual_home_win_rate": float(y[mask].mean()),
                "home_calibration_gap": float(prob[mask].mean() - y[mask].mean()),
            }
        )
    return rows


def _context_rows(frame: pd.DataFrame, candidate: str, slice_name: str) -> list[dict[str, Any]]:
    prob_col = f"{candidate}__prob"
    work = frame.loc[_slice_mask(frame, slice_name)].dropna(subset=["actual_home_win", prob_col]).copy()
    if work.empty:
        return []
    prob = _safe_prob(work[prob_col])
    y_home = work["actual_home_win"].to_numpy(dtype=int)
    favored_prob = np.maximum(prob, 1.0 - prob)
    favored_won = np.where(prob >= 0.5, y_home, 1 - y_home)
    contexts = {
        "favorite_tail_80": favored_prob >= 0.8,
        "heavy_favorite_90": favored_prob >= 0.9,
        "close_game_prob_band": np.abs(prob - 0.5) <= 0.05,
        "moderate_favorite_band": (favored_prob >= 0.65) & (favored_prob < 0.8),
    }
    market_prob = pd.to_numeric(work["repaired_market_home_prob"], errors="coerce").to_numpy(dtype=float)
    market_favored = np.maximum(market_prob, 1.0 - market_prob)
    rows: list[dict[str, Any]] = []
    for label, mask in contexts.items():
        if not mask.any():
            continue
        valid_market = np.isfinite(market_favored[mask])
        rows.append(
            {
                "candidate": candidate,
                "slice": slice_name,
                "context": label,
                "rows": int(mask.sum()),
                "avg_favored_prob": float(favored_prob[mask].mean()),
                "favored_win_rate": float(favored_won[mask].mean()),
                "favored_gap_actual": float(favored_prob[mask].mean() - favored_won[mask].mean()),
                "avg_market_favored_prob": float(market_favored[mask][valid_market].mean()) if valid_market.any() else np.nan,
                "favored_gap_market": float(favored_prob[mask][valid_market].mean() - market_favored[mask][valid_market].mean()) if valid_market.any() else np.nan,
            }
        )
    return rows


def _load_master() -> pd.DataFrame:
    df = pd.read_parquet(MASTER_CACHE_PATH).copy()
    df = df[df["season"].isin(ALL_SEASONS)].copy()
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    df["actual_home_win"] = pd.to_numeric(df["actual_home_win"], errors="coerce")
    df["actual_margin"] = pd.to_numeric(df["actual_margin"], errors="coerce")
    df["model_mu_home"] = pd.to_numeric(df["model_mu_home"], errors="coerce")
    df["pred_sigma_current"] = pd.to_numeric(df["pred_sigma_current"], errors="coerce")
    df["neutral_site"] = pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0.0)
    df["repaired_market_home_prob"] = _vig_free_home_prob(df["repaired_home_moneyline"], df["repaired_away_moneyline"])
    df["cap14_prob"] = mu_sigma_home_win_prob(df["model_mu_home"].to_numpy(dtype=float), df["pred_sigma_current"].to_numpy(dtype=float), sigma_mode="cap14")
    full_features = _feature_frame(df, "full")
    df["current_active_market_v1__prob"] = _prob_from_params(full_features, CURRENT_COEFFS).astype(np.float32)
    df["tapered_context_85_v1__prob"] = _prob_from_params(full_features, _tapered_current_params(0.85)).astype(np.float32)
    df["tapered_context_75_v1__prob"] = _prob_from_params(full_features, _tapered_current_params(0.75)).astype(np.float32)
    df["cap14_baseline__prob"] = df["cap14_prob"].astype(np.float32)
    return df.reset_index(drop=True)


def _candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec("cap14_baseline", "baseline", "Plain cap14 mu/sigma conversion.", "audit_only"),
        CandidateSpec("current_active_market_v1", "current", "Current live active market-refit transform.", "current_live"),
        CandidateSpec("tapered_context_85_v1", "taper", "Current transform with positive NCAA-neutral/context lift terms tapered to 85%.", "safe_live_candidate"),
        CandidateSpec("tapered_context_75_v1", "taper", "Current transform with positive NCAA-neutral/context lift terms tapered to 75%.", "research_only"),
        CandidateSpec("bounded_anchor_v1", "constrained", "Current family refit with bounded NCAA-neutral/context lift and anchor penalty.", "safe_live_candidate"),
        CandidateSpec("reduced_context_v1", "reduced", "Smaller active transform keeping only neutral_site and z14_x_neutral context.", "safe_live_candidate"),
        CandidateSpec("shape_penalized_v1", "penalized", "Current family refit with NCAA-neutral shape penalties on zero/mid/heavy zones.", "safe_live_candidate"),
    ]


def _tapered_current_params(scale: float) -> dict[str, float]:
    params = dict(CURRENT_COEFFS)
    for key in ["post_dec15", "neutral_site", "is_ncaa_neutral", "is_conf_tourney_neutral", "abs_mu_x_neutral"]:
        params[key] = float(params[key]) * scale
    return params


def _candidate_probs(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_master = master[master["season"].isin(EVAL_SEASONS)].copy().reset_index(drop=True)
    params_rows: list[dict[str, Any]] = []
    season_frames: list[pd.DataFrame] = []

    for season in EVAL_SEASONS:
        train_df = master[(master["season"] < season) & master["repaired_market_home_prob"].notna()].copy()
        season_df = eval_master[eval_master["season"] == season].copy().reset_index(drop=True)
        out = season_df[
            [
                "season",
                "gameId",
                "startDate",
                "actual_home_win",
                "actual_margin",
                "model_mu_home",
                "pred_sigma_current",
                "neutral_site",
                "tournament",
                "regime",
                "repaired_market_home_prob",
                "current_active_market_v1__prob",
                "tapered_context_85_v1__prob",
                "tapered_context_75_v1__prob",
                "cap14_baseline__prob",
            ]
        ].copy()

        full_train = _feature_frame(train_df, "full")
        full_eval = _feature_frame(season_df, "full")
        reduced_train = _feature_frame(train_df, "reduced")
        reduced_eval = _feature_frame(season_df, "reduced")

        bounded_bounds = {
            "intercept": (-0.35, 0.05),
            "mu": (0.14, 0.21),
            "sigma_cap14": (-0.02, -0.004),
            "z14": (-0.05, 0.08),
            "post_dec15": (0.02, 0.10),
            "abs_mu": (-0.01, 0.002),
            "neutral_site": (0.05, 0.18),
            "is_ncaa_neutral": (0.0, 0.04),
            "is_conf_tourney_neutral": (-0.04, 0.02),
            "z14_x_neutral": (-0.28, -0.02),
            "abs_mu_x_neutral": (0.0, 0.008),
        }
        params_bounded = _fit_transform(
            train_df,
            full_train,
            train_df["repaired_market_home_prob"].to_numpy(dtype=float),
            init_params=CURRENT_COEFFS,
            feature_set="bounded",
            bounds=bounded_bounds,
            l2=2e-3,
            anchor_penalty=15.0,
            moderate_penalty=12.0,
            heavy_penalty=8.0,
        )
        out["bounded_anchor_v1__prob"] = _prob_from_params(full_eval, params_bounded).astype(np.float32)
        params_rows.append({"candidate": "bounded_anchor_v1", "season": season, **params_bounded})

        reduced_init = {
            "intercept": CURRENT_COEFFS["intercept"],
            "mu": CURRENT_COEFFS["mu"],
            "sigma_cap14": CURRENT_COEFFS["sigma_cap14"],
            "z14": CURRENT_COEFFS["z14"],
            "post_dec15": CURRENT_COEFFS["post_dec15"],
            "abs_mu": CURRENT_COEFFS["abs_mu"],
            "neutral_site": 0.10,
            "z14_x_neutral": -0.18,
        }
        params_reduced = _fit_transform(
            train_df,
            reduced_train,
            train_df["repaired_market_home_prob"].to_numpy(dtype=float),
            init_params=reduced_init,
            feature_set="reduced",
            l2=8e-4,
            anchor_penalty=10.0,
            moderate_penalty=8.0,
            heavy_penalty=6.0,
        )
        out["reduced_context_v1__prob"] = _prob_from_params(reduced_eval, params_reduced).astype(np.float32)
        params_rows.append({"candidate": "reduced_context_v1", "season": season, **params_reduced})

        params_penalized = _fit_transform(
            train_df,
            full_train,
            train_df["repaired_market_home_prob"].to_numpy(dtype=float),
            init_params=CURRENT_COEFFS,
            feature_set="shape_penalized",
            l2=1.2e-3,
            shape_penalty=12.0,
            anchor_penalty=10.0,
            moderate_penalty=10.0,
            heavy_penalty=10.0,
        )
        out["shape_penalized_v1__prob"] = _prob_from_params(full_eval, params_penalized).astype(np.float32)
        params_rows.append({"candidate": "shape_penalized_v1", "season": season, **params_penalized})

        season_frames.append(out)

    return pd.concat(season_frames, ignore_index=True), pd.DataFrame(params_rows)


def _aggregate_metrics(frame: pd.DataFrame, candidate: str, slice_name: str) -> dict[str, Any] | None:
    prob_col = f"{candidate}__prob"
    work = frame.loc[_slice_mask(frame, slice_name)].dropna(subset=["actual_home_win", prob_col]).copy()
    if work.empty:
        return None
    prob = _safe_prob(work[prob_col])
    y = work["actual_home_win"].to_numpy(dtype=int)
    slope, intercept = _calibration_regression(prob, y)
    return {
        "candidate": candidate,
        "slice": slice_name,
        "rows": int(len(work)),
        "logloss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
        "home_overconfidence": float(prob.mean() - y.mean()),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _market_metrics(frame: pd.DataFrame, candidate: str, slice_name: str) -> dict[str, Any] | None:
    prob_col = f"{candidate}__prob"
    work = frame.loc[_slice_mask(frame, slice_name)].dropna(subset=["repaired_market_home_prob", prob_col]).copy()
    if work.empty:
        return None
    pred = _safe_prob(work[prob_col])
    market = _safe_prob(work["repaired_market_home_prob"])
    return {
        "candidate": candidate,
        "slice": slice_name,
        "rows": int(len(work)),
        "mae_vs_market": float(np.mean(np.abs(pred - market))),
        "rmse_vs_market": float(np.sqrt(np.mean((pred - market) ** 2))),
        "signed_bias_vs_market": float(np.mean(pred - market)),
    }


def _representative_case_rows(params_map: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in REPRESENTATIVE_CASES:
        mu = case["mu"]
        sigma = case["sigma"]
        cap14 = float(mu_sigma_home_win_prob(mu, sigma, sigma_mode="cap14"))
        row = {"case": case["case"], "mu": mu, "sigma": sigma, "cap14_baseline": cap14}
        for candidate, params in params_map.items():
            feature_set = "full"
            if candidate == "reduced_context_v1":
                feature_set = "reduced"
            row[candidate] = _evaluate_anchor_probability(
                params,
                feature_set,
                mu=mu,
                sigma=sigma,
                neutral=case["neutral"],
                is_ncaa=case["is_ncaa"],
                is_conf=case["is_conf"],
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _bracket_example_rows(params_map: dict[str, dict[str, float]]) -> pd.DataFrame:
    with CURRENT_NCAA_CACHE.open() as f:
        cache = json.load(f)["predictions"]
    rows: list[dict[str, Any]] = []
    for case in BRACKET_EXAMPLES:
        entry = cache[case["key"]]
        display_team_id = int(case["display_team_id"])
        sigma = float(entry["pred_sigma"])
        t1 = int(entry["team1_id"])
        direct = display_team_id == t1
        active_mu = float(entry["mu_team1_minus_team2"]) if direct else -float(entry["mu_team1_minus_team2"])
        internal_mu = (
            None
            if entry.get("mu_team1_minus_team2_team_ab_internal") is None
            else (float(entry["mu_team1_minus_team2_team_ab_internal"]) if direct else -float(entry["mu_team1_minus_team2_team_ab_internal"]))
        )
        row = {
            "label": case["label"],
            "key": case["key"],
            "sigma": sigma,
            "active_mu_display": active_mu,
            "internal_mu_display": internal_mu,
            "cap14_active_prob": float(mu_sigma_home_win_prob(active_mu, sigma, sigma_mode="cap14")),
        }
        for candidate, params in params_map.items():
            feature_set = "reduced" if candidate == "reduced_context_v1" else "full"
            active_prob = _evaluate_anchor_probability(
                params, feature_set, mu=active_mu, sigma=sigma, neutral=1.0, is_ncaa=1.0, is_conf=0.0
            )
            if internal_mu is None:
                display_prob = active_prob
            else:
                internal_prob = _evaluate_anchor_probability(
                    params, feature_set, mu=internal_mu, sigma=sigma, neutral=1.0, is_ncaa=1.0, is_conf=0.0
                )
                display_prob = 0.5 * (active_prob + internal_prob)
            row[f"{candidate}__active_prob"] = active_prob
            row[f"{candidate}__display_avg_prob"] = display_prob
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = _load_master()
    eval_df, params_df = _candidate_probs(master)
    registry = pd.DataFrame([spec.__dict__ for spec in _candidate_specs()])
    registry.to_csv(args.output_dir / "candidate_registry.csv", index=False)
    params_df.to_csv(args.output_dir / "candidate_params.csv", index=False)

    # Use median season params for representative examples.
    params_map: dict[str, dict[str, float]] = {
        "current_active_market_v1": CURRENT_COEFFS,
        "tapered_context_85_v1": _tapered_current_params(0.85),
        "tapered_context_75_v1": _tapered_current_params(0.75),
    }
    for candidate in ["bounded_anchor_v1", "reduced_context_v1", "shape_penalized_v1"]:
        sub = params_df[params_df["candidate"] == candidate].drop(columns=["candidate", "season"])
        params_map[candidate] = {col: float(sub[col].median()) for col in sub.columns}

    aggregate_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for candidate in registry["name"]:
        for slice_name in SLICE_ORDER:
            agg = _aggregate_metrics(eval_df, candidate, slice_name)
            if agg is not None:
                aggregate_rows.append(agg)
            market = _market_metrics(eval_df, candidate, slice_name)
            if market is not None:
                market_rows.append(market)
            bucket_rows.extend(_bucket_rows(eval_df, candidate, slice_name))
            context_rows.extend(_context_rows(eval_df, candidate, slice_name))

    aggregate_df = pd.DataFrame(aggregate_rows)
    market_df = pd.DataFrame(market_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    context_df = pd.DataFrame(context_rows)
    representative_df = _representative_case_rows(params_map)
    bracket_df = _bracket_example_rows(params_map)

    aggregate_df.to_csv(args.output_dir / "aggregate_results.csv", index=False)
    market_df.to_csv(args.output_dir / "market_alignment_results.csv", index=False)
    bucket_df.to_csv(args.output_dir / "bucket_results.csv", index=False)
    context_df.to_csv(args.output_dir / "context_results.csv", index=False)
    representative_df.to_csv(args.output_dir / "representative_cases.csv", index=False)
    bracket_df.to_csv(args.output_dir / "bracket_examples.csv", index=False)

    audit = {
        "current_transform": CURRENT_COEFFS,
        "eval_seasons": EVAL_SEASONS,
        "slice_rows": {slice_name: int(_slice_mask(eval_df, slice_name).sum()) for slice_name in SLICE_ORDER},
    }
    (args.output_dir / "transform_audit.json").write_text(json.dumps(audit, indent=2))

    summary_lines = ["# Active ML Transform Shape Refinement", "", "## Topline"]
    current = aggregate_df[aggregate_df["candidate"] == "current_active_market_v1"].set_index("slice")
    cap14 = aggregate_df[aggregate_df["candidate"] == "cap14_baseline"].set_index("slice")
    for slice_name in SLICE_ORDER:
        sub = aggregate_df[aggregate_df["slice"] == slice_name].sort_values(["logloss", "brier", "candidate"])
        if sub.empty:
            continue
        best = sub.iloc[0]
        summary_lines.append(
            f"- {slice_name}: best={best['candidate']} | logloss={best['logloss']:.6f} | brier={best['brier']:.6f}"
        )
        if slice_name in current.index:
            row = current.loc[slice_name]
            summary_lines.append(
                f"  current_active_market_v1: logloss={row['logloss']:.6f} | brier={row['brier']:.6f}"
            )
        if slice_name in cap14.index:
            row = cap14.loc[slice_name]
            summary_lines.append(
                f"  cap14_baseline: logloss={row['logloss']:.6f} | brier={row['brier']:.6f}"
            )
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
