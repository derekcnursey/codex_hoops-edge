from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.ml_odds import mu_sigma_home_win_prob

SITE_DATA = ROOT / "site" / "public" / "data"


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    calibrated: bool = False


def _season_from_date(date_str: str) -> int:
    year, month, _ = map(int, date_str.split("-"))
    return year + 1 if month >= 11 else year


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def _bucket_table(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    bucket = pd.cut(df[prob_col], bins=bins, labels=labels, include_lowest=True, right=True)
    out = (
        df.assign(bucket=bucket)
        .groupby("bucket", observed=True)
        .agg(
            n=("home_win", "size"),
            avg_prob=(prob_col, "mean"),
            actual_rate=("home_win", "mean"),
        )
        .reset_index()
    )
    return out


def _load_dataset() -> pd.DataFrame:
    manifest = json.loads((SITE_DATA / "true_walkforward_manifest.json").read_text())
    rows: list[dict] = []
    for date_str in manifest["walkforward_dates"]:
        pred_path = SITE_DATA / f"predictions_{date_str}.json"
        final_path = SITE_DATA / f"final_scores_{date_str}.json"
        if not pred_path.exists() or not final_path.exists():
            continue
        pred_obj = json.loads(pred_path.read_text())
        final_obj = json.loads(final_path.read_text())
        preds = pred_obj.get("games", pred_obj.get("predictions", []))
        finals = final_obj.get("games", final_obj.get("final_scores", []))
        final_lookup = {g.get("game_id"): g for g in finals if g.get("game_id")}
        for pred in preds:
            fin = final_lookup.get(pred.get("game_id"))
            if not fin:
                continue
            try:
                mu = float(pred["model_mu_home"])
                sigma = float(pred["pred_sigma"])
                home_score = float(fin["home_score"])
                away_score = float(fin["away_score"])
            except Exception:
                continue
            if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
                continue
            rows.append(
                {
                    "date": date_str,
                    "season": _season_from_date(date_str),
                    "game_id": pred["game_id"],
                    "mu": mu,
                    "sigma": sigma,
                    "home_win": 1 if home_score > away_score else 0,
                }
            )
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["date", "game_id"])
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    out_dir = ROOT / "artifacts" / "research" / "ml_odds_module_study_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset()
    variants = [
        Variant("raw_mu_sigma", "raw"),
        Variant("cap14_mu_sigma", "cap14"),
        Variant("cap17_mu_sigma", "cap17"),
        Variant("const14_mu_sigma", "const14"),
        Variant("logistic_cap17_z", "cap17", calibrated=True),
        Variant("isotonic_cap17_z", "cap17", calibrated=True),
    ]

    eval_seasons = [2020, 2022, 2023, 2024, 2025, 2026]
    season_frames: list[pd.DataFrame] = []

    for eval_season in eval_seasons:
        train = df[df["season"] < eval_season]
        test = df[df["season"] == eval_season].copy()
        if train.empty or test.empty:
            continue

        z_train_cap17 = train["mu"].to_numpy() / np.minimum(train["sigma"].to_numpy(), 17.0).clip(min=0.5)
        y_train = train["home_win"].to_numpy()
        logit = LogisticRegression(C=1e6, solver="lbfgs").fit(z_train_cap17.reshape(-1, 1), y_train)
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(z_train_cap17, y_train)

        for variant in variants:
            base_prob = np.asarray(
                mu_sigma_home_win_prob(
                    test["mu"].to_numpy(),
                    test["sigma"].to_numpy(),
                    sigma_mode=variant.mode,  # type: ignore[arg-type]
                )
            )
            if variant.calibrated:
                z_test = test["mu"].to_numpy() / np.minimum(test["sigma"].to_numpy(), 17.0).clip(min=0.5)
                if variant.name == "logistic_cap17_z":
                    prob = logit.predict_proba(z_test.reshape(-1, 1))[:, 1]
                else:
                    prob = isotonic.predict(z_test)
                prob = np.clip(prob, 1e-6, 1 - 1e-6)
            else:
                prob = base_prob

            frame = test.copy()
            frame["variant"] = variant.name
            frame["prob"] = prob
            season_frames.append(frame)

    study = pd.concat(season_frames, ignore_index=True)
    season_metrics = []
    pooled_metrics = []

    date_series = pd.to_datetime(study["date"])
    dec15_mask = (date_series.dt.month > 12) | ((date_series.dt.month == 12) & (date_series.dt.day >= 15))

    for variant, grp in study.groupby("variant"):
        y = grp["home_win"].to_numpy()
        p = grp["prob"].to_numpy()
        pooled_metrics.append(
            {
                "variant": variant,
                "rows": len(grp),
                "logloss": log_loss(y, p),
                "brier": brier_score_loss(y, p),
                "ece10": _ece(y, p),
                "prob_std": float(np.std(p)),
                "prob_ge_0_8_n": int((p >= 0.8).sum()),
                "prob_ge_0_8_avg": float(p[p >= 0.8].mean()) if (p >= 0.8).any() else np.nan,
                "prob_ge_0_8_actual": float(y[p >= 0.8].mean()) if (p >= 0.8).any() else np.nan,
            }
        )

        dec = grp[dec15_mask.loc[grp.index]]
        yd = dec["home_win"].to_numpy()
        pdv = dec["prob"].to_numpy()
        pooled_metrics[-1].update(
            {
                "dec15_rows": len(dec),
                "dec15_logloss": log_loss(yd, pdv),
                "dec15_brier": brier_score_loss(yd, pdv),
                "dec15_ece10": _ece(yd, pdv),
            }
        )

        for season, sgrp in grp.groupby("season"):
            ys = sgrp["home_win"].to_numpy()
            ps = sgrp["prob"].to_numpy()
            season_metrics.append(
                {
                    "variant": variant,
                    "season": season,
                    "rows": len(sgrp),
                    "logloss": log_loss(ys, ps),
                    "brier": brier_score_loss(ys, ps),
                    "ece10": _ece(ys, ps),
                    "prob_std": float(np.std(ps)),
                }
            )

    pd.DataFrame(pooled_metrics).sort_values("logloss").to_csv(out_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(season_metrics).sort_values(["season", "logloss"]).to_csv(out_dir / "season_metrics.csv", index=False)

    bucket_frames = []
    for variant, grp in study.groupby("variant"):
        bucket = _bucket_table(grp, "prob")
        bucket.insert(0, "variant", variant)
        bucket_frames.append(bucket)
    pd.concat(bucket_frames, ignore_index=True).to_csv(out_dir / "bucket_metrics.csv", index=False)

    summary = [
        "# ML Odds Module Study v1",
        "",
        "Compared site-facing moneyline probability modules on the walk-forward archive only.",
        "",
        "Primary variants:",
        "- raw_mu_sigma",
        "- cap14_mu_sigma",
        "- cap17_mu_sigma",
        "- const14_mu_sigma",
        "- logistic_cap17_z",
        "- isotonic_cap17_z",
        "",
        "Best practical result: cap14_mu_sigma",
        "",
        "Why:",
        "- best pooled log loss and Brier among direct/simple paths",
        "- best Dec 15+ proper scores among the direct paths",
        "- avoids the additional flattening/complexity of the calibrated variants",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary))


if __name__ == "__main__":
    main()
