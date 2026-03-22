#!/usr/bin/env python3
"""Train production artifacts for the configured mean-model variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.dataset import load_multi_season_features
from src.efficiency_blend import blend_enabled
from src.features import build_features, get_feature_matrix, get_targets
from src.mean_model_variants import (
    LEGACY_HOME_SLOT,
    TEAM_AB_ELITE_TAIL_ROUND64_V1,
    normalize_mean_model_variant,
    prepare_team_ab_training_frame,
    prepare_team_ab_training_matrices,
    variant_spec,
)
from src.model_hparams import load_best_hparams, production_mu_hparams
from src.prediction_sources import prediction_source_spec
from src.trainer import (
    fit_scaler,
    save_checkpoint,
    save_tree_regressor,
    train_classifier,
    train_lightgbm_regressor,
    train_regressor,
)

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
SEASONS = [season for season in range(2015, 2026) if season not in config.EXCLUDE_SEASONS]
VAL_FRAC = 0.15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mean-model-variant",
        default=config.MEAN_MODEL_VARIANT,
        choices=[LEGACY_HOME_SLOT, TEAM_AB_ELITE_TAIL_ROUND64_V1],
        help="Mu-model variant to train.",
    )
    parser.add_argument(
        "--prediction-source",
        default=config.PUBLIC_DEFAULT_PREDICTION_SOURCE,
        choices=["he", "torvik", "kenpom"],
        help="Source-specific Team A/B checkpoint to train.",
    )
    return parser.parse_args()


def _production_regressor_hparams(base: dict, efficiency_source: str) -> dict:
    """Use a safer sigma fit when training on the gold-backed feature set."""
    hp = {**base, "epochs": 150}
    if efficiency_source == "gold":
        hp["lr"] = min(float(hp.get("lr", 1e-3)), 1e-3)
        hp["batch_size"] = min(int(hp.get("batch_size", 1024)), 1024)
    return hp


def _temporal_train_val_split(
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    y_spread: np.ndarray,
    y_win: np.ndarray,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_val = max(1, int(len(X_raw) * val_frac))
    n_val = min(n_val, len(X_raw) - 1)
    split_idx = len(X_raw) - n_val
    return (
        X_raw[:split_idx],
        X_raw[split_idx:],
        X_scaled[:split_idx],
        X_scaled[split_idx:],
        y_spread[:split_idx],
        y_spread[split_idx:],
        y_win[:split_idx],
        y_win[split_idx:],
    )


def _compute_impute_means(X_raw: np.ndarray) -> np.ndarray:
    means = np.nanmean(X_raw, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    return means.astype(np.float32)


def _impute_with_means(X_raw: np.ndarray, means: np.ndarray) -> np.ndarray:
    out = X_raw.copy()
    nan_mask = np.isnan(out)
    if nan_mask.any():
        for j in range(out.shape[1]):
            out[nan_mask[:, j], j] = means[j]
    return out


def _feature_file_path(season: int, efficiency_source: str) -> Path:
    suffix = "_no_garbage"
    if efficiency_source != "gold":
        suffix += f"_{efficiency_source}"
    suffix += f"_{ADJ_SUFFIX}"
    return config.FEATURES_DIR / f"season_{season}{suffix}_features.parquet"


def _ensure_feature_files(seasons: list[int], *, efficiency_source: str, gold_table_name: str | None = None) -> None:
    for season in seasons:
        path = _feature_file_path(season, efficiency_source)
        if path.exists():
            continue
        print(f"Building missing feature file for season {season}: {path.name}")
        df = build_features(
            season,
            no_garbage=True,
            extra_features=config.EXTRA_FEATURES,
            adjust_ff=config.ADJUST_FF,
            adjust_alpha=config.ADJUST_ALPHA,
            adjust_prior_weight=config.ADJUST_PRIOR,
            efficiency_source=efficiency_source,
            gold_table_name=gold_table_name if efficiency_source == "gold" else None,
        )
        if df.empty:
            raise FileNotFoundError(
                f"Could not build training features for season {season} and source {efficiency_source}."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def _train_legacy(best_hp: dict[str, dict]) -> None:
    reg_hp = best_hp["regressor"]
    cls_hp = best_hp["classifier"]
    mu_hp = production_mu_hparams()

    print("=== Production Training: legacy_home_slot ===")
    print(f"Seasons: {SEASONS}")
    print(f"Efficiency source: {config.EFFICIENCY_SOURCE}")
    print(f"Adj suffix: {ADJ_SUFFIX}")
    print(f"Features: {len(config.FEATURE_ORDER)}")
    print(f"Val frac: {VAL_FRAC} (best-loss checkpointing)")
    print(f"Regressor HP: {reg_hp}")
    print(f"Classifier HP: {cls_hp}")
    print(f"Mu HP: {mu_hp}")

    print("\nLoading features...")
    df = load_multi_season_features(
        SEASONS,
        no_garbage=True,
        adj_suffix=ADJ_SUFFIX,
        efficiency_source=config.EFFICIENCY_SOURCE,
    )
    df = df.dropna(subset=["homeScore", "awayScore"])
    df = df[(df["homeScore"] != 0) | (df["awayScore"] != 0)]
    df["startDate"] = np.array(df["startDate"], dtype="datetime64[ns]")
    df = df.sort_values(["startDate", "gameId"]).reset_index(drop=True)
    print(f"  Training samples: {len(df)}")

    X = get_feature_matrix(df).values.astype(np.float32)
    targets = get_targets(df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    nan_count = np.isnan(X).sum()
    print(f"  NaN values: {nan_count}")
    impute_means = _compute_impute_means(X)
    X = _impute_with_means(X, impute_means)

    print("\nFitting StandardScaler...")
    scaler = fit_scaler(X)
    X_scaled = scaler.transform(X)
    (
        X_train_raw,
        X_val_raw,
        X_train_scaled,
        X_val_scaled,
        y_spread_train,
        y_spread_val,
        y_win_train,
        y_win_val,
    ) = _temporal_train_val_split(X, X_scaled, y_spread, y_win, VAL_FRAC)

    print("\nTraining MLPRegressor (Gaussian NLL)...")
    reg_hp_full = _production_regressor_hparams(reg_hp, config.EFFICIENCY_SOURCE)
    regressor = train_regressor(
        X_scaled,
        y_spread,
        hparams=reg_hp_full,
        val_frac=VAL_FRAC,
        temporal_val_split=True,
    )
    save_checkpoint(regressor, "regressor", hparams=reg_hp_full, feature_order=config.FEATURE_ORDER)

    print("\nTraining LightGBMRegressor (mu)...")
    tree_regressor = train_lightgbm_regressor(
        X_train_raw,
        y_spread_train,
        hparams=mu_hp,
        X_val=X_val_raw,
        y_val=y_spread_val,
    )
    mu_hp_saved = {
        **mu_hp,
        **(
            {"best_iteration": int(tree_regressor.best_iteration_)}
            if getattr(tree_regressor, "best_iteration_", None)
            else {}
        ),
    }
    tree_path = save_tree_regressor(
        tree_regressor,
        feature_order=config.FEATURE_ORDER,
        hparams=mu_hp_saved,
        impute_means=impute_means,
    )

    torvik_tree_path = None
    if blend_enabled():
        print("\nTraining Torvik LightGBMRegressor (mu blend side)...")
        torvik_df = load_multi_season_features(
            SEASONS,
            no_garbage=True,
            adj_suffix=ADJ_SUFFIX,
            efficiency_source="torvik",
        )
        torvik_df = torvik_df.dropna(subset=["homeScore", "awayScore"])
        torvik_df = torvik_df[(torvik_df["homeScore"] != 0) | (torvik_df["awayScore"] != 0)]
        torvik_df["startDate"] = np.array(torvik_df["startDate"], dtype="datetime64[ns]")
        torvik_df = torvik_df.sort_values(["startDate", "gameId"]).reset_index(drop=True)
        X_t_raw = get_feature_matrix(torvik_df).values.astype(np.float32)
        impute_means_t = _compute_impute_means(X_t_raw)
        X_t = _impute_with_means(X_t_raw, impute_means_t)
        y_t = get_targets(torvik_df)["spread_home"].values.astype(np.float32)
        y_t_win = get_targets(torvik_df)["home_win"].values.astype(np.float32)
        X_t_scaled = scaler.transform(X_t)
        (
            X_t_train_raw,
            X_t_val_raw,
            _x_t_train_scaled,
            _x_t_val_scaled,
            y_t_train,
            y_t_val,
            _yw_t_train,
            _yw_t_val,
        ) = _temporal_train_val_split(X_t, X_t_scaled, y_t, y_t_win, VAL_FRAC)
        torvik_tree = train_lightgbm_regressor(
            X_t_train_raw,
            y_t_train,
            hparams=mu_hp,
            X_val=X_t_val_raw,
            y_val=y_t_val,
        )
        torvik_mu_hp_saved = {
            **mu_hp,
            **(
                {"best_iteration": int(torvik_tree.best_iteration_)}
                if getattr(torvik_tree, "best_iteration_", None)
                else {}
            ),
        }
        torvik_tree_path = save_tree_regressor(
            torvik_tree,
            path=config.TORVIK_TREE_REGRESSOR_PATH,
            feature_order=config.FEATURE_ORDER,
            hparams=torvik_mu_hp_saved,
            impute_means=impute_means_t,
        )

    print("\nTraining MLPClassifier (BCE)...")
    cls_hp_full = {**cls_hp, "epochs": 150}
    classifier = train_classifier(
        X_scaled,
        y_win,
        hparams=cls_hp_full,
        val_frac=VAL_FRAC,
        temporal_val_split=True,
    )
    save_checkpoint(classifier, "classifier", hparams=cls_hp_full, feature_order=config.FEATURE_ORDER)

    print("\n=== Production training complete ===")
    print(f"  Tree regressor: {tree_path}")
    if torvik_tree_path is not None:
        print(f"  Torvik tree regressor: {torvik_tree_path}")
    print(f"  Regressor: {config.CHECKPOINTS_DIR / 'regressor.pt'}")
    print(f"  Classifier: {config.CHECKPOINTS_DIR / 'classifier.pt'}")
    print(f"  Scaler: {config.ARTIFACTS_DIR / 'scaler.pkl'}")


def _train_team_ab_variant(prediction_source: str) -> None:
    spec = variant_spec(TEAM_AB_ELITE_TAIL_ROUND64_V1)
    source_spec = prediction_source_spec(prediction_source)
    mu_hp = production_mu_hparams()

    print(f"=== Production Training: team_ab_elite_tail_round64_v1 [{source_spec.name}] ===")
    print(f"Seasons: {SEASONS}")
    print(f"Prediction source: {source_spec.name}")
    print(f"Training efficiency source: {source_spec.efficiency_source}")
    print(f"Adj suffix: {ADJ_SUFFIX}")
    print(f"Features: {len(spec.feature_order or [])}")
    print(f"Val frac: {VAL_FRAC}")
    print(f"Mu HP: {mu_hp}")
    print("Note: this variant trains only the live mu tree path. Legacy sigma/classifier artifacts remain unchanged.")

    print("\nLoading and enriching Team A/B training frame...")
    _ensure_feature_files(
        SEASONS,
        efficiency_source=source_spec.efficiency_source,
        gold_table_name=source_spec.gold_table_name,
    )
    full_df = prepare_team_ab_training_frame(
        SEASONS,
        no_garbage=True,
        adj_suffix=ADJ_SUFFIX,
        efficiency_source=source_spec.efficiency_source,
    )
    source_df, feature_df, y = prepare_team_ab_training_matrices(full_df)
    print(f"  Base rows: {len(full_df)}")
    print(f"  Pair-augmented rows: {len(source_df)}")

    X_raw = feature_df.to_numpy(dtype=np.float32)
    impute_means = _compute_impute_means(X_raw)
    X = _impute_with_means(X_raw, impute_means)

    source_df["startDate"] = pd.to_datetime(source_df["startDate"], errors="coerce", utc=True)
    source_df = source_df.reset_index(drop=True)
    split_idx = len(source_df) - min(max(1, int(len(source_df) * VAL_FRAC)), len(source_df) - 1)
    X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print("\nTraining LightGBMRegressor (mu Team A/B)...")
    tree_regressor = train_lightgbm_regressor(
        X_train_raw,
        y_train,
        hparams=mu_hp,
        X_val=X_val_raw,
        y_val=y_val,
    )
    mu_hp_saved = {
        **mu_hp,
        **(
            {"best_iteration": int(tree_regressor.best_iteration_)}
            if getattr(tree_regressor, "best_iteration_", None)
            else {}
        ),
    }
    tree_path = save_tree_regressor(
        tree_regressor,
        path=source_spec.checkpoint_path,
        feature_order=spec.feature_order,
        hparams=mu_hp_saved,
        impute_means=impute_means,
        metadata={
            "prediction_source": source_spec.name,
            "training_efficiency_source": source_spec.efficiency_source,
            "mean_model_variant": spec.name,
        },
    )

    print("\n=== Variant training complete ===")
    print(f"  Tree regressor: {tree_path}")
    if blend_enabled():
        print("  Secondary Torvik mu blend is not used by this Team A/B production candidate path.")


def main() -> None:
    args = _parse_args()
    variant = normalize_mean_model_variant(args.mean_model_variant)
    best_hp = load_best_hparams()
    print(json.dumps({"mean_model_variant": variant, "prediction_source": args.prediction_source}, indent=2))
    if variant == LEGACY_HOME_SLOT:
        _train_legacy(best_hp)
        return
    _train_team_ab_variant(args.prediction_source)


if __name__ == "__main__":
    main()
