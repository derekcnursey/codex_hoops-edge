"""Research helpers for swapped-slot feature augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EXPLICIT_SWAP_PAIRS: list[tuple[str, str]] = [
    ("home_opp_ft_rate", "away_def_ft_rate"),
    ("home_team_efg_home_split", "away_team_efg_away_split"),
]
NEGATED_FEATURES = ["rest_advantage"]
GLOBAL_FEATURES = ["neutral_site"]
UNSWAPPABLE_FEATURES = ["home_team_hca"]


@dataclass(frozen=True)
class FeatureSwapAudit:
    generic_pairs: list[tuple[str, str]]
    explicit_pairs: list[tuple[str, str]]
    negated: list[str]
    globals: list[str]
    unswappable: list[str]
    unclassified: list[str]


def audit_feature_order(feature_order: list[str]) -> FeatureSwapAudit:
    explicit_left = {left for left, _ in EXPLICIT_SWAP_PAIRS}
    explicit_right = {right for _, right in EXPLICIT_SWAP_PAIRS}
    used: set[str] = set(explicit_left | explicit_right)
    generic_pairs: list[tuple[str, str]] = []
    for col in feature_order:
        if col in used:
            continue
        if col.startswith("home_"):
            other = "away_" + col[len("home_") :]
            if other in feature_order:
                generic_pairs.append((col, other))
                used.add(col)
                used.add(other)

    tagged = used | set(NEGATED_FEATURES) | set(GLOBAL_FEATURES) | set(UNSWAPPABLE_FEATURES)
    unclassified = [col for col in feature_order if col not in tagged]
    return FeatureSwapAudit(
        generic_pairs=generic_pairs,
        explicit_pairs=[pair for pair in EXPLICIT_SWAP_PAIRS if pair[0] in feature_order and pair[1] in feature_order],
        negated=[col for col in NEGATED_FEATURES if col in feature_order],
        globals=[col for col in GLOBAL_FEATURES if col in feature_order],
        unswappable=[col for col in UNSWAPPABLE_FEATURES if col in feature_order],
        unclassified=unclassified,
    )


def swap_feature_frame(
    feature_df: pd.DataFrame,
    feature_order: list[str],
    *,
    neutral_only: bool = True,
) -> pd.DataFrame:
    """Swap home/away slots for a feature matrix.

    This helper is only semantically safe for neutral-site rows under the
    current 53-feature contract because ``home_team_hca`` has no mirrored
    away-team counterpart.
    """
    if neutral_only and "neutral_site" in feature_df.columns:
        neutral_mask = feature_df["neutral_site"].fillna(0).astype(float).eq(1.0)
        if not bool(neutral_mask.all()):
            raise ValueError("swap_feature_frame(neutral_only=True) requires all rows to be neutral-site")

    swapped = feature_df[feature_order].copy()
    audit = audit_feature_order(feature_order)

    for left, right in audit.explicit_pairs + audit.generic_pairs:
        tmp = swapped[left].copy()
        swapped[left] = swapped[right]
        swapped[right] = tmp

    for col in audit.negated:
        swapped[col] = -swapped[col]

    if "neutral_site" in swapped.columns:
        swapped["neutral_site"] = 1.0
    if "home_team_hca" in swapped.columns:
        swapped["home_team_hca"] = 0.0

    return swapped


def augment_swapped_slot_training(
    feature_df: pd.DataFrame,
    spread_home: np.ndarray,
    *,
    home_win: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Append swapped-slot rows for a subset of training rows."""
    if len(feature_df) != len(spread_home):
        raise ValueError("feature_df and spread_home must have the same length")
    if home_win is not None and len(home_win) != len(feature_df):
        raise ValueError("home_win and feature_df must have the same length")

    feature_order = list(feature_df.columns)
    if eligible_mask is None:
        eligible_mask = np.ones(len(feature_df), dtype=bool)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)

    base_features = feature_df.reset_index(drop=True).copy()
    base_spread = np.asarray(spread_home, dtype=np.float32).copy()
    base_home_win = None if home_win is None else np.asarray(home_win, dtype=np.float32).copy()

    if not eligible_mask.any():
        return base_features, base_spread, base_home_win

    swap_features = swap_feature_frame(base_features.loc[eligible_mask], feature_order, neutral_only=True)
    swap_spread = -base_spread[eligible_mask]
    if base_home_win is None:
        swap_home_win = None
    else:
        swap_home_win = 1.0 - base_home_win[eligible_mask]

    out_features = pd.concat([base_features, swap_features], ignore_index=True)
    out_spread = np.concatenate([base_spread, swap_spread]).astype(np.float32)
    if base_home_win is None:
        out_home_win = None
    else:
        out_home_win = np.concatenate([base_home_win, swap_home_win]).astype(np.float32)
    return out_features, out_spread, out_home_win
