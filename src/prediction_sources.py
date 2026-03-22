"""Source registry for public prediction families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class PredictionSourceSpec:
    name: str
    label: str
    efficiency_source: str
    checkpoint_path: Path
    gold_table_name: str | None = None


_SPECS = {
    "he": PredictionSourceSpec(
        name="he",
        label="HE",
        efficiency_source="gold",
        checkpoint_path=config.TREE_REGRESSOR_TEAM_AB_ELITE_TAIL_ROUND64_V1_HE_PATH,
        gold_table_name=config.PRODUCTION_GOLD_RATINGS_TABLE,
    ),
    "torvik": PredictionSourceSpec(
        name="torvik",
        label="Torvik",
        efficiency_source="torvik",
        checkpoint_path=config.TREE_REGRESSOR_TEAM_AB_ELITE_TAIL_ROUND64_V1_TORVIK_PATH,
    ),
    "kenpom": PredictionSourceSpec(
        name="kenpom",
        label="KenPom",
        efficiency_source="kenpom",
        checkpoint_path=config.TREE_REGRESSOR_TEAM_AB_ELITE_TAIL_ROUND64_V1_KENPOM_PATH,
    ),
}


def normalize_prediction_source(value: str | None) -> str:
    source = (value or config.PUBLIC_DEFAULT_PREDICTION_SOURCE).strip().lower()
    if source not in _SPECS:
        raise ValueError(
            f"Unsupported prediction source {source!r}. Expected one of {sorted(_SPECS)}."
        )
    return source


def prediction_source_spec(value: str | None) -> PredictionSourceSpec:
    return _SPECS[normalize_prediction_source(value)]


def public_prediction_sources() -> list[str]:
    return list(_SPECS)
