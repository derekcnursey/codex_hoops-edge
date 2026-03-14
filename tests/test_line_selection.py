from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

from src.line_selection import select_preferred_lines


def _load_rebuild_module():
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "rebuild_fct_lines_repaired_stage.py"
    )
    spec = importlib.util.spec_from_file_location("rebuild_fct_lines_repaired_stage", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rebuild_lines = _load_rebuild_module()


def test_select_preferred_lines_prefers_consensus_when_multiple_books_have_spreads():
    lines = pd.DataFrame(
        [
            {"gameId": 1, "provider": "Draft Kings", "spread": -4.5, "overUnder": 150.5},
            {"gameId": 1, "provider": "ESPN BET", "spread": -3.5, "overUnder": 149.5},
            {"gameId": 1, "provider": "Bovada", "spread": -4.0, "overUnder": 151.5},
        ]
    )

    selected = select_preferred_lines(lines)

    assert selected.loc[0, "provider"] == "consensus"
    assert selected.loc[0, "book_spread"] == -4.0
    assert selected.loc[0, "book_total"] == 150.5


def test_select_preferred_lines_prefers_other_books_over_bovada():
    lines = pd.DataFrame(
        [
            {"gameId": 2, "provider": "Bovada", "spread": None, "overUnder": 145.5},
            {"gameId": 2, "provider": "Circa", "spread": -2.0},
        ]
    )

    selected = select_preferred_lines(lines)

    assert selected.loc[0, "provider"] == "Circa"
    assert selected.loc[0, "book_spread"] == -2.0


def test_select_preferred_lines_prefers_hrb_over_consensus_and_more_complete_books():
    lines = pd.DataFrame(
        [
            {
                "gameId": 3,
                "provider": "Hard Rock Bet",
                "spread": -4.5,
            },
            {
                "gameId": 3,
                "provider": "Draft Kings",
                "spread": -4.0,
                "overUnder": 149.5,
                "homeMoneyline": -180,
                "awayMoneyline": 150,
            },
            {
                "gameId": 3,
                "provider": "ESPN BET",
                "spread": -4.5,
                "overUnder": 150.5,
            },
        ]
    )

    selected = select_preferred_lines(lines)

    assert selected.loc[0, "provider"] == "Hard Rock Bet"
    assert selected.loc[0, "book_spread"] == -4.5


def test_best_provider_rows_prefers_more_complete_snapshot_over_latest():
    rows = pd.DataFrame(
        [
            {
                "gameId": 10,
                "provider": "Draft Kings",
                "spread": -4.5,
                "overUnder": 145.5,
                "homeMoneyline": -190,
                "awayMoneyline": 160,
                "source_asof": pd.Timestamp("2026-03-10"),
                "source_key": "a",
            },
            {
                "gameId": 10,
                "provider": "Draft Kings",
                "spread": None,
                "overUnder": 146.5,
                "homeMoneyline": None,
                "awayMoneyline": None,
                "source_asof": pd.Timestamp("2026-03-11"),
                "source_key": "z",
            },
        ]
    )

    selected = rebuild_lines._best_provider_rows(rows)

    assert len(selected) == 1
    assert selected.loc[0, "spread"] == -4.5
    assert selected.loc[0, "homeMoneyline"] == -190
