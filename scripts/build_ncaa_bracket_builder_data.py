#!/usr/bin/env python3
"""Build NCAA bracket-builder data from a canonical tournament field input.

Default flow:
  1. Read `site/public/data/ncaa_field_input_<season>.json`
  2. Validate the 68-team bracket field and First Four structure
  3. Enrich the field with rankings metadata used by the current frontend
  4. Precompute neutral-site matchup predictions with the promoted model

Dev-only fallback:
  Pass `--use-rankings-fallback` to derive a temporary field from the current
  rankings JSON. This is never used implicitly when a canonical field file is
  missing; it must be requested explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch

# Avoid pathological CPU thread contention during batch inference.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.build_rankings_json import _load_latest_ratings
from scripts.rebuild_tourney_jsons import _build_synthetic_rows, _predict_pairwise_probability
from src import config, s3_reader
from src.features import load_lines, load_research_lines
from src.infer import (
    _fill_nan_with_scaler_means,
    american_to_breakeven,
    load_regressor,
    normal_cdf,
    prob_to_american,
)
from src.line_selection import select_preferred_lines
from src.tournament_adjustments import market_blended_display_margin
from src.trainer import load_scaler, load_tree_regressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "site" / "public" / "data"

CURRENT_SEASON = 2026
REGIONS = ["East", "West", "South", "Midwest"]
FIELD_INPUT_TEMPLATE = "ncaa_field_input_{season}.json"
FIELD_OUTPUT_TEMPLATE = "ncaa_bracket_builder_{season}.json"
MATCHUPS_OUTPUT_TEMPLATE = "ncaa_matchup_predictions_{season}.json"
RANKINGS_TEMPLATE = "rankings_{season}.json"


def _read_json(path: Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _season_path(template: str, season: int) -> Path:
    return DATA_DIR / template.format(season=season)


def _region_slug(region: str) -> str:
    return region.lower().replace(" ", "-")


def _main_bracket_slot(region: str, seed: int) -> str:
    return f"{_region_slug(region)}-{seed}"


def _seed_region_order(seed: int) -> list[str]:
    return REGIONS if seed % 2 == 1 else list(reversed(REGIONS))


def _load_rankings_rows(season: int) -> dict[int, dict[str, Any]]:
    rankings_path = _season_path(RANKINGS_TEMPLATE, season)
    payload = _read_json(rankings_path)
    rows = payload.get("teams", [])
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_id[int(row["team_id"])] = row
    return by_id


def _build_slot_plan() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reserved = {
        ("East", 11): "ff1",
        ("West", 11): "ff2",
        ("South", 16): "ff3",
        ("Midwest", 16): "ff4",
    }
    direct_slots: list[dict[str, Any]] = []
    play_in_slots: list[dict[str, Any]] = []
    for seed in range(1, 17):
        for region in _seed_region_order(seed):
            game_id = reserved.get((region, seed))
            if game_id:
                play_in_slots.append(
                    {
                        "region": region,
                        "seed": seed,
                        "winner_to_slot": _main_bracket_slot(region, seed),
                        "game_id": game_id,
                    }
                )
            else:
                direct_slots.append(
                    {
                        "region": region,
                        "seed": seed,
                        "slot": _main_bracket_slot(region, seed),
                    }
                )
    return direct_slots, play_in_slots


def _load_rankings_fallback_input(season: int) -> dict[str, Any]:
    rankings = list(_load_rankings_rows(season).values())
    rankings.sort(key=lambda row: int(row["rank"]))
    if len(rankings) < 68:
        raise ValueError(f"Rankings fallback requires at least 68 teams, found {len(rankings)}")

    direct_slots, play_in_slots = _build_slot_plan()
    direct_teams = rankings[:60]
    play_in_teams = rankings[60:68]

    entries: list[dict[str, Any]] = []
    for slot, team in zip(direct_slots, direct_teams):
        entries.append(
            {
                "team_id": int(team["team_id"]),
                "team_name": str(team["team"]),
                "seed": int(slot["seed"]),
                "region": str(slot["region"]),
                "slot": str(slot["slot"]),
                "is_first_four": False,
                "first_four_game_id": None,
                "feeder_slot": None,
            }
        )

    eleven_play_in = play_in_teams[:4]
    sixteen_play_in = play_in_teams[4:]
    play_in_pairs = {
        "ff1": [eleven_play_in[0], eleven_play_in[3]],
        "ff2": [eleven_play_in[1], eleven_play_in[2]],
        "ff3": [sixteen_play_in[0], sixteen_play_in[3]],
        "ff4": [sixteen_play_in[1], sixteen_play_in[2]],
    }

    first_four_games: list[dict[str, Any]] = []
    for slot in play_in_slots:
        first_four_games.append(
            {
                "id": str(slot["game_id"]),
                "region": str(slot["region"]),
                "seed": int(slot["seed"]),
                "winner_to_slot": str(slot["winner_to_slot"]),
            }
        )
        for index, team in enumerate(play_in_pairs[str(slot["game_id"])], start=1):
            entries.append(
                {
                    "team_id": int(team["team_id"]),
                    "team_name": str(team["team"]),
                    "seed": int(slot["seed"]),
                    "region": str(slot["region"]),
                    "slot": f"{slot['game_id']}-team-{index}",
                    "is_first_four": True,
                    "first_four_game_id": str(slot["game_id"]),
                    "feeder_slot": str(slot["winner_to_slot"]),
                }
            )

    return {
        "season": season,
        "source": "rankings_fallback",
        "note": (
            "Development helper only. Replace with the official Selection Sunday "
            "field input before publishing a real NCAA bracket."
        ),
        "first_four_games": sorted(first_four_games, key=lambda game: game["id"]),
        "entries": entries,
    }


def _validate_canonical_field_input(payload: dict[str, Any], season: int) -> None:
    errors: list[str] = []

    payload_season = payload.get("season")
    if payload_season != season:
        errors.append(f"Canonical field season {payload_season} does not match requested season {season}")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Canonical field input must contain an entries array")
    if len(entries) != 68:
        errors.append(f"Expected exactly 68 field entries, found {len(entries)}")

    first_four_games = payload.get("first_four_games")
    if not isinstance(first_four_games, list):
        raise ValueError("Canonical field input must contain a first_four_games array")
    if len(first_four_games) != 4:
        errors.append(f"Expected 4 First Four games, found {len(first_four_games)}")

    game_by_id: dict[str, dict[str, Any]] = {}
    winner_slot_to_game: dict[str, str] = {}
    for game in first_four_games:
        game_id = str(game.get("id") or "")
        region = str(game.get("region") or "")
        seed = game.get("seed")
        winner_to_slot = str(game.get("winner_to_slot") or "")
        if not game_id:
            errors.append("First Four game id is required")
            continue
        if game_id in game_by_id:
            errors.append(f"Duplicate First Four game id {game_id}")
        if region not in REGIONS:
            errors.append(f"First Four game {game_id} has invalid region {region}")
        if not isinstance(seed, int) or not 1 <= seed <= 16:
            errors.append(f"First Four game {game_id} has invalid seed {seed}")
        expected_slot = _main_bracket_slot(region, seed) if region in REGIONS and isinstance(seed, int) else None
        if winner_to_slot != expected_slot:
            errors.append(
                f"First Four game {game_id} winner_to_slot must be {expected_slot}, found {winner_to_slot}"
            )
        if winner_to_slot in winner_slot_to_game:
            errors.append(
                f"First Four winner slot {winner_to_slot} is assigned to both "
                f"{winner_slot_to_game[winner_to_slot]} and {game_id}"
            )
        winner_slot_to_game[winner_to_slot] = game_id
        game_by_id[game_id] = game

    team_ids: set[int] = set()
    occupied_slots: set[str] = set()
    direct_slot_owners: dict[str, int] = {}
    play_in_entries_by_game: dict[str, list[dict[str, Any]]] = {}
    main_bracket_assignments: dict[tuple[str, int], str] = {}
    region_seeds: dict[str, set[int]] = {region: set() for region in REGIONS}
    direct_count = 0
    first_four_count = 0

    for index, entry in enumerate(entries, start=1):
        prefix = f"Entry {index}"
        try:
            team_id = int(entry["team_id"])
        except Exception:
            errors.append(f"{prefix} is missing a valid team_id")
            continue
        team_name = str(entry.get("team_name") or "").strip()
        seed = entry.get("seed")
        region = str(entry.get("region") or "")
        slot = str(entry.get("slot") or "")
        is_first_four = bool(entry.get("is_first_four"))
        first_four_game_id = entry.get("first_four_game_id")
        feeder_slot = entry.get("feeder_slot")

        if not team_name:
            errors.append(f"{prefix} team_id {team_id} is missing team_name")
        if team_id in team_ids:
            errors.append(f"Duplicate team_id {team_id}")
        team_ids.add(team_id)

        if region not in REGIONS:
            errors.append(f"{prefix} team_id {team_id} has invalid region {region}")
        if not isinstance(seed, int) or not 1 <= seed <= 16:
            errors.append(f"{prefix} team_id {team_id} has invalid seed {seed}")
        if not slot:
            errors.append(f"{prefix} team_id {team_id} is missing slot")
        elif slot in occupied_slots:
            errors.append(f"Duplicate occupied slot {slot}")
        occupied_slots.add(slot)

        if not is_first_four:
            direct_count += 1
            expected_slot = _main_bracket_slot(region, seed) if region in REGIONS and isinstance(seed, int) else None
            if slot != expected_slot:
                errors.append(f"{prefix} team_id {team_id} must occupy slot {expected_slot}, found {slot}")
            if first_four_game_id not in (None, ""):
                errors.append(f"{prefix} team_id {team_id} must not include first_four_game_id")
            if feeder_slot not in (None, ""):
                errors.append(f"{prefix} team_id {team_id} must not include feeder_slot")
            if region in REGIONS and isinstance(seed, int):
                key = (region, seed)
                if key in main_bracket_assignments:
                    errors.append(
                        f"Main bracket seed {region} {seed} is assigned twice "
                        f"({main_bracket_assignments[key]} and direct team {team_id})"
                    )
                main_bracket_assignments[key] = f"direct team {team_id}"
                region_seeds[region].add(seed)
                direct_slot_owners[slot] = team_id
            continue

        first_four_count += 1
        game_id = str(first_four_game_id or "")
        feeder_slot_value = str(feeder_slot or "")
        if not game_id:
            errors.append(f"{prefix} team_id {team_id} is marked First Four but missing first_four_game_id")
            continue
        if game_id not in game_by_id:
            errors.append(f"{prefix} team_id {team_id} references unknown First Four game {game_id}")
            continue
        game = game_by_id[game_id]
        if region != game["region"]:
            errors.append(
                f"{prefix} team_id {team_id} region {region} does not match First Four game {game_id} region {game['region']}"
            )
        if seed != game["seed"]:
            errors.append(
                f"{prefix} team_id {team_id} seed {seed} does not match First Four game {game_id} seed {game['seed']}"
            )
        if feeder_slot_value != game["winner_to_slot"]:
            errors.append(
                f"{prefix} team_id {team_id} feeder_slot must be {game['winner_to_slot']}, found {feeder_slot_value}"
            )
        if region in REGIONS and isinstance(seed, int):
            key = (region, seed)
            owner = main_bracket_assignments.get(key)
            game_label = f"First Four game {game_id}"
            if owner and owner != game_label:
                errors.append(f"Main bracket seed {region} {seed} is assigned twice ({owner} and {game_label})")
            main_bracket_assignments[key] = game_label
            region_seeds[region].add(seed)
        play_in_entries_by_game.setdefault(game_id, []).append(entry)

    if direct_count != 60:
        errors.append(f"Expected 60 direct bracket teams, found {direct_count}")
    if first_four_count != 8:
        errors.append(f"Expected 8 First Four participants, found {first_four_count}")
    if len(team_ids) != 68:
        errors.append(f"Expected 68 unique team_ids, found {len(team_ids)}")

    for region in REGIONS:
        seeds = sorted(region_seeds[region])
        if seeds != list(range(1, 17)):
            errors.append(f"Region {region} must occupy bracket seeds 1-16 exactly once; found {seeds}")

    for game_id, game in game_by_id.items():
        linked = play_in_entries_by_game.get(game_id, [])
        if len(linked) != 2:
            errors.append(f"First Four game {game_id} must have exactly 2 participant entries, found {len(linked)}")
        if game["winner_to_slot"] in direct_slot_owners:
            errors.append(
                f"First Four game {game_id} winner slot {game['winner_to_slot']} conflicts with direct team "
                f"{direct_slot_owners[game['winner_to_slot']]}"
            )

    if errors:
        raise ValueError("Canonical NCAA field input is invalid:\n- " + "\n- ".join(errors))


def _enrich_entry(
    entry: dict[str, Any],
    season: int,
    rankings_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    team_id = int(entry["team_id"])
    rankings_row = rankings_by_id.get(team_id)
    if not rankings_row:
        raise ValueError(
            f"Canonical field team_id {team_id} ({entry['team_name']}) is missing from rankings_{season}.json"
        )

    return {
        "team_id": team_id,
        "team": str(rankings_row["team"]),
        "rank": int(rankings_row["rank"]),
        "conference": str(rankings_row.get("conference") or ""),
        "record": str(rankings_row.get("record") or ""),
        "conf_record": str(rankings_row.get("conf_record") or ""),
        "adj_oe": float(rankings_row["adj_oe"]),
        "adj_de": float(rankings_row["adj_de"]),
        "adj_margin": float(rankings_row["adj_margin"]),
        "adj_tempo": float(rankings_row["adj_tempo"]),
        "model_index": None
        if rankings_row.get("model_index") is None
        else float(rankings_row["model_index"]),
        "ft_pct": None
        if rankings_row.get("ft_pct") is None
        else float(rankings_row["ft_pct"]),
        "three_p_pct": None
        if rankings_row.get("three_p_pct") is None
        else float(rankings_row["three_p_pct"]),
    }


def _build_field_payload(
    canonical_input: dict[str, Any],
    season: int,
    rankings_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    regions: dict[str, list[dict[str, Any]]] = {region: [] for region in REGIONS}
    first_four_entries_by_game: dict[str, list[dict[str, Any]]] = {}

    for entry in canonical_input["entries"]:
        region = str(entry["region"])
        seed = int(entry["seed"])
        if entry["is_first_four"]:
            first_four_entries_by_game.setdefault(str(entry["first_four_game_id"]), []).append(entry)
            continue
        regions[region].append(
            {
                "seed": seed,
                "source": "team",
                **_enrich_entry(entry, season, rankings_by_id),
            }
        )

    first_four: list[dict[str, Any]] = []
    for game in sorted(canonical_input["first_four_games"], key=lambda row: row["id"]):
        game_id = str(game["id"])
        region = str(game["region"])
        seed = int(game["seed"])
        participants = sorted(first_four_entries_by_game[game_id], key=lambda row: row["slot"])
        first_four.append(
            {
                "id": game_id,
                "label": "First Four",
                "region": region,
                "seed": seed,
                "teams": [_enrich_entry(entry, season, rankings_by_id) for entry in participants],
            }
        )
        regions[region].append(
            {
                "seed": seed,
                "source": "play_in",
                "play_in_game_id": game_id,
            }
        )

    out_regions = []
    for region in REGIONS:
        entries = sorted(regions[region], key=lambda row: int(row["seed"]))
        out_regions.append({"name": region, "entries": entries})

    input_source = str(canonical_input.get("source") or "canonical_field_input")
    note_suffix = str(canonical_input.get("note") or "").strip()
    note = "Generated from canonical NCAA field input."
    if note_suffix:
        note = f"{note} {note_suffix}"

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "season": season,
        "source": input_source,
        "note": note,
        "regions": out_regions,
        "first_four": first_four,
    }


def _round_from_game_notes(note: object) -> tuple[str | None, str | None]:
    value = str(note or "").upper()
    if "FIRST FOUR" in value:
        return "first-four", "First Four"
    if "1ST ROUND" in value:
        return "round-of-64", "Round of 64"
    return None, None


def _preferred_ncaa_lines(season: int) -> pd.DataFrame:
    live_lines = select_preferred_lines(load_lines(season))
    research_lines = select_preferred_lines(load_research_lines(season))
    combined = pd.concat([live_lines, research_lines], ignore_index=True, sort=False)
    if combined.empty:
        return combined

    provider_rank = pd.Series(1, index=combined.index, dtype=int)
    provider_rank.loc[combined["provider"].fillna("").eq("Hard Rock Bet")] = 0
    provider_rank.loc[combined["provider"].fillna("").eq("consensus")] = 2
    provider_rank.loc[combined["book_spread"].isna()] = 3
    combined = combined.assign(_provider_rank=provider_rank)
    combined = combined.sort_values(
        ["gameId", "_provider_rank"],
        ascending=[True, True],
        kind="stable",
    )
    return combined.drop_duplicates("gameId", keep="first").drop(columns=["_provider_rank"])


def _load_opening_round_market_lookup(season: int) -> dict[str, dict[str, Any]]:
    games_table = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if games_table.num_rows == 0:
        return {}
    games = games_table.to_pandas()
    keep = [
        c
        for c in [
            "gameId",
            "homeTeamId",
            "awayTeamId",
            "homeTeam",
            "awayTeam",
            "startDate",
            "gameNotes",
            "tournament",
        ]
        if c in games.columns
    ]
    games = games[keep].drop_duplicates("gameId").copy()
    games = games[games["tournament"].eq("NCAA")].copy()
    if games.empty:
        return {}

    round_info = games["gameNotes"].map(_round_from_game_notes)
    games["scheduled_round_id"] = round_info.map(lambda item: item[0])
    games["scheduled_round_label"] = round_info.map(lambda item: item[1])
    games = games[games["scheduled_round_id"].isin(["first-four", "round-of-64"])].copy()
    if games.empty:
        return {}

    lines = _preferred_ncaa_lines(season)
    if lines.empty:
        return {}

    merged = games.merge(
        lines[
            [
                c
                for c in [
                    "gameId",
                    "book_spread",
                    "home_moneyline",
                    "away_moneyline",
                    "provider",
                ]
                if c in lines.columns
            ]
        ],
        on="gameId",
        how="left",
    )
    merged["book_spread"] = pd.to_numeric(merged["book_spread"], errors="coerce")
    merged = merged.dropna(subset=["book_spread", "homeTeamId", "awayTeamId"]).copy()
    if merged.empty:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in merged.iterrows():
        home_id = int(row["homeTeamId"])
        away_id = int(row["awayTeamId"])
        team1_id, team2_id = sorted([home_id, away_id])
        market_margin = -float(row["book_spread"]) if team1_id == home_id else float(row["book_spread"])
        key = f"{team1_id}::{team2_id}"
        lookup[key] = {
            "scheduled_game_id": int(row["gameId"]),
            "scheduled_round_id": row["scheduled_round_id"],
            "scheduled_round_label": row["scheduled_round_label"],
            "start_time": row.get("startDate"),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_name": row.get("homeTeam"),
            "away_team_name": row.get("awayTeam"),
            "market_mu_team1_minus_team2": market_margin,
            "market_spread_home": float(row["book_spread"]),
            "market_home_team_id": home_id,
            "market_away_team_id": away_id,
            "market_home_moneyline": None if pd.isna(row.get("home_moneyline")) else float(row["home_moneyline"]),
            "market_away_moneyline": None if pd.isna(row.get("away_moneyline")) else float(row["away_moneyline"]),
            "market_line_source": row.get("provider"),
        }
    return lookup


def _predict_pairwise_projection(
    team_a: pd.Series,
    team_b: pd.Series,
    feature_order: list[str],
    scaler,
    tree_model,
    sigma_model,
    sigma_param: str,
    month: int,
    day: int,
) -> tuple[float, float, float]:
    mu, win_prob_a = _predict_pairwise_probability(
        team_a,
        team_b,
        feature_order,
        scaler,
        tree_model,
        sigma_model,
        sigma_param,
        month,
        day,
    )

    row_ab, row_ba = _build_synthetic_rows(team_a, team_b, feature_order, scaler)
    X_ab = _fill_nan_with_scaler_means(row_ab, scaler)
    X_ba = _fill_nan_with_scaler_means(row_ba, scaler)
    mu_ab = float(tree_model.predict(X_ab.astype(np.float32))[0])
    mu_ba = float(tree_model.predict(X_ba.astype(np.float32))[0])

    X_ab_scaled = scaler.transform(X_ab)
    X_ba_scaled = scaler.transform(X_ba)
    X_ab_tensor = torch.tensor(X_ab_scaled, dtype=torch.float32)
    X_ba_tensor = torch.tensor(X_ba_scaled, dtype=torch.float32)
    with torch.no_grad():
        _, log_sigma_ab = sigma_model(X_ab_tensor)
        _, log_sigma_ba = sigma_model(X_ba_tensor)
        if sigma_param == "exp":
            sigma_ab = np.exp(log_sigma_ab.numpy())[0]
            sigma_ba = np.exp(log_sigma_ba.numpy())[0]
        else:
            sigma_ab = (torch.nn.functional.softplus(log_sigma_ab) + 1e-3).numpy()[0]
            sigma_ba = (torch.nn.functional.softplus(log_sigma_ba) + 1e-3).numpy()[0]
    sigma_var = 0.5 * (sigma_ab**2 + sigma_ba**2) + ((mu_ab + mu_ba) ** 2) / 4.0
    sigma = float(max(math.sqrt(max(sigma_var, 0.25)), 0.5))
    return float(mu), sigma, float(win_prob_a)


def _validate_field_payload(field: dict[str, Any]) -> None:
    if len(field["regions"]) != 4:
        raise ValueError(f"Expected 4 regions, found {len(field['regions'])}")
    if len(field["first_four"]) != 4:
        raise ValueError(f"Expected 4 First Four games, found {len(field['first_four'])}")

    seen_ids: set[int] = set()
    play_in_ids: set[str] = set()
    region_play_in_refs: set[str] = set()

    for region in field["regions"]:
        seeds = sorted(entry["seed"] for entry in region["entries"])
        if seeds != list(range(1, 17)):
            raise ValueError(f"Region {region['name']} must contain seeds 1-16 exactly once")
        for entry in region["entries"]:
            if entry["source"] == "team":
                team_id = int(entry["team_id"])
                if team_id in seen_ids:
                    raise ValueError(f"Duplicate team id {team_id} in field payload")
                seen_ids.add(team_id)
            else:
                region_play_in_refs.add(str(entry["play_in_game_id"]))

    for game in field["first_four"]:
        play_in_ids.add(str(game["id"]))
        if len(game["teams"]) != 2:
            raise ValueError(f"First Four game {game['id']} must contain 2 teams")
        for team in game["teams"]:
            team_id = int(team["team_id"])
            if team_id in seen_ids:
                raise ValueError(f"Duplicate play-in team id {team_id} in field payload")
            seen_ids.add(team_id)

    if play_in_ids != region_play_in_refs:
        raise ValueError("Generated field payload has mismatched First Four references")
    if len(seen_ids) != 68:
        raise ValueError(f"Expected 68 field teams, found {len(seen_ids)}")


def _build_matchup_payload(field: dict[str, Any], season: int) -> dict[str, Any]:
    print("Collecting selected NCAA teams", flush=True)
    team_rows = []
    for region in field["regions"]:
        for entry in region["entries"]:
            if entry["source"] == "team":
                team_rows.append(entry)
    for play_in in field["first_four"]:
        team_rows.extend(play_in["teams"])

    unique_rows: dict[int, dict[str, Any]] = {}
    for row in team_rows:
        unique_rows[int(row["team_id"])] = row

    selected_ids = sorted(unique_rows)
    print("Loading latest ratings", flush=True)
    ratings, _ = _load_latest_ratings(season)
    selected = pd.DataFrame.from_records(list(unique_rows.values()))
    merged = selected.merge(
        ratings,
        left_on="team_id",
        right_on="teamId",
        how="left",
        suffixes=("_public", ""),
    )
    missing = merged.loc[
        merged["adj_oe_y"].isna() if "adj_oe_y" in merged.columns else merged["adj_oe"].isna(),
        "team",
    ].tolist()
    if missing:
        raise ValueError(f"Missing ratings rows for selected NCAA teams: {missing[:10]}")

    if "adj_oe_y" in merged.columns:
        merged["adj_oe"] = merged["adj_oe_y"]
        merged["adj_de"] = merged["adj_de_y"]
        merged["adj_tempo"] = merged["adj_tempo_y"]
        merged = merged.drop(columns=[col for col in merged.columns if col.endswith("_x") or col.endswith("_y")])

    print("Loading model artifacts", flush=True)
    scaler = load_scaler()
    tree_model, feature_order, _ = load_tree_regressor()
    sigma_model, _, sigma_feature_order, sigma_param = load_regressor()
    if sigma_feature_order != feature_order:
        raise ValueError("Tree and sigma feature orders do not match")

    team_lookup = {int(row["team_id"]): row for _, row in merged.iterrows()}
    opening_round_lines = _load_opening_round_market_lookup(season)

    predictions: dict[str, Any] = {}
    total_pairs = len(selected_ids) * (len(selected_ids) - 1) // 2
    for pair_index, (team_a_id, team_b_id) in enumerate(combinations(selected_ids, 2), start=1):
        team_a = team_lookup[team_a_id]
        team_b = team_lookup[team_b_id]
        mu, sigma, win_prob_a = _predict_pairwise_projection(
            team_a,
            team_b,
            feature_order,
            scaler,
            tree_model,
            sigma_model,
            sigma_param,
            3,
            15,
        )
        canonical_key = f"{team_a_id}::{team_b_id}"
        line_info = opening_round_lines.get(canonical_key)
        display_mu = float(mu)
        if config.NCAA_TOURNAMENT_MARKET_BLEND_ENABLED and line_info is not None:
            display_mu = market_blended_display_margin(
                float(mu),
                float(line_info["market_mu_team1_minus_team2"]),
            )
        model_mu_home = None
        display_model_mu_home = None
        edge_home_points = None
        display_edge_home_points = None
        pick_side = None
        pick_cover_prob = None
        pick_prob_edge = None
        pick_fair_odds = None
        if line_info is not None:
            home_team_id = int(line_info["home_team_id"])
            book_spread = float(line_info["market_spread_home"])
            model_mu_home = float(mu) if home_team_id == team_a_id else -float(mu)
            display_model_mu_home = float(display_mu) if home_team_id == team_a_id else -float(display_mu)
            edge_home_points = model_mu_home + book_spread
            display_edge_home_points = display_model_mu_home + book_spread
            sigma_safe = max(float(sigma), 0.5)
            edge_z = edge_home_points / sigma_safe
            home_cover_prob = float(normal_cdf(edge_z))
            away_cover_prob = 1.0 - home_cover_prob
            pick_side = "HOME" if edge_home_points >= 0 else "AWAY"
            pick_cover_prob = home_cover_prob if pick_side == "HOME" else away_cover_prob
            pick_breakeven = float(american_to_breakeven(np.array([-110.0]))[0])
            pick_prob_edge = pick_cover_prob - pick_breakeven
            pick_fair_odds = float(prob_to_american(np.array([pick_cover_prob]))[0])
        predictions[f"{team_a_id}::{team_b_id}"] = {
            "team1_id": int(team_a_id),
            "team1_name": str(team_a["team"]),
            "team2_id": int(team_b_id),
            "team2_name": str(team_b["team"]),
            "mu_team1_minus_team2": float(mu),
            "display_mu_team1_minus_team2": display_mu,
            "win_prob_team1": float(win_prob_a),
            "pred_sigma": float(sigma),
            "scheduled_game_id": None if line_info is None else line_info["scheduled_game_id"],
            "scheduled_round_id": None if line_info is None else line_info["scheduled_round_id"],
            "scheduled_round_label": None if line_info is None else line_info["scheduled_round_label"],
            "start_time": None if line_info is None else line_info["start_time"],
            "home_team_id": None if line_info is None else line_info["home_team_id"],
            "away_team_id": None if line_info is None else line_info["away_team_id"],
            "home_team_name": None if line_info is None else line_info["home_team_name"],
            "away_team_name": None if line_info is None else line_info["away_team_name"],
            "model_mu_home": model_mu_home,
            "display_model_mu_home": display_model_mu_home,
            "edge_home_points": edge_home_points,
            "display_edge_home_points": display_edge_home_points,
            "pick_side": pick_side,
            "pick_cover_prob": pick_cover_prob,
            "pick_prob_edge": pick_prob_edge,
            "pick_fair_odds": pick_fair_odds,
            "market_mu_team1_minus_team2": None if line_info is None else line_info["market_mu_team1_minus_team2"],
            "market_spread_home": None if line_info is None else line_info["market_spread_home"],
            "market_home_team_id": None if line_info is None else line_info["market_home_team_id"],
            "market_away_team_id": None if line_info is None else line_info["market_away_team_id"],
            "market_home_moneyline": None if line_info is None else line_info["market_home_moneyline"],
            "market_away_moneyline": None if line_info is None else line_info["market_away_moneyline"],
            "market_line_source": None if line_info is None else line_info["market_line_source"],
        }
        if pair_index % 250 == 0 or pair_index == total_pairs:
            print(f"Computed {pair_index}/{total_pairs} matchup predictions", flush=True)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "season": season,
        "neutral_site": True,
        "source": "production_matchup_model",
        "note": (
            "Neutral-site pairwise predictions generated from the current promoted "
            "Hoops Edge production mean model plus sigma model and site ML correction. "
            "Opening-round NCAA games also carry optional display-spread and market-line metadata."
        ),
        "predictions": predictions,
    }


def _validate_matchup_payload(field: dict[str, Any], payload: dict[str, Any]) -> None:
    team_ids = set()
    for region in field["regions"]:
        for entry in region["entries"]:
            if entry["source"] == "team":
                team_ids.add(int(entry["team_id"]))
    for game in field["first_four"]:
        for team in game["teams"]:
            team_ids.add(int(team["team_id"]))

    expected = len(team_ids) * (len(team_ids) - 1) // 2
    if len(payload["predictions"]) != expected:
        raise ValueError(f"Expected {expected} matchup predictions, found {len(payload['predictions'])}")

    for key, entry in payload["predictions"].items():
        team_a_id, team_b_id = (int(part) for part in key.split("::"))
        if team_a_id >= team_b_id:
            raise ValueError(f"Non-canonical matchup key {key}")
        if entry["team1_id"] != team_a_id or entry["team2_id"] != team_b_id:
            raise ValueError(f"Mismatch between matchup key {key} and cached team ids")
        if team_a_id not in team_ids or team_b_id not in team_ids:
            raise ValueError(f"Matchup key {key} references team outside field")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument(
        "--field-input",
        type=Path,
        default=None,
        help="Canonical field input JSON path. Defaults to site/public/data/ncaa_field_input_<season>.json.",
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=None,
        help="Generated bracket-builder JSON output path.",
    )
    parser.add_argument(
        "--matchups-output",
        type=Path,
        default=None,
        help="Generated matchup cache JSON output path.",
    )
    parser.add_argument(
        "--use-rankings-fallback",
        action="store_true",
        help="Dev helper only. Build a temporary field from rankings instead of a canonical field input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    season = int(args.season)
    field_input_path = args.field_input or _season_path(FIELD_INPUT_TEMPLATE, season)
    field_output_path = args.field_output or _season_path(FIELD_OUTPUT_TEMPLATE, season)
    matchups_output_path = args.matchups_output or _season_path(MATCHUPS_OUTPUT_TEMPLATE, season)

    if args.use_rankings_fallback:
        print("Building canonical field input from rankings fallback", flush=True)
        canonical_input = _load_rankings_fallback_input(season)
    else:
        if not field_input_path.exists():
            raise FileNotFoundError(
                f"Canonical field input not found: {field_input_path}. "
                "Provide the official field file or rerun with --use-rankings-fallback."
            )
        print(f"Loading canonical field input from {field_input_path.name}", flush=True)
        canonical_input = _read_json(field_input_path)

    _validate_canonical_field_input(canonical_input, season)
    rankings_by_id = _load_rankings_rows(season)

    print("Building field payload", flush=True)
    field = _build_field_payload(canonical_input, season, rankings_by_id)
    _validate_field_payload(field)

    print("Building matchup payload", flush=True)
    matchup_payload = _build_matchup_payload(field, season)
    _validate_matchup_payload(field, matchup_payload)

    _write_json(field_output_path, field)
    _write_json(matchups_output_path, matchup_payload)
    print(f"Wrote {field_output_path.name} and {matchups_output_path.name}", flush=True)


if __name__ == "__main__":
    main()
