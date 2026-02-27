# Production Prediction Pipeline: Current State & Adjusted Four-Factors Plan

## 1. Current Prediction Flow

### Entry Point

**Command**: `poetry run python -m src.cli predict-today --season 2026 [--date YYYY-MM-DD]`
**Makefile**: `make predict-today SEASON=2026`
**Scheduling**: None — manually triggered, no GitHub Actions or cron

### Step-by-Step Flow

```
predict-today (src/cli.py:197)
  │
  ├─ 1. build_features(season, game_date)           src/features.py:337
  │     ├─ load_games(season)                        → S3 silver/fct_games
  │     ├─ load_efficiency_ratings(season)            → S3 gold/team_adjusted_efficiencies_no_garbage
  │     ├─ load_boxscores(season)                    → S3 silver/fct_pbp_game_teams_flat
  │     │
  │     ├─ compute_game_four_factors(boxscores)      src/four_factors.py:compute_game_four_factors
  │     │   └─ Produces 13 raw per-game stats (eff_fg_pct, ft_pct, ft_rate, etc.)
  │     │
  │     ├─ [adjust_ff=False → SKIPPED]               src/adjusted_four_factors.py:adjust_four_factors
  │     │
  │     ├─ compute_rolling_averages(ff)              src/rolling_averages.py:compute_rolling_averages
  │     │   └─ EWM(span=15) + shift(1) → 13 rolling_* columns per team-game
  │     │
  │     ├─ [extra_features=None → NO extras computed]
  │     │
  │     └─ Assemble per-game feature dict:
  │         ├─ Group 1 (11): efficiency ratings + neutral_site + home indicator
  │         └─ Group 2 (26): rolling four-factor averages (home + away × 13)
  │
  ├─ 2. load_lines(season)                          → S3 silver/fct_lines
  │
  ├─ 3. predict(df, lines_df)                       src/infer.py:54
  │     ├─ load_scaler()                             → artifacts/scaler.pkl
  │     ├─ load_regressor()                          → checkpoints/regressor.pt
  │     ├─ load_classifier()                         → checkpoints/classifier.pt
  │     │
  │     ├─ X = features_df[config.FEATURE_ORDER]     ← extracts feature columns
  │     ├─ Fill NaN with scaler column means
  │     ├─ X_scaled = scaler.transform(X)
  │     ├─ regressor(X_tensor) → (mu, log_sigma)    ← predicted spread + uncertainty
  │     ├─ classifier(X_tensor) → logit → P(home)   ← win probability
  │     └─ Merge betting lines, compute edge metrics
  │
  └─ 4. save_predictions(preds, game_date)           src/infer.py:133
        ├─ predictions/json/{game_date}.json
        └─ predictions/preds_today.csv
```

### How Predictions Reach hoops-edge.vercel.app

**Unknown / not automated.** Predictions are saved locally to `predictions/json/`. There is:
- No S3 upload in `save_predictions()`
- No GitHub Actions workflow in the predictor repo
- No Vercel deployment config
- No separate frontend repo found in `~/Desktop/ml_projects/`

The frontend data delivery mechanism needs to be built or documented separately.

---

## 2. Current Checkpoint / Feature State (IMPORTANT)

There is a **mismatch** between the active config and the root checkpoints:

| Artifact | Path | Features |
|----------|------|----------|
| **feature_order.json** (active config) | `artifacts/feature_order.json` | **10** (pruned) |
| Root regressor | `checkpoints/regressor.pt` | **37** (v1) |
| Root classifier | `checkpoints/classifier.pt` | **37** (v1) |
| Root scaler | `artifacts/scaler.pkl` | **37** (v1) |
| no_garbage regressor | `checkpoints/no_garbage/regressor.pt` | **10** (pruned) |
| no_garbage classifier | `checkpoints/no_garbage/classifier.pt` | **10** (pruned) |
| no_garbage scaler | `artifacts/no_garbage/scaler.pkl` | **10** (pruned) |

### Consequence

Running `predict-today` **right now** would fail:
1. `config.FEATURE_ORDER` = 10 features (from `artifacts/feature_order.json`)
2. `load_scaler()` loads `artifacts/scaler.pkl` which expects 37 features
3. Shape mismatch: `scaler.transform(X)` receives (n, 10) but expects (n, 37) → **crash**

The 10-feature pruned model lives in `checkpoints/no_garbage/` with its own scaler at `artifacts/no_garbage/scaler.pkl`, but `predict-today` doesn't use those paths.

### The 10 Pruned Features

```
away_team_adj_oe         ← Group 1 (efficiency)
away_sos_de              ← EXTRA (sos group) — NOT computed by predict-today
home_conf_strength       ← EXTRA (conf_strength group) — NOT computed by predict-today
home_team_adj_oe         ← Group 1 (efficiency)
away_team_adj_de         ← Group 1 (efficiency)
home_team_adj_de         ← Group 1 (efficiency)
away_conf_strength       ← EXTRA (conf_strength group) — NOT computed by predict-today
away_def_tov_rate        ← EXTRA (tov_rate group) — NOT computed by predict-today
away_def_eff_fg_pct      ← Group 2 (rolling four-factor)
home_def_eff_fg_pct      ← Group 2 (rolling four-factor)
```

4 of the 10 features require `extra_features=["sos", "conf_strength", "tov_rate"]` but `predict-today` doesn't pass this. Even if the scaler mismatch were fixed, these 4 features would be NaN (filled with scaler means = no signal).

### What This Means for Adjusted Four-Factors

The 2 rolling four-factor features in the pruned set (`away_def_eff_fg_pct`, `home_def_eff_fg_pct`) **would** benefit from opponent adjustment. These are the exact features where raw vs. adjusted matters.

---

## 3. Where Raw Four-Factors Are Computed

### Step 1: Per-game raw stats — `src/four_factors.py`

**Function**: `compute_game_four_factors(box: pd.DataFrame) → pd.DataFrame`

Called at `src/features.py:393`. Takes the full boxscore table and computes 13 stats per team-game:

| Stat | Formula | Category |
|------|---------|----------|
| `eff_fg_pct` | (FGM + 0.5×3PM) / FGA | Offensive |
| `ft_pct` | FTM / FTA | Offensive |
| `ft_rate` | FTA / FGA | Offensive |
| `three_pt_rate` | 3PA / FGA | Offensive |
| `three_p_pct` | 3PM / 3PA | Offensive |
| `off_rebound_pct` | Team_OREB / (Team_OREB + Opp_DREB) | Offensive |
| `def_rebound_pct` | Team_DREB / (Team_DREB + Opp_OREB) | Offensive |
| `def_eff_fg_pct` | (Opp_FGM + 0.5×Opp_3PM) / Opp_FGA | Defensive |
| `def_ft_rate` | Opp_FTA / Opp_FGA | Defensive |
| `def_3pt_rate` | Opp_3PA / Opp_FGA | Defensive |
| `def_3p_pct` | Opp_3PM / Opp_3PA | Defensive |
| `def_off_rebound_pct` | Opp_OREB / (Opp_OREB + Team_DREB) | Defensive |
| `def_def_rebound_pct` | Opp_DREB / (Opp_DREB + Team_OREB) | Defensive |

**Input**: `fct_pbp_game_teams_flat` from S3 silver layer.
**Output**: DataFrame with `gameid, teamid, opponentid, startdate, ishometeam` + 13 stat columns.

### Step 2: Rolling averages — `src/rolling_averages.py`

**Function**: `compute_rolling_averages(four_factors: pd.DataFrame) → pd.DataFrame`

Called at `src/features.py:400`. Per team, computes `EWM(span=15).shift(1)` for each of the 13 stats. Output columns: `rolling_eff_fg_pct`, `rolling_ft_pct`, ..., `rolling_def_def_rebound_pct`.

### Step 3: Assembly — `src/features.py:483-491`

Rolling values are mapped to feature names via `AWAY_ROLLING_MAP` and `HOME_ROLLING_MAP`:
- `away_def_eff_fg_pct` ← `rolling_def_eff_fg_pct` (away team)
- `home_def_eff_fg_pct` ← `rolling_def_eff_fg_pct` (home team)
- ... (26 total rolling features)

---

## 4. Where Adjusted Four-Factors Needs to Be Called

### Current wiring (already in place, just disabled)

The adjustment hook already exists in `build_features()` at `src/features.py:394-399`:

```python
ff = compute_game_four_factors(boxscores)       # line 393
if adjust_ff:                                    # line 394
    ff = adjust_four_factors(                    # line 395
        ff,
        prior_weight=adjust_prior_weight,        # line 397
        alpha=adjust_alpha,                      # line 398
    )
rolling_df = compute_rolling_averages(ff)        # line 400
```

**What `adjust_four_factors()` does** (`src/adjusted_four_factors.py`):
- Processes games date-by-date (causal ordering)
- For each game, computes `adj_stat = raw_stat × (league_avg / opp_season_avg) ^ alpha`
- `opp_season_avg` uses Bayesian shrinkage: `(opp_n × opp_raw + prior_weight × league_avg) / (opp_n + prior_weight)`
- 12 of 13 stats adjusted (ft_pct skipped — no defensive counterpart)
- Running averages use RAW values to prevent drift

### What needs to change for production

The adjustment code is done. The problem is the **call site** — `predict-today` doesn't pass `adjust_ff=True`:

```python
# src/cli.py:205 — CURRENT
df = build_features(season, game_date=game_date)

# NEEDED
df = build_features(
    season,
    game_date=game_date,
    adjust_ff=True,
    adjust_alpha=CHOSEN_ALPHA,
    adjust_prior_weight=CHOSEN_PRIOR,
    extra_features=["sos", "conf_strength", "tov_rate"],  # for pruned 10-feature model
)
```

### Changes needed (summary)

| File | Line | Change |
|------|------|--------|
| `src/cli.py` | 205 | `predict-today` must pass `adjust_ff=True` + alpha/prior + `extra_features` |
| `src/cli.py` | 259 | `predict-season` same changes |
| `src/cli.py` | 43 | `build-features` should support `--adjust-ff`, `--alpha`, `--prior`, `--extras` flags |
| `src/cli.py` | 95-96 | `train` command must build features with matching adjustment settings |
| `src/infer.py` | 67-69 | `predict()` should load models/scaler from `no_garbage/` subdir when appropriate |
| `src/config.py` | — | Add `ADJUST_FF`, `ADJUST_ALPHA`, `ADJUST_PRIOR_WEIGHT` defaults |
| `src/dataset.py` | 51-57 | `load_season_features()` needs to support adjusted parquet filenames |

---

## 5. Config Params to Plumb Through

### New config constants needed

```python
# src/config.py
ADJUST_FF = True                  # Enable opponent-adjusted four-factors
ADJUST_ALPHA = <TBD>              # SOS exponent (candidates: 0.85, 1.0) — from GPU eval Task 6
ADJUST_PRIOR_WEIGHT = <TBD>       # Bayesian shrinkage (candidates: 5, 10) — from GPU eval Task 6
DEFAULT_EXTRA_FEATURES = ["sos", "conf_strength", "tov_rate"]  # Required for pruned model
```

### Propagation path

```
config.py (defaults)
  └→ cli.py (CLI flags with config defaults)
       ├→ build-features: --adjust-ff, --alpha, --prior, --extras
       ├→ train: same flags (features must match inference)
       ├→ predict-today: same flags
       └→ predict-season: same flags
            └→ build_features(adjust_ff=True, adjust_alpha=..., adjust_prior_weight=..., extra_features=...)
                 └→ adjust_four_factors(ff, prior_weight=..., alpha=...)
```

### Open question: alpha and prior_weight values

GPU eval Task 6 will sweep 8 combos. Until that completes, we don't have production values. The plan should use config constants so they can be set once after Task 6 results are in.

---

## 6. S3 Data Dependencies

### Current daily dependencies (already satisfied)

| Table | S3 Path | Updated By | Refresh |
|-------|---------|------------|---------|
| `fct_games` | `silver/fct_games/season=YYYY/` | hoops_edge_database_etl | Daily 08:00 UTC |
| `fct_pbp_game_teams_flat` | `silver/fct_pbp_game_teams_flat/season=YYYY/` | hoops_edge_database_etl | Daily 08:00 UTC |
| `fct_lines` | `silver/fct_lines/season=YYYY/` | hoops_edge_database_etl | Daily 08:00 UTC |
| `team_adjusted_efficiencies_no_garbage` | `gold/team_adjusted_efficiencies_no_garbage/season=YYYY/` | Manual gold rebuild | On-demand |

### New dependencies for adjusted four-factors

**None.** The adjustment uses the same `fct_pbp_game_teams_flat` boxscore data already loaded. The computation is purely in-process:

```
fct_pbp_game_teams_flat (already loaded)
  → compute_game_four_factors()        # existing
  → adjust_four_factors()              # NEW — in-process, no new S3 reads
  → compute_rolling_averages()         # existing
```

### Performance impact

`adjust_four_factors()` processes games date-by-date with nested loops over teams and stats. For a full season (~6000 rows), this takes ~2-5 seconds on CPU. For a single game_date (predict-today), the entire season's boxscores are still loaded and processed because rolling averages need the full history. No additional S3 reads are needed.

---

## 7. Prerequisite: Fix predict-today Before Adding Adjustments

Before wiring in adjusted four-factors, `predict-today` needs to work with the current 10-feature pruned model. This requires:

### Option A: Point predict-today at no_garbage checkpoints

```python
# src/infer.py — load_regressor / load_classifier / load_scaler
# Add subdir parameter or detect from checkpoint
def load_regressor(subdir: str | None = "no_garbage") -> ...:
    path = config.CHECKPOINTS_DIR / subdir / "regressor.pt" if subdir else ...

def load_scaler(subdir: str | None = "no_garbage") -> StandardScaler:
    path = config.ARTIFACTS_DIR / subdir / "scaler.pkl" if subdir else ...
```

### Option B: Retrain root models with adjusted features + pruned set

This is the cleaner long-term path: once alpha/prior are chosen from GPU eval, retrain the pruned 10-feature model with adjusted four-factors and save to root paths.

### Option C: Restore v1 feature_order.json (37 features) temporarily

Copy `artifacts/feature_order_v1.json` → `artifacts/feature_order.json` to make predict-today work with root models again. This loses the pruned model but restores a working production path.

---

## 8. Recommended Sequence

1. **Run GPU eval Tasks 5-7** → determine best alpha/prior combo
2. **Fix predict-today** to use correct checkpoint/scaler paths (Option A or B)
3. **Wire adjust_ff=True** into CLI commands with chosen alpha/prior
4. **Retrain production models** with adjusted features
5. **Update artifacts/** with new scaler, feature_order, checkpoints
6. **Verify** with `predict-today --date <past_date>` and compare to known outcomes

---

## Appendix: File Reference

| File | Lines | Role |
|------|-------|------|
| `src/cli.py:197-247` | predict-today command | Entry point |
| `src/cli.py:252-272` | predict-season command | Bulk predictions |
| `src/cli.py:34-83` | build-features command | Feature building CLI |
| `src/cli.py:88-147` | train command | Model training CLI |
| `src/features.py:337-559` | `build_features()` | Core feature assembly |
| `src/features.py:393-400` | Four-factor + adjustment + rolling | **Where adjustment slots in** |
| `src/four_factors.py` | `compute_game_four_factors()` | Raw 13-stat computation |
| `src/adjusted_four_factors.py` | `adjust_four_factors()` | Opponent adjustment (already implemented) |
| `src/rolling_averages.py` | `compute_rolling_averages()` | EWM(span=15) + shift(1) |
| `src/infer.py:54-130` | `predict()` | Model inference |
| `src/infer.py:18-50` | `load_regressor/classifier()` | Checkpoint loading |
| `src/infer.py:133-166` | `save_predictions()` | JSON + CSV output |
| `src/config.py` | Configuration | Feature order, S3 paths, constants |
| `src/dataset.py:51-57` | `load_season_features()` | Parquet loading for training |
| `src/trainer.py:32-36` | `load_scaler()` | Scaler loading |
| `artifacts/feature_order.json` | Active feature order | **Currently 10 (pruned)** |
| `artifacts/feature_order_v1.json` | Original 37 features | Backup |
| `artifacts/feature_order_v2.json` | Expanded 54 features | Backup |
| `checkpoints/regressor.pt` | Root regressor | 37-feature model |
| `checkpoints/no_garbage/regressor.pt` | Pruned regressor | 10-feature model |
| `artifacts/scaler.pkl` | Root scaler | 37 features |
| `artifacts/no_garbage/scaler.pkl` | Pruned scaler | 10 features |
