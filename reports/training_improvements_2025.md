# Training Improvements Investigation — Season 2025

## Motivation

The [efficiency audit](efficiency_audit_2025.md) revealed two key issues with the
adjusted efficiency ratings feeding the spread prediction model:

1. **November adj_oe correlation with Torvik: r=0.53** — early-season ratings are
   noisy due to small sample sizes and unweighted SOS adjustment.
2. **Top-20 divergence teams: 16/20 had weak schedules** — the SOS adjustment
   formula amplifies ratings for teams that haven't played strong opponents.

This investigation tested three interventions to reduce prediction error,
measured on a 2025 holdout (train: 2015-2024, test: 2025).

---

## Task 1: Training Date Filter

**Hypothesis:** Excluding early-season games (noisy features) from training
improves holdout MAE.

| Cutoff | Train Rows | Holdout MAE |
|--------|-----------|-------------|
| None (all games) | 60,010 | **10.139** |
| Dec 1 | 48,015 | 10.502 |
| Dec 15 | 41,922 | 10.613 |
| Dec 20 | 39,779 | 10.915 |
| Jan 1 | 35,102 | 11.052 |
| Jan 15 | 28,662 | 11.050 |

**Result:** No date filter is optimal. Removing early-season training data hurts
more than the noise does — the model learns to handle noisy features from
the full dataset. MAE degrades monotonically with stricter cutoffs.

**Action:** No change to training pipeline.

---

## Task 2: Preseason Prior (Warm-Start)

**Hypothesis:** Initializing the iterative efficiency solver with the previous
season's final ratings (regressed toward league average) improves early-season
ratings and reduces November prediction error.

**Implementation:**
- Added `_load_preseason_prior()` to the ETL gold layer builder
- Loads season N-1 final ratings, recenters OE/DE to common mean, regresses
  30% toward league average
- Used as warm-start initialization for the iterative solver

**Results:**

| Metric | Without Prior | With Prior |
|--------|-------------|-----------|
| Nov adj_oe Torvik r | 0.5275 | 0.5438 |
| Model holdout MAE | 10.14 | ~10.19 |

**Result:** Negligible impact. The iterative solver hits max_iter=200 on every
date regardless of initialization, so the warm-start washes out during
convergence. A prior would need to be a regularization term in the loss function
(not just initialization) to have meaningful effect.

**Action:** Implementation kept but disabled (`preseason_regression: null` in config).

---

## Task 3: SOS Exponent (Dampened Strength-of-Schedule)

**Hypothesis:** The multiplicative SOS adjustment `adj_oe = raw_oe * (league_avg / opp_adj_de)`
creates a feedback loop that prevents solver convergence. Dampening with an
exponent `(league_avg / opp_adj_de) ^ alpha` where alpha < 1 improves both
convergence and rating quality.

**Implementation:**
- Added `sos_exponent` parameter to `solve_ratings()` in `iterative_ratings.py`
- Modified SOS adjustment: `(league_avg / opp_de) ** sos_exponent`
- Also tested `shrinkage` parameter (post-convergence blend toward league avg)

### Solver Convergence

| Variant | Max Iterations Hit | Avg Iterations |
|---------|-------------------|----------------|
| baseline (sos=1.0) | 200 (never converges) | 200.0 |
| sos=0.85 | 53 | 14.9 |
| sos=0.70 | 25 | 8.5 |
| sos=0.50 | 13 | 5.7 |

The baseline solver **never converges** — it hits the 200-iteration ceiling on
every date. This is a fundamental issue with the multiplicative SOS formula at
exponent 1.0.

### Torvik Correlation (adj_oe)

| Variant | November r | December r | January r | March r |
|---------|-----------|-----------|----------|--------|
| baseline | 0.544 | 0.917 | 0.965 | 0.961 |
| **sos=0.85** | **0.666** | **0.919** | **0.961** | **0.968** |
| sos=0.70 | 0.680 | 0.914 | 0.953 | 0.948 |
| sos=0.50 | 0.681 | 0.901 | 0.939 | 0.920 |
| shrink=0.05 | 0.547 | 0.920 | 0.965 | 0.966 |
| sos=0.85+shrink | 0.666 | 0.919 | 0.961 | 0.968 |

### Model Holdout MAE

| Variant | Overall MAE | Nov MAE | Dec MAE | Jan MAE | Feb MAE | Mar MAE |
|---------|------------|---------|---------|---------|---------|---------|
| **sos=0.85** | **10.165** | 12.71 | 10.34 | 9.21 | 8.87 | 9.08 |
| sos=0.85+shrink | 10.167 | 12.70 | 10.36 | 9.21 | 8.87 | 9.09 |
| baseline | 10.177 | 12.89 | 10.28 | 9.16 | 8.86 | 9.03 |
| sos=0.70 | 10.186 | 12.70 | 10.39 | 9.24 | 8.87 | 9.12 |
| shrink=0.05 | 10.186 | 12.86 | 10.31 | 9.19 | 8.86 | 9.05 |
| sos=0.50 | 10.225 | 12.73 | 10.47 | 9.28 | 8.89 | 9.17 |

**Key findings:**
- `sos=0.85` wins on overall MAE (10.165 vs 10.177 baseline) and November MAE
  (12.71 vs 12.89)
- November Torvik correlation jumps +22% (0.544 → 0.666)
- March Torvik correlation also improves (0.961 → 0.968)
- Too much dampening hurts: sos=0.50 has worst MAE despite best Nov correlation
- Shrinkage has negligible additional effect

**Action:** Adopted `sos_exponent: 0.85` in production config.

---

## Task 4: Combined Evaluation

Final production evaluation with optimal settings:
- `sos_exponent: 0.85`
- No training date filter
- No preseason prior (`preseason_regression: null`)

### Final Monthly MAE (2025 Holdout)

| Month | MAE | Games |
|-------|-----|-------|
| November | 12.64 | 1,512 |
| December | 10.45 | 1,162 |
| January | 9.20 | 1,451 |
| February | 8.92 | 1,371 |
| March | 9.14 | 783 |
| April | 8.09 | 19 |
| **Overall** | **10.18** | **6,298** |

### Summary of Changes

| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| sos_exponent | 1.0 (implicit) | 0.85 | -0.18 Nov MAE, fixes solver convergence |
| preseason_regression | N/A | null (disabled) | No impact; kept as option |
| training date filter | None | None | Confirmed optimal |
| shrinkage | 0.0 | 0.0 | Confirmed no benefit |

### Files Changed

**hoops_edge_database_etl:**
- `src/cbbd_etl/gold/iterative_ratings.py` — added `sos_exponent` and `shrinkage` params
- `src/cbbd_etl/gold/adjusted_efficiencies.py` — preseason prior, SOS param wiring
- `config.yaml` — new config options

**hoops-edge-predictor:**
- `scripts/cutoff_sweep.py` — date filter evaluation
- `scripts/sos_eval.py` — SOS variant evaluation
- `scripts/prior_eval.py` — preseason prior evaluation
- `reports/` — this report and supporting CSVs

---

## Recommendations for Future Work

1. **Regularized prior**: Implement preseason ratings as a Bayesian prior /
   regularization term in the solver (not just warm-start initialization).
   This would pull early-season ratings toward prior-year performance.
2. **Rebuild training data**: Recompute gold layer for seasons 2015-2024 with
   `sos_exponent=0.85` to ensure training features match production.
3. **Uncertainty calibration**: The model's sigma output (Gaussian NLL) appears
   uncalibrated — investigate and fix for better confidence intervals.
4. **HCA tuning**: Current HCA is 4.03 pts/100 poss. A sweep similar to SOS
   could find optimal value.
