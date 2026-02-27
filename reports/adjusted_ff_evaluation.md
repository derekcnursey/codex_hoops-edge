# Adjusted Four-Factors Evaluation — Session 12

**Date**: 2026-02-27 12:33
**Training**: Seasons 2015-2024 | **Holdout**: Season 2025
**Optuna**: 50 trials per config | **Metric**: Book-Spread MAE

## Master Comparison Table

All 27 configs ranked by Book-Spread MAE.

| Rank | Config | Features | Book-Spread MAE | Overall MAE |
|------|--------|----------|-----------------|-------------|
| 1 | a0.85_p10_54feat | 53 | 9.4214 | 10.1112 |
| 2 | a0.7_p5_54feat | 53 | 9.4399 | 10.0481 |
| 3 | a1.0_p3_54feat | 53 | 9.4404 | 9.9902 |
| 4 | raw_54feat | 53 | 9.4439 | 10.2065 |
| 5 | a0.5_p5_54feat | 53 | 9.4476 | 10.0557 |
| 6 | a1.0_p15_54feat | 53 | 9.4551 | 10.0048 |
| 7 | a1.0_p3_10feat | 10 | 9.4845 | 10.1344 |
| 8 | a0.85_p5_54feat | 53 | 9.5102 | 10.0319 |
| 9 | raw_10feat | 10 | 9.5105 | 10.0773 |
| 10 | a1.0_p10_54feat | 53 | 9.5169 | 10.0894 |
| 11 | a1.0_p5_10feat | 10 | 9.5200 | 10.1135 |
| 12 | a0.85_p5_10feat | 10 | 9.5324 | 10.0280 |
| 13 | a0.85_p10_10feat | 10 | 9.5325 | 10.0968 |
| 14 | a0.5_p5_10feat | 10 | 9.5402 | 10.0940 |
| 15 | a1.0_p15_10feat | 10 | 9.5518 | 10.0505 |
| 16 | a1.0_p10_10feat | 10 | 9.5605 | 10.0413 |
| 17 | a1.0_p3_37feat | 37 | 9.5695 | 10.1412 |
| 18 | a1.0_p15_37feat | 37 | 9.6108 | 10.2187 |
| 19 | a1.0_p5_54feat | 53 | 9.6140 | 10.1489 |
| 20 | a0.7_p5_10feat | 10 | 9.6165 | 10.0747 |
| 21 | a0.5_p5_37feat | 37 | 9.6209 | 10.2373 |
| 22 | a1.0_p10_37feat | 37 | 9.6220 | 10.2873 |
| 23 | a1.0_p5_37feat | 37 | 9.6700 | 10.4134 |
| 24 | a0.85_p5_37feat | 37 | 9.6840 | 10.3022 |
| 25 | a0.85_p10_37feat | 37 | 9.6908 | 10.3281 |
| 26 | a0.7_p5_37feat | 37 | 9.7098 | 10.3593 |
| 27 | raw_37feat | 37 | 9.7341 | 10.4742 |

## Best Configuration

**Winner**: `a0.85_p10_54feat` with Book-Spread MAE = **9.4214**

### vs Raw Baselines

| Comparison | Adjusted MAE | Raw MAE | Delta | % Improvement |
|------------|-------------|---------|-------|---------------|
| vs raw_37feat | 9.4214 | 9.6200 | -0.1986 | +2.06% |
| vs raw_54feat | 9.4214 | 9.3800 | +0.0414 | -0.44% |
| vs raw_10feat | 9.4214 | 9.4800 | -0.0586 | +0.62% |

### Best Hyperparameters

```json
{
  "hidden1": 128,
  "hidden2": 128,
  "dropout": 0.45076544350380515,
  "lr": 0.0073201276267666785,
  "weight_decay": 0.00018330231402948383,
  "batch_size": 256,
  "epochs": 100
}
```

## Results by Adjustment Combo

### alpha=1.0, prior=5

| Feature Set | N | Book-Spread MAE | Overall MAE |
|-------------|---|-----------------|-------------|
| 37feat | 37 | 9.6700 | 10.4134 |
| 54feat | 53 | 9.6140 | 10.1489 |
| 10feat | 10 | 9.5200 | 10.1135 |

### alpha=0.85, prior=5

| Feature Set | N | Book-Spread MAE | Overall MAE |
|-------------|---|-----------------|-------------|
| 37feat | 37 | 9.6840 | 10.3022 |
| 54feat | 53 | 9.5102 | 10.0319 |
| 10feat | 10 | 9.5324 | 10.0280 |

### alpha=0.7, prior=5

| Feature Set | N | Book-Spread MAE | Overall MAE |
|-------------|---|-----------------|-------------|
| 37feat | 37 | 9.7098 | 10.3593 |
| 54feat | 53 | 9.4399 | 10.0481 |
| 10feat | 10 | 9.6165 | 10.0747 |

### alpha=0.5, prior=5

| Feature Set | N | Book-Spread MAE | Overall MAE |
|-------------|---|-----------------|-------------|
| 37feat | 37 | 9.6209 | 10.2373 |
| 54feat | 53 | 9.4476 | 10.0557 |
| 10feat | 10 | 9.5402 | 10.0940 |

### alpha=1.0, prior=3

| Feature Set | N | Book-Spread MAE | Overall MAE |
|-------------|---|-----------------|-------------|
| 37feat | 37 | 9.5695 | 10.1412 |
| 54feat | 53 | 9.4404 | 9.9902 |
| 10feat | 10 | 9.4845 | 10.1344 |

### alpha=1.0, prior=10

| Feature Set | N | Book-Spread MAE | Overall MAE |
|-------------|---|-----------------|-------------|
| 37feat | 37 | 9.6220 | 10.2873 |
| 54feat | 53 | 9.5169 | 10.0894 |
| 10feat | 10 | 9.5605 | 10.0413 |

### alpha=1.0, prior=15

| Feature Set | N | Book-Spread MAE | Overall MAE |
|-------------|---|-----------------|-------------|
| 37feat | 37 | 9.6108 | 10.2187 |
| 54feat | 53 | 9.4551 | 10.0048 |
| 10feat | 10 | 9.5518 | 10.0505 |

### alpha=0.85, prior=10

| Feature Set | N | Book-Spread MAE | Overall MAE |
|-------------|---|-----------------|-------------|
| 37feat | 37 | 9.6908 | 10.3281 |
| 54feat | 53 | 9.4214 | 10.1112 |
| 10feat | 10 | 9.5325 | 10.0968 |

## Ablation Analysis (Best Combo)

### Permutation Importance

| Rank | Feature | MAE Increase | Std |
|------|---------|-------------|-----|
| 1 | home_conf_strength | +0.3941 | 0.0191 |
| 2 | away_conf_strength | +0.3693 | 0.0213 |
| 3 | away_team_adj_oe | +0.1730 | 0.0240 |
| 4 | home_team_adj_oe | +0.1701 | 0.0208 |
| 5 | away_team_adj_de | +0.1594 | 0.0186 |
| 6 | home_team_BARTHAG | +0.1592 | 0.0196 |
| 7 | away_team_BARTHAG | +0.1513 | 0.0208 |
| 8 | away_eff_fg_pct | +0.1204 | 0.0202 |
| 9 | home_eff_fg_pct | +0.1027 | 0.0171 |
| 10 | home_team_adj_de | +0.0992 | 0.0148 |
| 11 | home_def_tov_rate | +0.0679 | 0.0138 |
| 12 | away_def_tov_rate | +0.0576 | 0.0090 |
| 13 | away_def_eff_fg_pct | +0.0561 | 0.0084 |
| 14 | away_tov_rate | +0.0397 | 0.0074 |
| 15 | home_def_eff_fg_pct | +0.0392 | 0.0127 |
| 16 | home_team_home | +0.0354 | 0.0039 |
| 17 | neutral_site | +0.0354 | 0.0039 |
| 18 | away_def_3p_pct | +0.0336 | 0.0071 |
| 19 | away_sos_de | +0.0326 | 0.0085 |
| 20 | home_tov_rate | +0.0295 | 0.0082 |
| 21 | away_3p_pct | +0.0266 | 0.0079 |
| 22 | home_ft_pct | +0.0262 | 0.0090 |
| 23 | home_3p_pct | +0.0229 | 0.0082 |
| 24 | home_def_3p_pct | +0.0180 | 0.0035 |
| 25 | away_sos_oe | +0.0179 | 0.0083 |
| 26 | away_def_def_rebound_pct | +0.0150 | 0.0047 |
| 27 | away_def_rebound_pct | +0.0131 | 0.0047 |
| 28 | away_off_rebound_pct | +0.0119 | 0.0046 |
| 29 | home_sos_de | +0.0118 | 0.0079 |
| 30 | away_ft_pct | +0.0103 | 0.0129 |
| 31 | home_off_rebound_pct | +0.0097 | 0.0038 |
| 32 | home_def_rebound_pct | +0.0088 | 0.0055 |
| 33 | home_def_def_rebound_pct | +0.0085 | 0.0038 |
| 34 | home_3pt_rate | +0.0079 | 0.0021 |
| 35 | rest_advantage | +0.0070 | 0.0034 |
| 36 | away_def_off_rebound_pct | +0.0055 | 0.0046 |
| 37 | home_rest_days | +0.0047 | 0.0049 |
| 38 | home_def_3pt_rate | +0.0042 | 0.0026 |
| 39 | away_3pt_rate | +0.0035 | 0.0068 |
| 40 | away_def_3pt_rate | +0.0033 | 0.0080 |
| 41 | away_rest_days | +0.0030 | 0.0043 |
| 42 | home_sos_oe | +0.0027 | 0.0047 |
| 43 | home_ft_rate | +0.0024 | 0.0010 |
| 44 | home_margin_std | +0.0013 | 0.0037 |
| 45 | home_opp_ft_rate | +0.0007 | 0.0013 |
| 46 | home_form_delta | +0.0006 | 0.0020 |
| 47 | home_def_off_rebound_pct | +0.0006 | 0.0024 |
| 48 | away_def_ft_rate | +0.0006 | 0.0024 |
| 49 | away_ft_rate | +-0.0005 | 0.0018 |
| 50 | home_team_adj_pace | +-0.0007 | 0.0023 |
| 51 | away_form_delta | +-0.0035 | 0.0024 |
| 52 | away_margin_std | +-0.0038 | 0.0070 |
| 53 | away_team_adj_pace | +-0.0112 | 0.0101 |

### Backward Elimination

| Step | Features | MAE | Removed |
|------|----------|-----|---------|
| start | 53 | 9.4080 | — |
| coarse | 48 | 9.4353 | away_ft_rate, home_team_adj_pace, away_form_delta, away_margin_std, away_team_adj_pace |
| coarse | 43 | 9.4898 | home_margin_std, home_opp_ft_rate, home_form_delta, home_def_off_rebound_pct, away_def_ft_rate |
| coarse | 38 | 9.4403 | away_3pt_rate, away_def_3pt_rate, away_rest_days, home_sos_oe, home_ft_rate |
| coarse | 33 | 9.4695 | home_3pt_rate, rest_advantage, away_def_off_rebound_pct, home_rest_days, home_def_3pt_rate |
| coarse | 28 | 9.4540 | home_sos_de, away_ft_pct, home_off_rebound_pct, home_def_rebound_pct, home_def_def_rebound_pct |
| coarse | 23 | 9.5077 | home_def_3p_pct, away_sos_oe, away_def_def_rebound_pct, away_def_rebound_pct, away_off_rebound_pct |
| coarse | 18 | 9.4883 | away_sos_de, home_tov_rate, away_3p_pct, home_ft_pct, home_3p_pct |
| coarse | 13 | 9.5337 | away_tov_rate, home_def_eff_fg_pct, home_team_home, neutral_site, away_def_3p_pct |
| fine_remove_home_eff_fg_pct | 12 | 9.5027 | home_eff_fg_pct |
| fine_remove_home_team_adj_de | 11 | 9.5291 | home_team_adj_de |
| fine_remove_away_eff_fg_pct | 10 | 9.5269 | away_eff_fg_pct |
| fine_remove_away_def_tov_rate | 9 | 9.5407 | away_def_tov_rate |
| fine_remove_away_team_BARTHAG | 8 | 9.5407 | away_team_BARTHAG |

### Forward Selection

| Step | Features | MAE | Added |
|------|----------|-----|-------|
| seed | 8 | 9.5519 | — |
| add_neutral_site | 9 | 9.4920 | neutral_site |
| add_rest_advantage | 10 | 9.4823 | rest_advantage |
| add_home_def_rebound_pct | 11 | 9.4555 | home_def_rebound_pct |

### Optimal Feature Set (11 features)

```json
[
  "away_team_adj_oe",
  "away_team_adj_de",
  "home_team_adj_oe",
  "home_team_BARTHAG",
  "away_def_eff_fg_pct",
  "home_conf_strength",
  "away_conf_strength",
  "home_def_tov_rate",
  "neutral_site",
  "rest_advantage",
  "home_def_rebound_pct"
]
```

## Detailed Metrics — Best Configuration

### MAE

- **Overall MAE**: 10.249980926513672
- **Book-Spread MAE**: 9.4596 (n=5440)

### Calibration

- Within 1σ: **67.7%** (ideal: 68.3%)
- Within 2σ: **94.3%** (ideal: 95.4%)

### Sigma-Filtered ROI

| Threshold | Bets | Wins | Losses | Win Rate | ROI |
|-----------|------|------|--------|----------|-----|
| threshold_10 | 7 | 4 | 3 | 57.1% | +10.0% |
| threshold_5 | 155 | 73 | 82 | 47.1% | -11.1% |
| threshold_7 | 39 | 17 | 22 | 43.6% | -18.5% |

### Monthly MAE Breakdown

| Month | MAE | Games |
|-------|-----|-------|
| 2024-11 | 11.16 | 1213 |
| 2024-12 | 9.44 | 945 |
| 2025-01 | 8.90 | 1378 |
| 2025-02 | 8.63 | 1105 |
| 2025-03 | 9.05 | 780 |
| 2025-04 | 7.72 | 19 |

## Production Recommendation

Based on this evaluation, the recommended production configuration is:

- **Config**: `a0.85_p10_54feat`
- **Book-Spread MAE**: 9.4214
- **Architecture**: hidden1=128, hidden2=128
- **Features**: 11 (after ablation)
