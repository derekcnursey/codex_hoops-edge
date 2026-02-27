
# Data Audit — Pre-Session 13 Training

**Seasons**: 2015–2025
**Feature set**: 53 features (no_garbage_adj_a0.85_p10)
**Generated**: 2026-02-27 15:01

**Raw rows loaded**: 66,309
**After dropping unplayed**: 66,308

## Audit 1 — Feature Completeness

| Feature | NaN | NaN% | Zero | Zero% | Mean | Std | Min | Max | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| neutral_site | 0 | 0.0% | 58,995 | 89.0% | 0.1103 | 0.3133 | 0.0000 | 1.0000 | ZERO! |
| away_team_adj_oe | 7,914 | 11.9% | 0 | 0.0% | 103.5006 | 9.3782 | 50.1373 | 170.1644 | NaN! |
| away_team_BARTHAG | 7,914 | 11.9% | 0 | 0.0% | 0.4716 | 0.2752 | 0.0000 | 1.0000 | NaN! |
| away_team_adj_de | 7,914 | 11.9% | 0 | 0.0% | 104.9032 | 8.5326 | 57.7441 | 160.7729 | NaN! |
| away_team_adj_pace | 7,914 | 11.9% | 0 | 0.0% | 63.3248 | 3.9593 | 13.2457 | 149.4387 | NaN! |
| home_team_adj_oe | 3,552 | 5.4% | 0 | 0.0% | 104.6875 | 9.4486 | 46.8924 | 168.4845 | NaN! |
| home_team_adj_de | 3,552 | 5.4% | 0 | 0.0% | 103.6194 | 8.7898 | 54.2839 | 161.6118 | NaN! |
| home_team_adj_pace | 3,552 | 5.4% | 0 | 0.0% | 63.2974 | 3.9538 | 10.6220 | 150.4948 | NaN! |
| home_team_BARTHAG | 3,552 | 5.4% | 0 | 0.0% | 0.5167 | 0.2796 | 0.0000 | 0.9999 | NaN! |
| home_team_home | 0 | 0.0% | 7,313 | 11.0% | 0.8897 | 0.3133 | 0.0000 | 1.0000 |  |
| away_eff_fg_pct | 9,813 | 14.8% | 1 | 0.0% | 0.5076 | 0.0489 | 0.0000 | 0.8750 | NaN! |
| away_ft_pct | 9,822 | 14.8% | 6 | 0.0% | 0.7033 | 0.0649 | 0.0000 | 1.0000 | NaN! |
| away_ft_rate | 9,813 | 14.8% | 9 | 0.0% | 0.3454 | 0.1288 | 0.0000 | 5.1711 | NaN! |
| away_3pt_rate | 9,813 | 14.8% | 0 | 0.0% | 0.3745 | 0.0633 | 0.0449 | 0.7500 | NaN! |
| away_3p_pct | 9,813 | 14.8% | 17 | 0.0% | 0.3414 | 0.0497 | 0.0000 | 1.0000 | NaN! |
| away_off_rebound_pct | 9,813 | 14.8% | 3 | 0.0% | 0.3079 | 0.0557 | 0.0000 | 0.8000 | NaN! |
| away_def_rebound_pct | 9,813 | 14.8% | 0 | 0.0% | 0.7008 | 0.0508 | 0.2990 | 1.0000 | NaN! |
| away_def_eff_fg_pct | 9,813 | 14.8% | 2 | 0.0% | 0.4986 | 0.0465 | 0.0000 | 0.9224 | NaN! |
| away_def_ft_rate | 9,813 | 14.8% | 7 | 0.0% | 0.3411 | 0.1290 | 0.0000 | 5.3744 | NaN! |
| away_def_3pt_rate | 9,813 | 14.8% | 0 | 0.0% | 0.3748 | 0.0528 | 0.0526 | 0.7571 | NaN! |
| away_def_3p_pct | 9,813 | 14.8% | 14 | 0.0% | 0.3365 | 0.0479 | 0.0000 | 1.0000 | NaN! |
| away_def_off_rebound_pct | 9,813 | 14.8% | 3 | 0.0% | 0.2995 | 0.0501 | 0.0000 | 0.7014 | NaN! |
| away_def_def_rebound_pct | 9,813 | 14.8% | 0 | 0.0% | 0.6920 | 0.0552 | 0.2000 | 1.0000 | NaN! |
| home_eff_fg_pct | 7,455 | 11.2% | 0 | 0.0% | 0.5142 | 0.0442 | 0.2339 | 0.8333 | NaN! |
| home_ft_pct | 7,459 | 11.2% | 1 | 0.0% | 0.7055 | 0.0616 | 0.0000 | 1.0000 | NaN! |
| home_ft_rate | 7,455 | 11.2% | 4 | 0.0% | 0.3505 | 0.1196 | 0.0000 | 4.6881 | NaN! |
| home_3pt_rate | 7,455 | 11.2% | 0 | 0.0% | 0.3741 | 0.0620 | 0.0714 | 0.7571 | NaN! |
| home_3p_pct | 7,455 | 11.2% | 3 | 0.0% | 0.3444 | 0.0477 | 0.0000 | 1.0000 | NaN! |
| home_off_rebound_pct | 7,455 | 11.2% | 3 | 0.0% | 0.3133 | 0.0554 | 0.0000 | 1.0000 | NaN! |
| home_def_rebound_pct | 7,455 | 11.2% | 0 | 0.0% | 0.7065 | 0.0456 | 0.2000 | 1.0000 | NaN! |
| home_def_eff_fg_pct | 7,455 | 11.2% | 1 | 0.0% | 0.4911 | 0.0430 | 0.0000 | 0.8750 | NaN! |
| home_opp_ft_rate | 7,455 | 11.2% | 4 | 0.0% | 0.3365 | 0.1233 | 0.0000 | 5.2901 | NaN! |
| home_def_3pt_rate | 7,455 | 11.2% | 0 | 0.0% | 0.3740 | 0.0524 | 0.0556 | 0.7500 | NaN! |
| home_def_3p_pct | 7,455 | 11.2% | 10 | 0.0% | 0.3327 | 0.0463 | 0.0000 | 1.0334 | NaN! |
| home_def_off_rebound_pct | 7,455 | 11.2% | 2 | 0.0% | 0.2942 | 0.0446 | 0.0000 | 0.8000 | NaN! |
| home_def_def_rebound_pct | 7,455 | 11.2% | 1 | 0.0% | 0.6866 | 0.0548 | 0.0000 | 1.0000 | NaN! |
| home_rest_days | 0 | 0.0% | 8 | 0.0% | 3.9430 | 2.0017 | 0.0000 | 30.0000 |  |
| away_rest_days | 0 | 0.0% | 14 | 0.0% | 4.1607 | 3.0677 | 0.0000 | 30.0000 |  |
| rest_advantage | 0 | 0.0% | 11,783 | 17.8% | -0.2177 | 3.0201 | -30.0000 | 25.8542 |  |
| home_sos_oe | 3,552 | 5.4% | 0 | 0.0% | 104.1505 | 4.7111 | 66.2034 | 146.0898 | NaN! |
| home_sos_de | 3,552 | 5.4% | 0 | 0.0% | 104.1716 | 4.9692 | 67.1509 | 147.3684 | NaN! |
| away_sos_oe | 7,914 | 11.9% | 0 | 0.0% | 104.1213 | 4.6559 | 63.0069 | 145.1908 | NaN! |
| away_sos_de | 7,914 | 11.9% | 0 | 0.0% | 104.3284 | 4.7425 | 61.5729 | 146.8845 | NaN! |
| home_conf_strength | 1,700 | 2.6% | 18 | 0.0% | 0.5077 | 9.9425 | -47.3814 | 47.3814 |  |
| away_conf_strength | 6,478 | 9.8% | 3 | 0.0% | -1.4755 | 9.7137 | -47.3814 | 47.3814 | NaN! |
| home_form_delta | 7,459 | 11.2% | 2,214 | 3.3% | 0.0010 | 0.0147 | -0.0976 | 0.9168 | NaN! |
| away_form_delta | 9,822 | 14.8% | 2,568 | 3.9% | 0.0011 | 0.0153 | -0.1350 | 0.6893 | NaN! |
| home_tov_rate | 7,455 | 11.2% | 3 | 0.0% | 0.1966 | 0.0314 | 0.0000 | 0.4603 | NaN! |
| home_def_tov_rate | 7,455 | 11.2% | 0 | 0.0% | 0.2036 | 0.0354 | 0.0462 | 0.5714 | NaN! |
| away_tov_rate | 9,813 | 14.8% | 1 | 0.0% | 0.1996 | 0.0346 | 0.0000 | 0.5775 | NaN! |
| away_def_tov_rate | 9,813 | 14.8% | 2 | 0.0% | 0.2018 | 0.0351 | 0.0000 | 0.5000 | NaN! |
| home_margin_std | 6,942 | 10.5% | 23 | 0.0% | 15.0118 | 5.9931 | 0.0000 | 66.4254 | NaN! |
| away_margin_std | 9,602 | 14.5% | 38 | 0.1% | 15.1909 | 6.3061 | 0.0000 | 76.7224 | NaN! |

### Features with >5% NaN

- **away_team_adj_oe**: 11.9% NaN
- **away_team_BARTHAG**: 11.9% NaN
- **away_team_adj_de**: 11.9% NaN
- **away_team_adj_pace**: 11.9% NaN
- **home_team_adj_oe**: 5.4% NaN
- **home_team_adj_de**: 5.4% NaN
- **home_team_adj_pace**: 5.4% NaN
- **home_team_BARTHAG**: 5.4% NaN
- **away_eff_fg_pct**: 14.8% NaN
- **away_ft_pct**: 14.8% NaN
- **away_ft_rate**: 14.8% NaN
- **away_3pt_rate**: 14.8% NaN
- **away_3p_pct**: 14.8% NaN
- **away_off_rebound_pct**: 14.8% NaN
- **away_def_rebound_pct**: 14.8% NaN
- **away_def_eff_fg_pct**: 14.8% NaN
- **away_def_ft_rate**: 14.8% NaN
- **away_def_3pt_rate**: 14.8% NaN
- **away_def_3p_pct**: 14.8% NaN
- **away_def_off_rebound_pct**: 14.8% NaN
- **away_def_def_rebound_pct**: 14.8% NaN
- **home_eff_fg_pct**: 11.2% NaN
- **home_ft_pct**: 11.2% NaN
- **home_ft_rate**: 11.2% NaN
- **home_3pt_rate**: 11.2% NaN
- **home_3p_pct**: 11.2% NaN
- **home_off_rebound_pct**: 11.2% NaN
- **home_def_rebound_pct**: 11.2% NaN
- **home_def_eff_fg_pct**: 11.2% NaN
- **home_opp_ft_rate**: 11.2% NaN
- **home_def_3pt_rate**: 11.2% NaN
- **home_def_3p_pct**: 11.2% NaN
- **home_def_off_rebound_pct**: 11.2% NaN
- **home_def_def_rebound_pct**: 11.2% NaN
- **home_sos_oe**: 5.4% NaN
- **home_sos_de**: 5.4% NaN
- **away_sos_oe**: 11.9% NaN
- **away_sos_de**: 11.9% NaN
- **away_conf_strength**: 9.8% NaN
- **home_form_delta**: 11.2% NaN
- **away_form_delta**: 14.8% NaN
- **home_tov_rate**: 11.2% NaN
- **home_def_tov_rate**: 11.2% NaN
- **away_tov_rate**: 14.8% NaN
- **away_def_tov_rate**: 14.8% NaN
- **home_margin_std**: 10.5% NaN
- **away_margin_std**: 14.5% NaN

### Features with >50% Zero

- **neutral_site**: 89.0% zero

## Audit 2 — Target Variable (Home Margin)

- **Total games**: 66,308
- **Mean**: 7.15
- **Median**: 6.00
- **Std**: 16.46
- **Min**: -81
- **Max**: 108

- Within ±5: 28.1%
- Within ±10: 51.2%
- Within ±20: 79.9%
- Within ±30: 91.2%

**Games with |margin| > 60**: 453
| gameId | Date | Away | Home | AwayPts | HomePts | Margin |
| --- | --- | --- | --- | --- | --- | --- |
| 12182 | 2022-11-08 | 752 | 31 | 39 | 147 | +108 |
| 25235 | 2020-12-03 | 534 | 162 | 37 | 140 | +103 |
| 42157 | 2017-11-11 | 577 | 9 | 34 | 135 | +101 |
| 863 | 2024-11-19 | 685 | 130 | 19 | 119 | +100 |
| 6140 | 2023-11-10 | 867 | 237 | 34 | 130 | +96 |
| 18579 | 2021-11-11 | 1031 | 129 | 40 | 135 | +95 |
| 30309 | 2019-11-09 | 668 | 103 | 52 | 147 | +95 |
| 7756 | 2023-12-10 | 935 | 202 | 14 | 108 | +94 |
| 31018 | 2019-11-23 | 767 | 166 | 16 | 110 | +94 |
| 30321 | 2019-11-09 | 175 | 331 | 49 | 143 | +94 |
| 7951 | 2023-12-16 | 747 | 89 | 55 | 146 | +91 |
| 2483 | 2024-12-23 | 799 | 341 | 30 | 120 | +90 |
| 823 | 2024-11-17 | 668 | 274 | 42 | 131 | +89 |
| 38120 | 2018-12-22 | 559 | 139 | 32 | 121 | +89 |
| 348 | 2024-11-09 | 614 | 130 | 36 | 124 | +88 |
| 3504 | 2025-01-18 | 584 | 67 | 38 | 126 | +88 |
| 42335 | 2017-11-15 | 788 | 361 | 46 | 134 | +88 |
| 36068 | 2018-11-08 | 1016 | 99 | 51 | 139 | +88 |
| 30082 | 2019-11-06 | 631 | 317 | 46 | 134 | +88 |
| 13740 | 2022-12-03 | 587 | 199 | 40 | 127 | +87 |

**Games with margin = 0**: 1646

## Audit 3 — Duplicate / Corrupt Games

- **Duplicate gameIds**: 0
- **Home == Away team**: 0
- **Missing team IDs**: 0
- **0-0 games**: 1646
| gameId | Date | Away | Home |
| --- | --- | --- | --- |
| 61428 | 2014-12-07 | 176 | 222 |
| 65234 | 2015-03-01 | 192 | 326 |
| 65395 | 2015-03-06 | 14 | 175 |
| 60503 | 2014-11-22 | 180 | 33 |
| 63630 | 2015-01-28 | 65 | 235 |
| 63629 | 2015-01-28 | 169 | 245 |
| 63675 | 2015-01-29 | 111 | 152 |
| 63933 | 2015-02-03 | 123 | 83 |
| 64582 | 2015-02-15 | 168 | 26 |
| 64607 | 2015-02-17 | 7 | 14 |

**Total issues**: 1646

## Audit 4 — Season Distribution

| Season | Games | Flag |
| --- | --- | --- |
| 2015 | 5,949 |  |
| 2016 | 5,991 |  |
| 2017 | 5,982 |  |
| 2018 | 5,987 |  |
| 2019 | 6,066 |  |
| 2020 | 5,856 |  |
| 2021 | 5,282 |  |
| 2022 | 6,387 |  |
| 2023 | 6,261 |  |
| 2024 | 6,249 |  |
| 2025 | 6,298 |  |

All seasons have ≥1000 games.

## Audit 5 — Feature Correlation with Target

| Feature | Pearson r | Flag |
| --- | --- | --- |
| away_eff_fg_pct | -0.2468 |  |
| away_team_BARTHAG | -0.2123 |  |
| away_def_eff_fg_pct | +0.2115 |  |
| away_def_off_rebound_pct | +0.2097 |  |
| away_def_rebound_pct | -0.2092 |  |
| home_team_BARTHAG | +0.1971 |  |
| away_rest_days | +0.1953 |  |
| away_tov_rate | +0.1921 |  |
| away_team_adj_oe | -0.1912 |  |
| rest_advantage | -0.1691 |  |
| away_team_adj_de | +0.1626 |  |
| home_team_adj_oe | +0.1617 |  |
| home_team_adj_de | -0.1616 |  |
| away_3p_pct | -0.1410 |  |
| away_def_def_rebound_pct | +0.1377 |  |
| away_off_rebound_pct | -0.1375 |  |
| home_def_eff_fg_pct | -0.1370 |  |
| home_eff_fg_pct | +0.1198 |  |
| home_off_rebound_pct | +0.1154 |  |
| home_def_def_rebound_pct | -0.1153 |  |
| away_conf_strength | -0.1114 |  |
| home_def_tov_rate | +0.1052 |  |
| away_ft_pct | -0.1006 |  |
| neutral_site | -0.0913 |  |
| home_team_home | +0.0913 |  |
| away_def_3p_pct | +0.0864 |  |
| away_def_tov_rate | -0.0853 |  |
| home_def_3p_pct | -0.0794 |  |
| home_margin_std | +0.0756 |  |
| home_conf_strength | +0.0727 |  |
| home_tov_rate | -0.0718 |  |
| home_def_rebound_pct | +0.0623 |  |
| home_def_off_rebound_pct | -0.0591 |  |
| away_ft_rate | -0.0560 |  |
| away_margin_std | +0.0495 |  |
| home_3p_pct | +0.0456 |  |
| home_rest_days | +0.0442 |  |
| away_def_3pt_rate | +0.0380 |  |
| home_sos_oe | -0.0368 |  |
| away_def_ft_rate | +0.0341 |  |
| away_sos_oe | -0.0338 |  |
| home_opp_ft_rate | -0.0270 |  |
| home_form_delta | -0.0191 |  |
| away_3pt_rate | -0.0188 |  |
| home_sos_de | -0.0157 |  |
| home_3pt_rate | +0.0154 |  |
| home_team_adj_pace | -0.0153 |  |
| home_ft_pct | +0.0104 |  |
| home_def_3pt_rate | -0.0082 | NOISE |
| away_sos_de | +0.0082 | NOISE |
| home_ft_rate | +0.0063 | NOISE |
| away_form_delta | -0.0060 | NOISE |
| away_team_adj_pace | -0.0003 | NOISE |

### Near-zero correlation features (|r| < 0.01)

- **away_team_adj_pace**: r = -0.0003
- **home_ft_rate**: r = +0.0063
- **home_def_3pt_rate**: r = -0.0082
- **away_sos_de**: r = +0.0082
- **away_form_delta**: r = -0.0060

## Audit 6 — Feature-to-Feature Correlation

| Feature A | Feature B | r |
| --- | --- | --- |
| neutral_site | home_team_home | -1.0000 |
| home_off_rebound_pct | home_def_def_rebound_pct | -0.9991 |
| away_off_rebound_pct | away_def_def_rebound_pct | -0.9990 |
| away_def_rebound_pct | away_def_off_rebound_pct | -0.9750 |
| home_def_rebound_pct | home_def_off_rebound_pct | -0.9691 |

## Audit 7 — Book Spread Availability

| Season | Games | With Spread | % | Flag |
| --- | --- | --- | --- | --- |
| 2015 | 5,949 | 3,839 | 64.5% |  |
| 2016 | 5,991 | 3,890 | 64.9% |  |
| 2017 | 5,982 | 3,889 | 65.0% |  |
| 2018 | 5,987 | 4,050 | 67.6% |  |
| 2019 | 6,066 | 5,530 | 91.2% |  |
| 2020 | 5,856 | 5,388 | 92.0% |  |
| 2021 | 5,282 | 4,181 | 79.2% |  |
| 2022 | 6,387 | 5,556 | 87.0% |  |
| 2023 | 6,261 | 5,789 | 92.5% |  |
| 2024 | 6,249 | 5,311 | 85.0% |  |
| 2025 | 6,298 | 5,440 | 86.4% |  |

## Audit 8 — Scaler Sanity

| Feature | Scaler Mean | Scaler Std | Flag |
| --- | --- | --- | --- |
| neutral_site | 0.110288 | 0.313249 |  |
| away_team_adj_oe | 91.147559 | 34.690035 |  |
| away_team_BARTHAG | 0.415324 | 0.300088 |  |
| away_team_adj_de | 92.382819 | 34.939757 |  |
| away_team_adj_pace | 55.766865 | 20.863550 |  |
| home_team_adj_oe | 99.079605 | 25.300664 |  |
| home_team_adj_de | 98.068735 | 24.848976 |  |
| home_team_adj_pace | 59.906637 | 14.762170 |  |
| home_team_BARTHAG | 0.489064 | 0.295865 |  |
| home_team_home | 0.889712 | 0.313249 |  |
| away_eff_fg_pct | 0.432463 | 0.185808 |  |
| away_ft_pct | 0.599142 | 0.256924 |  |
| away_ft_rate | 0.294278 | 0.170837 |  |
| away_3pt_rate | 0.319050 | 0.145222 |  |
| away_3p_pct | 0.290905 | 0.129633 |  |
| away_off_rebound_pct | 0.262329 | 0.120834 |  |
| away_def_rebound_pct | 0.597068 | 0.253226 |  |
| away_def_eff_fg_pct | 0.424829 | 0.182186 |  |
| away_def_ft_rate | 0.290621 | 0.169858 |  |
| away_def_3pt_rate | 0.319296 | 0.141724 |  |
| away_def_3p_pct | 0.286737 | 0.127408 |  |
| away_def_off_rebound_pct | 0.255205 | 0.115971 |  |
| away_def_def_rebound_pct | 0.589593 | 0.250957 |  |
| home_eff_fg_pct | 0.456373 | 0.167689 |  |
| home_ft_pct | 0.626107 | 0.230345 |  |
| home_ft_rate | 0.311079 | 0.157970 |  |
| home_3pt_rate | 0.332084 | 0.131821 |  |
| home_3p_pct | 0.305703 | 0.117725 |  |
| home_off_rebound_pct | 0.278116 | 0.111905 |  |
| home_def_rebound_pct | 0.627083 | 0.227276 |  |
| home_def_eff_fg_pct | 0.435927 | 0.160349 |  |
| home_opp_ft_rate | 0.298668 | 0.157446 |  |
| home_def_3pt_rate | 0.331934 | 0.128034 |  |
| home_def_3p_pct | 0.295330 | 0.113787 |  |
| home_def_off_rebound_pct | 0.261084 | 0.101999 |  |
| home_def_def_rebound_pct | 0.609390 | 0.222958 |  |
| home_rest_days | 3.943032 | 2.001720 |  |
| away_rest_days | 4.160692 | 3.067694 |  |
| rest_advantage | -0.217660 | 3.020102 |  |
| home_sos_oe | 98.571305 | 23.894548 |  |
| home_sos_de | 98.591301 | 23.948633 |  |
| away_sos_oe | 91.694220 | 34.037960 |  |
| away_sos_de | 91.876611 | 34.115066 |  |
| home_conf_strength | 0.494705 | 9.814467 |  |
| away_conf_strength | -1.331313 | 9.237342 |  |
| home_form_delta | 0.000858 | 0.013865 |  |
| away_form_delta | 0.000938 | 0.014085 |  |
| home_tov_rate | 0.174520 | 0.068813 |  |
| home_def_tov_rate | 0.180671 | 0.072441 |  |
| away_tov_rate | 0.170040 | 0.077713 |  |
| away_def_tov_rate | 0.171968 | 0.078643 |  |
| home_margin_std | 13.440134 | 7.299241 |  |
| away_margin_std | 12.991151 | 7.911115 |  |

Scaler looks healthy — no near-zero stds or extreme means.

## Audit 9 — NaN→0 Fill Distortion (CRITICAL)

The current training pipeline fills NaN with 0 via `np.nan_to_num(X, nan=0.0)` BEFORE fitting the scaler.
This catastrophically distorts the scaler for features with natural ranges far from 0:

| Feature | True Mean | NaN% | NaN→0 Mean | Distortion |
| --- | ---: | ---: | ---: | ---: |
| away_team_adj_oe | 103.55 | 11.9% | 91.15 | 11.9% |
| away_team_adj_de | 104.90 | 11.9% | 92.38 | 11.9% |
| away_team_adj_pace | 63.32 | 11.9% | 55.77 | 11.9% |
| home_team_adj_oe | 104.69 | 5.4% | 99.08 | 5.4% |
| home_team_adj_de | 103.62 | 5.4% | 98.07 | 5.4% |
| away_eff_fg_pct | 0.51 | 14.8% | 0.43 | 14.8% |
| home_eff_fg_pct | 0.51 | 11.2% | 0.46 | 11.2% |
| away_margin_std | 15.19 | 14.5% | 12.99 | 14.5% |
| home_margin_std | 15.01 | 10.5% | 13.44 | 10.5% |

**Impact**: For `away_team_adj_oe`, the true std is 9.38 but the NaN→0 scaler std
is 34.69 (3.7x inflated). This compresses real signal into a narrow band while the
~12% of zero-filled NaN rows sit at z ≈ -2.6. The model sees these as teams with
impossibly low offensive efficiency, which is pure noise.

## Audit 10 — NaN Distribution Pattern

NaN values are NOT purely early-season cold-start. They persist throughout the season
for teams missing boxscore data (small-conference teams, D2/exhibition opponents):

| NaN per game | Games | Cumulative % |
| --- | ---: | ---: |
| 0 | 51,077 | 77.0% |
| 1–5 | 2,522 | 80.8% |
| 6–10 | 2,473 | 84.6% |
| 11–30 | 2,950 | 89.1% |
| 31–40 | 4,693 | 96.2% |
| 41–53 | 2,507 | 100.0% |

Effect of min-date filter + 0-0 removal:

| Filter | Games | Complete rows | Complete % |
| --- | ---: | ---: | ---: |
| None (0-0 removed) | 64,662 | 51,076 | 79.0% |
| min-date=12-01 | 51,252 | 45,954 | 89.7% |
| min-date=12-15 | 44,793 | 41,000 | 91.5% |

## Audit 11 — 0-0 Games Deep Dive

1,646 games have scores recorded as 0-0. These are NOT unplayed (they passed the
`dropna(homeScore, awayScore)` filter). They include real matchups like
DePaul @ Providence and Michigan St @ Rutgers — clearly not actual 0-0 results.

Season distribution:

| Season | 0-0 Games | Note |
| --- | ---: | --- |
| 2015–2019 | 111 | Scattered data gaps |
| 2020 | 88 | COVID cancellations |
| 2021 | 992 | COVID peak — 18.8% of season |
| 2022 | 410 | COVID tail |
| 2023–2025 | 45 | Residual gaps |

These games teach the model that margin=0 is common (it's not — only 0.03% of
real games end tied in regulation+OT). They add pure noise to the training target.

---

## Summary — Go / No-Go

### BLOCKING Issues (must fix before training)

1. **Remove 1,646 0-0 games** — Filter: `(homeScore != 0) | (awayScore != 0)`.
   These are data errors (missing scores recorded as 0), not real outcomes.

2. **Fix NaN imputation: column-mean instead of zero** — Change
   `np.nan_to_num(X, nan=0.0)` to fill with column means from the scaler.
   The current zero-fill distorts the scaler by up to 3.7x on key features
   (efficiency ratings, four-factors). This is silently corrupting 10-15% of
   training rows and biasing the scaler statistics.

### RECOMMENDED (significant improvement)

3. **Drop 1 redundant feature** — `home_team_home` is exactly `-1 * neutral_site`
   (r = -1.000). Drop `home_team_home` to eliminate perfect multicollinearity.

4. **Consider dropping 3 more redundant features** — `away_def_def_rebound_pct` is
   `1 - away_off_rebound_pct` (r = -0.999), and `home_def_def_rebound_pct` is
   `1 - home_off_rebound_pct` (r = -0.999). These add no information.
   Similarly `away_def_off_rebound_pct` ↔ `away_def_rebound_pct` (r = -0.975).

5. **Apply min-date=12-01 filter** — Drops early-season noise, brings completeness
   from 79% to 90%, and still keeps ~51K training games across 11 seasons.

### ACCEPTABLE (keep but monitor)

6. **47 features with >5% NaN** — Expected due to cold-start (rolling averages
   need prior games). With column-mean imputation + min-date filter, most NaN rows
   become reasonable approximations.

7. **5 noise features (|r| < 0.01)** — `away_team_adj_pace`, `home_ft_rate`,
   `home_def_3pt_rate`, `away_sos_de`, `away_form_delta`. These won't hurt with
   regularization but add no predictive value. Consider dropping in a future session.

8. **453 games with |margin| > 60** — Legitimate blowouts (D1 vs tiny programs).
   These are real but rare. The model should learn high sigma for these.

9. **neutral_site 89% zero** — Expected, binary indicator. Keep as-is.

10. **Book spread coverage 65-92%** — Lower in early seasons (2015-2018).
    Not an issue for training (spread is not a feature), only for evaluation.

### Effective training set after fixes

| Step | Games |
| --- | ---: |
| Raw loaded | 66,308 |
| Remove 0-0 scores | 64,662 |
| Apply min-date=12-01 | ~51,252 |
| Column-mean impute NaN | 51,252 (89.7% complete, rest imputed) |