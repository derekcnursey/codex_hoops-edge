# Session 13b: Ensemble + LightGBM + Feature Engineering

Date: 2026-02-27

## Baseline: C2-V2 BS-MAE = 9.129

## Best approach: Blend(0.65) BS-MAE = 9.060

## Full comparison:

| Model | BS-MAE | Overall MAE | σ_std | Cal | ROI@12% | σ useful? (ρ) |
|-------|--------|-------------|-------|-----|---------|---------------|
| C2-V2 (baseline) | 9.129 | 10.003 | 2.49 | 0.066 | +2.7% | +0.118 |
| Mean(5) | 9.144 | 10.042 | 2.55 | 0.056 | +0.9% | +0.116 |
| Top3(C2-V1,C5-V1,C2-V2) | 9.139 | 10.029 | 2.44 | 0.055 | +2.0% | +0.115 |
| Weighted | 9.144 | 10.042 | 2.55 | 0.056 | +0.9% | +0.116 |
| Median | 9.151 | 10.040 | 2.59 | 0.058 | -0.2% | +0.114 |
| LGB baseline | 9.175 | 10.324 | 2.49 | 0.081 | +1.5% | +0.137 |
| LGB tuned | 9.080 | 10.428 | 2.49 | 0.089 | +3.3% | +0.154 |
| LGB+NNσ | 9.080 | 10.428 | 2.49 | 0.089 | +3.3% | +0.154 |
| Blend50 | 9.065 | 10.144 | 2.49 | 0.075 | -1.2% | +0.136 |
| Blend(0.65) | 9.060 | 10.216 | 2.49 | 0.078 | -1.1% | +0.142 |
| LGB+feat eng | 9.081 | 10.319 | 2.49 | 0.083 | +2.0% | +0.146 |
| Blend eng (0.65) | 9.066 | 10.157 | 2.49 | 0.076 | +0.5% | +0.136 |
| NN temporal (d=0.85) | 9.135 | 10.101 | 2.17 | 0.075 | +2.8% | +0.132 |
| NN warmup | 9.182 | 10.005 | 3.06 | 0.062 | +0.1% | +0.107 |

## Walk-Forward Results

Pooled BS-MAE: 9.167
- 2019: BS-MAE=8.988
- 2020: BS-MAE=9.470
- 2021: BS-MAE=9.922
- 2022: BS-MAE=8.817
- 2023: BS-MAE=9.056
- 2024: BS-MAE=9.085
- 2025: BS-MAE=8.969

## Verdict
New winner: Blend(0.65) (val=9.060, WF pooled=9.167)

## Production Complexity Note
The best approach (Blend(0.65)) is an ensemble winning by 0.069 MAE. This is a meaningful margin. Ensemble complexity may be justified, but monitor for drift in component models.