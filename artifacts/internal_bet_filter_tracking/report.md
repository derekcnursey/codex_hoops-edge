# Internal Bet Filter Live Tracking

## Scope

- Tracked report directory: `/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/artifacts/daily_internal_bet_filter`
- Seasons included: `[2026]`
- Tracked slate dates: `['2026-02-15', '2026-02-16', '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20', '2026-02-21', '2026-02-22', '2026-02-23', '2026-02-24', '2026-02-25', '2026-02-26', '2026-02-27', '2026-02-28', '2026-03-01', '2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06', '2026-03-07', '2026-03-08', '2026-03-09', '2026-03-10', '2026-03-11', '2026-03-12', '2026-03-13', '2026-03-14', '2026-03-15', '2026-03-17', '2026-03-18', '2026-03-19', '2026-03-20', '2026-03-21', '2026-03-22', '2026-03-26', '2026-03-27', '2026-03-28', '2026-03-29', '2026-04-06']`
- Ledger rows across strategy buckets: `1215`

## Headline Strategy Summary

```csv
strategy,slice,tracked_rows,bets,wins,losses,pushes,pending,ats_hit_rate,roi_per_1_at_minus_110,avg_pick_prob_edge,avg_filter_score
filter_only,conference_tournaments,47,45,26,19,1,1,0.5778,0.103,0.0126,0.6389
internal_filter,conference_tournaments,57,55,32,23,1,1,0.5818,0.1107,0.032,0.6374
ncaa_caution,conference_tournaments,0,0,0,0,0,0,,,,
overlap,conference_tournaments,10,10,6,4,0,0,0.6,0.1455,0.1152,0.6303
raw_edge_baseline,conference_tournaments,55,55,33,22,0,0,0.6,0.1455,0.1277,0.4009
raw_only,conference_tournaments,45,45,27,18,0,0,0.6,0.1455,0.1305,0.35
filter_only,feb15_plus,171,90,48,42,1,80,0.5333,0.0182,0.0207,0.6514
internal_filter,feb15_plus,229,124,68,56,1,104,0.5484,0.0469,0.0487,0.6489
ncaa_caution,feb15_plus,4,4,2,2,0,0,0.5,-0.0455,0.1383,0.6052
overlap,feb15_plus,58,34,20,14,0,24,0.5882,0.123,0.1295,0.6414
raw_edge_baseline,feb15_plus,401,218,120,98,0,183,0.5505,0.0509,0.14,0.4075
raw_only,feb15_plus,340,181,98,83,0,159,0.5414,0.0337,0.1415,0.3658
filter_only,full_live,171,90,48,42,1,80,0.5333,0.0182,0.0207,0.6514
internal_filter,full_live,229,124,68,56,1,104,0.5484,0.0469,0.0487,0.6489
ncaa_caution,full_live,4,4,2,2,0,0,0.5,-0.0455,0.1383,0.6052
overlap,full_live,58,34,20,14,0,24,0.5882,0.123,0.1295,0.6414
raw_edge_baseline,full_live,407,224,121,103,0,183,0.5402,0.0312,0.1395,0.4089
raw_only,full_live,346,187,99,88,0,159,0.5294,0.0107,0.1409,0.3682
filter_only,march,81,47,26,21,1,33,0.5532,0.0561,0.0152,0.651
internal_filter,march,97,57,32,25,1,39,0.5614,0.0718,0.032,0.6474
ncaa_caution,march,4,4,2,2,0,0,0.5,-0.0455,0.1383,0.6052
overlap,march,16,10,6,4,0,6,0.6,0.1455,0.1127,0.6294
raw_edge_baseline,march,162,80,42,38,0,82,0.525,0.0023,0.1389,0.4021
raw_only,march,143,67,34,33,0,76,0.5075,-0.0312,0.1411,0.3723
filter_only,ncaa_tournament,0,0,0,0,0,0,,,,
internal_filter,ncaa_tournament,0,0,0,0,0,0,,,,
ncaa_caution,ncaa_tournament,4,4,2,2,0,0,0.5,-0.0455,0.1383,0.6052
overlap,ncaa_tournament,0,0,0,0,0,0,,,,
raw_edge_baseline,ncaa_tournament,24,24,8,16,0,0,0.3333,-0.3636,0.1409,0.4991
raw_only,ncaa_tournament,21,21,6,15,0,0,0.2857,-0.4545,0.1361,0.4833
```

## Recent By-Day Results

```csv
strategy,game_date,tracked_rows,bets,wins,losses,pushes,pending,ats_hit_rate,roi_per_1_at_minus_110,avg_pick_prob_edge,avg_filter_score
raw_edge_baseline,2026-03-18,3,3,1,2,0,0,0.3333,-0.3636,0.1113,0.506
raw_only,2026-03-18,3,3,1,2,0,0,0.3333,-0.3636,0.1113,0.506
ncaa_caution,2026-03-19,2,2,2,0,0,0,1.0,0.9091,0.176,0.5955
raw_edge_baseline,2026-03-19,6,6,4,2,0,0,0.6667,0.2727,0.1702,0.5617
raw_only,2026-03-19,4,4,2,2,0,0,0.5,-0.0455,0.1672,0.5448
ncaa_caution,2026-03-20,2,2,0,2,0,0,0.0,-1.0,0.1005,0.615
raw_edge_baseline,2026-03-20,6,6,1,5,0,0,0.1667,-0.6818,0.165,0.547
raw_only,2026-03-20,5,5,1,4,0,0,0.2,-0.6182,0.1638,0.5288
raw_edge_baseline,2026-03-21,2,2,1,1,0,0,0.5,-0.0455,0.099,0.4375
raw_only,2026-03-21,2,2,1,1,0,0,0.5,-0.0455,0.099,0.4375
raw_edge_baseline,2026-03-22,3,3,1,2,0,0,0.3333,-0.3636,0.109,0.4993
raw_only,2026-03-22,3,3,1,2,0,0,0.3333,-0.3636,0.109,0.4993
raw_edge_baseline,2026-03-26,2,2,1,1,0,0,0.5,-0.0455,0.166,0.4315
raw_only,2026-03-26,2,2,1,1,0,0,0.5,-0.0455,0.166,0.4315
raw_edge_baseline,2026-03-27,2,2,0,2,0,0,0.0,-1.0,0.108,0.45
raw_only,2026-03-27,2,2,0,2,0,0,0.0,-1.0,0.108,0.45
raw_edge_baseline,2026-03-28,1,1,0,1,0,0,0.0,-1.0,0.134,0.425
raw_only,2026-03-28,1,1,0,1,0,0,0.0,-1.0,0.134,0.425
raw_edge_baseline,2026-03-29,2,2,0,2,0,0,0.0,-1.0,0.0845,0.4515
raw_only,2026-03-29,2,2,0,2,0,0,0.0,-1.0,0.0845,0.4515
```

## Notes

- `internal_filter` excludes NCAA threshold-pass rows; NCAA is tracked separately under `ncaa_caution`.
- ROI uses the same `-110` vig convention as the research benchmark.
- Pushes are tracked separately and excluded from ATS hit-rate denominators.
- Pending rows are live picks that do not have final scores attached yet.
