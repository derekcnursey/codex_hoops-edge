# Internal Bet Filter Maintenance Report

## Recommendation

- Recommended action: **no action**
- Default policy is to leave the filter alone unless review thresholds are met.
- In-season recalibration is not recommended; any real recalibration should wait for offseason unless there is a severe data-integrity issue.

## Operational Monitoring

- Update the live tracker after each slate settles.
- Watch `internal_filter`, `filter_only`, `raw_edge_baseline`, and `raw_only` side by side.
- Keep NCAA rows separate as `ncaa_caution` and treat them as diagnostic only.

## Season-To-Date Summary

```csv
window,strategy,slice,tracked_rows,bets,wins,losses,pushes,pending,ats_hit_rate,roi_per_1_at_minus_110,avg_pick_prob_edge,avg_filter_score
season_to_date,filter_only,conference_tournaments,47,45,26,19,1,1,0.5778,0.103,0.0126,0.6389
season_to_date,internal_filter,conference_tournaments,57,55,32,23,1,1,0.5818,0.1107,0.032,0.6374
season_to_date,ncaa_caution,conference_tournaments,0,0,0,0,0,0,,,,
season_to_date,raw_edge_baseline,conference_tournaments,55,55,33,22,0,0,0.6,0.1455,0.1277,0.4009
season_to_date,raw_only,conference_tournaments,45,45,27,18,0,0,0.6,0.1455,0.1305,0.35
season_to_date,filter_only,feb15_plus,171,90,48,42,1,80,0.5333,0.0182,0.0207,0.6514
season_to_date,internal_filter,feb15_plus,229,124,68,56,1,104,0.5484,0.0469,0.0487,0.6489
season_to_date,ncaa_caution,feb15_plus,4,4,2,2,0,0,0.5,-0.0455,0.1383,0.6052
season_to_date,raw_edge_baseline,feb15_plus,401,218,120,98,0,183,0.5505,0.0509,0.14,0.4075
season_to_date,raw_only,feb15_plus,340,181,98,83,0,159,0.5414,0.0337,0.1415,0.3658
season_to_date,filter_only,full_live,171,90,48,42,1,80,0.5333,0.0182,0.0207,0.6514
season_to_date,internal_filter,full_live,229,124,68,56,1,104,0.5484,0.0469,0.0487,0.6489
season_to_date,ncaa_caution,full_live,4,4,2,2,0,0,0.5,-0.0455,0.1383,0.6052
season_to_date,raw_edge_baseline,full_live,407,224,121,103,0,183,0.5402,0.0312,0.1395,0.4089
season_to_date,raw_only,full_live,346,187,99,88,0,159,0.5294,0.0107,0.1409,0.3682
season_to_date,filter_only,march,81,47,26,21,1,33,0.5532,0.0561,0.0152,0.651
season_to_date,internal_filter,march,97,57,32,25,1,39,0.5614,0.0718,0.032,0.6474
season_to_date,ncaa_caution,march,4,4,2,2,0,0,0.5,-0.0455,0.1383,0.6052
season_to_date,raw_edge_baseline,march,162,80,42,38,0,82,0.525,0.0023,0.1389,0.4021
season_to_date,raw_only,march,143,67,34,33,0,76,0.5075,-0.0312,0.1411,0.3723
season_to_date,filter_only,ncaa_tournament,0,0,0,0,0,0,,,,
season_to_date,internal_filter,ncaa_tournament,0,0,0,0,0,0,,,,
season_to_date,ncaa_caution,ncaa_tournament,4,4,2,2,0,0,0.5,-0.0455,0.1383,0.6052
season_to_date,raw_edge_baseline,ncaa_tournament,24,24,8,16,0,0,0.3333,-0.3636,0.1409,0.4991
season_to_date,raw_only,ncaa_tournament,21,21,6,15,0,0,0.2857,-0.4545,0.1361,0.4833
```

## Trailing Window Summary

```csv
window,strategy,slice,tracked_rows,bets,wins,losses,pushes,pending,ats_hit_rate,roi_per_1_at_minus_110,avg_pick_prob_edge,avg_filter_score
trailing_14d,raw_edge_baseline,conference_tournaments,0,0,0,0,0,0,,,,
trailing_14d,raw_edge_baseline,full_live,30,30,9,21,0,0,0.3,-0.4273,0.134,0.4999
trailing_14d,raw_edge_baseline,march,24,24,8,16,0,0,0.3333,-0.3636,0.1409,0.4991
trailing_30d,filter_only,conference_tournaments,47,45,26,19,1,1,0.5778,0.103,0.0126,0.6389
trailing_30d,internal_filter,conference_tournaments,57,55,32,23,1,1,0.5818,0.1107,0.032,0.6374
trailing_30d,raw_edge_baseline,conference_tournaments,55,55,33,22,0,0,0.6,0.1455,0.1277,0.4009
trailing_30d,filter_only,full_live,98,47,26,21,1,50,0.5532,0.0561,0.0159,0.6531
trailing_30d,internal_filter,full_live,124,57,32,25,1,66,0.5614,0.0718,0.039,0.6496
trailing_30d,raw_edge_baseline,full_live,211,86,43,43,0,125,0.5,-0.0455,0.136,0.411
trailing_30d,filter_only,march,81,47,26,21,1,33,0.5532,0.0561,0.0152,0.651
trailing_30d,internal_filter,march,97,57,32,25,1,39,0.5614,0.0718,0.032,0.6474
trailing_30d,raw_edge_baseline,march,162,80,42,38,0,82,0.525,0.0023,0.1389,0.4021
```

## Benchmark Anchor Comparison

```csv
strategy,slice,live_bets,live_ats_hit_rate,live_roi_per_1_at_minus_110,benchmark_bets,benchmark_ats_hit_rate,benchmark_roi_per_1_at_minus_110,roi_delta_vs_benchmark,ats_delta_vs_benchmark
internal_filter,full_live,124,0.5484,0.0469,2884,0.5933,0.1326,-0.0857,-0.0449
internal_filter,feb15_plus,124,0.5484,0.0469,421,0.5534,0.0566,-0.0097,-0.0051
internal_filter,march,57,0.5614,0.0718,242,0.5579,0.065,0.0068,0.0036
internal_filter,conference_tournaments,55,0.5818,0.1107,105,0.619,0.1818,-0.0711,-0.0372
internal_filter,ncaa_tournament,0,,,86,0.4884,-0.0677,,
raw_edge_baseline,full_live,224,0.5402,0.0312,7639,0.5261,0.0044,0.0268,0.0141
raw_edge_baseline,feb15_plus,218,0.5505,0.0509,1540,0.489,-0.0665,0.1174,0.0615
raw_edge_baseline,march,80,0.525,0.0023,721,0.484,-0.0759,0.0782,0.041
raw_edge_baseline,conference_tournaments,55,0.6,0.1455,296,0.5135,-0.0197,0.1651,0.0865
raw_edge_baseline,ncaa_tournament,24,0.3333,-0.3636,91,0.4176,-0.2028,-0.1608,-0.0842
```

## Alerts

```csv
level,message
info,No review or recalibration thresholds met.
```

## Policy Guardrails

- Do not react to short runs smaller than `15` settled bets over `14` days.
- Do not treat March / conference-tournament underperformance as meaningful until at least `15` settled bets exist in that slice.
- Do not consider recalibration until there is a completed-season sample with at least `100` internal-filter bets and `50` filter-only bets.
- Do not change thresholds mid-season because of one bad week or one bad tournament pocket.
- Use review alerts to inspect grading, market-line integrity, and disagreement composition before touching model calibration.
