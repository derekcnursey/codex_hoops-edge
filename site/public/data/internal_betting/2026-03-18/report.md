# Internal Bet Filter Report: 2026-03-18

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `10`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `0`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `2`
- Filter-passing rows flagged mainly by disagreement features: `0`
- Slice mix: `{'ncaa_tournament': 2}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

_empty_

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372558,2026-03-18,2026-03-19 01:00:00+00:00,,New Mexico,Sam Houston,-11.5,7.18,AWAY,0.657,0.133,0.526,0.52,-0.006,False,none,0.0,0.0,0.0,0.0,False,raw edge only / filtered out,raw-edge watchlist only
2026,372555,2026-03-18,2026-03-19 00:00:00+00:00,,Illinois State,Kent State,-5.5,8.26,HOME,0.612,0.088,0.522,0.493,-0.029,False,none,0.0,0.0,0.0,0.0,False,raw edge only / filtered out,raw-edge watchlist only
```
