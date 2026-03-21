# Internal Bet Filter Report: 2026-03-22

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `14`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `0`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `4`
- Filter-passing rows flagged mainly by disagreement features: `0`
- Slice mix: `{'ncaa_tournament': 8}`

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
2026,372576,2026-03-22,2026-03-23 00:30:00+00:00,,Oklahoma State,Wichita State,-3.5,10.096,HOME,0.712,0.188,0.549,0.506,-0.043,False,none,0.0,0.0,0.0,0.0,False,raw edge only / filtered out,raw-edge watchlist only
2026,372572,2026-03-22,2026-03-22 20:30:00+00:00,,Wake Forest,Illinois State,-7.5,12.182,HOME,0.666,0.142,0.535,0.498,-0.037,False,none,0.0,0.0,0.0,0.0,False,raw edge only / filtered out,raw-edge watchlist only
2026,372583,2026-03-22,2026-03-22 23:50:00+00:00,ncaa_tournament,Arizona,Utah State,-12.5,7.147,AWAY,0.631,0.107,0.488,0.484,-0.004,False,none,1.116,1.116,0.0,0.0,True,raw edge only / filtered out,raw-edge watchlist only
2026,372582,2026-03-22,2026-03-22 23:10:00+00:00,ncaa_tournament,Florida,Iowa,-10.5,6.147,AWAY,0.613,0.089,0.481,0.511,0.03,False,none,3.236,3.236,0.0,0.0,True,raw edge only / filtered out,raw-edge watchlist only
```
