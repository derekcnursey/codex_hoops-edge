# Internal Bet Filter Report: 2026-03-22

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `14`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `0`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `3`
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
2026,372582,2026-03-22,2026-03-22 23:10:00+00:00,ncaa_tournament,Florida,Iowa,-10.5,4.238,AWAY,0.664,0.141,0.497,0.502,0.005,False,none,2.126,2.126,0.0,0.0,True,raw edge only / filtered out,raw-edge watchlist only
2026,372576,2026-03-22,2026-03-23 00:30:00+00:00,,Oklahoma State,Wichita State,-2.5,6.045,HOME,0.618,0.094,0.526,0.497,-0.029,False,none,0.0,0.0,0.0,0.0,False,raw edge only / filtered out,raw-edge watchlist only
2026,372584,2026-03-22,2026-03-23 00:45:00+00:00,ncaa_tournament,UConn,UCLA,-4.5,0.591,AWAY,0.616,0.092,0.479,0.499,0.021,False,none,2.909,2.909,1.0,0.0,True,raw edge only / filtered out,raw-edge watchlist only
```
