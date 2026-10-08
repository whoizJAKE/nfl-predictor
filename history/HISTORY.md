# Deeper public history

This is not a private archive. Renaissance-style edge is proprietary flow and microstructure. What exists for NFL, free, and actually joinable, is here.

Built 2026-10-07 from nflverse public releases.

## Files

| file | rows | what it is |
|---|---|---|
| `market_games.parquet` | 7,548 games | Schedules 1999–2026. Spread, total, moneylines, juice, rest, roof, surface, temp, wind, stadium, starter QB id/name, coach. `spread_line` is away-perspective, same gotcha as v1. |
| `team_games.parquet` | 10,990 team-games | 2006–2026 (2026 partial). Offensive/defensive EPA and success, plus early-down / pass / rush EPA splits. Real offensive snaps only (no kneels, no spikes). |
| `qb_games.parquet` | 13,380 QB-games | Dropbacks, EPA/play, success, CPOE, air yards. `is_primary` = most dropbacks for that team in that game. |
| `team_games_joined.parquet` | 10,990 | Team EPA joined to market, weather, rest, coach, schedule starter, and primary QB. 100% match after relocating STL→LA, SD→LAC, OAK→LV. |
| `closing_lines_2006_2018.parquet` | 20,490 | nflverse `closing_lines.csv`. Stale. Ends 2018. Do not treat it as the live close. `market_games.spread_line` is the line to use. |

## Keys

- `game_id` is `season_week_away_home` (`2006_01_MIA_PIT`) in both pbp aggregates and `market_games`.
- `old_game_id` is the numeric GSIS id.
- Pbp `team` is the modern abbreviation. `team_schedule` keeps the code used that season.

## How this plugs into v1

`features.py` already rolls `off_epa_pp`, `def_epa_pp`, points, rest. Point `FeatureBuilder` at `team_games_joined` instead of the 2020+ cache and add:

- `primary_qb_epa_pp` rolled, shifted, for the schedule starter (`starter_qb_id`), not the team average
- `early_epa_pp` the same way
- `spread_line` as a feature if the target is residual/cover, not raw win
- `wind`, `roof`, `div_game`, rest differential (rest is already there)

Do not feed `primary_qb_epa_pp` from the same game. Shift it. The starter id on a completed game is who played; for live weeks use the depth chart, or you leak.

EPA starts in 2006 because that is where the public EPA series and the posted lines are both usable. Scores and lines in `market_games` go back to 1999. Play-by-play exists to 1999 if you want the extra noisy seasons; it will not behave like 2018.
