# NFL Predictor

Team-level NFL game prediction (v1). Trains on nflverse data (2020–present),
predicts win probability and spread for each game, and backtests everything
walk-forward with no lookahead. Built so a player-props / fantasy module can
plug in later (see Roadmap).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib pyarrow requests tqdm appdirs
pip install --no-deps nfl_data_py
```

> Why `--no-deps`? `nfl_data_py` pins `pandas<2.0`, which has no wheel for
> modern Python and fails to build. It runs fine against pandas 3.x for
> everything this project uses (verified).

## Quickstart

```bash
python data.py        # download + cache nflverse data -> data/*.parquet (one-time)
python backtest.py    # walk-forward backtest, 2022-2026 (takes ~2 min)
python predict.py             # predict the next unplayed week
python predict.py --week 7    # predict a specific 2026 week
```

## Architecture

| File | Role |
|---|---|
| `data.py` | Downloads nflverse schedules + play-by-play, caches as parquet in `data/`. Builds `team_games`: one row per team per game with offensive/defensive EPA per play, success rates, points (from real offensive snaps only — no kneels/spikes). |
| `features.py` | Causally-safe features. Rolling last-8-game team stats (shifted: strictly pre-kickoff), rest-day differential, neutral-site flag, and 538-style margin-aware Elo. `FEATURES` is a **registry dict** — add a feature by decorating a function of the games frame; models and backtest pick it up automatically. |
| `models.py` | Three classifiers behind one `fit` / `predict_proba` interface: `EloBaseline` (fixed formula, no training), `LogisticModel` (L2, standardized), `GradientBoostingModel` (HGB). Plus `MarginModel` (HGB regressor) for fair-spread comparison. |
| `backtest.py` | Walk-forward: for each test season (2022–2025, plus 2026 weeks so far) and each week, train on all prior games, predict that week, refit. Reports accuracy / log loss / Brier, plus two betting sims (see below). |
| `predict.py` | CLI. Finds the next unplayed week from the schedule, trains on all completed games, prints matchup / kickoff (ET) / each model's home win prob / predicted spread / market spread / edge. Lists byes. |

### No-lookahead design

- Rolling features use `shift(1)`: a game's features only see games kicked off *before* it.
- Elo ratings are pre-game; updates apply after the final whistle.
- Season boundaries: rolling windows **carry across seasons** (no reset — a reset would leave September predictions on 2-game samples); Elo instead **regresses 1/3 toward 1500** each offseason, 538-style. Documented v1 tradeoff; a mean-reversion blend is a natural upgrade.
- Backtest refits weekly on strictly-prior data. Ties are excluded from modeling.

### The spread_line convention (gotcha)

nflverse's `spread_line` is quoted from the **away** team's perspective
(verified: it correlates −0.94 with `home_moneyline`). The code converts to
home perspective (`-spread_line`) wherever a spread is displayed or bet.

## Backtest results

Walk-forward, 1,148 regular-season games (2022 through 2026 Week 4).
Retrained every week on all prior games.

| model | accuracy | log loss | Brier | bets | profit | ROI | win rate |
|---|---|---|---|---|---|---|---|
| elo (baseline) | 0.636 | 0.639 | 0.224 | 667 | −63.73u | −9.6% | 47.4% |
| logreg | **0.645** | **0.637** | **0.224** | 638 | −76.73u | −12.0% | 46.1% |
| hgb | 0.605 | 0.755 | 0.256 | 1001 | +66.18u | +6.6% | 55.8% |
| margin (MAE 10.83 pts) | — | — | — | 842* | +15.18u | +1.8% | 53.3% |

\* *Margin row uses honest spread betting: take a side at −110 only when the
model's fair spread differs from market by >2 pts; pushes voided.*

Accuracy by season (logreg): 2022: 0.669, 2023: 0.632, 2024: 0.673,
2025: 0.616, 2026 (partial): 0.594.

**Honest read:**

1. **The market is efficient against these features.** The two well-calibrated
   models (Elo, logreg — best accuracy/log loss/Brier) both lose at −110 when
   betting 5%+ probability edges. There is no free lunch in team-level EPA +
   Elo vs the spread.
2. **Don't trust the HGB betting ROI.** HGB is badly miscalibrated
   (overconfident: predicts 0.82 mean in its top bin vs 0.65 actual), which
   makes it fire on 87% of games. Its +6.6% comes from that overconfidence
   combined with the simulation below — treat as variance, not edge.
3. **The probability-bet simulation is stylized.** It prices every bet at −110
   as if moneyline odds were flat; in reality a 70% favorite lays far worse
   than −110, so it overstates favorite-betting value. The margin-model
   spread sim (real −110 spread pricing, pushes voided) is the honest one:
   **+1.8% over 842 bets — roughly breakeven.**

## Sample output — Week 5, 2026 (generated 2026-10-07)

```
Week 5 predictions (all times ET)
  matchup        kickoff_et  p_home_elo  p_home_logreg  p_home_hgb  pred_spread  market_spread  edge_pts
 TB @ DAL Thu 10/08 8:15 PM       0.661          0.632       0.911         -7.1           -8.5      -1.4
PHI @ JAX Sun 10/11 9:30 AM       0.698          0.748       0.814         -4.6           -7.0      -2.4
NYG @ WAS Sun 10/11 1:00 PM       0.519          0.483       0.438          6.5           -3.5     -10.0
HOU @ TEN Sun 10/11 1:00 PM       0.300          0.322       0.166          4.9            7.5       2.6
IND @ PIT Sun 10/11 1:00 PM       0.618          0.578       0.619         -3.1           -2.5       0.6
CLE @ NYJ Sun 10/11 1:00 PM       0.443          0.404       0.179          2.6           -1.5      -4.1
  LV @ NE Sun 10/11 1:00 PM       0.769          0.711       0.828         -8.5           -3.5       5.0
CIN @ MIA Sun 10/11 1:00 PM       0.408          0.238       0.299         -2.0            6.5       8.5
 CHI @ GB Sun 10/11 1:00 PM       0.450          0.359       0.731         -5.3            2.5       7.8
 MIN @ NO Sun 10/11 1:00 PM       0.292          0.392       0.602          0.7            1.5       0.8
DEN @ LAC Sun 10/11 4:05 PM       0.369          0.377       0.324          1.7            3.5       1.8
DET @ ARI Sun 10/11 4:25 PM       0.388          0.373       0.545         -0.6            5.5       6.1
 SF @ SEA Sun 10/11 4:25 PM       0.647          0.578       0.626         -8.7           -2.5       6.2
BAL @ ATL Sun 10/11 8:20 PM       0.491          0.435       0.612          1.5           -3.5      -5.0
 BUF @ LA Mon 10/12 8:15 PM       0.525          0.490       0.280         -2.5           -3.0      -0.5

Byes: CAR, KC
```

`edge_pts = market_spread − predicted_spread` (home perspective):
positive = value on home, negative = value on away. Spreads shown home-team
perspective (negative = home favored).

## Roadmap — player props / fantasy module (v2)

v1 is deliberately team-level; the seams for player-level work are already in
place:

1. **`data.py` → `build_player_games()`**: nflverse pbp already has
   `rusher_player_id`, `receiver_player_id`, `passer_player_id`. Aggregate to
   one row per player per game (targets, carries, air yards, EPA) and cache as
   `player_games_<year>.parquet` next to `team_games_*`. Reuse the same
   `_load_or_fetch` caching.
2. **`features.py` → player registry**: the `FEATURES` registry pattern works
   unchanged — add `@feature("wr_target_share_last4")`-style functions over a
   player-games frame. Add injury/inactive handling (nflverse `injuries` +
   `snaps` data) since props are voided/void-adjacent on inactives.
3. **`models.py` → new heads**: same `fit`/`predict` interface — e.g. a
   Poisson/negative-binomial head for receiving yards, a classifier for
   anytime-TD. The backtest harness only needs (X, y, market_line).
4. **`backtest.py` → prop backtest**: swap the target to the prop line
   (over/under) with the same walk-forward loop; reuse `spread_cover_roi`
   logic with a push rule on exactly-the-line.
5. **Team-game features become inputs**: v1's `off_epa_diff` etc. are strong
   priors for player props (game script drives volume) — feed them as features
   into the prop models rather than starting from scratch.
6. **Calibration**: v1 showed HGB is overconfident — wrap v2 probability heads
   in `CalibratedClassifierCV` (isotonic) before any betting sim.
