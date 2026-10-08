# NFL Predictor

Team-level NFL game prediction. Walk-forward, no lookahead. Default train set is the public nflverse extract in `history/` (market lines back to 1999, EPA and QB tables from 2006). The 2020-only cache in `data.py` is still there if you want the original v1 run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib pyarrow requests
```

`nfl_data_py` is only required to rebuild the old `data/` cache or `history/`:

```bash
pip install --no-deps nfl_data_py   # pins pandas<2; runs fine on pandas 2/3 for this code
python history/build_history.py     # refresh the extract. ~400MB pbp cache, gitignored
```

## Quickstart

```bash
python backtest.py            # walk-forward on history/, test 2022-2026
python predict.py             # next unplayed week
python predict.py --week 7
```

No download on those two. The parquet tables are already in `history/`.

## Architecture

| File | Role |
|---|---|
| `history/` | Joinable extract. `market_games` (1999–2026 lines, juice, weather, rest, starter QB, coach), `team_games` (2006–2026 EPA splits), `qb_games`, `team_games_joined`. See `history/HISTORY.md`. |
| `history/build_history.py` | Rebuilds those tables from nflverse. Maps STL→LA, SD→LAC, OAK→LV so the join does not drop relocated teams. |
| `data.py` | `load_history_team_games()` / `load_market_games()` feed the model. The old downloader (`get_team_games`, 2020–present) is still there. |
| `features.py` | Causal features. Rolling last-8 team EPA (shifted), starter QB EPA shifted by `qb_id`, early-down EPA, rest, neutral site, 538-style Elo, and the market spread itself. |
| `models.py` | Elo baseline, L2 logistic, hist gradient boosting, plus a margin regressor. |
| `backtest.py` | Walk-forward. Trains on every prior game in the extract, predicts the week, refits. |
| `predict.py` | Next unplayed week from `market_games`. |

### No-lookahead design

- Team rolling stats use `shift(1)`. A game never sees its own EPA.
- QB EPA is shifted on `starter_qb_id`, not on the team. Same-game dropback EPA is not a feature.
- Elo is pre-game. Offseason regresses 1/3 toward 1500. Rolling windows carry across seasons.
- The posted `spread_line` is a feature (`market_spread_home`). That is the close, not a leak of the result. It will make accuracy look better. It does not make a bet better.
- Ties are dropped. `spread_line` in nflverse is away-perspective; display and bets flip it.

## Backtest

Walk-forward, 1,148 regular-season games, 2022 through 2026 week 4. Trained on 2006 through the previous week. Features include early-down EPA, starter QB EPA, and the market spread.

| model | accuracy | log loss | Brier | bets | profit | ROI | win rate |
|---|---|---|---|---|---|---|---|
| elo (baseline) | 0.636 | 0.639 | 0.224 | 668 | −66.64u | −10.0% | 47.2% |
| logreg | 0.676 | 0.609 | 0.211 | 171 | +56.18u | +32.9% | 69.6% |
| hgb | 0.637 | 0.652 | 0.228 | 802 | +24.64u | +3.1% | 54.0% |
| margin (MAE 10.01 pts) | — | — | — | 591* | −41.18u | −7.0% | 48.7% |

\* Margin row is the honest bet: −110, side only if fair spread differs from the market by more than 2 points, pushes voided.

Accuracy by season (logreg): 2022: 0.669, 2023: 0.688, 2024: 0.702, 2025: 0.653, 2026 (partial): 0.641.

v1 on the 2020 cache, no QB feature, no market feature: logreg accuracy 0.645, log loss 0.637, and the same honest margin sim was +1.8% on 842 bets. Longer history improved fit. It did not create a spread edge.

**Honest read:**

1. Logreg accuracy went up because the closing line is now a feature. You are mostly rediscovering the market. That is not a finding.
2. The +32.9% is the stylized probability sim (every bet priced at −110, edge vs a normal CDF of the spread). It is not a real moneyline price, and it double-counts the line once the line is a feature. Do not bet it.
3. The number that matters is the margin model's spread sim: **−7.0% on 591 bets.** Worse than the v1 breakeven. More seasons and a QB feature did not beat the close.
4. HGB is still the miscalibrated one. Ignore its ROI.

## Sample output — Week 5, 2026 (history model, 2026-10-07)

```
Week 5 predictions (all times ET)
  matchup        kickoff_et  p_home_elo  p_home_logreg  p_home_hgb  pred_spread  market_spread  edge_pts
 TB @ DAL Thu 10/08 8:15 PM       0.661          0.789       0.735         -4.1           -8.5      -4.4
PHI @ JAX Sun 10/11 9:30 AM       0.698          0.744       0.866        -11.6           -7.0       4.6
 CHI @ GB Sun 10/11 1:00 PM       0.450          0.404       0.442         -2.3            2.5       4.8
CIN @ MIA Sun 10/11 1:00 PM       0.408          0.269       0.211          5.4            6.5       1.1
  LV @ NE Sun 10/11 1:00 PM       0.769          0.604       0.528         -0.5           -3.5      -3.0
 MIN @ NO Sun 10/11 1:00 PM       0.292          0.459       0.575         -0.2            1.5       1.7
CLE @ NYJ Sun 10/11 1:00 PM       0.443          0.533       0.586         -2.8           -1.5       1.3
IND @ PIT Sun 10/11 1:00 PM       0.618          0.571       0.460         -5.2           -2.5       2.7
HOU @ TEN Sun 10/11 1:00 PM       0.300          0.260       0.087         10.5            7.5      -3.0
NYG @ WAS Sun 10/11 1:00 PM       0.519          0.615       0.647         -3.4           -3.5      -0.1
DEN @ LAC Sun 10/11 4:05 PM       0.369          0.371       0.397          3.2            3.5       0.3
DET @ ARI Sun 10/11 4:25 PM       0.388          0.293       0.129         11.9            5.5      -6.4
 SF @ SEA Sun 10/11 4:25 PM       0.647          0.535       0.414         -5.1           -2.5       2.6
BAL @ ATL Sun 10/11 8:20 PM       0.491          0.610       0.495          0.9           -3.5      -4.4
 BUF @ LA Mon 10/12 8:15 PM       0.525          0.579       0.543         -5.7           -3.0       2.7

Byes: CAR, KC
```

`edge_pts = market_spread − predicted_spread` (home perspective). Positive means the market gives home a better number than the model. Not a bet recommendation. The backtest says these edges do not pay at −110.

## Roadmap — player props

Team sides are done until the price changes. Extra samples live at player level:

1. Aggregate pbp to `player_games` (targets, carries, air yards, EPA). `history/build_history.py` already walks that pbp.
2. Register player features the same way. Injuries and inactives void props.
3. Poisson / negative-binomial head for yards, classifier for anytime TD.
4. Feed `off_epa_diff` and QB EPA in as game-script priors.
5. Calibrate before any betting sim. v1 HGB was overconfident. This run's probability ROI is the same class of mistake.
