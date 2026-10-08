"""Walk-forward backtest.

For each test season S (2022-2025, plus 2026 weeks played so far) and each
completed week w of S: train on every game before week w (all prior seasons
plus earlier weeks of S), then predict week w. Models are refit from scratch
each week, so no future information ever leaks into training.

Metrics per classifier: accuracy, log loss, Brier score.
Spread betting: convert the schedule's spread_line (home-team perspective)
to a market win probability via a normal CDF (margin std ~13.45 pts, the
classic Stern estimate), bet 1 unit at -110 whenever
|model_prob - market_prob| > threshold, and report ROI + bet count.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from data import get_team_games  # noqa: E402
from features import FeatureBuilder, make_X  # noqa: E402
from models import CLASSIFIERS, MarginModel  # noqa: E402

MARGIN_SD = 13.45  # std dev of NFL point margins (Stern)
BET_THRESHOLD = 0.05
ODDS_PAYOUT = 10 / 11  # profit per 1-unit stake at -110
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]

# IMPORTANT: nflverse `spread_line` is quoted from the AWAY team's perspective
# (verified empirically: spread_line correlates -0.94 with home_moneyline, so a
# positive spread_line means the HOME team is favored). I.e. the fair home
# spread is -spread_line, and the market's implied median home margin is
# +spread_line.


def _phi(x: float) -> float:
    """Standard normal CDF via math.erf (avoids a scipy dependency)."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def market_home_prob(spread_line: float) -> float:
    """Implied home win prob. spread_line is AWAY-perspective (see note above),
    so the market's median home margin is +spread_line."""
    return float(_phi(spread_line / MARGIN_SD))


def betting_roi(
    probs: np.ndarray, games: pd.DataFrame, threshold: float = BET_THRESHOLD
) -> dict:
    """Flat 1-unit bets at -110 when the model disagrees with the market.

    STYLIZED: every bet is priced at -110 as if moneyline odds were flat. In
    reality a 70% favorite lays much worse than -110, so this overstates the
    value of betting favorites. Directional only — not a trading strategy.
    """
    profit, n_bets, wins = 0.0, 0, 0
    for p_home, (_, g) in zip(probs, games.iterrows()):
        spread = g["spread_line"]
        if pd.isna(spread):
            continue
        mkt = market_home_prob(spread)
        home_win = bool(g["home_win"])
        if p_home - mkt > threshold:
            n_bets += 1
            if home_win:
                profit += ODDS_PAYOUT
                wins += 1
            else:
                profit -= 1.0
        elif mkt - p_home > threshold:
            n_bets += 1
            if not home_win:
                profit += ODDS_PAYOUT
                wins += 1
            else:
                profit -= 1.0
    return {
        "n_bets": n_bets,
        "wins": wins,
        "profit_units": round(profit, 2),
        "roi": round(profit / n_bets, 4) if n_bets else float("nan"),
        "win_rate": round(wins / n_bets, 3) if n_bets else float("nan"),
    }


def spread_cover_roi(
    pred_spreads_home: np.ndarray,
    games: pd.DataFrame,
    threshold_pts: float = 2.0,
) -> dict:
    """Honest spread betting: at -110, take a side only when the model's fair
    spread differs from the market spread by > threshold_pts. A bet wins when
    the side covers (pushes are voided)."""
    profit, n_bets, wins, pushes = 0.0, 0, 0, 0
    for ps, (_, g) in zip(pred_spreads_home, games.iterrows()):
        if pd.isna(g["spread_line"]) or pd.isna(g["home_margin"]):
            continue
        market_home = -g["spread_line"]  # schedule line is away-perspective
        side = None
        if ps < market_home - threshold_pts:
            side = "home"  # model rates home stronger than the market
        elif ps > market_home + threshold_pts:
            side = "away"
        if side is None:
            continue
        cover_margin = g["home_margin"] + market_home
        if cover_margin == 0:
            pushes += 1
            continue
        home_covers = cover_margin > 0
        won = (home_covers and side == "home") or (
            not home_covers and side == "away"
        )
        n_bets += 1
        if won:
            profit += ODDS_PAYOUT
            wins += 1
        else:
            profit -= 1.0
    return {
        "n_bets": n_bets,
        "wins": wins,
        "pushes": pushes,
        "profit_units": round(profit, 2),
        "roi": round(profit / n_bets, 4) if n_bets else float("nan"),
        "win_rate": round(wins / n_bets, 3) if n_bets else float("nan"),
    }


def run_backtest(test_seasons: list[int] = TEST_SEASONS, verbose: bool = True) -> dict:
    _, team_games = get_team_games()
    fb = FeatureBuilder(team_games)
    games = fb.build_games(game_type="REG")
    X_all = make_X(games)
    y_all = games["home_win"]

    results: dict[str, dict] = {
        cls().name: {"y_true": [], "p_home": [], "seasons": []} for cls in CLASSIFIERS
    }
    margin_true, margin_pred, margin_weeks = [], [], []
    test_games_ordered: list[pd.DataFrame] = []

    for season in test_seasons:
        season_games = games[games["season"] == season]
        weeks = sorted(season_games["week"].unique())
        for w in weeks:
            train_mask = (games["season"] < season) | (
                (games["season"] == season) & (games["week"] < w)
            )
            test_mask = (games["season"] == season) & (games["week"] == w)
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test = X_all[test_mask]
            g_test = games[test_mask]
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            test_games_ordered.append(g_test)

            for cls in CLASSIFIERS:
                model = cls().fit(X_train, y_train)
                p = model.predict_proba(X_test)[:, 1]
                r = results[model.name]
                r["y_true"].extend(g_test["home_win"].tolist())
                r["p_home"].extend(p.tolist())
                r["seasons"].extend([season] * len(p))

            mm = MarginModel().fit(X_train, games.loc[train_mask, "home_margin"])
            margin_pred.extend(mm.predict_margin(X_test).tolist())
            margin_true.extend(games.loc[test_mask, "home_margin"].tolist())
            margin_weeks.extend([season] * len(X_test))

        if verbose:
            print(f"  season {season} done", flush=True)

    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    ordered_games = (
        pd.concat(test_games_ordered, ignore_index=True)
        if test_games_ordered
        else games.iloc[0:0]
    )
    summary = {}
    for name, r in results.items():
        y_true = np.array(r["y_true"])
        p_home = np.clip(np.array(r["p_home"]), 1e-6, 1 - 1e-6)
        seasons = np.array(r["seasons"])
        by_season = {
            str(s): round(float(accuracy_score(y_true[seasons == s], (p_home[seasons == s] > 0.5).astype(int))), 3)
            for s in sorted(set(seasons.tolist()))
        }
        summary[name] = {
            "n_games": len(y_true),
            "accuracy": round(float(accuracy_score(y_true, (p_home > 0.5).astype(int))), 4),
            "log_loss": round(float(log_loss(y_true, p_home)), 4),
            "brier": round(float(brier_score_loss(y_true, p_home)), 4),
            "accuracy_by_season": by_season,
            "betting": betting_roi(p_home, ordered_games),
        }
    # Margin MAE
    mae = float(np.mean(np.abs(np.array(margin_true) - np.array(margin_pred))))
    summary["margin_mae"] = round(mae, 3)
    summary["margin_n"] = len(margin_true)
    summary["margin_spread_betting"] = spread_cover_roi(
        -np.array(margin_pred), ordered_games
    )
    return summary


def print_summary(summary: dict) -> None:
    print("\nWalk-forward backtest (train on all prior games, predict each week)")
    print("=" * 78)
    for name, s in summary.items():
        if name.startswith("margin"):
            continue
        print(f"\n{name}: n={s['n_games']}  acc={s['accuracy']}  "
              f"logloss={s['log_loss']}  brier={s['brier']}")
        print(f"  accuracy by season: {s['accuracy_by_season']}")
        b = s["betting"]
        print(f"  spread betting (thr={BET_THRESHOLD}, -110): "
              f"{b['n_bets']} bets, {b['wins']}W, profit {b['profit_units']}u, "
              f"ROI {b['roi']:.1%}, win rate {b['win_rate']:.1%}")
    print(f"\nmargin model MAE: {summary['margin_mae']} pts over {summary['margin_n']} games")
    b = summary["margin_spread_betting"]
    print(f"margin spread betting (>2pt edge, -110): {b['n_bets']} bets, {b['wins']}W, "
          f"{b['pushes']} pushes, profit {b['profit_units']}u, ROI {b['roi']:.1%}, "
          f"win rate {b['win_rate']:.1%}")


if __name__ == "__main__":
    print("Loading data and building features…")
    summary = run_backtest()
    print_summary(summary)
