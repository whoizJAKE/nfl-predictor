"""Predict the upcoming week.

Detects the next unplayed week from the schedule, trains every model on all
completed regular-season games (2020 through the current week), and prints a
table: matchup, kickoff (ET), each model's home win probability, predicted
spread, market spread, and edge.

Edge is in points from the home team's perspective:
    edge = market_spread_home - predicted_spread_home
where the schedule's spread_line is AWAY-perspective, so
market_spread_home = -spread_line. Positive edge means the market gives the
home team a *better* number than the model thinks is fair (value on home);
negative means value on the road team.

Usage:
    python predict.py            # next unplayed week
    python predict.py --week 7   # a specific 2026 week (must be unplayed)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from data import get_schedules, get_team_games  # noqa: E402
from features import FeatureBuilder, make_X  # noqa: E402
from models import CLASSIFIERS, MarginModel  # noqa: E402

ALL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LA", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB",
    "TEN", "WAS",
]  # nflverse abbreviations (Rams = LA, not LAR)


def fmt_kickoff(ts: pd.Timestamp) -> str:
    return ts.strftime("%a %m/%d %I:%M %p").replace(" 0", " ")


def predict_week(week: int | None = None, season: int = 2026) -> pd.DataFrame:
    schedules = get_schedules()
    _, team_games = get_team_games()

    sched = schedules[
        (schedules["season"] == season) & (schedules["game_type"] == "REG")
    ].copy()
    unplayed = sched[sched["home_score"].isna()]
    if unplayed.empty:
        raise SystemExit(f"No unplayed {season} regular-season games found.")
    if week is None:
        week = int(unplayed["week"].min())
    week_games = unplayed[unplayed["week"] == week].sort_values("kickoff")
    if week_games.empty:
        raise SystemExit(f"Week {week} has no unplayed games (already played?).")

    # Train on everything completed.
    fb = FeatureBuilder(team_games)
    games = fb.build_games(game_type="REG")
    X, y = make_X(games), games["home_win"]
    models = [cls().fit(X, y) for cls in CLASSIFIERS]
    margin_model = MarginModel().fit(X, games["home_margin"])

    rows = []
    for _, g in week_games.iterrows():
        neutral = 1 if str(g.get("location", "")).lower() == "neutral" else 0
        Xf = fb.features_for_matchup(
            g["home_team"], g["away_team"], g["kickoff"], season,
            is_neutral=neutral,
            home_rest=g.get("home_rest"), away_rest=g.get("away_rest"),
        )
        probs = {m.name: float(m.predict_proba(Xf)[0, 1]) for m in models}
        pred_margin = float(margin_model.predict_margin(Xf)[0])
        pred_spread_home = -pred_margin  # fair spread, home-team perspective
        market_spread_home = -g["spread_line"] if pd.notna(g["spread_line"]) else None
        edge = (market_spread_home - pred_spread_home) if market_spread_home is not None else float("nan")
        rows.append({
            "matchup": f"{g['away_team']} @ {g['home_team']}",
            "kickoff_et": fmt_kickoff(g["kickoff"]),
            "p_home_elo": round(probs["elo"], 3),
            "p_home_logreg": round(probs["logreg"], 3),
            "p_home_hgb": round(probs["hgb"], 3),
            "pred_spread": round(pred_spread_home, 1),
            "market_spread": round(market_spread_home, 1) if market_spread_home is not None else None,
            "edge_pts": round(edge, 1) if pd.notna(edge) else None,
        })

    result = pd.DataFrame(rows)
    playing = set(week_games["home_team"]) | set(week_games["away_team"])
    byes = sorted(set(ALL_TEAMS) - playing)
    result.attrs["week"] = week
    result.attrs["byes"] = byes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the upcoming NFL week.")
    parser.add_argument("--week", type=int, default=None, help="2026 week to predict")
    args = parser.parse_args()

    print("Loading data, building features, training models…", flush=True)
    table = predict_week(args.week)
    week, byes = table.attrs["week"], table.attrs["byes"]

    print(f"\nWeek {week} predictions (all times ET)")
    print("=" * 96)
    print(table.to_string(index=False))
    if byes:
        print(f"\nByes: {', '.join(byes)}")
    print("\nedge_pts: market_spread - predicted_spread (home perspective). "
          "+edge = value on home, -edge = value on away.")


if __name__ == "__main__":
    main()
