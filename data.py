"""Download and cache nflverse data locally as parquet.

Everything downstream reads from ``data/`` so the pipeline never re-downloads
unless you pass ``refresh=True``. Two artifacts are cached:

- ``schedules_<year>.parquet`` — one row per game (scores, spread_line, rest, …)
- ``team_games_<year>.parquet`` — one row per team per game, with team-level
  aggregates computed from play-by-play (offensive/defensive EPA per play,
  success rates, points). v1 is team-level on purpose; player-level data is
  the extension point for the props/fantasy module (see README Roadmap).
"""
from __future__ import annotations

import pathlib

import nfl_data_py as nfl
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
SEASONS = list(range(2020, 2027))  # 2020 .. 2026 (current season, partial)

# Only genuine offensive snaps count toward EPA aggregates.
PLAY_TYPES = ["pass", "run"]


def _load_or_fetch(path: pathlib.Path, fetch) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def get_schedules(years: list[int] = SEASONS, refresh: bool = False) -> pd.DataFrame:
    """League schedules with scores, lines and rest days, cached per season."""
    frames = []
    for year in years:
        path = DATA_DIR / f"schedules_{year}.parquet"

        def fetch(y=year):
            df = nfl.import_schedules([y])
            # Normalize kickoff to a single tz-naive ET timestamp for ordering.
            df["kickoff"] = pd.to_datetime(
                df["gameday"] + " " + df["gametime"].fillna("13:00"),
                errors="coerce",
            )
            return df

        if refresh and path.exists():
            path.unlink()
        frames.append(_load_or_fetch(path, fetch))
    sched = pd.concat(frames, ignore_index=True)
    return sched.sort_values("kickoff").reset_index(drop=True)


def get_pbp(years: list[int] = SEASONS, refresh: bool = False) -> pd.DataFrame:
    """Raw play-by-play, cached per season. Heavy — prefer team_games_*."""
    frames = []
    for year in years:
        path = DATA_DIR / f"pbp_{year}.parquet"

        def fetch(y=year):
            cols = [
                "game_id", "season", "week", "posteam", "defteam", "play_type",
                "epa", "success", "qb_kneel", "qb_spike", "wp",
            ]
            return nfl.import_pbp_data([y], columns=cols, downcast=True)

        if refresh and path.exists():
            path.unlink()
        frames.append(_load_or_fetch(path, fetch))
    return pd.concat(frames, ignore_index=True)


def build_team_games(
    schedules: pd.DataFrame, pbp: pd.DataFrame
) -> pd.DataFrame:
    """One row per team per game with team-level aggregates.

    Offensive numbers describe what the team's offense did; defensive numbers
    describe what its defense allowed. Points come from the schedule.
    """
    plays = pbp[
        pbp["play_type"].isin(PLAY_TYPES)
        & pbp["epa"].notna()
        & (pbp["qb_kneel"].fillna(0) == 0)
        & (pbp["qb_spike"].fillna(0) == 0)
    ].copy()

    off = (
        plays.groupby(["game_id", "posteam"], observed=True)
        .agg(
            off_plays=("epa", "size"),
            off_epa=("epa", "sum"),
            off_success=("success", "mean"),
        )
        .reset_index()
    )
    off["off_epa_pp"] = off["off_epa"] / off["off_plays"]

    defense = (
        plays.groupby(["game_id", "defteam"], observed=True)
        .agg(
            def_plays=("epa", "size"),
            def_epa_allowed=("epa", "sum"),
            def_success_allowed=("success", "mean"),
        )
        .reset_index()
    )
    defense["def_epa_pp"] = defense["def_epa_allowed"] / defense["def_plays"]

    rows = []
    for _, g in schedules.iterrows():
        base = {
            "game_id": g["game_id"],
            "season": g["season"],
            "week": g["week"],
            "game_type": g.get("game_type", "REG"),
            "kickoff": g["kickoff"],
            "spread_line": g.get("spread_line"),
            "total_line": g.get("total_line"),
            "is_neutral": 1 if str(g.get("location", "")).lower() == "neutral" else 0,
        }
        for side, team_col, score_col, opp_col, rest_col in [
            ("home", "home_team", "home_score", "away_team", "home_rest"),
            ("away", "away_team", "away_score", "home_team", "away_rest"),
        ]:
            team = g[team_col]
            o = off[(off["game_id"] == g["game_id"]) & (off["posteam"] == team)]
            d = defense[(defense["game_id"] == g["game_id"]) & (defense["defteam"] == team)]
            rows.append(
                {
                    **base,
                    "team": team,
                    "opp": g[opp_col],
                    "is_home": 1 if side == "home" else 0,
                    "points_for": g[score_col],
                    "points_against": g["away_score" if side == "home" else "home_score"],
                    "rest": g.get(rest_col),
                    "off_epa_pp": float(o["off_epa_pp"].iloc[0]) if len(o) else float("nan"),
                    "off_success": float(o["off_success"].iloc[0]) if len(o) else float("nan"),
                    "def_epa_pp": float(d["def_epa_pp"].iloc[0]) if len(d) else float("nan"),
                    "def_success_allowed": float(d["def_success_allowed"].iloc[0])
                    if len(d)
                    else float("nan"),
                }
            )
    team_games = pd.DataFrame(rows).sort_values("kickoff").reset_index(drop=True)
    return team_games


def get_team_games(
    years: list[int] = SEASONS, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (schedules, team_games), building the cache on first run."""
    schedules = get_schedules(years, refresh=refresh)
    frames = []
    for year in years:
        path = DATA_DIR / f"team_games_{year}.parquet"
        if refresh and path.exists():
            path.unlink()
        if not path.exists():
            pbp = get_pbp([year])
            tg = build_team_games(
                schedules[schedules["season"] == year], pbp
            )
            tg.to_parquet(path, index=False)
        frames.append(pd.read_parquet(path))
    team_games = pd.concat(frames, ignore_index=True)
    return schedules, team_games.sort_values("kickoff").reset_index(drop=True)


if __name__ == "__main__":
    import sys

    refresh = "--refresh" in sys.argv
    sched, tg = get_team_games(refresh=refresh)
    print(f"schedules: {len(sched)} games, team_games: {len(tg)} team-games")
    print(f"seasons: {sorted(sched['season'].unique())}")
