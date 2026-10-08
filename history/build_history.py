#!/usr/bin/env python3
"""Pull public nflverse history and write the joinable tables in history/.

Sources, all public, no key:
  games.csv          https://github.com/nflverse/nfldata  (1999-present lines, weather, QB, coach)
  closing_lines.csv  same repo, stale, ends 2018
  play_by_play_YYYY  https://github.com/nflverse/nflverse-data/releases/download/pbp/

Outputs next to this file:
  market_games.parquet
  team_games.parquet
  qb_games.parquet
  team_games_joined.parquet
  closing_lines_2006_2018.parquet

Raw pbp is cached under ./cache and is not committed. Re-run anytime; existing
year files are skipped.

    python history/build_history.py
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = pathlib.Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUT = ROOT

PBP_YEARS = list(range(2006, 2027))
# pbp uses today's abbreviation. Schedule team columns keep the code from that season.
HIST_TO_PBP = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CLOSING_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/closing_lines.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"

WANT = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
    "epa", "success", "qb_kneel", "qb_spike", "passer_player_id", "passer_player_name",
    "pass_attempt", "rush_attempt", "sack", "down", "cpoe", "air_yards",
]


def _download(url: str, dest: pathlib.Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {dest.name}", flush=True)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    dest.write_bytes(r.content)


def fetch_pbp(year: int) -> pd.DataFrame:
    path = CACHE / f"play_by_play_{year}.parquet"
    _download(PBP_URL.format(year=year), path)
    schema = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in WANT if c in schema])


def team_and_qb(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pbp.copy()
    for c in ("qb_kneel", "qb_spike", "pass_attempt", "rush_attempt", "sack", "down"):
        if c not in df.columns:
            df[c] = 0
    if "cpoe" not in df.columns:
        df["cpoe"] = float("nan")
    if "air_yards" not in df.columns:
        df["air_yards"] = float("nan")
    if "season_type" not in df.columns:
        df["season_type"] = "REG"

    plays = df[
        df["play_type"].isin(["pass", "run"])
        & df["epa"].notna()
        & (df["qb_kneel"].fillna(0) == 0)
        & (df["qb_spike"].fillna(0) == 0)
        & df["posteam"].notna()
    ].copy()
    plays["early"] = plays["down"].isin([1, 2])
    plays["is_pass"] = plays["play_type"].eq("pass")
    plays["is_run"] = plays["play_type"].eq("run")

    off = (
        plays.groupby(["game_id", "season", "week", "season_type", "posteam"], observed=True)
        .apply(lambda x: pd.Series({
            "off_plays": len(x),
            "off_epa": x["epa"].sum(),
            "off_success": x["success"].mean(),
            "early_plays": int(x["early"].sum()),
            "early_epa": x.loc[x["early"], "epa"].sum(),
            "pass_plays": int(x["is_pass"].sum()),
            "pass_epa": x.loc[x["is_pass"], "epa"].sum(),
            "rush_plays": int(x["is_run"].sum()),
            "rush_epa": x.loc[x["is_run"], "epa"].sum(),
        }), include_groups=False)
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    defense = (
        plays.groupby(["game_id", "defteam"], observed=True)
        .apply(lambda x: pd.Series({
            "def_plays": len(x),
            "def_epa_allowed": x["epa"].sum(),
            "def_success_allowed": x["success"].mean(),
        }), include_groups=False)
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    team = off.merge(defense, on=["game_id", "team"], how="left")
    for num, den, name in (
        ("off_epa", "off_plays", "off_epa_pp"),
        ("early_epa", "early_plays", "early_epa_pp"),
        ("pass_epa", "pass_plays", "pass_epa_pp"),
        ("rush_epa", "rush_plays", "rush_epa_pp"),
        ("def_epa_allowed", "def_plays", "def_epa_pp"),
    ):
        team[name] = team[num] / team[den].replace(0, pd.NA)

    drops = plays[(plays["pass_attempt"].fillna(0) == 1) | (plays["sack"].fillna(0) == 1)]
    qb = (
        drops.groupby(
            ["game_id", "season", "week", "season_type", "posteam", "passer_player_id", "passer_player_name"],
            observed=True,
        )
        .apply(lambda x: pd.Series({
            "dropbacks": len(x),
            "qb_epa": x["epa"].sum(),
            "qb_epa_pp": x["epa"].mean(),
            "qb_success": x["success"].mean(),
            "cpoe": x["cpoe"].mean(),
            "air_yards": x["air_yards"].sum(),
        }), include_groups=False)
        .reset_index()
        .rename(columns={"posteam": "team", "passer_player_id": "qb_id", "passer_player_name": "qb_name"})
    )
    qb = qb[qb["qb_id"].notna()].copy()
    qb["is_primary"] = (
        qb.groupby(["game_id", "team"])["dropbacks"].rank(method="first", ascending=False).eq(1)
    )
    return team, qb


def _side(games: pd.DataFrame, home: bool) -> pd.DataFrame:
    team_col, opp_col = ("team_home_pbp", "team_away_pbp") if home else ("team_away_pbp", "team_home_pbp")
    sched_team, sched_opp = ("home_team", "away_team") if home else ("away_team", "home_team")
    qb_id, qb_name = ("home_qb_id", "home_qb_name") if home else ("away_qb_id", "away_qb_name")
    rest, opp_rest = ("home_rest", "away_rest") if home else ("away_rest", "home_rest")
    coach = "home_coach" if home else "away_coach"
    return pd.DataFrame({
        "game_id": games["game_id"],
        "old_game_id": games["old_game_id"],
        "season": games["season"],
        "week": games["week"],
        "game_type": games["game_type"],
        "kickoff": games["kickoff"],
        "team": games[team_col],
        "team_schedule": games[sched_team],
        "opp": games[opp_col],
        "opp_schedule": games[sched_opp],
        "is_home": 1 if home else 0,
        "starter_qb_id": games[qb_id],
        "starter_qb_name": games[qb_name],
        "rest": games[rest],
        "opp_rest": games[opp_rest],
        "coach": games[coach],
        "spread_line": games["spread_line"],
        "total_line": games["total_line"],
        "home_moneyline": games["home_moneyline"],
        "away_moneyline": games["away_moneyline"],
        "home_spread_odds": games["home_spread_odds"],
        "away_spread_odds": games["away_spread_odds"],
        "div_game": games["div_game"],
        "roof": games["roof"],
        "surface": games["surface"],
        "temp": games["temp"],
        "wind": games["wind"],
        "location": games["location"],
        "stadium": games["stadium"],
        "home_score": games["home_score"],
        "away_score": games["away_score"],
    })


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    _download(GAMES_URL, CACHE / "games.csv")
    _download(CLOSING_URL, CACHE / "closing_lines.csv")

    games = pd.read_csv(CACHE / "games.csv")
    games["kickoff"] = pd.to_datetime(
        games["gameday"].astype(str) + " " + games["gametime"].fillna("13:00").astype(str),
        errors="coerce",
    )
    games["team_home_pbp"] = games["home_team"].replace(HIST_TO_PBP)
    games["team_away_pbp"] = games["away_team"].replace(HIST_TO_PBP)
    games.to_parquet(OUT / "market_games.parquet", index=False)
    pd.read_csv(CACHE / "closing_lines.csv").to_parquet(OUT / "closing_lines_2006_2018.parquet", index=False)
    print(f"market_games {len(games)} seasons {games.season.min()}-{games.season.max()}", flush=True)

    team_frames, qb_frames = [], []
    for year in PBP_YEARS:
        team, qb = team_and_qb(fetch_pbp(year))
        team_frames.append(team)
        qb_frames.append(qb)
        print(f"  {year}: team-rows {len(team)} qb-rows {len(qb)}", flush=True)

    team_games = pd.concat(team_frames, ignore_index=True)
    qb_games = pd.concat(qb_frames, ignore_index=True)
    team_games.to_parquet(OUT / "team_games.parquet", index=False)
    qb_games.to_parquet(OUT / "qb_games.parquet", index=False)

    primary = qb_games[qb_games["is_primary"]][[
        "game_id", "team", "qb_id", "qb_name", "dropbacks", "qb_epa_pp", "qb_success", "cpoe",
    ]].rename(columns={
        "qb_id": "primary_qb_id",
        "qb_name": "primary_qb_name",
        "dropbacks": "primary_dropbacks",
        "qb_epa_pp": "primary_qb_epa_pp",
        "qb_success": "primary_qb_success",
        "cpoe": "primary_cpoe",
    })
    side = pd.concat([_side(games, True), _side(games, False)], ignore_index=True)
    joined = team_games.merge(primary, on=["game_id", "team"], how="left")
    joined = joined.merge(side, on=["game_id", "team"], how="left")
    joined.to_parquet(OUT / "team_games_joined.parquet", index=False)
    matched = joined["spread_line"].notna().mean()
    print(f"joined {len(joined)} rows, market match {matched:.1%}", flush=True)
    if matched < 0.99:
        raise SystemExit("join dropped games — team-code map is stale")


if __name__ == "__main__":
    main()
