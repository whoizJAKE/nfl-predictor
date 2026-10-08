"""Causally-safe per-game features.

Every feature for a game uses only information available *before* that game's
kickoff — rolling windows are shifted by one game, Elo ratings are pre-game.

Season-boundary policy (documented v1 choice): rolling 8-game windows carry
across seasons with NO reset and NO decay, so September games lean on the
prior season's form. Elo instead regresses 1/3 toward 1500 each offseason
(538-style). Rationale: a hard reset would leave early-season games with
~2-game samples; carrying over is the lesser evil, and Elo's reversion
handles regime change. A mean-reversion blend is a natural v2 upgrade.

The FEATURES registry maps a feature name -> function(games_df) -> Series.
To add a feature, write a function of the games frame and register it; the
models and backtest pick it up automatically via FEATURE_COLS.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

ROLL_N = 8  # rolling window: last 8 games
ELO_START = 1500.0
ELO_K = 20.0
ELO_HFA = 55.0  # home-field edge in Elo points (constant v1 simplification)
ELO_CARRYOVER = 2 / 3  # offseason: keep 2/3 of rating, revert 1/3 to 1500

ROLL_STATS = [
    "off_epa_pp",
    "off_success",
    "def_epa_pp",
    "def_success_allowed",
    "points_for",
    "points_against",
]

# Constants used ONLY to fill the first games of the dataset (2020 Week 1),
# where no prior games exist. They are fixed, so no leakage.
FILL_VALUES = {
    "off_epa_pp": 0.0,
    "off_success": 0.45,
    "def_epa_pp": 0.0,
    "def_success_allowed": 0.45,
    "points_for": 22.0,
    "points_against": 22.0,
}


# ---------------------------------------------------------------------------
# 538-style margin-aware Elo
# ---------------------------------------------------------------------------
def elo_win_prob(elo_a: float, elo_b: float, hfa: float = ELO_HFA) -> float:
    """Pre-game win probability for A over B (hfa added when A is home)."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b + hfa) / 400.0))


def elo_update(
    elo_w: float, elo_l: float, margin: float, winner_is_home: bool, neutral: bool
) -> tuple[float, float]:
    """Margin-of-victory-multiplied Elo update (538 NFL formula)."""
    hfa = 0.0 if neutral else (ELO_HFA if winner_is_home else -ELO_HFA)
    expected_w = 1.0 / (1.0 + 10.0 ** (-((elo_w - elo_l) + hfa) / 400.0))
    denom = (elo_w - elo_l) * 0.001 + 2.2
    mov_mult = math.log(abs(margin) + 1.0) * (2.2 / max(denom, 0.5))
    delta = ELO_K * mov_mult * (1.0 - expected_w)
    return elo_w + delta, elo_l - delta


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------
FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def feature(name: str):
    """Decorator registering a per-game feature builder."""
    def decorator(fn: Callable[[pd.DataFrame], pd.Series]):
        FEATURES[name] = fn
        return fn
    return decorator


@feature("elo_diff")
def _elo_diff(g: pd.DataFrame) -> pd.Series:
    return g["elo_home"] - g["elo_away"]


@feature("off_epa_diff")
def _off_epa_diff(g: pd.DataFrame) -> pd.Series:
    return g["home_r_off_epa_pp"] - g["away_r_off_epa_pp"]


@feature("def_epa_diff")
def _def_epa_diff(g: pd.DataFrame) -> pd.Series:
    # Defensive EPA allowed: lower is better, so flip the sign.
    return g["away_r_def_epa_pp"] - g["home_r_def_epa_pp"]


@feature("off_success_diff")
def _off_success_diff(g: pd.DataFrame) -> pd.Series:
    return g["home_r_off_success"] - g["away_r_off_success"]


@feature("def_success_diff")
def _def_success_diff(g: pd.DataFrame) -> pd.Series:
    return g["away_r_def_success_allowed"] - g["home_r_def_success_allowed"]


@feature("net_points_diff")
def _net_points_diff(g: pd.DataFrame) -> pd.Series:
    home_net = g["home_r_points_for"] - g["home_r_points_against"]
    away_net = g["away_r_points_for"] - g["away_r_points_against"]
    return home_net - away_net


@feature("rest_diff")
def _rest_diff(g: pd.DataFrame) -> pd.Series:
    return (g["home_rest"] - g["away_rest"]).fillna(0.0)


@feature("is_neutral")
def _is_neutral(g: pd.DataFrame) -> pd.Series:
    return g["is_neutral"].astype(float)


FEATURE_COLS: list[str] = list(FEATURES)


def make_X(games: pd.DataFrame) -> pd.DataFrame:
    """Assemble the model feature matrix from the games frame."""
    return pd.DataFrame({name: FEATURES[name](games) for name in FEATURE_COLS})


# ---------------------------------------------------------------------------
# Builder: rolling stats + Elo, then one row per game
# ---------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, team_games: pd.DataFrame):
        self.tg = team_games.sort_values("kickoff").reset_index(drop=True).copy()
        self._add_rolling()
        self._add_elo()

    # -- rolling -----------------------------------------------------------
    def _add_rolling(self) -> None:
        tg = self.tg
        for stat in ROLL_STATS:
            col = f"r_{stat}"
            tg[col] = (
                tg.groupby("team", observed=True)[stat]
                .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean())
            )
            tg[col] = tg[col].fillna(FILL_VALUES[stat])
        self.tg = tg

    # -- Elo ---------------------------------------------------------------
    def _add_elo(self) -> None:
        """Sequential pre-game Elo for every team-game row (causal by construction).

        Iterates game-by-game (not row-by-row) so that *both* teams' pre-game
        ratings are recorded before either rating is updated.
        """
        ratings: dict[str, float] = {}
        last_season: dict[str, int] = {}
        elo_before: dict[int, float] = {}
        elo_after: dict[int, float] = {}

        for _, grp in self.tg.groupby("game_id", sort=False):
            by_side = {r.is_home: r for r in grp.itertuples()}
            if len(by_side) != 2:
                continue
            home, away = by_side[1], by_side[0]

            for r in (home, away):
                team, season = r.team, int(r.season)
                if team not in ratings:
                    ratings[team] = ELO_START
                    last_season[team] = season
                elif season != last_season[team]:
                    # Offseason regression toward the mean (538-style).
                    ratings[team] = ELO_START + (ratings[team] - ELO_START) * ELO_CARRYOVER
                    last_season[team] = season
                elo_before[r.Index] = ratings[team]
                elo_after[r.Index] = ratings[team]  # overwritten below if game played

            # Post-game update, only when the game has a result (no ties).
            if pd.notna(home.points_for) and home.points_for != home.points_against:
                margin = abs(home.points_for - home.points_against)
                neutral = bool(home.is_neutral)
                if home.points_for > home.points_against:
                    new_h, new_a = elo_update(
                        ratings[home.team], ratings[away.team],
                        margin, True, neutral,
                    )
                else:
                    new_a, new_h = elo_update(
                        ratings[away.team], ratings[home.team],
                        margin, False, neutral,
                    )
                ratings[home.team], ratings[away.team] = new_h, new_a
                elo_after[home.Index] = new_h
                elo_after[away.Index] = new_a

        self.tg["elo_before"] = self.tg.index.map(elo_before)
        self.tg["elo_after"] = self.tg.index.map(elo_after)

    # -- games frame --------------------------------------------------------
    def build_games(self, game_type: str = "REG") -> pd.DataFrame:
        """One row per game with home/away features and targets (REG only by default)."""
        tg = self.tg
        home = tg[tg["is_home"] == 1].copy()
        away = tg[tg["is_home"] == 0].copy()
        if game_type:
            home = home[home["game_type"] == game_type]
            away = away[away["game_type"] == game_type]

        games = home[
            ["game_id", "season", "week", "kickoff", "spread_line", "total_line",
             "is_neutral", "points_for", "points_against"]
        ].rename(columns={"points_for": "home_score", "points_against": "away_score"})
        games = games.merge(
            away[["game_id", "team"]].rename(columns={"team": "away_team"}),
            on="game_id", how="left",
        )
        games = games.merge(
            home[["game_id", "team"]].rename(columns={"team": "home_team"}),
            on="game_id", how="left",
        )
        for prefix in ("home", "away"):
            src = home if prefix == "home" else away
            cols = ["game_id"] + [f"r_{s}" for s in ROLL_STATS] + ["elo_before"]
            rename = {f"r_{s}": f"{prefix}_r_{s}" for s in ROLL_STATS}
            rename["elo_before"] = f"elo_{prefix}"
            m = src[cols].rename(columns=rename)
            games = games.merge(m, on="game_id", how="left")

        rest = home[["game_id", "rest"]].rename(columns={"rest": "home_rest"})
        games = games.merge(rest, on="game_id", how="left")
        rest = away[["game_id", "rest"]].rename(columns={"rest": "away_rest"})
        games = games.merge(rest, on="game_id", how="left")

        games["home_margin"] = games["home_score"] - games["away_score"]
        games = games[games["home_score"].notna()]  # only completed games
        games["home_win"] = (games["home_margin"] > 0).astype(int)
        games = games[games["home_margin"] != 0]  # drop ties from modeling
        return games.sort_values("kickoff").reset_index(drop=True)

    # -- upcoming games ------------------------------------------------------
    def current_elo(self, team: str, season: int) -> float:
        """Latest post-game Elo for a team, regressed if a new season started."""
        rows = self.tg[(self.tg["team"] == team) & self.tg["elo_after"].notna()]
        if not rows.empty:
            last = rows.iloc[-1]
            elo = float(last["elo_after"])
            if int(last["season"]) < season:
                elo = ELO_START + (elo - ELO_START) * ELO_CARRYOVER
            return elo
        return ELO_START

    def rolling_for(self, team: str, kickoff: pd.Timestamp) -> dict[str, float]:
        """Last-8 rolling stats for a team using only completed games before kickoff."""
        past = self.tg[
            (self.tg["team"] == team)
            & (self.tg["kickoff"] < kickoff)
            & self.tg["points_for"].notna()
        ].tail(ROLL_N)
        out = {}
        for stat in ROLL_STATS:
            vals = past[stat].dropna()
            out[f"r_{stat}"] = float(vals.mean()) if len(vals) else FILL_VALUES[stat]
        return out

    def features_for_matchup(
        self,
        home_team: str,
        away_team: str,
        kickoff: pd.Timestamp,
        season: int,
        is_neutral: int = 0,
        home_rest: float | None = None,
        away_rest: float | None = None,
    ) -> pd.DataFrame:
        """Single-row feature matrix for a future game (same columns as make_X)."""
        h = self.rolling_for(home_team, kickoff)
        a = self.rolling_for(away_team, kickoff)
        row = {
            "elo_home": self.current_elo(home_team, season),
            "elo_away": self.current_elo(away_team, season),
            "is_neutral": is_neutral,
            "home_rest": home_rest if home_rest is not None else 7.0,
            "away_rest": away_rest if away_rest is not None else 7.0,
        }
        for stat in ROLL_STATS:
            row[f"home_r_{stat}"] = h[f"r_{stat}"]
            row[f"away_r_{stat}"] = a[f"r_{stat}"]
        return make_X(pd.DataFrame([row]))
