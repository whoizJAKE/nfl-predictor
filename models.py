"""Models behind a common fit / predict_proba interface.

- EloBaseline: no training; fixed 538-style formula on the elo_diff feature.
- LogisticModel: L2 logistic regression on the feature set (scaled).
- GradientBoostingModel: HistGradientBoostingClassifier on the feature set.
- MarginModel: HistGradientBoostingRegressor predicting home margin, used for
  spread comparison (fair spread = -predicted margin, home-team perspective).

All classifiers return predict_proba with columns [P(away win), P(home win)].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import ELO_HFA, ELO_START, elo_win_prob


class EloBaseline:
    """Fixed-formula baseline: win prob from pre-game Elo diff (+ home edge)."""

    name = "elo"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EloBaseline":
        return self  # nothing to learn

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # elo_win_prob(a, b) depends only on (a - b); shifting both by
        # ELO_START keeps the call exact: (ELO_START + diff) - ELO_START == diff.
        hfa = np.where(X["is_neutral"] == 1.0, 0.0, ELO_HFA)
        p_home = np.array([
            elo_win_prob(ELO_START + d, ELO_START, h)
            for d, h in zip(X["elo_diff"], hfa)
        ])
        return np.column_stack([1.0 - p_home, p_home])


class LogisticModel:
    """L2 logistic regression on standardized features."""

    name = "logreg"

    def __init__(self):
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(X)


class GradientBoostingModel:
    """Histogram gradient boosting classifier (handles raw feature scales)."""

    name = "hgb"

    def __init__(self, random_state: int = 42):
        self.clf = HistGradientBoostingClassifier(random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X)


class MarginModel:
    """Regresses home-team margin; fair spread = -predicted margin."""

    name = "margin"

    def __init__(self, random_state: int = 42):
        self.reg = HistGradientBoostingRegressor(random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarginModel":
        self.reg.fit(X, y)
        return self

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(X)


CLASSIFIERS = [EloBaseline, LogisticModel, GradientBoostingModel]
