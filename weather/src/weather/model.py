"""Baseline forecasters for an hourly series (pure Python, no dependencies).

Every model exposes `fit(series)` and `predict(history, horizon)` so they can
be compared with `evaluate` and swapped for an ML model later.
"""

from __future__ import annotations

import math


def clean(series: list[float | None]) -> list[float]:
    """Fill gaps by carrying the last value forward (leading gaps take the first value)."""
    first = next((v for v in series if v is not None), None)
    if first is None:
        raise ValueError("series has no values")
    out, last = [], first
    for v in series:
        last = v if v is not None else last
        out.append(last)
    return out


class Persistence:
    """Tomorrow looks like today: repeats the last `season` values."""

    def __init__(self, season: int = 24):
        self.season = season

    def fit(self, series: list[float]) -> "Persistence":
        return self

    def predict(self, history: list[float], horizon: int) -> list[float]:
        if len(history) < self.season:
            raise ValueError(f"need at least {self.season} values of history")
        window = history[-self.season:]
        return [window[i % self.season] for i in range(horizon)]


class AutoRegressive:
    """Linear autoregression on the last `lags` values, fit by least squares."""

    def __init__(self, lags: int = 24, ridge: float = 1e-3):
        self.lags = lags
        self.ridge = ridge
        self.coef: list[float] | None = None

    def fit(self, series: list[float]) -> "AutoRegressive":
        n = self.lags + 1  # lags + intercept
        if len(series) <= self.lags:
            raise ValueError(f"need more than {self.lags} values to fit")
        xtx = [[0.0] * n for _ in range(n)]
        xty = [0.0] * n
        for t in range(self.lags, len(series)):
            row = series[t - self.lags:t] + [1.0]
            for i in range(n):
                xty[i] += row[i] * series[t]
                for j in range(n):
                    xtx[i][j] += row[i] * row[j]
        for i in range(self.lags):
            xtx[i][i] += self.ridge
        self.coef = _solve(xtx, xty)
        return self

    def predict(self, history: list[float], horizon: int) -> list[float]:
        if self.coef is None:
            raise RuntimeError("call fit() first")
        window = list(history[-self.lags:])
        if len(window) < self.lags:
            raise ValueError(f"need at least {self.lags} values of history")
        out = []
        for _ in range(horizon):
            y = sum(c * x for c, x in zip(self.coef, window + [1.0]))
            out.append(y)
            window = window[1:] + [y]
        return out


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def evaluate(model, series: list[float], horizon: int = 24) -> dict[str, float]:
    """Fit on everything except the last `horizon` points, score on those."""
    train, test = series[:-horizon], series[-horizon:]
    pred = model.fit(train).predict(train, horizon)
    return {"mae": mae(test, pred), "rmse": rmse(test, pred)}
