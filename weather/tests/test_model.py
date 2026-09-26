import math
import unittest

from weather.model import AutoRegressive, Persistence, clean, evaluate, mae, rmse


def daily_cycle(hours: int) -> list[float]:
    return [28 + 4 * math.sin(2 * math.pi * h / 24) for h in range(hours)]


class CleanTest(unittest.TestCase):
    def test_forward_fills_gaps(self):
        self.assertEqual(clean([None, 1.0, None, 3.0]), [1.0, 1.0, 1.0, 3.0])

    def test_all_missing_raises(self):
        with self.assertRaises(ValueError):
            clean([None, None])


class ModelTest(unittest.TestCase):
    def test_persistence_repeats_last_season(self):
        series = list(range(48))
        self.assertEqual(Persistence(season=24).predict(series, 3), [24, 25, 26])

    def test_persistence_is_exact_on_periodic_series(self):
        scores = evaluate(Persistence(), daily_cycle(24 * 10))
        self.assertAlmostEqual(scores["mae"], 0.0, places=9)

    def test_autoregressive_learns_periodic_series(self):
        scores = evaluate(AutoRegressive(lags=24), daily_cycle(24 * 10))
        self.assertLess(scores["mae"], 0.05)

    def test_autoregressive_requires_fit(self):
        with self.assertRaises(RuntimeError):
            AutoRegressive().predict([0.0] * 24, 1)


class MetricsTest(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(mae([1, 2], [2, 4]), 1.5)
        self.assertAlmostEqual(rmse([1, 2], [2, 4]), math.sqrt(2.5))


if __name__ == "__main__":
    unittest.main()
