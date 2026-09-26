import unittest
from unittest import mock

from weather import client


PAYLOAD = {
    "hourly": {
        "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
        "temperature_2m": [25.1, 24.8],
    }
}


class ParseHourlyTest(unittest.TestCase):
    def test_parses_columns(self):
        series = client.parse_hourly(PAYLOAD)
        self.assertEqual(series.times, ["2026-01-01T00:00", "2026-01-01T01:00"])
        self.assertEqual(series.column("temperature_2m"), [25.1, 24.8])

    def test_rejects_missing_hourly(self):
        with self.assertRaises(ValueError):
            client.parse_hourly({})

    def test_rejects_misaligned_column(self):
        bad = {"hourly": {"time": ["a", "b"], "temperature_2m": [1.0]}}
        with self.assertRaises(ValueError):
            client.parse_hourly(bad)


class FetchTest(unittest.TestCase):
    def test_fetch_forecast_builds_query(self):
        with mock.patch.object(client, "_get_json", return_value=PAYLOAD) as get:
            client.fetch_forecast(13.75, 100.5, variables=("temperature_2m",), days=2)
        url = get.call_args.args[0]
        self.assertTrue(url.startswith(client.FORECAST_URL))
        self.assertIn("latitude=13.75", url)
        self.assertIn("hourly=temperature_2m", url)
        self.assertIn("forecast_days=2", url)


if __name__ == "__main__":
    unittest.main()
