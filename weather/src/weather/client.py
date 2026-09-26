"""Client for the Open-Meteo API (free, no API key required)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_VARIABLES = ("temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m")


@dataclass
class HourlySeries:
    """Hourly observations: one list of values per variable, aligned with `times`."""

    times: list[str]
    values: dict[str, list[float | None]]

    def column(self, name: str) -> list[float | None]:
        return self.values[name]


def build_url(base: str, params: dict[str, object]) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"


def parse_hourly(payload: dict) -> HourlySeries:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("response has no hourly data")
    times = hourly["time"]
    values = {k: v for k, v in hourly.items() if k != "time"}
    for name, column in values.items():
        if len(column) != len(times):
            raise ValueError(f"column {name!r} length {len(column)} != {len(times)} timestamps")
    return HourlySeries(times=times, values=values)


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def fetch_forecast(
    latitude: float,
    longitude: float,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    days: int = 7,
    timezone: str = "auto",
    timeout: float = 30.0,
) -> HourlySeries:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "forecast_days": days,
        "timezone": timezone,
    }
    return parse_hourly(_get_json(build_url(FORECAST_URL, params), timeout))


def fetch_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    timezone: str = "auto",
    timeout: float = 60.0,
) -> HourlySeries:
    """Fetch historical hourly data; dates are ISO `YYYY-MM-DD`."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
    }
    return parse_hourly(_get_json(build_url(ARCHIVE_URL, params), timeout))
