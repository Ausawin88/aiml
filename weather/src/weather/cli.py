"""Command line entry point: `python -m weather.cli --help`."""

from __future__ import annotations

import argparse
import datetime as dt

from . import client
from .model import AutoRegressive, Persistence, clean, evaluate

# Bangkok
DEFAULT_LAT, DEFAULT_LON = 13.7563, 100.5018


def cmd_forecast(args: argparse.Namespace) -> None:
    data = client.fetch_forecast(args.lat, args.lon, days=args.days)
    temps = data.column("temperature_2m")
    for t, v in list(zip(data.times, temps))[: args.hours]:
        print(f"{t}  {'n/a' if v is None else f'{v:.1f}':>6} °C")


def cmd_evaluate(args: argparse.Namespace) -> None:
    end = dt.date.today() - dt.timedelta(days=5)  # archive lags a few days
    start = end - dt.timedelta(days=args.days)
    data = client.fetch_history(args.lat, args.lon, start.isoformat(), end.isoformat())
    series = clean(data.column(args.variable))
    for name, model in [("persistence", Persistence()), ("autoregressive", AutoRegressive())]:
        scores = evaluate(model, series, horizon=args.horizon)
        print(f"{name:<15} MAE={scores['mae']:.3f}  RMSE={scores['rmse']:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="weather")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("forecast", help="show the Open-Meteo hourly temperature forecast")
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--hours", type=int, default=24)
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("evaluate", help="backtest baseline models on recent history")
    p.add_argument("--days", type=int, default=60, help="days of history to fetch")
    p.add_argument("--horizon", type=int, default=24, help="hours to forecast")
    p.add_argument("--variable", default="temperature_2m")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
