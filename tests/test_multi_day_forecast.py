from pathlib import Path

import pandas as pd

from src.inference.predict import load_model
from src.inference.multi_day_forecast import forecast_multiple_days


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_seven_day_recursive_forecast():
    """Verify that the production pipeline generates 7 valid future forecasts."""

    history_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ca_1_daily_features_calendar.csv"
    )

    calendar_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "m5"
        / "calendar.csv"
    )

    history = pd.read_csv(history_path)
    calendar = pd.read_csv(calendar_path)

    history["date"] = pd.to_datetime(history["date"])

    last_historical_date = history["date"].max()

    model = load_model()

    forecasts = forecast_multiple_days(
        model=model,
        history=history,
        calendar=calendar,
        days=7,
    )

    # Exactly seven forecasts should be produced
    assert len(forecasts) == 7

    # Required output columns
    required_columns = {"date", "p10", "p50", "p90", "mean"}
    assert required_columns.issubset(forecasts.columns)

    # Forecasting must begin after the historical data
    assert forecasts["date"].min() > last_historical_date

    # Forecast dates must be consecutive
    expected_dates = pd.date_range(
        start=last_historical_date + pd.Timedelta(days=1),
        periods=7,
        freq="D",
    )

    actual_dates = pd.to_datetime(forecasts["date"])

    assert list(actual_dates) == list(expected_dates)

    # Demand predictions must be non-negative
    assert (forecasts[["p10", "p50", "p90", "mean"]] >= 0).all().all()

    # Quantile ordering must hold for every forecast day
    assert (forecasts["p10"] <= forecasts["p50"]).all()
    assert (forecasts["p50"] <= forecasts["p90"]).all()