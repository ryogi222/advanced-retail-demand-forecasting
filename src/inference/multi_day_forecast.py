from pathlib import Path

import pandas as pd

from src.inference.predict import load_model, predict_quantiles
from src.inference.future_features import (
    build_next_day_features,
    add_calendar_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def forecast_multiple_days(model, history, calendar, days=7):
    """
    Generate recursive multi-day demand forecasts.

    After each forecast, the P50 prediction is appended to the
    demand history and used to construct features for the next day.
    """

    history = history[["date", "sales"]].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)

    forecasts = []

    for _ in range(days):

        # Build features for the next unseen day
        features = build_next_day_features(history)

        # Add calendar and event information
        features = add_calendar_features(features, calendar)

        # Convert the feature dictionary into a one-row DataFrame
        feature_frame = pd.DataFrame([features])

        # Generate probabilistic prediction
        prediction = predict_quantiles(model, feature_frame)

        forecast_date = pd.to_datetime(features["date"])

        forecasts.append(
            {
                "date": forecast_date,
          "p10": float(prediction["p10"][0]),
          "p50": float(prediction["p50"][0]),
          "p90": float(prediction["p90"][0]),
          "mean": float(prediction["mean"][0]),
            }
        )

        # P50 becomes the assumed demand for the next recursive step
        new_history_row = pd.DataFrame(
            {
                "date": [forecast_date],
           "sales": [float(prediction["p50"][0])],
            }
        )

        history = pd.concat(
            [history, new_history_row],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)

if __name__ == "__main__":

    feature_path = (
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

    history = pd.read_csv(feature_path)
    calendar = pd.read_csv(calendar_path)

    model = load_model()

    forecasts = forecast_multiple_days(
        model=model,
        history=history,
        calendar=calendar,
        days=7,
    )

    print("\n7-DAY FUTURE FORECAST")
    print("=" * 60)
    print(forecasts.to_string(index=False))