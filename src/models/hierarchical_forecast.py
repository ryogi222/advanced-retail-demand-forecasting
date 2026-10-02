from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_hierarchy_daily.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hierarchical_forecasts.csv"
)


# ---------------------------------------------------------
# MAPE
# ---------------------------------------------------------

def calculate_mape(actual, predicted):
    """
    Calculate MAPE while ignoring observations where
    actual demand is zero.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


# ---------------------------------------------------------
# Forecast one time series
# ---------------------------------------------------------

def forecast_series(series_df, horizon=28):
    """
    Forecast the final `horizon` days using a
    28-day moving-average baseline.
    """

    series_df = series_df.sort_values("date").copy()

    train = series_df.iloc[:-horizon].copy()
    test = series_df.iloc[-horizon:].copy()

    history = train["sales"].tolist()

    predictions = []

    for actual in test["sales"]:
        window = history[-28:]

        forecast = np.mean(window)

        predictions.append(forecast)

        # Walk-forward validation:
        # add the real observation to history.
        history.append(actual)

    test["forecast"] = predictions

    return test


# ---------------------------------------------------------
# Main hierarchical forecasting pipeline
# ---------------------------------------------------------

def run_hierarchical_forecast():
    print("Loading hierarchy dataset...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print(f"Rows: {len(df):,}")
    print(f"Series: {df['series_id'].nunique()}")
    print()

    all_forecasts = []
    metrics = []

    grouped = df.groupby(
        ["level", "series_id"],
        sort=False
    )

    for (level, series_id), group in grouped:

        print(
            f"Forecasting {level}: "
            f"{series_id}"
        )

        forecast_df = forecast_series(
            group,
            horizon=28
        )

        actual = forecast_df["sales"]
        predicted = forecast_df["forecast"]

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        mape = calculate_mape(
            actual,
            predicted
        )

        forecast_df["level"] = level
        forecast_df["series_id"] = series_id

        all_forecasts.append(
            forecast_df[
                [
                    "date",
                    "level",
                    "series_id",
                    "sales",
                    "forecast",
                ]
            ]
        )

        metrics.append(
            {
                "level": level,
                "series_id": series_id,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
            }
        )

    # -----------------------------------------------------
    # Combine forecasts
    # -----------------------------------------------------

    forecasts = pd.concat(
        all_forecasts,
        ignore_index=True
    )

    metrics_df = pd.DataFrame(metrics)

    # -----------------------------------------------------
    # Save forecasts
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    forecasts.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("HIERARCHICAL FORECAST RESULTS")
    print("=" * 70)

    print(
        metrics_df.round(2).to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("AVERAGE METRICS BY LEVEL")
    print("=" * 70)

    level_metrics = (
        metrics_df
        .groupby("level")[
            ["MAE", "RMSE", "MAPE"]
        ]
        .mean()
        .round(2)
    )

    print(level_metrics)

    print()
    print(
        "Forecast dataset saved:"
    )

    print(OUTPUT_PATH)

    print()
    print(
        f"Forecast rows: "
        f"{len(forecasts):,}"
    )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    run_hierarchical_forecast()