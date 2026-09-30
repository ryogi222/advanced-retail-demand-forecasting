from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"


def calculate_mape(actual, predicted):
    """Calculate MAPE, excluding zero actual values."""

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    return (
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def run_baseline():

    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    # Use the final 28 days as unseen test data
    test_days = 28

    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()

    print("\n--- DATA SPLIT ---")

    print(
        "Train:",
        train["date"].min(),
        "to",
        train["date"].max()
    )

    print(
        "Test:",
        test["date"].min(),
        "to",
        test["date"].max()
    )

    # Seasonal naive:
    # predict today's demand using demand 7 days ago
    test["prediction"] = test["lag_7"]

    mae = mean_absolute_error(
        test["sales"],
        test["prediction"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["sales"],
            test["prediction"]
        )
    )

    mape = calculate_mape(
        test["sales"],
        test["prediction"]
    )

    print("\n--- BASELINE PERFORMANCE ---")

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    comparison = test[
        ["date", "sales", "prediction"]
    ].copy()

    print("\n--- SAMPLE FORECASTS ---")
    print(comparison.head(10))

    plt.figure(figsize=(12, 6))

    plt.plot(
        comparison["date"],
        comparison["sales"],
        marker="o",
        label="Actual"
    )

    plt.plot(
        comparison["date"],
        comparison["prediction"],
        marker="o",
        label="Seasonal Naive Forecast"
    )

    plt.title("CA_1 Demand Forecast - Baseline")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    output_path = REPORT_DIR / "baseline_forecast.png"

    plt.savefig(output_path, dpi=150)
    plt.close()

    print("\nForecast chart saved:")
    print(output_path)


if __name__ == "__main__":
    run_baseline()
