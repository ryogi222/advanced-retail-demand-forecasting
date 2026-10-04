from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------------------------------
# Paths
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BASELINE_FILE = PROCESSED_DIR / "hierarchical_forecasts.csv"
GRU_FILE = PROCESSED_DIR / "gru_ca1_forecast.csv"
LSTM_FILE = PROCESSED_DIR / "lstm_ca1_forecast.csv"
RF_FILE = PROCESSED_DIR / "random_forest_ca1_forecast.csv"

OUTPUT_FILE = PROCESSED_DIR / "model_comparison.csv"


# -----------------------------------------------------
# Metrics
# -----------------------------------------------------

def calculate_mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

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


def calculate_metrics(actual, predicted):

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

    return mae, rmse, mape


# -----------------------------------------------------
# Main comparison
# -----------------------------------------------------

def compare_models():

    print("=" * 70)
    print("CA_1 STORE-LEVEL MODEL COMPARISON")
    print("=" * 70)

    # -------------------------------------------------
    # Load predictions
    # -------------------------------------------------

    baseline = pd.read_csv(
        BASELINE_FILE,
        parse_dates=["date"]
    )

    gru = pd.read_csv(
        GRU_FILE,
        parse_dates=["date"]
    )

    lstm = pd.read_csv(
        LSTM_FILE,
        parse_dates=["date"]
    )
    rf = pd.read_csv(
        RF_FILE,
        parse_dates=["date"]
    )

    rf = rf.sort_values(
        "date"
    ).reset_index(drop=True)

    # Keep only store-level CA_1 baseline
    baseline = baseline[
        (baseline["level"] == "store")
        & (baseline["series_id"] == "CA_1")
    ].copy()

    baseline = baseline.sort_values(
        "date"
    ).reset_index(drop=True)

    gru = gru.sort_values(
        "date"
    ).reset_index(drop=True)

    lstm = lstm.sort_values(
        "date"
    ).reset_index(drop=True)

    print(f"\nBaseline rows: {len(baseline)}")
    print(f"GRU rows:      {len(gru)}")
    print(f"LSTM rows:     {len(lstm)}")
    print(f"Random Forest rows: {len(rf)}")

    # -------------------------------------------------
    # Validate lengths
    # -------------------------------------------------

    expected_rows = 28

    if not (
        len(baseline)
        == len(gru)
        == len(lstm)
        == len(rf)
        == expected_rows
    ):
        raise ValueError(
            "Forecast files do not contain the same "
            "28-day test period."
        )

    # -------------------------------------------------
    # Validate dates
    # -------------------------------------------------

    if not baseline["date"].equals(gru["date"]):
        raise ValueError(
            "Baseline and GRU dates do not match."
        )

    if not baseline["date"].equals(lstm["date"]):
        raise ValueError(
            "Baseline and LSTM dates do not match."
        )
    if not baseline["date"].equals(rf["date"]):
        raise ValueError(
            "Baseline and Random Forest dates do not match."
        )

    # -------------------------------------------------
    # Validate actual values
    # -------------------------------------------------

    baseline_actual = baseline[
        "sales"
    ].to_numpy(dtype=float)

    gru_actual = gru[
        "actual"
    ].to_numpy(dtype=float)

    lstm_actual = lstm[
        "actual"
    ].to_numpy(dtype=float)
    rf_actual = rf[
        "actual"
    ].to_numpy(dtype=float)

    if not np.allclose(
        baseline_actual,
        gru_actual
    ):
        raise ValueError(
            "Baseline and GRU actual values do not match."
        )

    if not np.allclose(
        baseline_actual,
        lstm_actual
    ):
        raise ValueError(
            "Baseline and LSTM actual values do not match."
        )
    if not np.allclose(
        baseline_actual,
        rf_actual
    ):
        raise ValueError(
            "Baseline and Random Forest actual values do not match."
        )
    print("\nValidation passed:")
    print(
        "All models use the same 28 test dates "
        "and actual CA_1 sales values."
    )

    print(
        f"Test period: "
        f"{baseline['date'].min().date()} to "
        f"{baseline['date'].max().date()}"
    )

    # -------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------

    baseline_metrics = calculate_metrics(
        baseline_actual,
        baseline["forecast"]
    )

    gru_metrics = calculate_metrics(
        gru_actual,
        gru["gru_forecast"]
    )

    lstm_metrics = calculate_metrics(
        lstm_actual,
        lstm["lstm_forecast"]
    )
    rf_metrics = calculate_metrics(
        rf_actual,
        rf["random_forest_forecast"]
    )
    # -------------------------------------------------
    # Comparison table
    # -------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "Moving Average Baseline",
                "MAE": baseline_metrics[0],
                "RMSE": baseline_metrics[1],
                "MAPE": baseline_metrics[2]
            },
            {
                "model": "GRU",
                "MAE": gru_metrics[0],
                "RMSE": gru_metrics[1],
                "MAPE": gru_metrics[2]
            },
                        {
                "model": "LSTM",
                "MAE": lstm_metrics[0],
                "RMSE": lstm_metrics[1],
                "MAPE": lstm_metrics[2]
            },
            {
                "model": "Random Forest",
                "MAE": rf_metrics[0],
                "RMSE": rf_metrics[1],
                "MAPE": rf_metrics[2]
            }
        ]
    )

    comparison = comparison.sort_values(
        "MAPE"
    ).reset_index(drop=True)

    # -------------------------------------------------
    # Display
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        comparison.round(2).to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # GRU improvement over baseline
    # -------------------------------------------------

    baseline_mape = baseline_metrics[2]
    gru_mape = gru_metrics[2]

    mape_reduction = (
        baseline_mape - gru_mape
    )

    relative_reduction = (
        mape_reduction
        / baseline_mape
        * 100
    )

    print("\nGRU vs Moving Average Baseline")

    print(
        f"MAPE reduction: "
        f"{mape_reduction:.2f} percentage points"
    )

    print(
        f"Relative MAPE reduction: "
        f"{relative_reduction:.2f}%"
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    comparison.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nComparison saved:")
    print(OUTPUT_FILE)


# -----------------------------------------------------
# Run
# -----------------------------------------------------

if __name__ == "__main__":
    compare_models()
