from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------------------------------
# Paths
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GRU_FILE = PROCESSED_DIR / "gru_ca1_forecast.csv"
LSTM_FILE = PROCESSED_DIR / "lstm_ca1_forecast.csv"

OUTPUT_FILE = PROCESSED_DIR / "deep_learning_model_comparison.csv"


# -----------------------------------------------------
# MAPE
# -----------------------------------------------------

def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error.

    Zero actual values are excluded to avoid division
    by zero.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = actual != 0

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


# -----------------------------------------------------
# Metrics
# -----------------------------------------------------

def calculate_metrics(actual, predicted):
    """
    Calculate MAE, RMSE and MAPE.
    """

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

    print("=" * 65)
    print("DEEP LEARNING MODEL COMPARISON")
    print("=" * 65)

    # -------------------------------------------------
    # Load forecasts
    # -------------------------------------------------

    gru = pd.read_csv(
        GRU_FILE,
        parse_dates=["date"]
    )

    lstm = pd.read_csv(
        LSTM_FILE,
        parse_dates=["date"]
    )

    print(f"\nGRU rows:  {len(gru)}")
    print(f"LSTM rows: {len(lstm)}")

    # -------------------------------------------------
    # Validate test periods
    # -------------------------------------------------

    if len(gru) != len(lstm):
        raise ValueError(
            "GRU and LSTM forecast files have "
            "different numbers of rows."
        )

    if not gru["date"].equals(lstm["date"]):
        raise ValueError(
            "GRU and LSTM test dates do not match."
        )

    if not np.allclose(
        gru["actual"].to_numpy(),
        lstm["actual"].to_numpy()
    ):
        raise ValueError(
            "GRU and LSTM actual values do not match."
        )

    print("\nValidation passed:")
    print("Same test dates and actual values.")

    print(
        f"Test period: "
        f"{gru['date'].min().date()} to "
        f"{gru['date'].max().date()}"
    )

    # -------------------------------------------------
    # Calculate GRU metrics
    # -------------------------------------------------

    gru_mae, gru_rmse, gru_mape = calculate_metrics(
        gru["actual"],
        gru["gru_forecast"]
    )

    # -------------------------------------------------
    # Calculate LSTM metrics
    # -------------------------------------------------

    lstm_mae, lstm_rmse, lstm_mape = calculate_metrics(
        lstm["actual"],
        lstm["lstm_forecast"]
    )

    # -------------------------------------------------
    # Comparison table
    # -------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "GRU",
                "MAE": gru_mae,
                "RMSE": gru_rmse,
                "MAPE": gru_mape
            },
            {
                "model": "LSTM",
                "MAE": lstm_mae,
                "RMSE": lstm_rmse,
                "MAPE": lstm_mape
            }
        ]
    )

    comparison = comparison.sort_values(
        "MAPE"
    ).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE")
    print("=" * 65)

    print(
        comparison.round(2).to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # Relative difference
    # -------------------------------------------------

    mape_difference = (
        lstm_mape - gru_mape
    )

    print("\nMAPE difference:")
    print(
        f"LSTM - GRU = "
        f"{mape_difference:.2f} percentage points"
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