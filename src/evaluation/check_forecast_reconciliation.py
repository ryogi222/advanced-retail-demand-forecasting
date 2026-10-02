from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hierarchical_forecasts.csv"
)


# ---------------------------------------------------------
# Main reconciliation check
# ---------------------------------------------------------

def check_reconciliation():

    print("Loading hierarchical forecasts...")

    df = pd.read_csv(FORECAST_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Rows: {len(df):,}")
    print(f"Forecast dates: {df['date'].nunique()}")
    print(f"Series: {df['series_id'].nunique()}")
    print()

    # -----------------------------------------------------
    # Store forecast
    # -----------------------------------------------------

    store = (
        df[df["level"] == "store"]
        .set_index("date")["forecast"]
        .sort_index()
    )

    # -----------------------------------------------------
    # Sum category forecasts
    # -----------------------------------------------------

    category_sum = (
        df[df["level"] == "category"]
        .groupby("date")["forecast"]
        .sum()
        .sort_index()
    )

    # -----------------------------------------------------
    # Sum department forecasts
    # -----------------------------------------------------

    department_sum = (
        df[df["level"] == "department"]
        .groupby("date")["forecast"]
        .sum()
        .sort_index()
    )

    # -----------------------------------------------------
    # Compare with store
    # -----------------------------------------------------

    comparison = pd.DataFrame({
        "store_forecast": store,
        "category_sum": category_sum,
        "department_sum": department_sum,
    })

    comparison["store_category_diff"] = (
        comparison["store_forecast"]
        - comparison["category_sum"]
    )

    comparison["store_department_diff"] = (
        comparison["store_forecast"]
        - comparison["department_sum"]
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    max_category_diff = (
        comparison["store_category_diff"]
        .abs()
        .max()
    )

    max_department_diff = (
        comparison["store_department_diff"]
        .abs()
        .max()
    )

    mean_category_diff = (
        comparison["store_category_diff"]
        .abs()
        .mean()
    )

    mean_department_diff = (
        comparison["store_department_diff"]
        .abs()
        .mean()
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("=" * 70)
    print("FORECAST RECONCILIATION CHECK")
    print("=" * 70)

    print()
    print("Store vs Category forecasts")
    print("---------------------------")
    print(
        f"Maximum difference: "
        f"{max_category_diff:.6f}"
    )
    print(
        f"Mean absolute difference: "
        f"{mean_category_diff:.6f}"
    )

    print()
    print("Store vs Department forecasts")
    print("-----------------------------")
    print(
        f"Maximum difference: "
        f"{max_department_diff:.6f}"
    )
    print(
        f"Mean absolute difference: "
        f"{mean_department_diff:.6f}"
    )

    # -----------------------------------------------------
    # Pass / fail
    # -----------------------------------------------------

    tolerance = 1e-6

    category_pass = (
        max_category_diff <= tolerance
    )

    department_pass = (
        max_department_diff <= tolerance
    )

    print()
    print("=" * 70)

    if category_pass and department_pass:
        print("FORECAST RECONCILIATION: PASSED")
        print(
            "Forecasts are coherent across "
            "all hierarchy levels."
        )
    else:
        print("FORECAST RECONCILIATION: FAILED")
        print(
            "Independent forecasts do not "
            "perfectly reconcile."
        )

    print("=" * 70)

    # -----------------------------------------------------
    # Show sample
    # -----------------------------------------------------

    print()
    print("Sample reconciliation results:")
    print(
        comparison.head(10)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    check_reconciliation()