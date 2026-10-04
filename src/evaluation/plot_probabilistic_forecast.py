from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------
# Paths
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_FILE = (
    PROJECT_ROOT
    / "reports"
    / "quantile_random_forest_forecast.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "probabilistic_forecast.png"
)


# -----------------------------------------------------
# Plot probabilistic forecast
# -----------------------------------------------------

def plot_probabilistic_forecast():

    print("=" * 70)
    print("PROBABILISTIC DEMAND FORECAST VISUALISATION")
    print("=" * 70)

    # Load forecast
    df = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    print(f"\nForecast rows: {len(df)}")

    print(
        f"Period: "
        f"{df['date'].min().date()} to "
        f"{df['date'].max().date()}"
    )

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    # Prediction interval
    ax.fill_between(
        df["date"],
        df["p10"],
        df["p90"],
        alpha=0.25,
        label="P10-P90 Prediction Interval"
    )

    # Actual sales
    ax.plot(
        df["date"],
        df["actual"],
        marker="o",
        linewidth=2,
        label="Actual Sales"
    )

    # Median forecast
    ax.plot(
        df["date"],
        df["p50"],
        marker="o",
        linewidth=2,
        label="P50 Forecast"
    )

    # -------------------------------------------------
    # Formatting
    # -------------------------------------------------

    ax.set_title(
        "CA_1 Probabilistic Demand Forecast"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Unit Sales")

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nChart saved:")
    print(OUTPUT_FILE)

    plt.show()


# -----------------------------------------------------
# Run
# -----------------------------------------------------

if __name__ == "__main__":
    plot_probabilistic_forecast()