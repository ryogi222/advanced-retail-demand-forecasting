from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ca_1_sales_long.csv"
REPORT_DIR = PROJECT_ROOT / "reports"


def run_eda():

    print("Loading processed CA_1 dataset...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=["date", "sales"]
    )

    df["date"] = pd.to_datetime(df["date"])

    print("Rows:", len(df))
    print("Start:", df["date"].min())
    print("End:", df["date"].max())

    # -----------------------------
    # Daily store demand
    # -----------------------------

    daily_sales = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    print("\n--- DAILY STORE DEMAND ---")
    print(daily_sales.head())

    print("\n--- SALES STATISTICS ---")
    print(daily_sales["sales"].describe())

    # -----------------------------
    # 7-day moving average
    # -----------------------------

    daily_sales["moving_average_7"] = (
        daily_sales["sales"]
        .rolling(window=7)
        .mean()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        daily_sales["date"],
        daily_sales["sales"],
        label="Daily Sales",
        alpha=0.5
    )

    plt.plot(
        daily_sales["date"],
        daily_sales["moving_average_7"],
        label="7-Day Moving Average",
        linewidth=2
    )

    plt.title("CA_1 Store Daily Demand")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()

    daily_chart = REPORT_DIR / "ca_1_daily_demand.png"

    plt.savefig(
        daily_chart,
        dpi=150
    )

    plt.close()

    print("\nDaily demand chart saved:")
    print(daily_chart)

    # -----------------------------
    # Weekly seasonality
    # -----------------------------

    daily_sales["day_of_week"] = (
        daily_sales["date"].dt.day_name()
    )

    daily_sales["day_number"] = (
        daily_sales["date"].dt.dayofweek
    )

    weekly_pattern = (
        daily_sales
        .groupby(
            ["day_number", "day_of_week"]
        )["sales"]
        .mean()
        .reset_index()
        .sort_values("day_number")
    )

    print("\n--- AVERAGE SALES BY DAY OF WEEK ---")
    print(weekly_pattern)

    plt.figure(figsize=(10, 5))

    plt.bar(
        weekly_pattern["day_of_week"],
        weekly_pattern["sales"]
    )

    plt.title("CA_1 Average Demand by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Units Sold")
    plt.xticks(rotation=45)
    plt.tight_layout()

    weekly_chart = (
        REPORT_DIR /
        "ca_1_weekly_seasonality.png"
    )

    plt.savefig(
        weekly_chart,
        dpi=150
    )

    plt.close()

    print("\nWeekly seasonality chart saved:")
    print(weekly_chart)

    # -----------------------------
    # Investigate zero-sales days
    # -----------------------------

    zero_sales = daily_sales[
        daily_sales["sales"] == 0
    ]

    print("\n--- ZERO SALES DAYS ---")

    if zero_sales.empty:
        print("No zero-sales days found.")
    else:
        print(
            zero_sales[
                ["date", "day_of_week", "sales"]
            ]
        )


if __name__ == "__main__":
    run_eda()