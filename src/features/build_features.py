from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_sales_long.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features.csv"
)


def build_features():

    print("Loading CA_1 sales data...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=["date", "sales"]
    )

    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------
    # Aggregate to store/day level
    # --------------------------------

    daily = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    print("Daily observations:", len(daily))

    # --------------------------------
    # Calendar features
    # --------------------------------

    daily["day_of_week"] = daily["date"].dt.dayofweek

    daily["day_of_month"] = daily["date"].dt.day

    daily["month"] = daily["date"].dt.month

    daily["year"] = daily["date"].dt.year

    daily["week_of_year"] = (
        daily["date"].dt.isocalendar().week.astype(int)
    )

    daily["is_weekend"] = (
        daily["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------
    # Lag features
    # --------------------------------

    daily["lag_1"] = daily["sales"].shift(1)

    daily["lag_7"] = daily["sales"].shift(7)

    daily["lag_14"] = daily["sales"].shift(14)

    daily["lag_28"] = daily["sales"].shift(28)

    # --------------------------------
    # Rolling features
    #
    # shift(1) is important:
    # don't allow today's sales to
    # predict today's sales.
    # --------------------------------

    historical_sales = daily["sales"].shift(1)

    daily["rolling_mean_7"] = (
        historical_sales
        .rolling(7)
        .mean()
    )

    daily["rolling_mean_28"] = (
        historical_sales
        .rolling(28)
        .mean()
    )

    daily["rolling_std_7"] = (
        historical_sales
        .rolling(7)
        .std()
    )

    daily["rolling_std_28"] = (
        historical_sales
        .rolling(28)
        .std()
    )

    # --------------------------------
    # Remove rows without full history
    # --------------------------------

    features = daily.dropna().copy()

    print("\nFeature dataset shape:")
    print(features.shape)

    print("\nColumns:")
    print(features.columns.tolist())

    print("\nSample:")
    print(features.head())

    print("\nDate range:")
    print(features["date"].min())
    print(features["date"].max())

    features.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build_features()