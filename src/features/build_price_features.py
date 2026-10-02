from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRICE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "m5"
    / "sell_prices.csv"
)

CALENDAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "m5"
    / "calendar.csv"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features_calendar.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features_calendar_price.csv"
)


def build_price_features():

    print("Loading CA_1 price data...")

    prices = pd.read_csv(PRICE_PATH)

    prices = (
        prices[
            prices["store_id"] == "CA_1"
        ]
        .copy()
    )

    prices = prices.sort_values(
        ["item_id", "wm_yr_wk"]
    )

    print("Price rows:", len(prices))
    print(
        "Products:",
        prices["item_id"].nunique()
    )

    # -----------------------------------------
    # Previous product price
    # -----------------------------------------

    prices["previous_price"] = (
        prices
        .groupby("item_id")["sell_price"]
        .shift(1)
    )

    # -----------------------------------------
    # Price change
    # -----------------------------------------

    prices["price_change_pct"] = (
        (
            prices["sell_price"]
            - prices["previous_price"]
        )
        / prices["previous_price"]
        * 100
    )

    # -----------------------------------------
    # Change indicators
    # -----------------------------------------

    prices["price_decreased"] = (
        prices["price_change_pct"] < 0
    ).astype(int)

    prices["price_increased"] = (
        prices["price_change_pct"] > 0
    ).astype(int)

    prices["price_changed"] = (
        prices["price_change_pct"] != 0
    ).astype(int)

    # First observation has no previous price.
    prices.loc[
        prices["previous_price"].isna(),
        [
            "price_decreased",
            "price_increased",
            "price_changed"
        ]
    ] = 0

    # -----------------------------------------
    # Weekly aggregation
    # -----------------------------------------

    print("\nCreating weekly price features...")

    weekly = (
        prices
        .groupby("wm_yr_wk")
        .agg(
            products_with_price=(
                "item_id",
                "nunique"
            ),
            mean_sell_price=(
                "sell_price",
                "mean"
            ),
            median_sell_price=(
                "sell_price",
                "median"
            ),
            median_price_change_pct=(
                "price_change_pct",
                "median"
            ),
            price_decrease_count=(
                "price_decreased",
                "sum"
            ),
            price_increase_count=(
                "price_increased",
                "sum"
            ),
            price_change_count=(
                "price_changed",
                "sum"
            )
        )
        .reset_index()
    )

    # -----------------------------------------
    # Convert counts to shares
    # -----------------------------------------

    weekly["discount_share"] = (
        weekly["price_decrease_count"]
        / weekly["products_with_price"]
    )

    weekly["price_increase_share"] = (
        weekly["price_increase_count"]
        / weekly["products_with_price"]
    )

    weekly["price_change_share"] = (
        weekly["price_change_count"]
        / weekly["products_with_price"]
    )

    # -----------------------------------------
    # Product coverage
    # -----------------------------------------

    total_products = (
        prices["item_id"].nunique()
    )

    weekly["price_coverage"] = (
        weekly["products_with_price"]
        / total_products
    )

    # -----------------------------------------
    # Robust median change
    # -----------------------------------------

    weekly[
        "median_price_change_pct"
    ] = (
        weekly[
            "median_price_change_pct"
        ]
        .fillna(0)
    )

    print(
        "Weekly observations:",
        len(weekly)
    )

    print(
        "\nWeekly feature sample:"
    )

    print(
        weekly.head(10).round(4)
    )

    # -----------------------------------------
    # Load M5 calendar
    # -----------------------------------------

    print("\nLoading calendar...")

    calendar = pd.read_csv(
        CALENDAR_PATH,
        usecols=[
            "date",
            "wm_yr_wk"
        ],
        parse_dates=["date"]
    )

    # -----------------------------------------
    # Load existing daily features
    # -----------------------------------------

    print(
        "Loading calendar-enhanced "
        "daily features..."
    )

    daily = pd.read_csv(
        FEATURE_PATH,
        parse_dates=["date"]
    )

    original_rows = len(daily)

    # -----------------------------------------
    # Add week ID
    # -----------------------------------------

    daily = daily.merge(
        calendar,
        on="date",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------
    # Add weekly price features
    # -----------------------------------------

    price_features = [
        "wm_yr_wk",
        "products_with_price",
        "mean_sell_price",
        "median_sell_price",
        "median_price_change_pct",
        "discount_share",
        "price_increase_share",
        "price_change_share",
        "price_coverage"
    ]

    enhanced = daily.merge(
        weekly[price_features],
        on="wm_yr_wk",
        how="left",
        validate="many_to_one"
    )

    # -----------------------------------------
    # Validation
    # -----------------------------------------

    if len(enhanced) != original_rows:
        raise ValueError(
            "Row count changed after price merge."
        )

    print(
        "\nOriginal rows:",
        original_rows
    )

    print(
        "Enhanced rows:",
        len(enhanced)
    )

    print(
        "Final columns:",
        len(enhanced.columns)
    )

    # -----------------------------------------
    # Missing price information
    # -----------------------------------------

    model_price_features = [
        "products_with_price",
        "mean_sell_price",
        "median_sell_price",
        "median_price_change_pct",
        "discount_share",
        "price_increase_share",
        "price_change_share",
        "price_coverage"
    ]

    print(
        "\nMissing values:"
    )

    print(
        enhanced[
            model_price_features
        ].isna().sum()
    )

    # -----------------------------------------
    # Diagnostics
    # -----------------------------------------

    print(
        "\n--- PRICE FEATURE SUMMARY ---"
    )

    print(
        enhanced[
            model_price_features
        ]
        .describe()
        .round(4)
    )

    # -----------------------------------------
    # Save
    # -----------------------------------------

    enhanced.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nPrice-enhanced dataset saved:"
    )

    print(OUTPUT_PATH)

    print(
        "\nFinal shape:",
        enhanced.shape
    )


if __name__ == "__main__":
    build_price_features()