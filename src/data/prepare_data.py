from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "m5"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_store_data(store_id="CA_1"):
    """Convert M5 sales data from wide to long format for one store."""

    print(f"Preparing data for store: {store_id}")

    sales = pd.read_csv(
        RAW_DIR / "sales_train_evaluation.csv"
    )

    calendar = pd.read_csv(
        RAW_DIR / "calendar.csv"
    )
    prices = pd.read_csv(
    RAW_DIR / "sell_prices.csv"
    )

    # Select one store
    store_sales = sales[
        sales["store_id"] == store_id
    ].copy()

    print("Products:", store_sales["item_id"].nunique())

    # Identify daily sales columns
    day_columns = [
        column for column in store_sales.columns
        if column.startswith("d_")
    ]

    print("Sales days:", len(day_columns))

    # Wide → long
    long_sales = store_sales.melt(
        id_vars=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id"
        ],
        value_vars=day_columns,
        var_name="d",
        value_name="sales"
    )

    # Add actual calendar date
    calendar_lookup = calendar[
        ["d", "date", "wm_yr_wk"]
    ]

    long_sales = long_sales.merge(
        calendar_lookup,
        on="d",
        how="left"
    )

    long_sales["date"] = pd.to_datetime(
        long_sales["date"]
    )
    # Select prices for this store
    store_prices = prices[
    prices["store_id"] == store_id
    ].copy()

    # Join weekly selling prices
    long_sales = long_sales.merge(
    store_prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
    )

    print("\nMissing prices:")
    print(long_sales["sell_price"].isna().sum())

    print("\nPrice statistics:")
    print(long_sales["sell_price"].describe())
    print("\nPrepared shape:", long_sales.shape)

    print("\nSample:")
    print(long_sales.head())

    print("\nDate range:")
    print(long_sales["date"].min())
    print(long_sales["date"].max())

    # Save processed dataset
    output_path = (
        PROCESSED_DIR /
        f"{store_id.lower()}_sales_long.csv"
    )

    long_sales.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    prepare_store_data()