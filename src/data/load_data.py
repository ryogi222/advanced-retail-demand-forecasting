from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "m5"


def load_m5_data():
    """Load the core M5 retail forecasting datasets."""

    calendar_path = DATA_DIR / "calendar.csv"
    sales_path = DATA_DIR / "sales_train_evaluation.csv"
    prices_path = DATA_DIR / "sell_prices.csv"

    print("Loading M5 datasets...")

    calendar = pd.read_csv(calendar_path)
    sales = pd.read_csv(sales_path)
    prices = pd.read_csv(prices_path)

    print("\nDatasets loaded successfully.")

    return calendar, sales, prices


def validate_data(calendar, sales, prices):
    """Validate and inspect the M5 datasets."""

    print("\n--- DATASET SHAPES ---")
    print("Calendar:", calendar.shape)
    print("Sales:", sales.shape)
    print("Prices:", prices.shape)

    print("\n--- MISSING VALUES ---")
    print("Calendar:", calendar.isna().sum().sum())
    print("Sales:", sales.isna().sum().sum())
    print("Prices:", prices.isna().sum().sum())

    print("\n--- SALES IDENTIFIERS ---")
    print("Products:", sales["item_id"].nunique())
    print("Departments:", sales["dept_id"].nunique())
    print("Categories:", sales["cat_id"].nunique())
    print("Stores:", sales["store_id"].nunique())
    print("States:", sales["state_id"].nunique())

    print("\n--- CALENDAR COLUMNS ---")
    print(calendar.columns.tolist())

    print("\n--- SALES FIRST 10 COLUMNS ---")
    print(sales.columns[:10].tolist())

    print("\n--- PRICE COLUMNS ---")
    print(prices.columns.tolist())

    print("\n--- SAMPLE PRICE DATA ---")
    print(prices.head())

    print("\n--- DATE RANGE ---")
    print("Start:", calendar["date"].min())
    print("End:", calendar["date"].max())


if __name__ == "__main__":
    calendar_df, sales_df, prices_df = load_m5_data()

    validate_data(
        calendar_df,
        sales_df,
        prices_df
    )