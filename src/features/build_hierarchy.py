from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SALES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "m5"
    / "sales_train_evaluation.csv"
)

CALENDAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "m5"
    / "calendar.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def build_hierarchy():

    print("Loading M5 sales data...")

    sales = pd.read_csv(SALES_PATH)

    # -----------------------------------
    # Select CA_1
    # -----------------------------------

    sales = (
        sales[
            sales["store_id"] == "CA_1"
        ]
        .copy()
    )

    print("CA_1 products:", len(sales))

    # -----------------------------------
    # Identify daily sales columns
    # -----------------------------------

    day_columns = [
        col
        for col in sales.columns
        if col.startswith("d_")
    ]

    print(
        "Sales days:",
        len(day_columns)
    )

    # -----------------------------------
    # Store level
    # -----------------------------------

    print("\nBuilding store hierarchy...")

    store_daily = (
        sales[day_columns]
        .sum(axis=0)
        .reset_index()
    )

    store_daily.columns = [
        "d",
        "sales"
    ]

    store_daily["level"] = "store"
    store_daily["series_id"] = "CA_1"

    # -----------------------------------
    # Category level
    # -----------------------------------

    print("Building category hierarchy...")

    category_wide = (
        sales
        .groupby("cat_id")[day_columns]
        .sum()
        .reset_index()
    )

    category_daily = (
        category_wide
        .melt(
            id_vars=["cat_id"],
            value_vars=day_columns,
            var_name="d",
            value_name="sales"
        )
    )

    category_daily["level"] = "category"

    category_daily["series_id"] = (
        category_daily["cat_id"]
    )

    category_daily = (
        category_daily[
            [
                "d",
                "sales",
                "level",
                "series_id"
            ]
        ]
    )

    # -----------------------------------
    # Department level
    # -----------------------------------

    print("Building department hierarchy...")

    department_wide = (
        sales
        .groupby(
            ["cat_id", "dept_id"]
        )[day_columns]
        .sum()
        .reset_index()
    )

    department_daily = (
        department_wide
        .melt(
            id_vars=[
                "cat_id",
                "dept_id"
            ],
            value_vars=day_columns,
            var_name="d",
            value_name="sales"
        )
    )

    department_daily["level"] = "department"

    department_daily["series_id"] = (
        department_daily["dept_id"]
    )

    department_daily = (
        department_daily[
            [
                "d",
                "sales",
                "level",
                "series_id"
            ]
        ]
    )

    # -----------------------------------
    # Combine hierarchy
    # -----------------------------------

    hierarchy = pd.concat(
        [
            store_daily,
            category_daily,
            department_daily,
        ],
        ignore_index=True
    )

    # -----------------------------------
    # Add actual dates
    # -----------------------------------

    print("Adding calendar dates...")

    calendar = pd.read_csv(
        CALENDAR_PATH,
        usecols=[
            "date",
            "d"
        ],
        parse_dates=["date"]
    )

    hierarchy = hierarchy.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one"
    )

    hierarchy = (
        hierarchy[
            [
                "date",
                "d",
                "level",
                "series_id",
                "sales"
            ]
        ]
        .sort_values(
            [
                "level",
                "series_id",
                "date"
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------
    # Validation
    # -----------------------------------

    print(
        "\n================================"
    )

    print(
        "HIERARCHY SUMMARY"
    )

    print(
        "================================"
    )

    print(
        "Rows:",
        len(hierarchy)
    )

    print(
        "Date range:",
        hierarchy["date"].min().date(),
        "to",
        hierarchy["date"].max().date()
    )

    print(
        "\nSeries by level:"
    )

    print(
        hierarchy
        .groupby("level")["series_id"]
        .nunique()
    )

    print(
        "\nTotal series:",
        hierarchy["series_id"].nunique()
    )

    # -----------------------------------
    # Check reconciliation
    # -----------------------------------

    store_check = (
        hierarchy[
            hierarchy["level"] == "store"
        ]
        .set_index("date")["sales"]
    )

    category_check = (
        hierarchy[
            hierarchy["level"] == "category"
        ]
        .groupby("date")["sales"]
        .sum()
    )

    department_check = (
        hierarchy[
            hierarchy["level"] == "department"
        ]
        .groupby("date")["sales"]
        .sum()
    )

    category_difference = (
        store_check - category_check
    ).abs().max()

    department_difference = (
        store_check - department_check
    ).abs().max()

    print(
        "\nMaximum Store vs Category difference:",
        category_difference
    )

    print(
        "Maximum Store vs Department difference:",
        department_difference
    )

    if category_difference != 0:
        raise ValueError(
            "Category hierarchy does not reconcile."
        )

    if department_difference != 0:
        raise ValueError(
            "Department hierarchy does not reconcile."
        )

    print(
        "\nHierarchy reconciliation: PASSED"
    )

    # -----------------------------------
    # Average daily demand
    # -----------------------------------

    summary = (
        hierarchy
        .groupby(
            [
                "level",
                "series_id"
            ]
        )["sales"]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max"
            ]
        )
        .round(2)
    )

    print(
        "\nSeries demand summary:"
    )

    print(summary)

    # -----------------------------------
    # Save
    # -----------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "ca_1_hierarchy_daily.csv"
    )

    hierarchy.to_csv(
        output_path,
        index=False
    )

    print(
        "\nHierarchy dataset saved:"
    )

    print(output_path)

    print(
        "\nFinal shape:",
        hierarchy.shape
    )


if __name__ == "__main__":
    build_hierarchy()