from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features_calendar_price.csv"
)


BASE_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]


CALENDAR_FEATURES = [
    "snap_CA",
    "has_event_1",
    "has_event_2",
    "has_any_event",
    "event_type_cultural",
    "event_type_national",
    "event_type_religious",
    "event_type_sporting",
    "is_superbowl",
    "is_valentines",
    "is_christmas",
]


PRICE_FEATURES = [
    "products_with_price",
    "mean_sell_price",
    "median_sell_price",
    "median_price_change_pct",
    "discount_share",
    "price_increase_share",
    "price_change_share",
    "price_coverage",
]


FEATURES = (
    BASE_FEATURES
    + CALENDAR_FEATURES
    + PRICE_FEATURES
)


def calculate_mape(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    return (
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


def calculate_wape(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        / denominator
        * 100
    )


def price_feature_validation():

    print(
        "Loading calendar + price dataset..."
    )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    print(
        "Observations:",
        len(df)
    )

    print(
        "Total model features:",
        len(FEATURES)
    )

    print(
        "Base features:",
        len(BASE_FEATURES)
    )

    print(
        "Calendar features:",
        len(CALENDAR_FEATURES)
    )

    print(
        "Price features:",
        len(PRICE_FEATURES)
    )

    # ---------------------------------------
    # Walk-forward configuration
    # ---------------------------------------

    test_days = 28
    n_folds = 5

    first_test_start = (
        len(df)
        - test_days * n_folds
    )

    results = []
    importance_results = []

    print(
        "\n--- PRICE FEATURE WALK-FORWARD ---"
    )

    # ---------------------------------------
    # Run folds
    # ---------------------------------------

    for fold in range(n_folds):

        test_start = (
            first_test_start
            + fold * test_days
        )

        test_end = (
            test_start
            + test_days
        )

        train = df.iloc[
            :test_start
        ]

        test = df.iloc[
            test_start:test_end
        ]

        X_train = train[
            FEATURES
        ]

        y_train = train[
            "sales"
        ]

        X_test = test[
            FEATURES
        ]

        y_test = test[
            "sales"
        ]

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        # -----------------------------------
        # Metrics
        # -----------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        mape = calculate_mape(
            y_test,
            predictions
        )

        wape = calculate_wape(
            y_test,
            predictions
        )

        print(
            f"\nFold {fold + 1}"
        )

        print(
            "Test:",
            test["date"].min().date(),
            "to",
            test["date"].max().date()
        )

        print(
            f"MAE:  {mae:.2f}"
        )

        print(
            f"RMSE: {rmse:.2f}"
        )

        print(
            f"MAPE: {mape:.2f}%"
        )

        print(
            f"WAPE: {wape:.2f}%"
        )

        results.append(
            {
                "fold": fold + 1,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "wape": wape,
            }
        )

        # -----------------------------------
        # Feature importance
        # -----------------------------------

        importance = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance":
                    model.feature_importances_,
            }
        )

        importance["fold"] = (
            fold + 1
        )

        importance_results.append(
            importance
        )

    # ---------------------------------------
    # Summary
    # ---------------------------------------

    results_df = pd.DataFrame(
        results
    )

    average_mae = (
        results_df["mae"].mean()
    )

    average_rmse = (
        results_df["rmse"].mean()
    )

    average_mape = (
        results_df["mape"].mean()
    )

    mape_std = (
        results_df["mape"].std()
    )

    average_wape = (
        results_df["wape"].mean()
    )

    print(
        "\n================================"
    )

    print(
        "PRICE-ENHANCED RF SUMMARY"
    )

    print(
        "================================"
    )

    print(
        results_df[
            [
                "fold",
                "mae",
                "rmse",
                "mape",
                "wape",
            ]
        ].round(2)
    )

    print(
        f"\nAverage MAE: "
        f"{average_mae:.2f}"
    )

    print(
        f"Average RMSE: "
        f"{average_rmse:.2f}"
    )

    print(
        f"Average MAPE: "
        f"{average_mape:.2f}%"
    )

    print(
        f"MAPE Std Dev: "
        f"{mape_std:.2f}%"
    )

    print(
        f"Average WAPE: "
        f"{average_wape:.2f}%"
    )

    # ---------------------------------------
    # Compare models
    # ---------------------------------------

    original_rf_mape = 5.93
    calendar_rf_mape = 5.48

    original_rf_wape = 6.14
    calendar_rf_wape = 5.58

    improvement_from_calendar = (
        calendar_rf_mape
        - average_mape
    )

    relative_improvement = (
        improvement_from_calendar
        / calendar_rf_mape
        * 100
    )

    print(
        "\n================================"
    )

    print(
        "PRICE FEATURE IMPACT"
    )

    print(
        "================================"
    )

    print(
        f"Original RF MAPE: "
        f"{original_rf_mape:.2f}%"
    )

    print(
        f"Calendar RF MAPE: "
        f"{calendar_rf_mape:.2f}%"
    )

    print(
        f"Calendar + Price RF MAPE: "
        f"{average_mape:.2f}%"
    )

    print(
        f"\nOriginal RF WAPE: "
        f"{original_rf_wape:.2f}%"
    )

    print(
        f"Calendar RF WAPE: "
        f"{calendar_rf_wape:.2f}%"
    )

    print(
        f"Calendar + Price RF WAPE: "
        f"{average_wape:.2f}%"
    )

    print(
        f"\nMAPE improvement vs calendar model: "
        f"{improvement_from_calendar:.2f} "
        f"percentage points"
    )

    print(
        f"Relative improvement vs calendar model: "
        f"{relative_improvement:.2f}%"
    )

    # ---------------------------------------
    # Feature importance
    # ---------------------------------------

    importance_df = pd.concat(
        importance_results,
        ignore_index=True
    )

    average_importance = (
        importance_df
        .groupby("feature")["importance"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\n================================"
    )

    print(
        "TOP 20 FEATURE IMPORTANCES"
    )

    print(
        "================================"
    )

    print(
        average_importance
        .head(20)
        .round(4)
    )

    print(
        "\nPRICE FEATURE IMPORTANCE"
    )

    price_importance = (
        average_importance[
            average_importance.index.isin(
                PRICE_FEATURES
            )
        ]
        .sort_values(
            ascending=False
        )
    )

    print(
        price_importance.round(4)
    )

    # ---------------------------------------
    # Save outputs
    # ---------------------------------------

    reports_dir = (
        PROJECT_ROOT
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        reports_dir
        / "price_feature_walk_forward.csv"
    )

    importance_path = (
        reports_dir
        / "price_feature_importance.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    average_importance.reset_index().to_csv(
        importance_path,
        index=False
    )

    print(
        "\nResults saved:"
    )

    print(
        results_path
    )

    print(
        "\nFeature importance saved:"
    )

    print(
        importance_path
    )


if __name__ == "__main__":
    price_feature_validation()