from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features.csv"
)


FEATURES = [
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
    "rolling_std_28"
]


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


def walk_forward_validation():

    print(
        "Loading feature dataset..."
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
        "Date range:",
        df["date"].min().date(),
        "to",
        df["date"].max().date()
    )

    # ----------------------------------------
    # Walk-forward configuration
    # ----------------------------------------

    test_days = 28
    n_folds = 5

    total_test_days = (
        test_days * n_folds
    )

    first_test_start = (
        len(df)
        - total_test_days
    )

    results = []

    print(
        "\n--- WALK-FORWARD VALIDATION ---"
    )

    print(
        "Folds:",
        n_folds
    )

    print(
        "Days per fold:",
        test_days
    )

    print(
        "Total evaluation days:",
        total_test_days
    )

    # ----------------------------------------
    # Validation loop
    # ----------------------------------------

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

        X_train = train[FEATURES]
        y_train = train["sales"]

        X_test = test[FEATURES]
        y_test = test["sales"]

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

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
            y_test.values,
            predictions
        )

        wape = calculate_wape(
            y_test.values,
            predictions
        )

        print(
            f"\nFold {fold + 1}"
        )

        print(
            "Train:",
            train["date"].min().date(),
            "to",
            train["date"].max().date()
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
                "train_end":
                    train["date"]
                    .max()
                    .date(),

                "test_start":
                    test["date"]
                    .min()
                    .date(),

                "test_end":
                    test["date"]
                    .max()
                    .date(),

                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "wape": wape
            }
        )

    # ----------------------------------------
    # Summary
    # ----------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n================================="
    )

    print(
        "RANDOM FOREST VALIDATION SUMMARY"
    )

    print(
        "================================="
    )

    print(
        results_df[
            [
                "fold",
                "mae",
                "rmse",
                "mape",
                "wape"
            ]
        ].round(2)
    )

    print(
        "\nAverage MAE:",
        round(
            results_df["mae"].mean(),
            2
        )
    )

    print(
        "Average RMSE:",
        round(
            results_df["rmse"].mean(),
            2
        )
    )

    print(
        "Average MAPE:",
        round(
            results_df["mape"].mean(),
            2
        ),
        "%"
    )

    print(
        "MAPE Std Dev:",
        round(
            results_df["mape"].std(),
            2
        ),
        "%"
    )

    print(
        "Average WAPE:",
        round(
            results_df["wape"].mean(),
            2
        ),
        "%"
    )

    # ----------------------------------------
    # Save results
    # ----------------------------------------

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "random_forest_walk_forward.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nResults saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    walk_forward_validation()