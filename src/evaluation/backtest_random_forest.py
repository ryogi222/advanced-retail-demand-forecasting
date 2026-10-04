from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features_calendar.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "random_forest_backtest.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "random_forest_backtest_summary.csv"
)


TEST_DAYS = 28
N_FOLDS = 4


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
    "rolling_std_28",
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


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    mape = (
        mean_absolute_percentage_error(
            actual,
            predicted,
        )
        * 100
    )

    return mae, rmse, mape


def run_backtest():
    print("=" * 70)
    print("RANDOM FOREST ROLLING-ORIGIN BACKTEST")
    print("=" * 70)

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    print(f"\nRows: {len(df)}")
    print(
        f"Period: "
        f"{df['date'].min().date()} to "
        f"{df['date'].max().date()}"
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features: "
            + ", ".join(missing_features)
        )

    all_predictions = []
    fold_summaries = []

    total_test_days = (
        TEST_DAYS * N_FOLDS
    )

    first_test_start = (
        len(df) - total_test_days
    )

    for fold in range(N_FOLDS):
        test_start = (
            first_test_start
            + fold * TEST_DAYS
        )

        test_end = (
            test_start
            + TEST_DAYS
        )

        train = df.iloc[:test_start].copy()
        test = df.iloc[
            test_start:test_end
        ].copy()

        X_train = train[FEATURES]
        y_train = train["sales"]

        X_test = test[FEATURES]
        y_test = test["sales"]

        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        tree_predictions = np.array(
            [
                tree.predict(
                    X_test.to_numpy()
                )
                for tree in model.estimators_
            ]
        )

        mean_forecast = (
            tree_predictions.mean(axis=0)
        )

        p10 = np.quantile(
            tree_predictions,
            0.10,
            axis=0,
        )

        p50 = np.quantile(
            tree_predictions,
            0.50,
            axis=0,
        )

        p90 = np.quantile(
            tree_predictions,
            0.90,
            axis=0,
        )

        actual = y_test.to_numpy()

        mae, rmse, mape = (
            calculate_metrics(
                actual,
                p50,
            )
        )

        covered = (
            (actual >= p10)
            & (actual <= p90)
        )

        coverage = (
            covered.mean() * 100
        )

        fold_number = fold + 1

        print("\n" + "-" * 70)
        print(f"Fold {fold_number}")
        print(
            f"Train end: "
            f"{train['date'].max().date()}"
        )
        print(
            f"Test period: "
            f"{test['date'].min().date()} to "
            f"{test['date'].max().date()}"
        )
        print(f"MAE:      {mae:.2f}")
        print(f"RMSE:     {rmse:.2f}")
        print(f"MAPE:     {mape:.2f}%")
        print(
            f"Coverage: {coverage:.2f}%"
        )

        fold_result = pd.DataFrame(
            {
                "fold": fold_number,
                "date": test["date"].values,
                "actual": actual,
                "mean_forecast": mean_forecast,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "covered": covered,
            }
        )

        all_predictions.append(
            fold_result
        )

        fold_summaries.append(
            {
                "fold": fold_number,
                "train_end":
                    train["date"].max(),
                "test_start":
                    test["date"].min(),
                "test_end":
                    test["date"].max(),
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "coverage": coverage,
            }
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        fold_summaries
    )

    overall_mae, overall_rmse, overall_mape = (
        calculate_metrics(
            predictions["actual"],
            predictions["p50"],
        )
    )

    overall_coverage = (
        predictions["covered"].mean()
        * 100
    )

    print("\n" + "=" * 70)
    print("OVERALL BACKTEST PERFORMANCE")
    print("=" * 70)

    print(
        f"Test observations: "
        f"{len(predictions)}"
    )
    print(
        f"P50 MAE:     "
        f"{overall_mae:.2f}"
    )
    print(
        f"P50 RMSE:    "
        f"{overall_rmse:.2f}"
    )
    print(
        f"P50 MAPE:    "
        f"{overall_mape:.2f}%"
    )
    print(
        f"P10-P90 Coverage: "
        f"{overall_coverage:.2f}%"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\nPredictions saved:")
    print(OUTPUT_FILE)

    print("\nFold summary saved:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    run_backtest()