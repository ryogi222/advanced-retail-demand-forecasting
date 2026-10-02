from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features_calendar.csv"
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


def calculate_mape(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    return (
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def pinball_loss(actual, predicted, quantile):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    error = actual - predicted

    return np.mean(
        np.maximum(
            quantile * error,
            (quantile - 1) * error
        )
    )


def quantile_walk_forward():

    print(
        "Loading calendar-enhanced features..."
    )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    print("Observations:", len(df))
    print("Features:", len(FEATURES))

    test_days = 28
    n_folds = 5

    first_test_start = (
        len(df)
        - test_days * n_folds
    )

    results = []
    forecast_results = []

    print(
        "\n================================"
    )

    print(
        "PROBABILISTIC WALK-FORWARD"
    )

    print(
        "================================"
    )

    for fold in range(n_folds):

        test_start = (
            first_test_start
            + fold * test_days
        )

        test_end = (
            test_start
            + test_days
        )

        train = df.iloc[:test_start]
        test = df.iloc[test_start:test_end]

        X_train = train[FEATURES]
        y_train = train["sales"]

        X_test = test[FEATURES]
        y_test = test["sales"]

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

        # Convert to NumPy to avoid
        # individual-tree feature-name warnings.
        X_test_array = X_test.to_numpy()

        tree_predictions = np.array(
            [
                tree.predict(X_test_array)
                for tree in model.estimators_
            ]
        )

        p10 = np.quantile(
            tree_predictions,
            0.10,
            axis=0
        )

        p50 = np.quantile(
            tree_predictions,
            0.50,
            axis=0
        )

        p90 = np.quantile(
            tree_predictions,
            0.90,
            axis=0
        )

        actual = y_test.to_numpy()

        # ----------------------------------
        # Point forecast metrics
        # ----------------------------------

        mae = mean_absolute_error(
            actual,
            p50
        )

        mape = calculate_mape(
            actual,
            p50
        )

        # ----------------------------------
        # Interval metrics
        # ----------------------------------

        covered = (
            (actual >= p10)
            & (actual <= p90)
        )

        coverage = (
            covered.mean()
            * 100
        )

        interval_width = (
            p90 - p10
        )

        average_width = (
            interval_width.mean()
        )

        # ----------------------------------
        # Quantile losses
        # ----------------------------------

        p10_loss = pinball_loss(
            actual,
            p10,
            0.10
        )

        p50_loss = pinball_loss(
            actual,
            p50,
            0.50
        )

        p90_loss = pinball_loss(
            actual,
            p90,
            0.90
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
            f"P50 MAE: {mae:.2f}"
        )

        print(
            f"P50 MAPE: {mape:.2f}%"
        )

        print(
            f"P10-P90 Coverage: "
            f"{coverage:.2f}%"
        )

        print(
            f"Average Width: "
            f"{average_width:.2f}"
        )

        print(
            "Pinball Loss:"
        )

        print(
            f"  P10: {p10_loss:.2f}"
        )

        print(
            f"  P50: {p50_loss:.2f}"
        )

        print(
            f"  P90: {p90_loss:.2f}"
        )

        results.append(
            {
                "fold": fold + 1,
                "test_start":
                    test["date"].min(),
                "test_end":
                    test["date"].max(),
                "p50_mae": mae,
                "p50_mape": mape,
                "coverage": coverage,
                "average_width":
                    average_width,
                "p10_pinball":
                    p10_loss,
                "p50_pinball":
                    p50_loss,
                "p90_pinball":
                    p90_loss,
            }
        )

        fold_forecasts = pd.DataFrame(
            {
                "fold": fold + 1,
                "date": test["date"].to_numpy(),
                "actual": actual,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "interval_width":
                    interval_width,
                "covered": covered,
            }
        )

        forecast_results.append(
            fold_forecasts
        )

    # --------------------------------------
    # Summary
    # --------------------------------------

    results_df = pd.DataFrame(
        results
    )

    forecasts_df = pd.concat(
        forecast_results,
        ignore_index=True
    )

    print(
        "\n================================"
    )

    print(
        "PROBABILISTIC SUMMARY"
    )

    print(
        "================================"
    )

    print(
        results_df[
            [
                "fold",
                "p50_mae",
                "p50_mape",
                "coverage",
                "average_width",
            ]
        ].round(2)
    )

    print(
        f"\nAverage P50 MAE: "
        f"{results_df['p50_mae'].mean():.2f}"
    )

    print(
        f"Average P50 MAPE: "
        f"{results_df['p50_mape'].mean():.2f}%"
    )

    print(
        f"P50 MAPE Std Dev: "
        f"{results_df['p50_mape'].std():.2f}%"
    )

    print(
        f"\nAverage P10-P90 Coverage: "
        f"{results_df['coverage'].mean():.2f}%"
    )

    print(
        f"Target Coverage: 80.00%"
    )

    print(
        f"Coverage Gap: "
        f"{results_df['coverage'].mean() - 80:.2f} "
        f"percentage points"
    )

    print(
        f"\nAverage Interval Width: "
        f"{results_df['average_width'].mean():.2f} "
        f"units"
    )

    print(
        "\nAverage Pinball Loss"
    )

    print(
        f"P10: "
        f"{results_df['p10_pinball'].mean():.2f}"
    )

    print(
        f"P50: "
        f"{results_df['p50_pinball'].mean():.2f}"
    )

    print(
        f"P90: "
        f"{results_df['p90_pinball'].mean():.2f}"
    )

    # --------------------------------------
    # Save reports
    # --------------------------------------

    reports_dir = (
        PROJECT_ROOT
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        reports_dir
        / "quantile_walk_forward_summary.csv"
    )

    forecast_path = (
        reports_dir
        / "quantile_walk_forward_forecasts.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False
    )

    forecasts_df.to_csv(
        forecast_path,
        index=False
    )

    print(
        "\nSummary saved:"
    )

    print(summary_path)

    print(
        "\nForecasts saved:"
    )

    print(forecast_path)


if __name__ == "__main__":
    quantile_walk_forward()