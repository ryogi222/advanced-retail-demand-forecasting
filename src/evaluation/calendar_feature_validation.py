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
    / "ca_1_daily_features_calendar.csv"
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

FEATURES = BASE_FEATURES + CALENDAR_FEATURES


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


def calculate_wape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(actual - predicted))
        / denominator
        * 100
    )


def calendar_feature_validation():

    print("Loading enhanced feature dataset...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    print("Observations:", len(df))
    print("Total model features:", len(FEATURES))
    print("Base features:", len(BASE_FEATURES))
    print("Calendar features:", len(CALENDAR_FEATURES))

    test_days = 28
    n_folds = 5

    first_test_start = (
        len(df)
        - (test_days * n_folds)
    )

    results = []
    importance_results = []

    print("\n--- ENHANCED RANDOM FOREST WALK-FORWARD ---")

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

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

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

        print(f"\nFold {fold + 1}")
        print(
            "Test:",
            test["date"].min().date(),
            "to",
            test["date"].max().date(),
        )
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"WAPE: {wape:.2f}%")

        results.append(
            {
                "fold": fold + 1,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "wape": wape,
            }
        )

        importance = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": model.feature_importances_,
            }
        )

        importance["fold"] = fold + 1

        importance_results.append(
            importance
        )

    results_df = pd.DataFrame(results)

    average_mae = results_df["mae"].mean()
    average_rmse = results_df["rmse"].mean()
    average_mape = results_df["mape"].mean()
    mape_std = results_df["mape"].std()
    average_wape = results_df["wape"].mean()

    print("\n================================")
    print("ENHANCED RANDOM FOREST SUMMARY")
    print("================================")

    print(
        results_df[
            ["fold", "mae", "rmse", "mape", "wape"]
        ].round(2)
    )

    print(f"\nAverage MAE: {average_mae:.2f}")
    print(f"Average RMSE: {average_rmse:.2f}")
    print(f"Average MAPE: {average_mape:.2f}%")
    print(f"MAPE Std Dev: {mape_std:.2f}%")
    print(f"Average WAPE: {average_wape:.2f}%")

    original_mape = 5.93
    original_wape = 6.14

    mape_change = (
        original_mape
        - average_mape
    )

    relative_improvement = (
        mape_change
        / original_mape
        * 100
    )

    print("\n================================")
    print("CALENDAR FEATURE IMPACT")
    print("================================")

    print(
        f"Original RF MAPE: {original_mape:.2f}%"
    )

    print(
        f"Enhanced RF MAPE: {average_mape:.2f}%"
    )

    print(
        f"Original RF WAPE: {original_wape:.2f}%"
    )

    print(
        f"Enhanced RF WAPE: {average_wape:.2f}%"
    )

    print(
        f"MAPE improvement: "
        f"{mape_change:.2f} percentage points"
    )

    print(
        f"Relative MAPE improvement: "
        f"{relative_improvement:.2f}%"
    )

    importance_df = pd.concat(
        importance_results,
        ignore_index=True
    )

    average_importance = (
        importance_df
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n================================")
    print("TOP 15 FEATURE IMPORTANCES")
    print("================================")

    print(
        average_importance
        .head(15)
        .round(4)
    )

    print("\nCalendar features:")

    calendar_importance = (
        average_importance[
            average_importance.index.isin(
                CALENDAR_FEATURES
            )
        ]
        .sort_values(ascending=False)
    )

    print(
        calendar_importance.round(4)
    )

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
        / "calendar_feature_walk_forward.csv"
    )

    importance_path = (
        reports_dir
        / "calendar_feature_importance.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    average_importance.reset_index().to_csv(
        importance_path,
        index=False
    )

    print("\nResults saved:")
    print(results_path)

    print("\nFeature importance saved:")
    print(importance_path)


if __name__ == "__main__":
    calendar_feature_validation()