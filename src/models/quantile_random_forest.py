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


def quantile_random_forest():

    print("Loading calendar-enhanced features...")

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

    # -----------------------------------
    # Final 28-day holdout
    # -----------------------------------

    test_days = 28

    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]

    X_train = train[FEATURES]
    y_train = train["sales"]

    X_test = test[FEATURES]
    y_test = test["sales"]

    print(
        "\nTraining observations:",
        len(train)
    )

    print(
        "Test observations:",
        len(test)
    )

    print(
        "Test period:",
        test["date"].min().date(),
        "to",
        test["date"].max().date()
    )

    # -----------------------------------
    # Random Forest
    # -----------------------------------

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------
    # Individual tree predictions
    # -----------------------------------

    print(
        "Generating prediction distribution..."
    )

    tree_predictions = np.array(
        [
            tree.predict(X_test)
            for tree in model.estimators_
        ]
    )

    print(
        "Prediction matrix shape:",
        tree_predictions.shape
    )

    # -----------------------------------
    # Quantiles
    # -----------------------------------

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

    # Standard RF mean forecast
    mean_forecast = (
        tree_predictions.mean(axis=0)
    )

    # -----------------------------------
    # Accuracy
    # -----------------------------------

    mae = mean_absolute_error(
        y_test,
        p50
    )
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            p50
        )
    )

    mape = calculate_mape(
        y_test,
        p50
    )

    print(
        f"\nP50 MAE:  {mae:.2f}"
    )
    print(
        f"P50 RMSE: {rmse:.2f}"
    )
    print(
        f"P50 MAPE: {mape:.2f}%"
    )

    # -----------------------------------
    # Interval coverage
    # -----------------------------------

    actual = y_test.to_numpy()

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

    print(
        f"\nP10-P90 coverage: "
        f"{coverage:.2f}%"
    )

    print(
        f"Average interval width: "
        f"{average_width:.2f} units"
    )

    # -----------------------------------
    # Pinball loss
    # -----------------------------------

    def pinball_loss(
        actual,
        predicted,
        quantile
    ):

        error = actual - predicted

        return np.mean(
            np.maximum(
                quantile * error,
                (quantile - 1) * error
            )
        )

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
        "\nPinball Loss"
    )

    print(
        f"P10: {p10_loss:.2f}"
    )

    print(
        f"P50: {p50_loss:.2f}"
    )

    print(
        f"P90: {p90_loss:.2f}"
    )

    # -----------------------------------
    # Results
    # -----------------------------------

    results = pd.DataFrame(
        {
            "date": test["date"],
            "actual": actual,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "mean_forecast": mean_forecast,
            "interval_width": interval_width,
            "covered": covered,
        }
    )

    print(
        "\nForecast sample:"
    )

    print(
        results.head(10).round(2)
    )

    # -----------------------------------
    # Save
    # -----------------------------------

    reports_dir = (
        PROJECT_ROOT
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        reports_dir
        / "quantile_random_forest_forecast.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        "\nForecast saved:"
    )

    print(output_path)


if __name__ == "__main__":
    quantile_random_forest()