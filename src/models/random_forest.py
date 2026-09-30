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
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def train_random_forest():

    print("Loading feature dataset...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    # Final 28 days remain unseen
    test_days = 28

    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()

    X_train = train[FEATURES]
    y_train = train["sales"]

    X_test = test[FEATURES]
    y_test = test["sales"]

    print("\nTraining observations:", len(X_train))
    print("Testing observations:", len(X_test))
    print("Number of features:", len(FEATURES))

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

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

    print("\n--- RANDOM FOREST PERFORMANCE ---")

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    comparison = pd.DataFrame({
        "date": test["date"],
        "actual": y_test,
        "prediction": predictions
    })

    print("\n--- SAMPLE FORECASTS ---")
    print(comparison.head(10))

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\n--- FEATURE IMPORTANCE ---")
    print(importance)


if __name__ == "__main__":
    train_random_forest()