from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "quantile_random_forest.joblib"
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


def load_model():
    """Load the trained Quantile Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_quantiles(model, features):
    """
    Generate P10, P50, P90 and mean forecasts
    from the Random Forest tree ensemble.
    """

    if isinstance(features, dict):
        features = pd.DataFrame([features])

    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "features must be a dictionary or pandas DataFrame"
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = features[FEATURES]

    tree_predictions = np.array(
        [
            tree.predict(X.to_numpy())
            for tree in model.estimators_
        ]
    )

    return {
        "p10": np.quantile(
            tree_predictions,
            0.10,
            axis=0
        ),
        "p50": np.quantile(
            tree_predictions,
            0.50,
            axis=0
        ),
        "p90": np.quantile(
            tree_predictions,
            0.90,
            axis=0
        ),
        "mean": tree_predictions.mean(axis=0),
    }


if __name__ == "__main__":
    model = load_model()

    print("Model loaded successfully.")
    print("Trees:", len(model.estimators_))
    print("Expected features:", len(FEATURES))