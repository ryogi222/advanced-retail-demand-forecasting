from pathlib import Path

import pandas as pd

from src.inference.predict import (
    FEATURES,
    load_model,
    predict_quantiles,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_production_model_inference():
    """Verify that the saved production model can generate valid quantiles."""

    feature_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ca_1_daily_features_calendar.csv"
    )

    data = pd.read_csv(feature_path)

    # Use one known feature row
    row = data.iloc[[-1]]

    model = load_model()

    prediction = predict_quantiles(
        model,
        row[FEATURES],
    )

    assert "p10" in prediction
    assert "p50" in prediction
    assert "p90" in prediction
    assert "mean" in prediction

    p10 = float(prediction["p10"][0])
    p50 = float(prediction["p50"][0])
    p90 = float(prediction["p90"][0])

    assert p10 >= 0
    assert p50 >= 0
    assert p90 >= 0

    # Quantiles must be ordered correctly
    assert p10 <= p50 <= p90