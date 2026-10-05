import numpy as np
import pandas as pd


def build_next_day_features(history):
    """
    Build lag and rolling demand features for the next day
    using only historically observed sales.
    """

    if not isinstance(history, pd.DataFrame):
        raise TypeError(
            "history must be a pandas DataFrame"
        )

    required_columns = [
        "date",
        "sales",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    history = history.copy()

    history["date"] = pd.to_datetime(
        history["date"]
    )

    history = history.sort_values(
        "date"
    ).reset_index(drop=True)

    if len(history) < 28:
        raise ValueError(
            "At least 28 days of sales history are required."
        )

    sales = history["sales"]

    next_date = (
        history["date"].iloc[-1]
        + pd.Timedelta(days=1)
    )

    features = {
        "date": next_date,
        "day_of_week": next_date.dayofweek,
        "day_of_month": next_date.day,
        "month": next_date.month,
        "year": next_date.year,
        "week_of_year": int(
            next_date.isocalendar().week
        ),
        "is_weekend": int(
            next_date.dayofweek >= 5
        ),
        "lag_1": sales.iloc[-1],
        "lag_7": sales.iloc[-7],
        "lag_14": sales.iloc[-14],
        "lag_28": sales.iloc[-28],
        "rolling_mean_7": sales.iloc[-7:].mean(),
        "rolling_mean_28": sales.iloc[-28:].mean(),
        "rolling_std_7": sales.iloc[-7:].std(),
        "rolling_std_28": sales.iloc[-28:].std(),
    }

    return features
def add_calendar_features(features, calendar):
    """
    Add SNAP and event features for the forecast date
    using the M5 calendar.
    """

    calendar = calendar.copy()

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    forecast_date = pd.to_datetime(
        features["date"]
    )

    calendar_row = calendar[
        calendar["date"] == forecast_date
    ]

    if calendar_row.empty:
        raise ValueError(
            f"Calendar data not found for {forecast_date.date()}"
        )

    row = calendar_row.iloc[0]

    event_name_1 = row["event_name_1"]
    event_name_2 = row["event_name_2"]

    event_type_1 = row["event_type_1"]
    event_type_2 = row["event_type_2"]

    features["snap_CA"] = int(
        row["snap_CA"]
    )

    features["has_event_1"] = int(
        pd.notna(event_name_1)
    )

    features["has_event_2"] = int(
        pd.notna(event_name_2)
    )

    features["has_any_event"] = int(
        features["has_event_1"]
        or features["has_event_2"]
    )

    event_types = {
        str(event_type_1).lower(),
        str(event_type_2).lower(),
    }

    features["event_type_cultural"] = int(
        "cultural" in event_types
    )

    features["event_type_national"] = int(
        "national" in event_types
    )

    features["event_type_religious"] = int(
        "religious" in event_types
    )

    features["event_type_sporting"] = int(
        "sporting" in event_types
    )

    event_names = {
        str(event_name_1).lower(),
        str(event_name_2).lower(),
    }

    features["is_superbowl"] = int(
        "superbowl" in event_names
    )

    features["is_valentines"] = int(
        "valentinesday" in event_names
    )

    features["is_christmas"] = int(
        "christmas" in event_names
    )

    return features