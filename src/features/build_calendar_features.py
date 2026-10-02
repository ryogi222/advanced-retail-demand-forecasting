from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SALES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features.csv"
)

CALENDAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "m5"
    / "calendar.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features_calendar.csv"
)


def build_calendar_features():

    print("Loading existing daily features...")

    sales = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"]
    )

    print(
        "Sales rows:",
        len(sales)
    )

    print(
        "Sales date range:",
        sales["date"].min().date(),
        "to",
        sales["date"].max().date()
    )

    # --------------------------------------------------
    # Load M5 calendar
    # --------------------------------------------------

    print("\nLoading M5 calendar...")

    calendar = pd.read_csv(
        CALENDAR_PATH,
        parse_dates=["date"]
    )

    calendar_columns = [
        "date",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA"
    ]

    calendar = calendar[
        calendar_columns
    ].copy()

    print(
        "Calendar rows:",
        len(calendar)
    )

    # --------------------------------------------------
    # Missing event values
    # --------------------------------------------------

    event_columns = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2"
    ]

    calendar[event_columns] = (
        calendar[event_columns]
        .fillna("None")
    )

    # --------------------------------------------------
    # Simple event indicators
    # --------------------------------------------------

    calendar["has_event_1"] = (
        calendar["event_name_1"]
        != "None"
    ).astype(int)

    calendar["has_event_2"] = (
        calendar["event_name_2"]
        != "None"
    ).astype(int)

    calendar["has_any_event"] = (
        (
            calendar["has_event_1"] == 1
        )
        |
        (
            calendar["has_event_2"] == 1
        )
    ).astype(int)

    # --------------------------------------------------
    # Event type indicators
    # --------------------------------------------------

    event_types = [
        "Cultural",
        "National",
        "Religious",
        "Sporting"
    ]

    for event_type in event_types:

        column_name = (
            "event_type_"
            + event_type.lower()
        )

        calendar[column_name] = (
            (
                calendar["event_type_1"]
                == event_type
            )
            |
            (
                calendar["event_type_2"]
                == event_type
            )
        ).astype(int)

    # --------------------------------------------------
    # Specific high-level event indicators
    # --------------------------------------------------

    calendar["is_superbowl"] = (
        (
            calendar["event_name_1"]
            == "SuperBowl"
        )
        |
        (
            calendar["event_name_2"]
            == "SuperBowl"
        )
    ).astype(int)

    calendar["is_valentines"] = (
        (
            calendar["event_name_1"]
            == "ValentinesDay"
        )
        |
        (
            calendar["event_name_2"]
            == "ValentinesDay"
        )
    ).astype(int)

    calendar["is_christmas"] = (
        (
            calendar["event_name_1"]
            == "Christmas"
        )
        |
        (
            calendar["event_name_2"]
            == "Christmas"
        )
    ).astype(int)

    # --------------------------------------------------
    # SNAP feature
    # --------------------------------------------------

    calendar["snap_CA"] = (
        calendar["snap_CA"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------
    # Keep numerical model features
    # --------------------------------------------------

    calendar_features = [
        "date",
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
        "is_christmas"
    ]

    calendar_model = calendar[
        calendar_features
    ].copy()

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------

    print(
        "\nMerging calendar features..."
    )

    enhanced = sales.merge(
        calendar_model,
        on="date",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print(
        "Enhanced rows:",
        len(enhanced)
    )

    print(
        "Enhanced columns:",
        len(enhanced.columns)
    )

    missing_calendar = (
        enhanced[
            calendar_features[1:]
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Missing calendar feature values:",
        missing_calendar
    )

    if len(enhanced) != len(sales):

        raise ValueError(
            "Row count changed after calendar merge."
        )

    if missing_calendar != 0:

        raise ValueError(
            "Missing calendar features detected."
        )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    print(
        "\n--- EVENT SUMMARY ---"
    )

    print(
        "Days with events:",
        enhanced["has_any_event"].sum()
    )

    print(
        "SNAP CA days:",
        enhanced["snap_CA"].sum()
    )

    print(
        "Cultural event days:",
        enhanced[
            "event_type_cultural"
        ].sum()
    )

    print(
        "National event days:",
        enhanced[
            "event_type_national"
        ].sum()
    )

    print(
        "Religious event days:",
        enhanced[
            "event_type_religious"
        ].sum()
    )

    print(
        "Sporting event days:",
        enhanced[
            "event_type_sporting"
        ].sum()
    )

    print(
        "Christmas days:",
        enhanced[
            "is_christmas"
        ].sum()
    )

    # --------------------------------------------------
    # Compare demand
    # --------------------------------------------------

    event_demand = (
        enhanced.groupby(
            "has_any_event"
        )["sales"]
        .agg(
            [
                "count",
                "mean",
                "median"
            ]
        )
    )

    print(
        "\n--- DEMAND BY EVENT STATUS ---"
    )

    print(
        event_demand.round(2)
    )

    snap_demand = (
        enhanced.groupby(
            "snap_CA"
        )["sales"]
        .agg(
            [
                "count",
                "mean",
                "median"
            ]
        )
    )

    print(
        "\n--- DEMAND BY SNAP STATUS ---"
    )

    print(
        snap_demand.round(2)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    enhanced.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nEnhanced feature dataset saved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nFinal shape:",
        enhanced.shape
    )

    print(
        "\nNew calendar features:"
    )

    for column in calendar_features[1:]:

        print(
            "-",
            column
        )


if __name__ == "__main__":

    build_calendar_features()