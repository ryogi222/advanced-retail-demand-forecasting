from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPARISON_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "model_comparison_mape.png"
)


def plot_model_comparison():
    print("=" * 70)
    print("FORECASTING MODEL COMPARISON VISUALISATION")
    print("=" * 70)

    df = pd.read_csv(
        COMPARISON_FILE
    )

    required_columns = [
        "model",
        "MAE",
        "RMSE",
        "MAPE",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    df = (
        df.sort_values(
            "MAPE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    print(
        f"\nModels: {len(df)}"
    )

    print("\nModel ranking by MAPE:")

    for index, row in df.iterrows():
        print(
            f"{index + 1}. "
            f"{row['model']}: "
            f"{row['MAPE']:.2f}%"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.barh(
        df["model"],
        df["MAPE"],
    )

    ax.invert_yaxis()

    for bar, value in zip(
        bars,
        df["MAPE"],
    ):
        ax.text(
            value + 0.15,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
        )

    ax.set_title(
        "CA_1 Forecasting Model Comparison"
    )

    ax.set_xlabel(
        "MAPE (%) — Lower Is Better"
    )

    ax.set_ylabel(
        "Forecasting Model"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    max_mape = df["MAPE"].max()

    ax.set_xlim(
        0,
        max_mape * 1.15,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    print("\nChart saved:")
    print(OUTPUT_FILE)

    plt.show()


if __name__ == "__main__":
    plot_model_comparison()