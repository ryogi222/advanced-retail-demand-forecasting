from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "random_forest_backtest_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "random_forest_backtest_mape.png"
)


def plot_backtest_performance():
    print("=" * 70)
    print("RANDOM FOREST BACKTEST PERFORMANCE VISUALISATION")
    print("=" * 70)

    df = pd.read_csv(
        SUMMARY_FILE,
        parse_dates=[
            "train_end",
            "test_start",
            "test_end",
        ],
    )

    df = df.sort_values(
        "fold"
    ).reset_index(drop=True)

    overall_mape = (
        df["mape"].mean()
    )

    print(f"\nFolds: {len(df)}")

    print("\nMAPE by fold:")

    for _, row in df.iterrows():
        print(
            f"Fold {int(row['fold'])}: "
            f"{row['mape']:.2f}%"
        )

    print(
        f"\nMean fold MAPE: "
        f"{overall_mape:.2f}%"
    )

    labels = [
        f"Fold {int(fold)}"
        for fold in df["fold"]
    ]

    values = df["mape"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.axhline(
        overall_mape,
        linestyle="--",
        linewidth=2,
        label=(
            f"Mean MAPE "
            f"({overall_mape:.2f}%)"
        ),
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    ax.set_title(
        "Random Forest Rolling-Origin Backtest"
    )

    ax.set_xlabel(
        "28-Day Test Fold"
    )

    ax.set_ylabel(
        "MAPE (%)"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
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
    plot_backtest_performance()
