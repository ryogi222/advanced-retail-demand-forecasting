from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# Configuration
# =========================================================

SEED = 42
LOOKBACK = 28
TEST_DAYS = 28
EPOCHS = 100
BATCH_SIZE = 32


# =========================================================
# Reproducibility
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_hierarchy_daily.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gru_ca1_forecast.csv"
)


# =========================================================
# MAPE
# =========================================================

def calculate_mape(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


# =========================================================
# Create sequences
# =========================================================

def create_sequences(data, lookback):

    X = []
    y = []

    for i in range(lookback, len(data)):

        X.append(
            data[i - lookback:i]
        )

        y.append(
            data[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# =========================================================
# Main GRU pipeline
# =========================================================

def run_gru_forecast():

    print("=" * 65)
    print("GRU STORE-LEVEL DEMAND FORECAST")
    print("=" * 65)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("\nLoading hierarchy dataset...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # Keep only CA_1 store-level series

    store_df = (
        df[
            (df["level"] == "store")
            & (df["series_id"] == "CA_1")
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Observations: {len(store_df):,}"
    )

    print(
        f"Date range: "
        f"{store_df['date'].min().date()} "
        f"to "
        f"{store_df['date'].max().date()}"
    )

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    train_df = store_df.iloc[
        :-TEST_DAYS
    ].copy()

    test_df = store_df.iloc[
        -TEST_DAYS:
    ].copy()

    print(
        f"Training observations: "
        f"{len(train_df):,}"
    )

    print(
        f"Test observations: "
        f"{len(test_df):,}"
    )

    # -----------------------------------------------------
    # Scaling
    # IMPORTANT:
    # Fit scaler only on training data.
    # -----------------------------------------------------

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_df[["sales"]]
    )

    # -----------------------------------------------------
    # Training sequences
    # -----------------------------------------------------

    X_train, y_train = create_sequences(
        train_scaled,
        LOOKBACK
    )

    print(
        f"\nX_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"y_train shape: "
        f"{y_train.shape}"
    )

    # -----------------------------------------------------
    # Validation split
    # Keep temporal order.
    # -----------------------------------------------------

    validation_size = int(
        len(X_train) * 0.1
    )

    X_fit = X_train[
        :-validation_size
    ]

    y_fit = y_train[
        :-validation_size
    ]

    X_val = X_train[
        -validation_size:
    ]

    y_val = y_train[
        -validation_size:
    ]

    print(
        f"Training sequences: "
        f"{len(X_fit):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(X_val):,}"
    )

    # -----------------------------------------------------
    # GRU model
    # -----------------------------------------------------

    model = Sequential([
        tf.keras.Input(
            shape=(LOOKBACK, 1)
        ),

        GRU(
            64,
            return_sequences=True
        ),

        Dropout(0.2),

        GRU(
            32
        ),

        Dropout(0.2),

        Dense(
            16,
            activation="relu"
        ),

        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    print("\nModel architecture:")
    model.summary()

    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    print("\nTraining GRU...")

    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[
            early_stopping
        ],
        verbose=1
    )

    # -----------------------------------------------------
    # Walk-forward forecast
    #
    # At each test date:
    # 1. Use previous 28 actual observations.
    # 2. Predict next day.
    # 3. Add actual observation to history.
    #
    # This evaluates one-step-ahead forecasting.
    # -----------------------------------------------------

    print(
        "\nGenerating 28-day "
        "walk-forward forecast..."
    )

    history_values = (
        train_df["sales"]
        .astype(float)
        .tolist()
    )

    predictions = []

    for actual in test_df["sales"]:

        input_window = np.array(
            history_values[-LOOKBACK:]
        ).reshape(-1, 1)

        input_df = pd.DataFrame(
            input_window,
            columns=["sales"]
        )

        input_scaled = scaler.transform(
            input_df
        )

        X_input = input_scaled.reshape(
            1,
            LOOKBACK,
            1
        )

        prediction_scaled = model.predict(
            X_input,
            verbose=0
        )

        prediction = scaler.inverse_transform(
            prediction_scaled
        )[0, 0]

        # Demand cannot be negative
        prediction = max(
            0.0,
            float(prediction)
        )

        predictions.append(
            prediction
        )

        # Add actual observation for
        # walk-forward evaluation
        history_values.append(
            float(actual)
        )


    # Evaluation
    # -----------------------------------------------------

    actual = (
        test_df["sales"]
        .to_numpy(dtype=float)
    )

    predictions = np.array(
        predictions
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    mape = calculate_mape(
        actual,
        predictions
    )

    print()
    print("=" * 65)
    print("GRU FORECAST RESULTS")
    print("=" * 65)

    print(
        f"MAE:  {mae:.2f}"
    )

    print(
        f"RMSE: {rmse:.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results = pd.DataFrame({
        "date": test_df["date"].values,
        "actual": actual,
        "gru_forecast": predictions
    })

    results["absolute_error"] = np.abs(
        results["actual"]
        - results["gru_forecast"]
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("Forecast saved:")
    print(OUTPUT_PATH)

    print()
    print("Sample predictions:")

    sample = results.head(10).copy()

    sample["gru_forecast"] = (
        sample["gru_forecast"].round(2)
    )

    sample["absolute_error"] = (
        sample["absolute_error"].round(2)
    )

    print(
        sample.to_string(index=False)
    )

    print()
    print(
        "Epochs completed:",
        len(history.history["loss"])
    )

# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    run_gru_forecast()
