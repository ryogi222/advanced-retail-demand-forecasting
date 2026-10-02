from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features.csv"
)


FEATURES = [
    "sales",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]


torch.manual_seed(42)
np.random.seed(42)


class TemporalCNN(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                in_channels=input_features,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        # Input:
        # batch, sequence, features

        # Conv1D expects:
        # batch, features, sequence

        x = x.permute(
            0,
            2,
            1
        )

        x = self.network(x)

        x = x.squeeze(-1)

        return self.fc(x)


def create_sequences(
    features,
    targets,
    sequence_length
):

    X = []
    y = []

    for i in range(
        sequence_length,
        len(features)
    ):

        X.append(
            features[
                i-sequence_length:i
            ]
        )

        y.append(
            targets[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


def calculate_mape(
    actual,
    predicted
):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    return (
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


def train_temporal_cnn():

    print(
        "Loading multivariate feature dataset..."
    )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    print(
        "Observations:",
        len(df)
    )

    print(
        "Features:",
        len(FEATURES)
    )

    test_days = 28
    sequence_length = 28

    train_size = (
        len(df)
        - test_days
    )

    # ---------------------------------
    # Scaling
    # ---------------------------------

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(
        df[FEATURES]
        .iloc[:train_size]
    )

    target_scaler.fit(
        df[["sales"]]
        .iloc[:train_size]
    )

    scaled_features = (
        feature_scaler.transform(
            df[FEATURES]
        )
    )

    scaled_target = (
        target_scaler
        .transform(
            df[["sales"]]
        )
        .flatten()
    )

    # ---------------------------------
    # Create sequences
    # ---------------------------------

    X, y = create_sequences(
        scaled_features,
        scaled_target,
        sequence_length
    )

    target_indices = np.arange(
        sequence_length,
        len(df)
    )

    train_mask = (
        target_indices
        < train_size
    )

    test_mask = (
        target_indices
        >= train_size
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]

    print(
        "\n--- SEQUENCE DATA ---"
    )

    print(
        "Training sequences:",
        len(X_train)
    )

    print(
        "Testing sequences:",
        len(X_test)
    )

    print(
        "Input features:",
        X_train.shape[2]
    )

    # ---------------------------------
    # Convert to tensors
    # ---------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    ).reshape(-1, 1)

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = TemporalCNN(
        input_features=len(FEATURES)
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 150

    print(
        "\nTraining Temporal CNN..."
    )

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        predictions = model(
            X_train
        )

        loss = criterion(
            predictions,
            y_train
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):

            print(
                f"Epoch "
                f"{epoch + 1:3d}/"
                f"{epochs} "
                f"- Loss: "
                f"{loss.item():.6f}"
            )

    # ---------------------------------
    # Prediction
    # ---------------------------------

    model.eval()

    with torch.no_grad():

        predictions_scaled = (
            model(X_test)
            .numpy()
        )

    predictions = (
        target_scaler
        .inverse_transform(
            predictions_scaled
        )
        .flatten()
    )

    actual = (
        df["sales"]
        .iloc[train_size:]
        .values
    )

    # ---------------------------------
    # Metrics
    # ---------------------------------

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

    print(
        "\n--- TEMPORAL CNN PERFORMANCE ---"
    )

    print(
        f"MAE:  {mae:.2f}"
    )

    print(
        f"RMSE: {rmse:.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    # ---------------------------------
    # Forecast comparison
    # ---------------------------------

    comparison = pd.DataFrame({

        "date":
            df["date"]
            .iloc[train_size:]
            .values,

        "actual":
            actual,

        "prediction":
            predictions
    })

    print(
        "\n--- SAMPLE FORECASTS ---"
    )

    print(
        comparison.head(10)
    )

    # ---------------------------------
    # Save model
    # ---------------------------------

    model_path = (
        PROJECT_ROOT
        / "models"
        / "temporal_cnn_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        "\nModel saved:"
    )

    print(
        model_path
    )


if __name__ == "__main__":

    train_temporal_cnn()