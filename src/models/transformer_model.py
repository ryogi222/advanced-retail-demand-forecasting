from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
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


# --------------------------------------------------
# Positional Encoding
# --------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=500
    ):

        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return (
            x
            + self.pe[:, :x.size(1)]
        )


# --------------------------------------------------
# Transformer Model
# --------------------------------------------------

class DemandTransformer(nn.Module):

    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,
                dropout=dropout,
                batch_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        self.fc = nn.Linear(
            d_model,
            1
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        x = self.transformer(x)

        # Use representation from final day
        x = x[:, -1, :]

        return self.fc(x)


# --------------------------------------------------
# Sequence creation
# --------------------------------------------------

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


# --------------------------------------------------
# MAPE
# --------------------------------------------------

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


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_transformer():

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

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Sequences
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Tensors
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = DemandTransformer(
        input_size=len(FEATURES),
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.2
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    epochs = 150

    print(
        "\nTraining Transformer..."
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

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
        "\n--- TRANSFORMER PERFORMANCE ---"
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

    # --------------------------------------------------
    # Forecast comparison
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = (
        PROJECT_ROOT
        / "models"
        / "transformer_demand_model.pt"
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

    train_transformer()
