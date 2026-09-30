from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ca_1_daily_features.csv"
)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------
# LSTM Model
# --------------------------------------------------

class DemandLSTM(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Take final time step
        output = output[:, -1, :]

        return self.fc(output)


# --------------------------------------------------
# Create sequences
# --------------------------------------------------

def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(sequence_length, len(data)):

        X.append(
            data[i-sequence_length:i]
        )

        y.append(
            data[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# --------------------------------------------------
# MAPE
# --------------------------------------------------

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


# --------------------------------------------------
# Train
# --------------------------------------------------

def train_lstm():

    print("Loading daily sales data...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    sales = df[
        ["date", "sales"]
    ].copy()

    sales = sales.sort_values(
        "date"
    ).reset_index(drop=True)

    print("Observations:", len(sales))

    # --------------------------------------------------
    # Train/Test split
    # --------------------------------------------------

    test_days = 28
    sequence_length = 28

    train_size = len(sales) - test_days

    train_sales = sales[
        "sales"
    ].iloc[:train_size].values.reshape(-1, 1)

    # --------------------------------------------------
    # Scaling
    # Fit ONLY on training data
    # --------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(train_sales)

    all_scaled = scaler.transform(
        sales["sales"].values.reshape(-1, 1)
    )

    # --------------------------------------------------
    # Sequences
    # --------------------------------------------------

    X, y = create_sequences(
        all_scaled,
        sequence_length
    )

    target_indices = np.arange(
        sequence_length,
        len(sales)
    )

    train_mask = target_indices < train_size
    test_mask = target_indices >= train_size

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("\n--- SEQUENCE DATA ---")
    print("Sequence length:", sequence_length)
    print("Training sequences:", len(X_train))
    print("Testing sequences:", len(X_test))

    # --------------------------------------------------
    # Convert to PyTorch tensors
    # --------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = DemandLSTM()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    epochs = 100

    print("\nTraining LSTM...")

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
            (epoch + 1) % 10 == 0
            or epoch == 0
        ):

            print(
                f"Epoch {epoch + 1:3d}/{epochs} "
                f"- Loss: {loss.item():.6f}"
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

    predictions = scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual = sales[
        "sales"
    ].iloc[train_size:].values

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

    print("\n--- LSTM PERFORMANCE ---")

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # --------------------------------------------------
    # Sample forecasts
    # --------------------------------------------------

    comparison = pd.DataFrame({

        "date":
            sales["date"].iloc[
                train_size:
            ].values,

        "actual":
            actual,

        "prediction":
            predictions
    })

    print("\n--- SAMPLE FORECASTS ---")

    print(
        comparison.head(10)
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = (
        PROJECT_ROOT
        / "models"
        / "lstm_demand_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print("\nModel saved:")
    print(model_path)


if __name__ == "__main__":
    train_lstm()
