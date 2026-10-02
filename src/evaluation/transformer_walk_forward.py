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


SEQUENCE_LENGTH = 28
TEST_DAYS = 28
N_FOLDS = 5
EPOCHS = 150


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
# Transformer
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
    indices = []

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

        indices.append(i)

    return (
        np.array(X),
        np.array(y),
        np.array(indices)
    )


# --------------------------------------------------
# Metrics
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


def calculate_wape(
    actual,
    predicted
):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return np.nan

    return (
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        / denominator
        * 100
    )


# --------------------------------------------------
# Walk-forward validation
# --------------------------------------------------

def transformer_walk_forward():

    print(
        "Loading feature dataset..."
    )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    print(
        "Observations:",
        len(df)
    )

    print(
        "Date range:",
        df["date"].min().date(),
        "to",
        df["date"].max().date()
    )

    total_test_days = (
        TEST_DAYS * N_FOLDS
    )

    first_test_start = (
        len(df)
        - total_test_days
    )

    results = []

    print(
        "\n--- TRANSFORMER WALK-FORWARD ---"
    )

    print(
        "Folds:",
        N_FOLDS
    )

    print(
        "Days per fold:",
        TEST_DAYS
    )

    print(
        "Epochs per fold:",
        EPOCHS
    )

    # --------------------------------------------------
    # Fold loop
    # --------------------------------------------------

    for fold in range(N_FOLDS):

        print(
            "\n================================="
        )

        print(
            f"FOLD {fold + 1}/{N_FOLDS}"
        )

        print(
            "================================="
        )

        test_start = (
            first_test_start
            + fold * TEST_DAYS
        )

        test_end = (
            test_start
            + TEST_DAYS
        )

        print(
            "Train:",
            df["date"].iloc[0].date(),
            "to",
            df["date"]
            .iloc[test_start - 1]
            .date()
        )

        print(
            "Test:",
            df["date"]
            .iloc[test_start]
            .date(),
            "to",
            df["date"]
            .iloc[test_end - 1]
            .date()
        )

        # ----------------------------------------------
        # Scaling
        # Fit ONLY on current fold training data
        # ----------------------------------------------

        feature_scaler = StandardScaler()

        target_scaler = StandardScaler()

        feature_scaler.fit(
            df[FEATURES]
            .iloc[:test_start]
        )

        target_scaler.fit(
            df[["sales"]]
            .iloc[:test_start]
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

        # ----------------------------------------------
        # Create sequences
        # ----------------------------------------------

        X, y, target_indices = (
            create_sequences(
                scaled_features,
                scaled_target,
                SEQUENCE_LENGTH
            )
        )

        train_mask = (
            target_indices
            < test_start
        )

        test_mask = (
            (target_indices >= test_start)
            &
            (target_indices < test_end)
        )

        X_train = X[
            train_mask
        ]

        y_train = y[
            train_mask
        ]

        X_test = X[
            test_mask
        ]

        print(
            "Training sequences:",
            len(X_train)
        )

        print(
            "Testing sequences:",
            len(X_test)
        )

        # ----------------------------------------------
        # Tensors
        # ----------------------------------------------

        X_train_tensor = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        y_train_tensor = torch.tensor(
            y_train,
            dtype=torch.float32
        ).reshape(-1, 1)

        X_test_tensor = torch.tensor(
            X_test,
            dtype=torch.float32
        )

        # ----------------------------------------------
        # Reset random seed for reproducibility
        # ----------------------------------------------

        torch.manual_seed(42)

        # ----------------------------------------------
        # New model for every fold
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Train
        # ----------------------------------------------

        for epoch in range(EPOCHS):

            model.train()

            optimizer.zero_grad()

            predictions = model(
                X_train_tensor
            )

            loss = criterion(
                predictions,
                y_train_tensor
            )

            loss.backward()

            optimizer.step()

            if (
                epoch == 0
                or (epoch + 1) % 25 == 0
            ):

                print(
                    f"Epoch "
                    f"{epoch + 1:3d}/"
                    f"{EPOCHS} "
                    f"- Loss: "
                    f"{loss.item():.6f}"
                )

        # ----------------------------------------------
        # Predict
        # ----------------------------------------------

        model.eval()

        with torch.no_grad():

            predictions_scaled = (
                model(
                    X_test_tensor
                )
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
            .iloc[
                test_start:test_end
            ]
            .values
        )

        # ----------------------------------------------
        # Metrics
        # ----------------------------------------------

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

        wape = calculate_wape(
            actual,
            predictions
        )

        print(
            "\nFold performance:"
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

        print(
            f"WAPE: {wape:.2f}%"
        )

        results.append(
            {
                "fold":
                    fold + 1,

                "train_end":
                    df["date"]
                    .iloc[
                        test_start - 1
                    ]
                    .date(),

                "test_start":
                    df["date"]
                    .iloc[
                        test_start
                    ]
                    .date(),

                "test_end":
                    df["date"]
                    .iloc[
                        test_end - 1
                    ]
                    .date(),

                "mae":
                    mae,

                "rmse":
                    rmse,

                "mape":
                    mape,

                "wape":
                    wape
            }
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n================================="
    )

    print(
        "TRANSFORMER VALIDATION SUMMARY"
    )

    print(
        "================================="
    )

    print(
        results_df[
            [
                "fold",
                "mae",
                "rmse",
                "mape",
                "wape"
            ]
        ].round(2)
    )

    print(
        "\nAverage MAE:",
        round(
            results_df["mae"].mean(),
            2
        )
    )

    print(
        "Average RMSE:",
        round(
            results_df["rmse"].mean(),
            2
        )
    )

    print(
        "Average MAPE:",
        round(
            results_df["mape"].mean(),
            2
        ),
        "%"
    )

    print(
        "MAPE Std Dev:",
        round(
            results_df["mape"].std(),
            2
        ),
        "%"
    )

    print(
        "Average WAPE:",
        round(
            results_df["wape"].mean(),
            2
        ),
        "%"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "transformer_walk_forward.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nResults saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    transformer_walk_forward()