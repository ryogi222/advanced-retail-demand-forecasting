from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.inference.predict import load_model, predict_quantiles
from src.inference.future_features import (
    build_next_day_features,
    add_calendar_features,
)
from src.inference.multi_day_forecast import forecast_multiple_days

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide",
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Forecasting Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Comparison",
        "Probabilistic Forecast",
        "Backtesting",
        "Hierarchical Forecasting",
        "Live Prediction",
    ],
)
st.sidebar.divider()

st.sidebar.markdown(
    """
    **Dataset:** M5 Forecasting Accuracy

    **Store:** CA_1

    **Best Model:** Quantile Random Forest

    **Best MAPE:** 5.07%
    """
)


# --------------------------------------------------
# Reusable header
# --------------------------------------------------

def show_header():
    st.title("Advanced Retail Demand Forecasting")

    st.write(
        """
        Interactive forecasting dashboard for the M5 retail dataset,
        comparing machine learning and deep learning forecasting models
        for the CA_1 store.
        """
    )

    st.divider()


# --------------------------------------------------
# Overview page
# --------------------------------------------------

def show_overview():

    show_header()

    st.subheader("Forecasting Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Model", "Quantile RF")

    with col2:
        st.metric("Holdout MAPE", "5.07%")

    with col3:
        st.metric("Backtest MAPE", "5.05%")

    with col4:
        st.metric("Models Compared", "7")

    st.subheader("Project Overview")

    st.markdown(
        """
        This project evaluates multiple approaches to retail demand forecasting:

        - Moving Average Baseline
        - Random Forest
        - GRU
        - LSTM
        - Temporal CNN
        - Transformer
        - Probabilistic Random Forest

        The forecasting pipeline also includes **hierarchical forecasting**,
        **prediction intervals**, and **rolling-origin backtesting**.
        """
    )

    st.subheader("Business Value")

    st.markdown(
        """
        Accurate demand forecasting can support:

        - inventory replenishment
        - product availability
        - safety-stock planning
        - supply-chain planning
        - workforce planning
        - promotion planning
        - waste reduction

        Probabilistic forecasting also helps decision-makers understand
        uncertainty around expected demand.
        """
    )


# --------------------------------------------------
# Model comparison page
# --------------------------------------------------

def show_model_comparison():

    show_header()

    st.subheader("Model Comparison")

    comparison_path = DATA_DIR / "model_comparison.csv"

    if not comparison_path.exists():
        st.error(f"File not found: {comparison_path}")
        return

    df = pd.read_csv(comparison_path)
    df = df.sort_values("MAPE").reset_index(drop=True)

    st.markdown(
        """
        Seven forecasting approaches are compared using **MAE, RMSE and MAPE**.

        Lower values indicate better forecasting performance.
        """
    )

    col1, col2, col3 = st.columns(3)

    best = df.iloc[0]

    with col1:
        st.metric("Best Model", best["model"])

    with col2:
        st.metric("Best MAPE", f"{best['MAPE']:.2f}%")

    with col3:
        st.metric("Models Evaluated", len(df))

    display_df = df.copy()

    display_df["MAE"] = display_df["MAE"].round(2)
    display_df["RMSE"] = display_df["RMSE"].round(2)
    display_df["MAPE"] = display_df["MAPE"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("MAPE Ranking")

    chart_df = (
        df[["model", "MAPE"]]
        .sort_values("MAPE", ascending=False)
        .set_index("model")
    )

    st.bar_chart(
        chart_df,
        horizontal=True,
    )

    st.success(
        f"{best['model']} achieved the best holdout "
        f"MAPE at {best['MAPE']:.2f}%."
    )


# --------------------------------------------------
# Probabilistic forecast page
# --------------------------------------------------

def show_probabilistic_forecast():

    show_header()

    st.subheader("Probabilistic Demand Forecast")

    forecast_path = REPORTS_DIR / "quantile_random_forest_forecast.csv"

    if not forecast_path.exists():
        st.error(f"File not found: {forecast_path}")
        return

    df = pd.read_csv(forecast_path)

    df["date"] = pd.to_datetime(df["date"])

    required = ["date", "actual", "p10", "p50", "p90"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        st.error(
            f"Missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
        return

    st.markdown(
        """
        Instead of producing only one demand estimate, the probabilistic
        Random Forest provides a range of possible demand outcomes.

        - **P10** — lower demand estimate
        - **P50** — median forecast
        - **P90** — upper demand estimate
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("P50 MAPE", "5.07%")

    with col2:
        st.metric("P10-P90 Coverage", "71.43%")

    with col3:
        st.metric("Average Interval Width", "700.31 units")

    st.subheader("28-Day Forecast")

    chart_df = (
        df[["date", "actual", "p10", "p50", "p90"]]
        .set_index("date")
    )

    st.line_chart(chart_df)

    with st.expander("View forecast data"):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# Backtesting page
# --------------------------------------------------

def show_backtesting():

    show_header()

    st.subheader("Rolling-Origin Backtesting")

    summary_path = REPORTS_DIR / "random_forest_backtest_summary.csv"

    if not summary_path.exists():
        st.error(f"File not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)

    st.markdown(
        """
        Rolling-origin backtesting evaluates the forecasting model across
        several historical periods instead of relying on one holdout window.

        Four folds are used, each containing **28 out-of-sample days**.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Test Observations", "112")

    with col2:
        st.metric("Overall MAE", "236.08")

    with col3:
        st.metric("Overall RMSE", "303.42")

    with col4:
        st.metric("Overall MAPE", "5.05%")

    st.subheader("Backtest Results")

    display_df = df.copy()

    for column in ["mae", "rmse", "mape", "coverage"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    if "fold" in df.columns and "mape" in df.columns:

        st.subheader("MAPE by Backtest Fold")

        chart_df = (
            df[["fold", "mape"]]
            .assign(fold=lambda x: "Fold " + x["fold"].astype(str))
            .set_index("fold")
        )

        st.bar_chart(chart_df)

    st.success(
        "The model maintained approximately 5% MAPE across "
        "four historical forecasting windows."
    )


# --------------------------------------------------
# Hierarchical forecasting page
# --------------------------------------------------

def show_hierarchical_forecasting():

    show_header()

    st.subheader("Hierarchical Forecasting")

    hierarchy_path = DATA_DIR / "hierarchical_forecasts.csv"

    if not hierarchy_path.exists():
        st.error(f"File not found: {hierarchy_path}")
        return

    df = pd.read_csv(hierarchy_path)

    st.markdown(
        """
        Retail demand naturally exists at multiple aggregation levels.

        This project evaluates the hierarchy:

        **CA_1 Store → Category → Department**
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Store Series", "1")

    with col2:
        st.metric("Category Series", "3")

    with col3:
        st.metric("Department Series", "7")

    st.markdown(
        """
        ### Categories

        - FOODS
        - HOBBIES
        - HOUSEHOLD
        """
    )

    st.success(
        "Forecast reconciliation validated: category and department "
        "forecasts aggregate exactly to the CA_1 store forecast."
    )

    with st.expander("View hierarchical forecasts"):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

# --------------------------------------------------
# Live Prediction page
# --------------------------------------------------
 
@st.cache_resource
def get_prediction_model():
    return load_model()

def show_live_prediction():

    show_header()

    st.subheader("Historical Demand Prediction")

    st.write(
        "Select a historical date to inspect the model prediction against known demand features."
    )

    st.info(
        "The model returns P10, P50 and P90 demand estimates "
        "to represent forecast uncertainty."
    )

    feature_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ca_1_daily_features_calendar.csv"
    )

    feature_data = pd.read_csv(feature_path)

    feature_data["date"] = pd.to_datetime(
        feature_data["date"]
    )

    selected_date = st.selectbox(
        "Select forecast date",
        options=feature_data["date"].dt.date.tolist(),
        index=len(feature_data) - 1,
    )

    if st.button("Generate Prediction"):

        try:
            model = get_prediction_model()

            selected_row = feature_data[
                feature_data["date"].dt.date == selected_date
            ]

            prediction = predict_quantiles(
                model,
                selected_row
            )

            p10 = float(prediction["p10"][0])
            p50 = float(prediction["p50"][0])
            p90 = float(prediction["p90"][0])
            mean = float(prediction["mean"][0])

            st.success("Prediction generated successfully.")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("P10", f"{p10:,.0f}")
            col2.metric("P50", f"{p50:,.0f}")
            col3.metric("P90", f"{p90:,.0f}")
            col4.metric("Mean", f"{mean:,.0f}")

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )

    st.divider()

    st.subheader("Next-Day Forecast")

    st.write(
        "Forecast the first unseen day after the latest available historical data. "
        "This forecast does not depend on the historical date selected above."
    )
    if st.button("Generate Next-Day Forecast"):

        try:
            model = get_prediction_model()

            history = pd.read_csv(
                PROJECT_ROOT
                / "data"
                / "processed"
                / "ca_1_daily_features_calendar.csv"
            )

            calendar = pd.read_csv(
                PROJECT_ROOT
                / "data"
                / "raw"
                / "m5"
                / "calendar.csv"
            )

            future_features = build_next_day_features(
                history[["date", "sales"]]
            )

            future_features = add_calendar_features(
                future_features,
                calendar
            )

            future_row = pd.DataFrame(
                [future_features]
            )

            prediction = predict_quantiles(
                model,
                future_row
            )

            p10 = float(prediction["p10"][0])
            p50 = float(prediction["p50"][0])
            p90 = float(prediction["p90"][0])
            mean = float(prediction["mean"][0])

            forecast_date = pd.to_datetime(
                future_features["date"]
            ).date()

            st.success(
                f"Next-day forecast generated for {forecast_date}"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("P10", f"{p10:,.0f}")
            col2.metric("P50", f"{p50:,.0f}")
            col3.metric("P90", f"{p90:,.0f}")
            col4.metric("Mean", f"{mean:,.0f}")

        except Exception as error:
            st.error(
                f"Future forecast failed: {error}"
            )


    st.divider()

    st.subheader("7-Day Future Forecast")

    st.write(
        "Generate a recursive 7-day demand forecast. "
        "Each day's P50 prediction is used to construct the demand features "
        "for the following day."
    )

    if st.button("Generate 7-Day Forecast"):

        try:
            model = get_prediction_model()

            history = pd.read_csv(
                PROJECT_ROOT
                / "data"
                / "processed"
                / "ca_1_daily_features_calendar.csv"
            )

            calendar = pd.read_csv(
                PROJECT_ROOT
                / "data"
                / "raw"
                / "m5"
                / "calendar.csv"
            )

            forecasts = forecast_multiple_days(
                model=model,
                history=history,
                calendar=calendar,
                days=7,
            )

            display_forecasts = forecasts.copy()

            display_forecasts["date"] = (
                pd.to_datetime(display_forecasts["date"])
                .dt.strftime("%Y-%m-%d")
            )

            for column in ["p10", "p50", "p90", "mean"]:
                display_forecasts[column] = (
                    display_forecasts[column].round(0)
                )

            display_forecasts = display_forecasts.rename(
                columns={
                    "date": "Date",
                    "p10": "P10",
                    "p50": "P50",
                    "p90": "P90",
                    "mean": "Mean",
                }
            )

            st.success(
                "7-day recursive forecast generated successfully."
            )

            st.dataframe(
                display_forecasts,
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("7-Day Forecast with Uncertainty")

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(
                forecasts["date"],
                forecasts["p50"],
                marker="o",
                label="P50 Forecast",
            )

            ax.fill_between(
                forecasts["date"],
                forecasts["p10"],
                forecasts["p90"],
                alpha=0.2,
                label="P10–P90 Interval",
            )

            ax.set_title("7-Day Retail Demand Forecast")
            ax.set_xlabel("Forecast Date")
            ax.set_ylabel("Demand (Units)")
            ax.legend()
            ax.grid(alpha=0.3)

            fig.autofmt_xdate()

            st.pyplot(fig)

            plt.close(fig)
        except Exception as error:
            st.error(
                f"7-day forecast failed: {error}"
            )


# --------------------------------------------------
# Page router
# --------------------------------------------------

if page == "Overview":
    show_overview()

elif page == "Model Comparison":
    show_model_comparison()

elif page == "Probabilistic Forecast":
    show_probabilistic_forecast()

elif page == "Backtesting":
    show_backtesting()

elif page == "Hierarchical Forecasting":
    show_hierarchical_forecasting()

elif page == "Live Prediction":
    show_live_prediction()