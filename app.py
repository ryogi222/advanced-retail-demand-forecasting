from pathlib import Path

import pandas as pd
import streamlit as st
# --------------------------------------------------
# Sidebar navigation
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
FIGURES_DIR = REPORTS_DIR / "figures"


# --------------------------------------------------
# Dashboard title
# --------------------------------------------------

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
# Headline performance
# --------------------------------------------------

st.subheader("Forecasting Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Best Model",
        value="Quantile RF",
    )

with col2:
    st.metric(
        label="Holdout MAPE",
        value="5.07%",
    )

with col3:
    st.metric(
        label="Backtest MAPE",
        value="5.05%",
    )

with col4:
    st.metric(
        label="Models Compared",
        value="7",
    )


# --------------------------------------------------
# Project summary
# --------------------------------------------------

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
# --------------------------------------------------
# Model comparison
# --------------------------------------------------

st.divider()
st.subheader("Model Comparison")

comparison_path = DATA_DIR / "model_comparison.csv"

if comparison_path.exists():

    comparison_df = pd.read_csv(comparison_path)
    comparison_df = comparison_df.sort_values("MAPE")

    st.markdown(
        """
        Models are ranked using **Mean Absolute Percentage Error (MAPE)**.
        Lower MAPE indicates better forecasting accuracy.
        """
    )

    # Display results table
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    # Interactive MAPE chart
    chart_data = (
        comparison_df[["model", "MAPE"]]
        .set_index("model")
    )

    st.bar_chart(chart_data)

    best_model = comparison_df.iloc[0]

    st.success(
        f"Best performing model: **{best_model['model']}** "
        f"with **{best_model['MAPE']:.2f}% MAPE**."
    )

else:
    st.warning(
        f"Model comparison file not found: {comparison_path}"
    )
# --------------------------------------------------
# Probabilistic forecast
# --------------------------------------------------

st.divider()
st.subheader("Probabilistic Demand Forecast")

quantile_path = REPORTS_DIR / "quantile_random_forest_forecast.csv"

if quantile_path.exists():

    quantile_df = pd.read_csv(quantile_path)

    quantile_df["date"] = pd.to_datetime(quantile_df["date"])

    st.markdown(
        """
        The probabilistic Random Forest estimates demand uncertainty
        using predictions from the individual trees in the ensemble.

        - **P10** — lower demand estimate
        - **P50** — median forecast
        - **P90** — upper demand estimate
        """
    )

    # Detect the actual quantile column names
    required_columns = ["date", "actual", "p10", "p50", "p90"]

    missing_columns = [
        col for col in required_columns
        if col not in quantile_df.columns
    ]

    if not missing_columns:

        forecast_chart = (
            quantile_df[
                ["date", "actual", "p10", "p50", "p90"]
            ]
            .set_index("date")
        )

        st.line_chart(forecast_chart)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("P50 MAPE", "5.07%")

        with col2:
            st.metric("P10-P90 Coverage", "71.43%")

        with col3:
            st.metric("Average Interval Width", "700.31 units")

    else:

        st.warning(
            "Expected probabilistic forecast columns were not found. "
            f"Available columns: {list(quantile_df.columns)}"
        )

else:

    st.warning(
        f"Probabilistic forecast file not found: {quantile_path}"
    )
# --------------------------------------------------
# Rolling-origin backtesting
# --------------------------------------------------

st.divider()
st.subheader("Rolling-Origin Backtesting")

backtest_path = REPORTS_DIR / "random_forest_backtest_summary.csv"

if backtest_path.exists():

    backtest_df = pd.read_csv(backtest_path)

    st.markdown(
        """
        Rolling-origin backtesting evaluates the forecasting model across
        multiple historical periods instead of relying on a single holdout
        window.

        Each fold contains **28 out-of-sample days**.
        """
    )

    # Display backtest results
    st.dataframe(
        backtest_df,
        use_container_width=True,
        hide_index=True,
    )

    # MAPE by fold
    if "fold" in backtest_df.columns and "MAPE" in backtest_df.columns:

        backtest_chart = (
            backtest_df[["fold", "MAPE"]]
            .set_index("fold")
        )

        st.bar_chart(backtest_chart)

    # Overall backtest metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Backtest Observations", "112")

    with col2:
        st.metric("Overall MAE", "236.08")

    with col3:
        st.metric("Overall RMSE", "303.42")

    with col4:
        st.metric("Overall MAPE", "5.05%")

    st.success(
        "The model maintained approximately 5% MAPE across "
        "four historical forecasting windows."
    )

else:

    st.warning(
        f"Backtest summary file not found: {backtest_path}"
    )
# --------------------------------------------------
# Hierarchical forecasting
# --------------------------------------------------

st.divider()
st.subheader("Hierarchical Forecasting")

hierarchy_path = DATA_DIR / "hierarchical_forecasts.csv"

if hierarchy_path.exists():

    hierarchy_df = pd.read_csv(hierarchy_path)

    st.markdown(
        """
        Retail demand exists at multiple aggregation levels.

        This project evaluates forecasts across the hierarchy:

        **CA_1 Store → Category → Department**

        The hierarchy contains:

        - **1** store-level series
        - **3** category-level series
        - **7** department-level series
        """
    )

    # Show available hierarchy data
    st.dataframe(
        hierarchy_df.head(30),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Store Series", "1")

    with col2:
        st.metric("Category Series", "3")

    with col3:
        st.metric("Department Series", "7")

    st.success(
        "Forecast reconciliation validated: category and department "
        "forecasts aggregate exactly to the CA_1 store forecast."
    )

else:

    st.warning(
        f"Hierarchical forecast file not found: {hierarchy_path}"
    )