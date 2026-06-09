import streamlit as st
import pandas as pd
import numpy as np
import os

# Import custom backend modules
from backend.data_stream import load_local_data
from backend.portfolio_math import calculate_regime_matrices, optimize_portfolio
from backend.strategy_library import moving_average_crossover, inverse_momentum, bollinger_band_reversion
from backend.ledger import simulate_macro_portfolio, simulate_micro_trades
from ml_engine.inference import RegimeInferenceEngine
from ml_engine.asset_ranker import MultiAssetRanker

# Page Configuration
st.set_page_config(page_title="AI-Driven Regime Switching Simulator", layout="wide")
st.title("📈 Multi-Asset Regime Switching & Portfolio Optimization Engine")
st.markdown("---")

# Initialize Engines safely
try:
    inference_engine = RegimeInferenceEngine(model_path="data/regime_hmm_model.pkl")
    ranker_engine = MultiAssetRanker(lookback_window=60)
except Exception as e:
    st.error(f"Initialization Error: Ensure 'data/regime_hmm_model.pkl' exists. Details: {e}")
    st.stop()

# 1. SIDEBAR CONTROLS
st.sidebar.header("🎛️ Control Panel")
mode = st.sidebar.radio("Select Simulation Mode:",
                        ["Mode 1: Macro Portfolio Optimization", "Mode 2: Micro Tactical Trading"])

# Load Dataset
df_prices = load_local_data()
if df_prices is None:
    st.error("Data Error: Could not load historical data from local storage.")
    st.stop()

# Ensure datetime index
df_prices.index = pd.to_datetime(df_prices.index)
tickers = list(df_prices.columns)

# 2. MODE 1: MACRO PORTFOLIO OPTIMIZATION
if mode == "Mode 1: Macro Portfolio Optimization":
    st.header("🏢 Mode 1: Dynamic Portfolio Allocation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Leaderboard & Parameters")
        starting_capital = st.number_input("Starting Capital ($):", min_value=1000.0, value=10000.0, step=1000.0)

        # Compute and display predictive ranking scores
        with st.spinner("Calculating asset momentum rankings..."):
            scores = ranker_engine.generate_ranking_scores(df_prices)
            ranking_df = scores.to_frame(name='Predictive Score')
            st.dataframe(ranking_df.style.format("{:.2f}"))

    with col2:
        st.subheader("Macro Backtest Simulation")

        # Generate dummy/simulated regimes across the timeline for demonstration
        # In a full deployment, this tracks the historical state array
        market_proxy = df_prices.mean(axis=1)
        log_returns = np.log(market_proxy / market_proxy.shift(1)).dropna()
        rolling_vol = log_returns.rolling(window=20).std().dropna()
        idx = rolling_vol.index

        X_hist = np.column_stack([log_returns.loc[idx], rolling_vol])
        historical_regimes = pd.Series(inference_engine.model.predict(X_hist), index=idx)

        # Generate target weights for each discovered state dynamically
        # Calculate matrices for all discovered states efficiently
        matrices = calculate_regime_matrices(df_prices.loc[idx], historical_regimes)

        # Generate optimal target weights for each regime dynamically
        regime_weights_dict = {}
        for state, mats in matrices.items():
            regime_weights_dict[state] = optimize_portfolio(mats['mu'], mats['sigma'], mats['assets'])

        # Run Ledger Accountant
        equity_curve, metrics = simulate_macro_portfolio(
            price_data=df_prices.loc[idx],
            regime_series=historical_regimes,
            regime_weights_dict=regime_weights_dict,
            starting_capital=starting_capital
        )

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Return", f"{metrics['Total Return (%)']}%")
        m2.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']}%")
        m3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']}")

        # Chart Equity Curve
        st.line_chart(equity_curve, use_container_width=True)

# 3. MODE 2: MICRO TACTICAL TRADING
else:
    st.header("⚡ Mode 2: Micro Regime-Based Trading Switchboard")

    selected_asset = st.selectbox("Select Ticker to Trade:", tickers)
    asset_prices = df_prices[selected_asset].dropna()

    # Run Real-Time ML Inference for the final data point
    try:
        recent_window = asset_prices.iloc[-25:]
        current_state_desc = inference_engine.predict_current_regime(recent_window)
        current_state_int = int(current_state_desc.split("State ")[1].split(" ")[0])
        st.info(f"🔮 **Machine Learning Engine Live Output:** {current_state_desc}")
    except Exception as e:
        st.warning(f"Inference window warming up. Defaulting to baseline state. Details: {e}")
        current_state_int = 0

    # Route entire asset history through the correct strategy based on the active regime
    st.subheader(f"Tactical Execution Loop: {selected_asset}")

    if current_state_int == 0:
        st.write("🟢 Market flagged as **Bull**. Executing **Moving Average Crossover** Strategy.")
        signals_df = moving_average_crossover(asset_prices)
    elif current_state_int == 1:
        st.write("🔴 Market flagged as **Bear**. Executing **Inverse Momentum** Strategy.")
        signals_df = inverse_momentum(asset_prices)
    else:
        st.write("🔵 Market flagged as **Chop**. Executing **Bollinger Band Mean Reversion** Strategy.")
        signals_df = bollinger_band_reversion(asset_prices)

    # Run Micro Ledger Accountant
    trades_df, total_profit = simulate_micro_trades(signals_df, trade_capital=1000.0)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Strategy Net Profit ($)", f"${total_profit:,.2f}")
        if not trades_df.empty:
            st.dataframe(trades_df.tail(10))
        else:
            st.write("No recent tactical trades executed in this window.")

    with col2:
        # Plot price tracking and signals
        chart_data = signals_df[['Price']].copy()
        st.line_chart(chart_data, use_container_width=True)