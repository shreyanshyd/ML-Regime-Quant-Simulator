import streamlit as st
import pandas as pd
import numpy as np

from backend.data_stream import load_local_data
from backend.portfolio_math import calculate_regime_matrices, optimize_portfolio
from backend.strategy_library import moving_average_crossover, inverse_momentum, bollinger_band_reversion
from backend.ledger import simulate_macro_portfolio, simulate_micro_trades
from ml_engine.inference import RegimeInferenceEngine
from ml_engine.asset_ranker import MultiAssetRanker

# 1. Page Setup & Sidebar State
st.set_page_config(page_title="Regime Quant Simulator", layout="wide", initial_sidebar_state="collapsed")

# 2. Informational Sidebar (Matches Reference UI)
with st.sidebar:
    st.markdown("### 📊 About the Simulator")
    st.markdown(
        "Think of this as a virtual quantitative research lab. Instead of relying on static buy-and-hold strategies, we trained a Hidden Markov Model to 'look' at raw market volatility and instantly recognize underlying economic regimes.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ How it works")
    st.markdown(
        "**1. Screen:** We evaluate the stock universe to find assets with high risk-adjusted momentum.<br><br>**2. Identify:** The AI scans price action to classify the current market state (Bull, Bear, or Chop).<br><br>**3. Allocate:** We use Markowitz optimization to dynamically shift capital weights based on the AI's state prediction.<br><br>**4. Execute:** Isolated micro-strategies are triggered to handle entry and exit timing.",
        unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small>Developed by Shreyansh Yadav and Shrinidhi Walvekar | IIT Dharwad</small>",
                unsafe_allow_html=True)

# 3. Engine Initialization
try:
    inference_engine = RegimeInferenceEngine(model_path="data/regime_hmm_model.pkl")
    ranker_engine = MultiAssetRanker(lookback_window=60)
except Exception as e:
    st.error(f"Engine initialization failed: {e}")
    st.stop()

df_prices = load_local_data()
if df_prices is None:
    st.stop()

df_prices.index = pd.to_datetime(df_prices.index)
tickers = list(df_prices.columns)

# 4. Centralized Main Header & Navigation
st.markdown("<h1 style='text-align: center;'>📈 Multi-Asset Regime Switching Engine</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>A live playground to see how AI adapts portfolio allocation to shifting market states.</p>",
    unsafe_allow_html=True)

st.write("")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    mode = st.radio("Operating Mode", ["🏢 Mode 1: Macro Portfolio Allocation", "⚡ Mode 2: Micro Tactical Trading"],
                    horizontal=True, label_visibility="collapsed")
st.markdown("---")

# 5. MODE 1: MACRO PORTFOLIO
if mode == "🏢 Mode 1: Macro Portfolio Allocation":

    left_col, right_col = st.columns([1, 2.5])

    with left_col:
        st.subheader("Simulation Parameters")
        st.info("Adjust your starting capital and run the Markowitz optimization engine.")
        starting_capital = st.number_input("Starting Capital ($)", value=10000.0, step=1000.0)
        run_sim = st.button("Run Portfolio Simulation", use_container_width=True)

        st.write("")
        st.subheader("Asset Momentum Leaderboard")

        scores = ranker_engine.generate_ranking_scores(df_prices)
        ranking_df = scores.to_frame(name='Momentum Score (0-100)')
        ranking_df.index.name = 'Stock Ticker'

        st.dataframe(ranking_df.style.format("{:.2f}"), use_container_width=True, height=550)

    with right_col:
        st.subheader("Backtest Results")

        if run_sim:
            with st.spinner("Executing dynamic allocation across AI regimes..."):
                market_proxy = df_prices.mean(axis=1)
                log_returns = np.log(market_proxy / market_proxy.shift(1)).dropna()
                rolling_vol = log_returns.rolling(window=20).std().dropna()
                idx = rolling_vol.index

                X_hist = np.column_stack([log_returns.loc[idx], rolling_vol])
                historical_regimes = pd.Series(inference_engine.model.predict(X_hist), index=idx)

                matrices = calculate_regime_matrices(df_prices.loc[idx], historical_regimes)

                regime_weights_dict = {}
                for state, mats in matrices.items():
                    regime_weights_dict[state] = optimize_portfolio(mats['mu'], mats['sigma'], mats['assets'])

                equity_curve, metrics = simulate_macro_portfolio(
                    price_data=df_prices.loc[idx],
                    regime_series=historical_regimes,
                    regime_weights_dict=regime_weights_dict,
                    starting_capital=starting_capital
                )

                # Display explicit financial metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Return", f"{metrics['Total Return (%)']}%")
                m2.metric("Maximum Drawdown", f"{metrics['Max Drawdown (%)']}%")
                m3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']}")

                st.line_chart(equity_curve)
        else:
            st.write("Awaiting simulation trigger. Click 'Run Portfolio Simulation' to begin.")

# 6. MODE 2: MICRO TACTICAL
else:
    st.subheader("Tactical Execution Playground")
    st.info(
        "Select an individual asset to see how the AI adjusts the underlying trading strategy based on current market conditions.")

    selected_asset = st.selectbox("Select Target Stock Ticker", tickers)
    asset_prices = df_prices[selected_asset].dropna()

    try:
        current_state_desc = inference_engine.predict_current_regime(asset_prices.iloc[-25:])
        current_state_int = int(current_state_desc.split("State ")[1].split(" ")[0])
        st.success(f"**AI Inference Engine:** {current_state_desc}")
    except:
        current_state_int = 0
        st.warning("Inference window warming up. Defaulting to baseline state.")

    if current_state_int == 0:
        signals_df = moving_average_crossover(asset_prices)
    elif current_state_int == 1:
        signals_df = inverse_momentum(asset_prices)
    else:
        signals_df = bollinger_band_reversion(asset_prices)

    trades_df, total_profit = simulate_micro_trades(signals_df)

    st.write("")
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.metric("Strategy Net Profit", f"${total_profit:,.2f}")
        st.write("**Recent Trade Log**")
        if not trades_df.empty:
            st.dataframe(trades_df.tail(10), use_container_width=True)
        else:
            st.write("No recent tactical trades executed in this window.")

    with right_col:
        st.write("**Price Action & Signals**")
        st.line_chart(signals_df[['Price']])