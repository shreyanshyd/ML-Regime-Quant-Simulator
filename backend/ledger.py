import pandas as pd
import numpy as np


def calculate_performance_metrics(equity_curve, risk_free_rate=0.02):
    """
    Grades the final simulation based on standard institutional risk metrics.
    """
    returns = equity_curve.pct_change().dropna()

    # Cumulative Return (Total % Growth)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Maximum Drawdown (Steepest peak-to-trough drop)
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Annualized Sharpe Ratio
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    if excess_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2)
    }


def simulate_macro_portfolio(price_data, regime_series, regime_weights_dict, starting_capital=10000.0, fee_pct=0.0002):
    """
    Mode 1 Accountant: Steps through time, applying regime weights and compounding cash.
    """
    daily_returns = price_data.pct_change().dropna()
    aligned_regimes = regime_series.loc[daily_returns.index]

    cash = starting_capital
    equity_curve = pd.Series(index=daily_returns.index, dtype=float)

    current_regime = None
    current_weights = pd.Series(0.0, index=price_data.columns)

    for date in daily_returns.index:
        today_regime = aligned_regimes.loc[date]

        # Check for Regime Shift (Triggers a Portfolio Rebalance)
        if today_regime != current_regime:
            target_weights_dict = regime_weights_dict.get(today_regime, {})
            current_weights = pd.Series(target_weights_dict).reindex(price_data.columns).fillna(0.0)

            # Deduct transaction fee for moving massive amounts of capital
            cash = cash * (1 - fee_pct)
            current_regime = today_regime

        # Apply today's market performance to the portfolio
        daily_port_return = (current_weights * daily_returns.loc[date]).sum()
        cash = cash * (1 + daily_port_return)

        # Log the balance for the UI Chart
        equity_curve.loc[date] = cash

    metrics = calculate_performance_metrics(equity_curve)

    return equity_curve, metrics


def simulate_micro_trades(signals_df, trade_capital=1000.0, fee_pct=0.0002):
    """
    Mode 2 Accountant: Logs absolute dollar profit/loss for isolated tactical trades.
    """
    trades_log = []
    current_position = 0.0  # 0 for Cash, 1 for Long, -1 for Short
    entry_price = 0.0

    for date, row in signals_df.iterrows():
        pos_change = row['Position_Change']
        current_price = row['Price']

        # Entry Logic
        if pos_change != 0 and current_position == 0:
            current_position = row['Signal']
            entry_price = current_price
            fee = trade_capital * fee_pct

            direction = "LONG" if current_position == 1.0 else "SHORT"
            trades_log.append({'Date': date, 'Action': f'ENTRY {direction}', 'Price': current_price, 'PnL': -fee})

        # Exit Logic
        elif pos_change != 0 and current_position != 0:
            # Calculate raw Profit/Loss based on direction
            if current_position == 1.0:
                pnl = ((current_price - entry_price) / entry_price) * trade_capital
            elif current_position == -1.0:
                pnl = ((entry_price - current_price) / entry_price) * trade_capital
            else:
                pnl = 0

            fee = trade_capital * fee_pct
            net_pnl = pnl - fee

            trades_log.append({'Date': date, 'Action': 'EXIT', 'Price': current_price, 'PnL': net_pnl})

            # Update state (Signal might flip directly from 1 to -1)
            current_position = row['Signal']
            entry_price = current_price if current_position != 0 else 0.0

            # If immediately flipping, log the new entry fee
            if current_position != 0:
                direction = "LONG" if current_position == 1.0 else "SHORT"
                trades_log.append({'Date': date, 'Action': f'ENTRY {direction}', 'Price': current_price,
                                   'PnL': -(trade_capital * fee_pct)})

    trades_df = pd.DataFrame(trades_log)
    total_net_profit = trades_df['PnL'].sum() if not trades_df.empty else 0.0

    return trades_df, total_net_profit