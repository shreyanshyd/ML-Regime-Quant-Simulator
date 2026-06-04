import pandas as pd
import numpy as np


def moving_average_crossover(prices_series, short_window=10, long_window=50):
    """
    State 0 (Bull): Trend Following.
    Stays Long (+1) while the short trend is above the long trend.
    """
    df = pd.DataFrame(index=prices_series.index)
    df['Price'] = prices_series

    df['SMA_Short'] = df['Price'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_Long'] = df['Price'].rolling(window=long_window, min_periods=1).mean()

    # 1.0 = Buy/Long, 0.0 = Cash/Hold
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1.0, 0.0)
    df['Position_Change'] = df['Signal'].diff().fillna(0)

    return df


def inverse_momentum(prices_series, window=20):
    """
    State 1 (Bear): Defensive / Shorting.
    Moves to Cash (0) or Shorts (-1) when the asset is bleeding.
    """
    df = pd.DataFrame(index=prices_series.index)
    df['Price'] = prices_series

    # Calculate the percentage drop over the window
    df['Momentum'] = df['Price'].pct_change(periods=window)

    # If momentum is severely negative, Short it (-1.0). Otherwise, stay in Cash (0.0).
    df['Signal'] = np.where(df['Momentum'] < -0.02, -1.0, 0.0)
    df['Position_Change'] = df['Signal'].diff().fillna(0)

    return df


def bollinger_band_reversion(prices_series, window=20, num_std_dev=2):
    """
    State 2 (Chop): Mean Reversion.
    Buys at the bottom band, Shorts at the top band.
    """
    df = pd.DataFrame(index=prices_series.index)
    df['Price'] = prices_series

    df['SMA'] = df['Price'].rolling(window=window).mean()
    df['Rolling_Std'] = df['Price'].rolling(window=window).std()
    df['Upper_Band'] = df['SMA'] + (df['Rolling_Std'] * num_std_dev)
    df['Lower_Band'] = df['SMA'] - (df['Rolling_Std'] * num_std_dev)

    df['Signal'] = 0.0
    # Buy when price drops below the bottom band
    df.loc[df['Price'] < df['Lower_Band'], 'Signal'] = 1.0
    # Short when price spikes above the top band
    df.loc[df['Price'] > df['Upper_Band'], 'Signal'] = -1.0

    # Forward fill to hold the trade until it hits the opposite band
    df['Signal'] = df['Signal'].replace(to_replace=0, method='ffill')
    df['Position_Change'] = df['Signal'].diff().fillna(0)

    return df