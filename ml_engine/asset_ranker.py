import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class MultiAssetRanker:
    def __init__(self, lookback_window=60):
        """
        Initializes the ranking engine. 
        lookback_window: Days to measure momentum (default 60 days / ~3 months).
        """
        self.lookback_window = lookback_window

    def generate_ranking_scores(self, historical_prices):
        """
        Evaluates the entire stock universe simultaneously and outputs predictive scores.
        historical_prices: A pandas DataFrame of asset prices (Columns = Tickers, Rows = Dates).
        """
        # Ensure we have enough data to look back
        if len(historical_prices) < self.lookback_window:
            raise ValueError(f"Need at least {self.lookback_window} days of data to rank assets.")
            
        # Isolate the relevant timeframe (e.g., the last 3 months)
        window_data = historical_prices.iloc[-self.lookback_window:]
        
        # 1. Calculate Absolute Momentum (Total Return over the window)
        total_return = (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]
        
        # 2. Calculate Volatility (Risk profile over the window)
        daily_returns = window_data.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        
        # 3. Core Feature: Risk-Adjusted Momentum Score
        # We reward stocks that go up smoothly, and penalize stocks that are chaotic
        risk_adjusted_scores = total_return / (volatility + 1e-6)
        
        # 4. Normalize to a clean 0-100 predictive scale for the UI
        min_score = risk_adjusted_scores.min()
        max_score = risk_adjusted_scores.max()
        
        # Edge case protection
        if max_score == min_score:
            normalized_scores = risk_adjusted_scores * 0 + 50 
        else:
            normalized_scores = ((risk_adjusted_scores - min_score) / (max_score - min_score)) * 100
            
        # Return sorted rankings (Highest score = predicted outperformer)
        final_rankings = normalized_scores.sort_values(ascending=False).round(2)
        
        return final_rankings