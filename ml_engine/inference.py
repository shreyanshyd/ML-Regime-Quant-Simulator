import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

class RegimeInferenceEngine:
    def __init__(self, model_path="ml_engine/regime_hmm_model.pkl"):
        """Loads the pre-trained Hidden Markov Model into memory."""
        self.model = joblib.load(model_path)
        
        # Hardcoded mapping based on our specific unsupervised training results
        
        self.regime_map = {
            0: "State 0 = Bull (Low Volatility Trend)",
            2: "State 1 = Bear (Negative Drift)",
            1: "State 2 = Chop (High Volatility Panic)"
        }
    def predict_current_regime(self, recent_prices):
        """
        Accepts a trailing window of market prices, computes the live features,
        and executes a zero-latency prediction of the current market state.
        
        recent_prices: A pandas Series or 1D numpy array of the last ~25 days of prices.
        """
        # We need at least 21 days of data to calculate a 20-day rolling standard deviation
        if len(recent_prices) < 21:
            raise ValueError("Inference requires a minimum of 21 days of trailing price history.")

        prices = pd.Series(recent_prices)
        
        # 1. Zero-Latency Feature Engineering on the live feed
        log_returns = np.log(prices / prices.shift(1)).dropna()
        rolling_vol = log_returns.rolling(window=20).std().dropna()
        
        # 2. Isolate the exact features for "Today"
        current_return = log_returns.iloc[-1]
        current_vol = rolling_vol.iloc[-1]
        
        # Reshape for scikit-learn/hmmlearn format: [[feature1, feature2]]
        X_live = np.array([[current_return, current_vol]])
        
        # 3. Execute Inference
        predicted_state_id = self.model.predict(X_live)[0]
        
        return self.regime_map[predicted_state_id]