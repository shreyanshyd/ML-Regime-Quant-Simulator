import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
import os
import warnings

# Suppress annoying hmmlearn deprecation warnings
warnings.filterwarnings("ignore")

print("🧠 Starting ML Regime Clustering Pipeline...")

# 1. LOAD HISTORICAL TRAINING DATA (2010 - 2024)
csv_path = "data/historical_training_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing {csv_path}. Ensure your training data is inside the data/ folder.")

# Read data and compute a broad market proxy index (average of all tracking stocks)
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
market_proxy = df.mean(axis=1)

# 2. FEATURE ENGINEERING (Extracting Market Signatures)
print("⚙️ Engineering Features (Log Returns & Rolling Volatility)...")
# Feature 1: Log Returns (captures trajectory/direction)
log_returns = np.log(market_proxy / market_proxy.shift(1)).dropna()

# Feature 2: Rolling Volatility (captures risk/uncertainty)
rolling_vol = log_returns.rolling(window=20).std().dropna()

# Align vectors due to rolling windows
idx = rolling_vol.index
X = np.column_stack([log_returns.loc[idx], rolling_vol])

print(f"📊 Extracted {X.shape[0]} training data points.")

# 3. INITIALIZE AND TRAIN THE HIDDEN MARKOV MODEL
print("🤖 Training 3-State Gaussian Hidden Markov Model...")

# We use init_params="smc" so it initializes Starting probabilities, Means, and Covariances, 
# but it leaves the Transition Matrix ('t') alone so we can manually tune it!
model = GaussianHMM(
    n_components=3, 
    covariance_type="full", 
    n_iter=1000, 
    random_state=42,
    init_params="smc" 
)

# --- TRANSITION MATRIX TUNING (Eliminating the "Flickering") ---
# We force a 95% probability that tomorrow's market regime will be the same as today's.
# There is only a 2.5% chance of flipping states, which forces the AI to ignore micro-noise.
model.transmat_ = np.array([
    [0.95, 0.025, 0.025], # If in State 0, 95% chance to stay in State 0
    [0.025, 0.95, 0.025], # If in State 1, 95% chance to stay in State 1
    [0.025, 0.025, 0.95]  # If in State 2, 95% chance to stay in State 2
])

model.fit(X)
print("✅ Unsupervised Clustering Training & Matrix Tuning Complete!")

# 4. PREDICT STATES & MAP CHARACTERISTICS
hidden_states = model.predict(X)

print("\n📊 Discovered Regime Profiles:")
for i in range(model.n_components):
    state_mask = (hidden_states == i)
    mean_return = model.means_[i][0] * 100  # Scale to %
    mean_vol = model.means_[i][1] * 100
    print(f"  Regime {i}: Mean Daily Return = {mean_return:+.4f}%, Mean Volatility = {mean_vol:.4f}%")

# 5. EXPORT TRAINED BRAIN FOR INFERENCE OVERLAY
model_output_path = "data/regime_hmm_model.pkl"
joblib.dump(model, model_output_path)
print(f"\n💾 Model successfully saved to {model_output_path}!")