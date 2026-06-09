import os
import yfinance as yf
import pandas as pd

# Defining our choice of stocks, chose 15 highly liquid stocks across the different sectors
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'JPM', 'V', 'XOM',
    'JNJ', 'WMT', 'UNH', 'PG', 'LLY'
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def download_market_data():
    """ Downloads both training and simulation data for the asset universe """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Starting data download via yfinance...")

    # Gathering data from 2010-01-01 to 2024-12-31, both years being inclusive to train our ML model
    print("Downloading historical data (2010-2024) for ML training...")
    # Explicitly set auto_adjust=False to force yfinance to generate the 'Adj Close' column
    raw_train = yf.download(TICKERS, start="2010-01-01", end="2025-01-01", auto_adjust=False)
    train_data = raw_train['Adj Close']
    train_path = os.path.join(DATA_DIR, 'historical_training_data.csv')
    train_data.to_csv(train_path)
    print(f"Saved training data to {train_path}")

    #  The simulation window is from January 2025 to April 2026, both inclusive, data for that is separate
    print("Downloading evaluation data (2025-2026) for Simulator...")
    raw_sim = yf.download(TICKERS, start="2025-01-01", end="2026-05-01", auto_adjust=False)
    sim_data = raw_sim['Adj Close']
    sim_path = os.path.join(DATA_DIR, 'simulation_testing_data.csv')
    sim_data.to_csv(sim_path)
    print(f"Saved simulation data to {sim_path}")

if __name__ == "__main__":
    download_market_data()

# Additional code to load the CSV directly onto the memory
import os
import pandas as pd

def load_local_data(filepath="data/historical_training_data.csv"):
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return None

    # Reads the CSV, making sure the dates are treated as actual Datetime objects
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df