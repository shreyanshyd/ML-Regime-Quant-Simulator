import numpy as np
import cvxpy as cp
import pandas as pd


def calculate_regime_matrices(price_data, regime_series):
    """
    It calculates the expected returns and covariance matrices for EACH dynamic regime.
    """
    # Converting raw prices to daily percentage returns
    returns = price_data.pct_change().dropna()

    # Aligning the AI's regime labels perfectly with our return dates
    aligned_data = returns.copy()
    aligned_data['Regime'] = regime_series

    regime_matrices = {}
    unique_states = aligned_data['Regime'].dropna().unique()

    # Calculating the matrices for each state independently
    for state in unique_states:
        state_returns = aligned_data[aligned_data['Regime'] == state].drop(columns=['Regime'])

        mu = state_returns.mean() * 252
        sigma = state_returns.cov() * 252

        regime_matrices[state] = {
            'mu': mu.values,
            'sigma': sigma.values,
            'assets': state_returns.columns.tolist()  # Capturing the stock names
        }

    return regime_matrices


def optimize_portfolio(expected_returns, cov_matrix, asset_names, risk_aversion=1.0):
    """
    It calculates the optimal portfolio weights and maps them to asset tickers for the UI.
    """
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)

    # Mathematical Objective
    port_return = expected_returns @ weights
    port_variance = cp.quad_form(weights, cov_matrix)
    objective = cp.Maximize(port_return - (risk_aversion / 2) * port_variance)

    # Constraints
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0
    ]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except cp.error.SolverError:
        return {asset: 1.0 / n_assets for asset in asset_names}

    # Cleaning up weights
    clean_weights = np.clip(weights.value, 0, 1)
    if np.sum(clean_weights) > 0:
        clean_weights /= np.sum(clean_weights)
    else:
        clean_weights = np.ones(n_assets) / n_assets

    weight_dictionary = dict(zip(asset_names, clean_weights))

    return weight_dictionary