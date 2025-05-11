import pandas as pd
import numpy as np
import cvxpy as cp


def mean_variance_sharpe_maximization(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """
    Compute traditional mean-variance optimal weights to maximize Sharpe ratio, per Bemis et al. (2009).

    Parameters:
    -----------
    returns : pd.DataFrame
        Daily (or periodic) return series for each asset, shape (T, n).
    rf : float
        Risk-free rate (in same periodic units as returns).

    Returns:
    --------
    pd.Series
        Optimal portfolio weights summing to 1, long-only.
    """
    # Sample mean vector and covariance matrix
    mu = returns.mean().values
    Sigma = returns.cov().values
    n = len(mu)

    # Decision variable: portfolio weights (unnormalized)
    w = cp.Variable(n)

    # Objective: minimize portfolio variance w^T Sigma w
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    # Constraints:
    #  1) Achieve unit excess return: mu^T w - rf == 1  (transformed from max Sharpe)
    #  2) No short-selling: w >= 0
    constraints = [mu.T @ w - rf == 1,
                   w >= 0]

    # Form and solve QP
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    # Retrieve and normalize weights
    w_opt = w.value
    if w_opt is None:
        raise ValueError("Optimization failed: no solution found.")
    w_norm = w_opt / np.sum(w_opt)

    return pd.Series(w_norm, index=returns.columns)


if __name__ == "__main__":
    # Load previously computed equity returns
    returns = pd.read_csv('equity_returns.csv', index_col=0, parse_dates=True)

    # Set risk-free rate (daily rf ~ 0)
    rf = 0.0

    # Compute optimal weights
    weights = mean_variance_sharpe_maximization(returns, rf)

    # Save weights to CSV
    weights.to_csv('mv_optimal_weights.csv', header=['weight'])
    print('Mean-variance optimal weights saved to mv_optimal_weights.csv')
