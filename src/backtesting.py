import numpy as np

def cumulative_returns(weights, returns_df):
    """
    Calculate cumulative returns of a weighted portfolio.
    """
    strat_rets = (returns_df * weights).sum(axis=1)
    return (1 + strat_rets).cumprod()

def performance_metrics(cumrets):
    """
    Compute total return, annualized return, Sharpe ratio, and max drawdown.
    """
    total_return = cumrets.iloc[-1] - 1
    annualized  = cumrets.resample("YE").last().pct_change().iloc[-1]
    sharpe_ratio = (cumrets.pct_change().mean() / cumrets.pct_change().std()) * np.sqrt(252)
    max_dd = (cumrets.cummax() - cumrets).max()
    return total_return, annualized, sharpe_ratio, max_dd
