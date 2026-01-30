from pypfopt import expected_returns, risk_models, EfficientFrontier

def build_return_cov(price_df, tsla_exp_ret):
    """
    Compute expected return vector and covariance matrix.
    """
    mu_hist = expected_returns.mean_historical_return(price_df)
    cov = risk_models.sample_cov(price_df)
    mu_hist["TSLA"] = tsla_exp_ret
    return mu_hist, cov

def optimize_max_sharpe(mu, cov):
    ef = EfficientFrontier(mu, cov)
    ef.max_sharpe()
    return ef.clean_weights(), ef.portfolio_performance(verbose=True)

def optimize_min_vol(mu, cov):
    ef = EfficientFrontier(mu, cov)
    ef.min_volatility()
    return ef.clean_weights(), ef.portfolio_performance(verbose=True)
