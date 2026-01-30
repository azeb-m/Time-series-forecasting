import numpy as np
import pandas as pd

def lstm_recursive_forecast(model, scaler, series_scaled, horizon=180):
    """
    Multi-step recursive forecast for LSTM.
    """
    last_seq = series_scaled[-60:]
    future_scaled = []
    current = last_seq.copy()
    for _ in range(horizon):
        pred = model.predict(current.reshape(1, current.shape[0], 1))
        future_scaled.append(pred[0,0])
        current = np.append(current[1:], pred[0,0])
    future_scaled = np.array(future_scaled).reshape(-1,1)
    future = scaler.inverse_transform(future_scaled).flatten()
    return future

def compute_confidence_intervals(residuals, future_series):
    """
    Approximate confidence intervals based on residual standard deviation.
    """
    std = np.std(residuals)
    upper = future_series + 1.96 * std
    lower = future_series - 1.96 * std
    return upper, lower
