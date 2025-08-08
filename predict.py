"""
Prediction logic for stock price forecasting.
Handles future predictions and model inference.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stock_utils import create_sequences, calculate_metrics

def predict_future_prices(model, last_sequence, scaler, days=30, model_type='lstm'):
    """
    Predict future stock prices.

    Args:
        model: Trained model (LSTM or Regression)
        last_sequence (np.array): Last sequence of data
        scaler: Data scaler
        days (int): Number of days to predict
        model_type (str): Type of model ('lstm' or 'regression')

    Returns:
        np.array: Predicted prices
    """
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days):
        if model_type == 'lstm':
            # Reshape for LSTM
            current_input = current_sequence.reshape((1, current_sequence.shape[0], 1))
            prediction = model.predict(current_input)

            # Update sequence
            current_sequence = np.append(current_sequence[1:], prediction)
        else:
            # For regression models
            current_input = current_sequence.reshape(1, -1)
            prediction = model.predict(current_input)

        predictions.append(prediction[0])

    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    return predictions.flatten()

def create_prediction_dates(start_date, num_days):
    """
    Create dates for predictions.

    Args:
        start_date (datetime): Starting date
        num_days (int): Number of days to predict

    Returns:
        list: List of dates
    """
    dates = []
    current_date = start_date

    for i in range(num_days):
        # Skip weekends
        while current_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            current_date += timedelta(days=1)

        dates.append(current_date)
        current_date += timedelta(days=1)

    return dates

def evaluate_model(model, X_test, y_test, scaler, model_type='lstm'):
    """
    Evaluate model performance on test data.

    Args:
        model: Trained model
        X_test (np.array): Test features
        y_test (np.array): Test targets
        scaler: Data scaler
        model_type (str): Type of model
        Returns:
            tuple: (predictions, metrics)
        """
    if model_type == 'lstm':
        # Reshape for LSTM
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        predictions = model.predict(X_test_reshaped, verbose=0)
    else:
        predictions = model.predict(X_test)

    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    metrics = calculate_metrics(y_test_actual.flatten(), predictions.flatten())

    return predictions.flatten(), metrics


def create_prediction_dataframe(historical_data, predictions, prediction_dates):
    """
    Create a dataframe combining historical and predicted data.

    Args:
        historical_data (pd.DataFrame): Historical stock data
        predictions (np.array): Predicted values
        prediction_dates (list): Dates for predictions

    Returns:
        pd.DataFrame: Combined dataframe
    """
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted_Close': predictions
    })

    # Add a column to historical data to distinguish it
    historical_data['Type'] = 'Historical'
    pred_df['Type'] = 'Predicted'

    # Select relevant columns from historical data
    hist_subset = historical_data[['Date', 'Close']].copy()
    hist_subset['Predicted_Close'] = hist_subset['Close']
    hist_subset['Type'] = 'Historical'

    # Combine dataframes
    combined_df = pd.concat([
        hist_subset[['Date', 'Predicted_Close', 'Type']],
        pred_df
    ], ignore_index=True)

    return combined_df


def generate_trading_signals(predictions, current_price, threshold=0.02):
    """
    Generate trading signals based on predictions.

    Args:
        predictions (np.array): Predicted prices
        current_price (float): Current stock price
        threshold (float): Percentage threshold for signals

    Returns:
        dict: Trading signals and analysis
    """
    # Calculate percentage changes
    final_prediction = predictions[-1]
    price_change = (final_prediction - current_price) / current_price

    # Generate signal
    if price_change > threshold:
        signal = 'BUY'
        confidence = min(price_change / threshold, 2.0) * 50  # Cap at 100%
    elif price_change < -threshold:
        signal = 'SELL'
        confidence = min(abs(price_change) / threshold, 2.0) * 50
    else:
        signal = 'HOLD'
        confidence = 50

    # Calculate additional metrics
    max_price = np.max(predictions)
    min_price = np.min(predictions)
    avg_price = np.mean(predictions)
    volatility = np.std(predictions)

    return {
        'signal': signal,
        'confidence': confidence,
        'current_price': current_price,
        'predicted_price': final_prediction,
        'price_change_percent': price_change * 100,
        'max_predicted': max_price,
        'min_predicted': min_price,
        'avg_predicted': avg_price,
        'volatility': volatility
    }


def backtest_predictions(model, data, lookback_days, prediction_days, scaler, model_type='lstm'):
    """
    Backtest model predictions on historical data.

    Args:
        model: Trained model
        data (pd.DataFrame): Historical data
        lookback_days (int): Days to look back
        prediction_days (int): Days to predict forward
        scaler: Data scaler
        model_type (str): Model type

    Returns:
        pd.DataFrame: Backtest results
    """
    results = []

    # Use last 20% of data for backtesting
    test_start_idx = int(len(data) * 0.8)

    for i in range(test_start_idx, len(data) - prediction_days, prediction_days):
        # Get sequence for prediction
        sequence = data['Close'].iloc[i - lookback_days:i].values
        scaled_sequence = scaler.transform(sequence.reshape(-1, 1)).flatten()

        # Make predictions
        predictions = predict_future_prices(
            model, scaled_sequence, scaler,
            days=prediction_days, model_type=model_type
        )

        # Get actual values
        actual_values = data['Close'].iloc[i:i + prediction_days].values

        # Calculate error
        if len(actual_values) == len(predictions):
            mae = np.mean(np.abs(predictions - actual_values))
            mape = np.mean(np.abs((predictions - actual_values) / actual_values)) * 100

            results.append({
                'start_date': data['Date'].iloc[i],
                'end_date': data['Date'].iloc[i + prediction_days - 1],
                'mae': mae,
                'mape': mape,
                'predicted_trend': 'up' if predictions[-1] > predictions[0] else 'down',
                'actual_trend': 'up' if actual_values[-1] > actual_values[0] else 'down'
            })

    return pd.DataFrame(results)