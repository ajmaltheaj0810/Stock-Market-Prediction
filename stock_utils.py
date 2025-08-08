"""
Stock data fetching and preprocessing utilities.
Handles data retrieval from Yahoo Finance and data preparation for ML models.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import ta


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        # Reset index to have Date as a column
        data.reset_index(inplace=True)

        # Add technical indicators
        data = add_technical_indicators(data)

        return data

    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")


def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.

    Args:
        df (pd.DataFrame): Stock data

    Returns:
        pd.DataFrame: Stock data with technical indicators
    """
    # Moving averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()

    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()

    return df


def prepare_data_for_lstm(data, lookback_days=60):
    """
    Prepare data for LSTM model training.

    Args:
        data (pd.DataFrame): Stock data
        lookback_days (int): Number of days to look back for prediction

    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler, train_size)
    """
    # Use closing price for prediction
    dataset = data['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Split data into train and test sets (80-20 split)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - lookback_days:]

    # Create sequences for LSTM
    X_train, y_train = create_sequences(train_data, lookback_days)
    X_test, y_test = create_sequences(test_data, lookback_days)

    return X_train, y_train, X_test, y_test, scaler, train_size


def create_sequences(data, lookback_days):
    """
    Create sequences for time series prediction.

    Args:
        data (np.array): Scaled data
        lookback_days (int): Number of days to look back

    Returns:
        tuple: (X, y) sequences
    """
    X, y = [], []

    for i in range(lookback_days, len(data)):
        X.append(data[i - lookback_days:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


def prepare_data_for_regression(data, feature_days=30):
    """
    Prepare data for regression model.

    Args:
        data (pd.DataFrame): Stock data
        feature_days (int): Number of days to use for features

    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    # Create features from historical data
    features = []
    targets = []

    for i in range(feature_days, len(data)):
        # Use multiple features
        feature_data = data.iloc[i - feature_days:i][['Close', 'Volume', 'MA_10', 'RSI']].values.flatten()
        features.append(feature_data)
        targets.append(data.iloc[i]['Close'])

    X = np.array(features)
    y = np.array(targets)

    # Split and scale
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.

    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values

    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }