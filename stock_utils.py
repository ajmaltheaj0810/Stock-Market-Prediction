"""
Enhanced Stock data fetching and preprocessing utilities with comprehensive NaN handling.
Handles data retrieval from Yahoo Finance and data preparation for ML models.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import ta
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance with enhanced error handling.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: Stock data with OHLCV columns and technical indicators
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        # Reset index to have Date as a column
        data.reset_index(inplace=True)

        # Add technical indicators with comprehensive NaN handling
        data = add_technical_indicators_safe(data)

        return data

    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def add_technical_indicators_safe(df):
    """
    Add technical indicators to the dataframe with comprehensive NaN handling.

    Args:
        df (pd.DataFrame): Stock data

    Returns:
        pd.DataFrame: Stock data with technical indicators (NaN values properly handled)
    """
    # Create a copy to avoid modifying original data
    df = df.copy()

    # Ensure we have required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Moving averages with min_periods to avoid initial NaN values
    df['MA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    # RSI with proper NaN handling
    try:
        rsi_indicator = ta.momentum.RSIIndicator(df['Close'], window=14, fillna=True)
        df['RSI'] = rsi_indicator.rsi()
        # Additional fallback for any remaining NaN values
        if df['RSI'].isna().any():
            df['RSI'].fillna(method='bfill', inplace=True)
            df['RSI'].fillna(50.0, inplace=True)  # Neutral RSI value
    except Exception as e:
        print(f"Warning: RSI calculation failed, using manual calculation")
        df['RSI'] = calculate_rsi_manual(df['Close'], window=14)

    # MACD with proper NaN handling
    try:
        macd_indicator = ta.trend.MACD(df['Close'],
                                     window_slow=26,
                                     window_fast=12,
                                     window_sign=9,
                                     fillna=True)
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()

        # Handle any remaining NaN values
        df['MACD'].fillna(method='bfill', inplace=True)
        df['MACD'].fillna(0.0, inplace=True)
        df['MACD_signal'].fillna(method='bfill', inplace=True)
        df['MACD_signal'].fillna(0.0, inplace=True)

    except Exception as e:
        print(f"Warning: MACD calculation failed, using fallback values")
        df['MACD'] = 0.0
        df['MACD_signal'] = 0.0

    # Bollinger Bands with proper NaN handling
    try:
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, fillna=True)
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()

        # Handle any remaining NaN values
        if df['BB_upper'].isna().any() or df['BB_lower'].isna().any():
            # Fallback: use rolling mean ± 2*std with min_periods
            rolling_mean = df['Close'].rolling(window=20, min_periods=1).mean()
            rolling_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_upper'].fillna(rolling_mean + (rolling_std * 2), inplace=True)
            df['BB_lower'].fillna(rolling_mean - (rolling_std * 2), inplace=True)

    except Exception as e:
        print(f"Warning: Bollinger Bands calculation failed, using manual calculation")
        rolling_mean = df['Close'].rolling(window=20, min_periods=1).mean()
        rolling_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_upper'] = rolling_mean + (rolling_std * 2)
        df['BB_lower'] = rolling_mean - (rolling_std * 2)

    # Volume indicators with min_periods
    df['Volume_MA'] = df['Volume'].rolling(window=10, min_periods=1).mean()

    # Final comprehensive NaN handling
    df = handle_remaining_nans(df)

    return df

def calculate_rsi_manual(prices, window=14):
    """
    Calculate RSI manually when ta library fails.

    Args:
        prices (pd.Series): Price series
        window (int): RSI calculation window

    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Fill any remaining NaN values
    rsi.fillna(method='bfill', inplace=True)
    rsi.fillna(50.0, inplace=True)  # Neutral RSI value

    return rsi

def handle_remaining_nans(df):
    """
    Handle remaining NaN values using multiple strategies.

    Args:
        df (pd.DataFrame): DataFrame with potential NaN values

    Returns:
        pd.DataFrame: DataFrame with NaN values handled
    """
    # Get numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Strategy 1: Forward fill for time series continuity
    for col in numeric_columns:
        if df[col].isna().any():
            df[col].fillna(method='ffill', inplace=True)

    # Strategy 2: Backward fill for any remaining leading NaNs
    for col in numeric_columns:
        if df[col].isna().any():
            df[col].fillna(method='bfill', inplace=True)

    # Strategy 3: Fill any remaining NaNs with column means or appropriate defaults
    for col in numeric_columns:
        if df[col].isna().any():
            if col in ['RSI']:
                df[col].fillna(50.0, inplace=True)  # Neutral RSI
            elif col in ['MACD', 'MACD_signal']:
                df[col].fillna(0.0, inplace=True)  # Zero for MACD
            else:
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    df[col].fillna(0.0, inplace=True)
                else:
                    df[col].fillna(mean_val, inplace=True)

    # Strategy 4: Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill any NaN values created by inf replacement
    for col in numeric_columns:
        if df[col].isna().any():
            df[col].fillna(0.0, inplace=True)

    return df

def validate_data_quality(df):
    """
    Validate data quality and provide diagnostics.

    Args:
        df (pd.DataFrame): DataFrame to validate

    Returns:
        dict: Validation results and recommendations
    """
    results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_data': {},
        'infinite_values': {},
        'data_types': df.dtypes.to_dict(),
        'recommendations': []
    }

    # Check for missing data
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            results['missing_data'][col] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100
            }

    # Check for infinite values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            results['infinite_values'][col] = inf_count

    # Generate recommendations
    if results['missing_data']:
        results['recommendations'].append("Missing data detected - applying comprehensive NaN handling")

    if results['infinite_values']:
        results['recommendations'].append("Infinite values detected - replacing with NaN and handling")

    if not results['missing_data'] and not results['infinite_values']:
        results['recommendations'].append("Data quality is good - no NaN or infinite values detected")

    return results

def prepare_data_for_lstm(data, lookback_days=60):
    """
    Prepare data for LSTM model training with comprehensive NaN handling.

    Args:
        data (pd.DataFrame): Stock data
        lookback_days (int): Number of days to look back for prediction

    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler, train_size)
    """
    # Ensure data is clean before processing
    data = handle_remaining_nans(data.copy())

    # Use closing price for prediction
    dataset = data['Close'].values.reshape(-1, 1)

    # Check for any remaining NaN or infinite values
    if np.any(np.isnan(dataset)) or np.any(np.isinf(dataset)):
        # Remove problematic rows
        mask = np.isfinite(dataset.flatten())
        dataset = dataset[mask]
        print(f"Removed {(~mask).sum()} rows with NaN/inf values from LSTM data")

    if len(dataset) < lookback_days + 10:
        raise ValueError(f"Insufficient data after cleaning. Need at least {lookback_days + 10} points, got {len(dataset)}")

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
    Prepare data for regression model with comprehensive NaN handling.

    Args:
        data (pd.DataFrame): Stock data
        feature_days (int): Number of days to use for features

    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    # Ensure data is clean before processing
    data = handle_remaining_nans(data.copy())

    # Verify required columns exist
    required_features = ['Close', 'Volume', 'MA_10', 'RSI']
    missing_features = [f for f in required_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Create features from historical data
    features = []
    targets = []

    for i in range(feature_days, len(data)):
        try:
            # Use multiple features
            feature_data = data.iloc[i - feature_days:i][required_features].values.flatten()

            # Check for NaN in feature data (should be rare after cleaning)
            if np.any(np.isnan(feature_data)) or np.any(np.isinf(feature_data)):
                print(f"Skipping sample {i} due to NaN/inf values")
                continue

            features.append(feature_data)
            targets.append(data.iloc[i]['Close'])

        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")
            continue

    if len(features) == 0:
        raise ValueError("No valid features could be created. Check your data quality.")

    X = np.array(features)
    y = np.array(targets)

    # Final validation check
    nan_mask = np.any(np.isnan(X), axis=1) | np.isnan(y) | np.any(np.isinf(X), axis=1) | np.isinf(y)
    if np.any(nan_mask):
        X = X[~nan_mask]
        y = y[~nan_mask]
        print(f"Removed {nan_mask.sum()} samples with NaN/inf values from regression data")

    if len(X) == 0:
        raise ValueError("No valid samples remain after NaN removal")

    # Split and scale
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features with NaN handling
    scaler = MinMaxScaler()

    # Use SimpleImputer as additional safety
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics with NaN handling.

    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values

    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Remove any NaN values from predictions
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Backward compatibility - keep original function name
def add_technical_indicators(df):
    """
    Backward compatibility wrapper for add_technical_indicators_safe.
    """
    return add_technical_indicators_safe(df)
