"""
Machine Learning models for stock price prediction.
Includes LSTM and Linear Regression implementations.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import joblib
import os


class StockLSTM:
    """LSTM model for stock price prediction."""

    def __init__(self, lookback_days=60, units=50, dropout=0.2):
        """
        Initialize LSTM model.

        Args:
            lookback_days (int): Number of days to look back
            units (int): Number of LSTM units
            dropout (float): Dropout rate
        """
        self.lookback_days = lookback_days
        self.units = units
        self.dropout = dropout
        self.model = None
        self.history = None

    def build_model(self, input_shape):
        """
        Build LSTM model architecture.

        Args:
            input_shape (tuple): Shape of input data
        """
        self.model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(units=self.units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(units=self.units),
            Dropout(self.dropout),
            Dense(units=1)
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model.

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio

        Returns:
            History object
        """
        # Reshape X_train for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        if self.model is None:
            self.build_model((X_train.shape[1], 1))

        # Add early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        return self.history

    def predict(self, X, verbose=0):
        """
        Make predictions.

        Args:
            X (np.array): Input features
            verbose (int): Verbosity level for predictions (default: 0)

        Returns:
            np.array: Predictions
        """
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X, verbose=verbose)

    def save_model(self, filepath):
        """Save the model to disk."""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)


class StockRegressor:
    """Regression models for stock price prediction."""

    def __init__(self, model_type='linear'):
        """
        Initialize regression model.

        Args:
            model_type (str): Type of regression model ('linear' or 'random_forest')
        """
        self.model_type = model_type

        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train, y_train):
        """
        Train the regression model.

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
        """
        self.model.fit(X_train, y_train)

    def predict(self, X, verbose=0):
        """
        Make predictions.

        Args:
            X (np.array): Input features
            verbose (int): Verbosity level for predictions (default: 0)

        Returns:
            np.array: Predictions
        """
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X, verbose=verbose)

    def save_model(self, filepath):
        """Save the model to disk."""
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Load model from disk."""
        self.model = joblib.load(filepath)


def create_ensemble_predictions(models, X):
    """
    Create ensemble predictions from multiple models.

    Args:
        models (list): List of trained models
        X (np.array): Input features

    Returns:
        np.array: Ensemble predictions (average)
    """
    predictions = []

    for model in models:
        if hasattr(model, 'predict'):
            pred = model.predict(X)
            predictions.append(pred.flatten())

    # Return average of all predictions
    return np.mean(predictions, axis=0)