"""
Enhanced Machine Learning models for stock price prediction with comprehensive NaN handling.
Includes LSTM and Linear Regression implementations.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class StockLSTM:
    """LSTM model for stock price prediction with enhanced NaN handling."""

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
        Train the LSTM model with comprehensive NaN validation.

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio

        Returns:
            History object
        """
        # Comprehensive input validation
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            raise ValueError("Training data contains NaN values. Please clean data using stock_utils functions first.")

        if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
            raise ValueError("Training data contains infinite values. Please clean data first.")

        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty.")

        print(f"Training LSTM with {len(X_train)} samples")

        # Reshape X_train for LSTM [samples, time steps, features]
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))

        # Enhanced callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=7,
                min_lr=0.0001,
                verbose=0
            )
        ]

        try:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            print("✅ LSTM model trained successfully!")

        except Exception as e:
            print(f"❌ Error during LSTM training: {e}")
            raise

        return self.history

    def predict(self, X, verbose=0):
        """
        Make predictions with comprehensive error handling.

        Args:
            X (np.array): Input features
            verbose (int): Verbosity level for predictions (default: 0)

        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Input validation
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values. Please clean data first.")

        if np.any(np.isinf(X)):
            raise ValueError("Input data contains infinite values. Please clean data first.")

        # Reshape for LSTM if needed
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))

        try:
            predictions = self.model.predict(X, verbose=verbose)

            # Validate predictions
            if np.any(np.isnan(predictions)):
                print("⚠️ Warning: Model produced NaN predictions. Using fallback values.")
                predictions = np.nan_to_num(predictions, nan=0.0)

            return predictions

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            raise

    def save_model(self, filepath):
        """Save the model to disk with error handling."""
        if self.model is not None:
            try:
                self.model.save(filepath)
                print(f"✅ Model saved to {filepath}")
            except Exception as e:
                print(f"❌ Error saving model: {e}")
                raise
        else:
            raise ValueError("No model to save. Train the model first.")

    def load_model(self, filepath):
        """Load model from disk with error handling."""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"✅ Model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

class StockRegressor:
    """Enhanced regression models for stock price prediction with comprehensive NaN handling."""

    def __init__(self, model_type='linear'):
        """
        Initialize regression model with NaN handling capabilities.

        Args:
            model_type (str): Type of regression model
                            ('linear', 'random_forest', 'hist_gradient_boosting')
        """
        self.model_type = model_type
        self.imputer = None

        if model_type == 'linear':
            self.model = LinearRegression()
            self.imputer = SimpleImputer(strategy='mean')

        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.imputer = SimpleImputer(strategy='mean')

        elif model_type == 'hist_gradient_boosting':
            # This model can handle NaN natively
            self.model = HistGradientBoostingRegressor(
                random_state=42,
                max_iter=100
            )
            # No imputer needed for this model

        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'linear', 'random_forest', or 'hist_gradient_boosting'")

        print(f"✅ Initialized {model_type} regressor with NaN handling")

    def train(self, X_train, y_train):
        """
        Train the regression model with comprehensive NaN handling.

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
        """
        print(f"Training {self.model_type} model...")

        # Handle NaN values in target
        if np.any(np.isnan(y_train)):
            valid_mask = ~np.isnan(y_train)
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            print(f"Removed {(~valid_mask).sum()} samples with NaN targets")

        # Handle infinite values in target
        if np.any(np.isinf(y_train)):
            finite_mask = np.isfinite(y_train)
            X_train = X_train[finite_mask]
            y_train = y_train[finite_mask]
            print(f"Removed {(~finite_mask).sum()} samples with infinite targets")

        if len(X_train) == 0:
            raise ValueError("No valid training samples remain after cleaning")

        # Handle NaN values in features based on model type
        if self.model_type == 'hist_gradient_boosting':
            # HistGradientBoostingRegressor can handle NaN natively
            # Just replace infinite values with NaN
            X_train_processed = np.where(np.isinf(X_train), np.nan, X_train)
            print("Using native NaN handling for HistGradientBoostingRegressor")

        else:
            # For other models, use imputation
            if self.imputer is not None:
                if np.any(np.isnan(X_train)):
                    X_train_processed = self.imputer.fit_transform(X_train)
                    print(f"Imputed {np.isnan(X_train).sum()} NaN values in features")
                else:
                    X_train_processed = X_train
                    self.imputer.fit(X_train)  # Fit for consistency in transform

                # Handle infinite values
                X_train_processed = np.where(np.isinf(X_train_processed), 0, X_train_processed)
            else:
                X_train_processed = X_train

        try:
            # Train the model
            self.model.fit(X_train_processed, y_train)
            print(f"✅ {self.model_type} model trained successfully on {len(X_train_processed)} samples!")

        except Exception as e:
            print(f"❌ Error training {self.model_type} model: {e}")
            raise

    def predict(self, X, verbose=0):
        """
        Make predictions with comprehensive NaN handling.

        Args:
            X (np.array): Input features
            verbose (int): Verbosity level for predictions (default: 0)

        Returns:
            np.array: Predictions
        """
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model not trained. Please train the model first.")

        # Handle NaN values in features based on model type
        if self.model_type == 'hist_gradient_boosting':
            # HistGradientBoostingRegressor can handle NaN natively
            X_processed = np.where(np.isinf(X), np.nan, X)

        else:
            # For other models, use imputation
            if self.imputer is not None:
                if np.any(np.isnan(X)):
                    X_processed = self.imputer.transform(X)
                    if verbose > 0:
                        print(f"Imputed {np.isnan(X).sum()} NaN values for prediction")
                else:
                    X_processed = X

                # Handle infinite values
                X_processed = np.where(np.isinf(X_processed), 0, X_processed)
            else:
                X_processed = X

        try:
            predictions = self.model.predict(X_processed)

            # Validate predictions
            if np.any(np.isnan(predictions)):
                print("⚠️ Warning: Model produced NaN predictions. Using fallback values.")
                predictions = np.nan_to_num(predictions, nan=0.0)

            if np.any(np.isinf(predictions)):
                print("⚠️ Warning: Model produced infinite predictions. Clipping values.")
                predictions = np.clip(predictions, -1e10, 1e10)

            return predictions

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            raise

    def save_model(self, filepath):
        """Save the model and imputer to disk."""
        try:
            model_data = {
                'model': self.model,
                'imputer': self.imputer,
                'model_type': self.model_type
            }
            joblib.dump(model_data, filepath)
            print(f"✅ Model saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise

    def load_model(self, filepath):
        """Load model and imputer from disk."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.imputer = model_data['imputer']
            self.model_type = model_data['model_type']
            print(f"✅ Model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

def create_ensemble_predictions(models, X):
    """
    Create ensemble predictions from multiple models with comprehensive NaN handling.

    Args:
        models (list): List of trained models
        X (np.array): Input features

    Returns:
        np.array: Ensemble predictions (average)
    """
    predictions = []
    successful_models = 0

    for i, model in enumerate(models):
        if hasattr(model, 'predict'):
            try:
                pred = model.predict(X)

                # Handle NaN predictions
                if np.any(np.isnan(pred)):
                    print(f"⚠️ Model {i} produced NaN predictions, using fallback")
                    pred = np.nan_to_num(pred, nan=0.0)

                predictions.append(pred.flatten())
                successful_models += 1

            except Exception as e:
                print(f"⚠️ Model {i} prediction failed: {e}, skipping")
                continue

    if successful_models == 0:
        raise ValueError("No models produced valid predictions")

    # Return average of successful predictions
    ensemble_pred = np.mean(predictions, axis=0)

    # Final validation
    if np.any(np.isnan(ensemble_pred)):
        print("⚠️ Ensemble produced NaN predictions, using fallback")
        ensemble_pred = np.nan_to_num(ensemble_pred, nan=0.0)

    print(f"✅ Ensemble prediction from {successful_models} models")
    return ensemble_pred
