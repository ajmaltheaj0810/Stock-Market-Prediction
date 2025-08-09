"""
Enhanced Streamlit application for stock market prediction with comprehensive NaN handling.
Main interface for users to interact with the prediction system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# Import enhanced custom modules
from stock_utils import (
    fetch_stock_data, prepare_data_for_lstm, prepare_data_for_regression,
    validate_data_quality, handle_remaining_nans
)
from model import StockLSTM, StockRegressor
from predict import (
    predict_future_prices, create_prediction_dates, evaluate_model,
    create_prediction_dataframe, generate_trading_signals, backtest_predictions
)

# Page configuration
st.set_page_config(
    page_title="📈 Enhanced Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'data_quality' not in st.session_state:
    st.session_state.data_quality = None

# Title and description
st.markdown('<div class="main-header">📈 Enhanced Stock Market Prediction App</div>', unsafe_allow_html=True)
st.markdown("### Predict future stock prices using Machine Learning with robust NaN handling")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Stock selection
    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter a valid stock ticker (e.g., AAPL, GOOGL, MSFT, TSLA)"
    ).upper()

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365 * 2),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )

    # Model selection
    st.subheader("🤖 Model Configuration")
    model_type = st.selectbox(
        "Select Model Type",
        [
            "LSTM (Deep Learning)",
            "Linear Regression (Enhanced)",
            "Random Forest (Enhanced)",
            "Gradient Boosting (NaN-Native)"
        ],
        help="Enhanced models include comprehensive NaN handling"
    )

    # Prediction parameters
    prediction_days = st.slider(
        "Days to Predict",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Number of future days to predict"
    )

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        if "LSTM" in model_type:
            lookback_days = st.slider("Lookback Days", 30, 120, 60)
            epochs = st.slider("Training Epochs", 10, 100, 50)
            batch_size = st.slider("Batch Size", 16, 64, 32)
        else:
            feature_days = st.slider("Feature Days", 10, 60, 30)

        # Data quality settings
        st.subheader("📊 Data Quality")
        show_data_quality = st.checkbox("Show Data Quality Report", value=True)
        auto_fix_nan = st.checkbox("Auto-fix NaN Issues", value=True)

    # Action buttons
    st.markdown("---")
    fetch_button = st.button("📊 Fetch Data", type="primary", use_container_width=True)
    train_button = st.button("🧠 Train Model", type="secondary", use_container_width=True)

# Main content area
if fetch_button or train_button:
    try:
        # Fetch stock data
        with st.spinner(f"Fetching data for {ticker}..."):
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            st.session_state.stock_data = stock_data

        # Data quality assessment
        if show_data_quality:
            with st.spinner("Analyzing data quality..."):
                data_quality = validate_data_quality(stock_data)
                st.session_state.data_quality = data_quality

        # Display data quality report
        if show_data_quality and st.session_state.data_quality:
            quality_report = st.session_state.data_quality

            st.subheader("📊 Data Quality Report")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", quality_report['total_rows'])
            with col2:
                st.metric("Total Columns", quality_report['total_columns'])
            with col3:
                missing_count = sum(
                    [v['count'] if isinstance(v, dict) else v for v in quality_report['missing_data'].values()]) if \
                quality_report['missing_data'] else 0
                st.metric("Missing Values", missing_count)

            # Show recommendations
            if quality_report['recommendations']:
                st.info("📋 **Data Quality Recommendations:**\n" + "\n".join(
                    f"• {rec}" for rec in quality_report['recommendations']))

        # Auto-fix NaN issues if enabled
        if auto_fix_nan:
            with st.spinner("Auto-fixing data quality issues..."):
                stock_data = handle_remaining_nans(stock_data)
                st.session_state.stock_data = stock_data
                st.success("✅ Data quality issues automatically resolved!")

        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = stock_data['Close'].iloc[-1] if len(stock_data) > 0 else 0
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            if len(stock_data) > 1:
                change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                change_pct = (change / stock_data['Close'].iloc[-2]) * 100
                st.metric("Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
        with col3:
            volume = stock_data['Volume'].iloc[-1] if len(stock_data) > 0 else 0
            st.metric("Volume", f"{volume:,.0f}")
        with col4:
            st.metric("Data Points", len(stock_data))

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Historical Data",
            "🔮 Predictions",
            "📈 Technical Analysis",
            "📉 Model Performance",
            "💾 Export Data"
        ])

        with tab1:
            st.subheader("📊 Historical Stock Data")

            # Enhanced candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                row_heights=[0.7, 0.3]
            )

            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=stock_data['Date'],
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name="OHLC"
                ),
                row=1, col=1
            )

            # Add moving averages if available
            for ma_col, color in [('MA_20', 'orange'), ('MA_50', 'red')]:
                if ma_col in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data['Date'],
                            y=stock_data[ma_col],
                            name=ma_col.replace('_', ' '),
                            line=dict(color=color, width=1)
                        ),
                        row=1, col=1
                    )

            # Add volume
            fig.add_trace(
                go.Bar(
                    x=stock_data['Date'],
                    y=stock_data['Volume'],
                    name="Volume",
                    marker_color='lightblue'
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                title=f"{ticker} Stock Analysis - Enhanced with NaN Handling",
                yaxis_title="Price ($)",
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show recent data
            st.subheader("📋 Recent Data")
            recent_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
            st.dataframe(recent_data, use_container_width=True)

        with tab2:
            st.subheader("🔮 Stock Price Predictions")

            if train_button:
                # Model training with enhanced error handling
                with st.spinner("Training model with NaN handling... This may take a few minutes."):
                    try:
                        if "LSTM" in model_type:
                            # Prepare LSTM data
                            X_train, y_train, X_test, y_test, scaler, train_size = prepare_data_for_lstm(
                                stock_data, lookback_days
                            )

                            # Create and train LSTM model
                            model = StockLSTM(lookback_days=lookback_days)
                            history = model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

                            # Store in session state
                            st.session_state.model = model
                            st.session_state.scaler = scaler
                            st.session_state.model_type = 'lstm'
                            st.session_state.lookback_days = lookback_days

                        else:
                            # Prepare regression data
                            X_train, y_train, X_test, y_test, scaler = prepare_data_for_regression(
                                stock_data, feature_days
                            )

                            # Create and train regression model
                            if "Linear" in model_type:
                                model_name = 'linear'
                            elif "Random Forest" in model_type:
                                model_name = 'random_forest'
                            else:  # Gradient Boosting
                                model_name = 'hist_gradient_boosting'

                            model = StockRegressor(model_type=model_name)
                            model.train(X_train, y_train)

                            # Store in session state
                            st.session_state.model = model
                            st.session_state.scaler = scaler
                            st.session_state.model_type = 'regression'
                            st.session_state.feature_days = feature_days

                        st.session_state.model_trained = True
                        st.success("✅ Model trained successfully with enhanced NaN handling!")

                    except Exception as e:
                        st.error(f"❌ Error during model training: {str(e)}")
                        st.info("💡 **Troubleshooting Tips:**")
                        st.info("• Try a different date range with more data")
                        st.info("• Select a different model type")
                        st.info("• Enable auto-fix NaN issues in Advanced Settings")

            # Make predictions if model is trained
            if st.session_state.model_trained and st.session_state.model is not None:
                with st.spinner("Generating predictions..."):
                    try:
                        if st.session_state.model_type == 'lstm':
                            # Get last sequence for LSTM prediction
                            last_sequence = stock_data['Close'].iloc[-st.session_state.lookback_days:].values
                            scaled_sequence = st.session_state.scaler.transform(last_sequence.reshape(-1, 1)).flatten()

                            future_predictions = predict_future_prices(
                                st.session_state.model, scaled_sequence, st.session_state.scaler,
                                days=prediction_days, model_type='lstm'
                            )
                        else:
                            # For regression models
                            required_cols = ['Close', 'Volume', 'MA_10', 'RSI']
                            available_cols = [col for col in required_cols if col in stock_data.columns]

                            if len(available_cols) < len(required_cols):
                                st.warning(f"⚠️ Missing columns: {set(required_cols) - set(available_cols)}")
                                st.info("Using available features for prediction...")

                            last_features = stock_data[available_cols].iloc[
                                            -st.session_state.feature_days:].values.flatten()
                            scaled_features = st.session_state.scaler.transform(last_features.reshape(1, -1))

                            future_predictions = predict_future_prices(
                                st.session_state.model, scaled_features[0], st.session_state.scaler,
                                days=prediction_days, model_type='regression'
                            )

                        # Create prediction dates
                        last_date = stock_data['Date'].iloc[-1]
                        prediction_dates = create_prediction_dates(last_date + timedelta(days=1), prediction_days)

                        # Generate trading signals
                        current_price = stock_data['Close'].iloc[-1]
                        signals = generate_trading_signals(future_predictions, current_price)

                        # Display trading signal
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            signal_color = {
                                'BUY': 'green',
                                'SELL': 'red',
                                'HOLD': 'gray'
                            }.get(signals['signal'], 'gray')

                            st.markdown(
                                f'<div style="background-color: {signal_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">'
                                f'<h3>Signal: {signals["signal"]}</h3></div>',
                                unsafe_allow_html=True
                            )
                        with col2:
                            st.metric("Confidence", f"{signals['confidence']:.1f}%")
                        with col3:
                            st.metric("Expected Change", f"{signals['price_change_percent']:.2f}%")

                        # Create prediction visualization
                        fig = go.Figure()

                        # Add historical data (last 60 days)
                        hist_data = stock_data.tail(min(60, len(stock_data)))
                        fig.add_trace(go.Scatter(
                            x=hist_data['Date'],
                            y=hist_data['Close'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))

                        # Add predictions
                        fig.add_trace(go.Scatter(
                            x=prediction_dates,
                            y=future_predictions,
                            mode='lines+markers',
                            name='Predictions',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))

                        # Add confidence interval
                        std_dev = np.std(future_predictions) * 0.1
                        upper_bound = future_predictions + std_dev
                        lower_bound = future_predictions - std_dev

                        fig.add_trace(go.Scatter(
                            x=prediction_dates + prediction_dates[::-1],
                            y=list(upper_bound) + list(lower_bound[::-1]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.1)',
                            line=dict(color='rgba(255,0,0,0)'),
                            name='Confidence Interval',
                            showlegend=True
                        ))

                        # Update layout
                        fig.update_layout(
                            title=f"{ticker} Enhanced Price Prediction - Next {prediction_days} Days",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Display prediction statistics
                        st.subheader("📊 Prediction Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Predicted Price", f"${signals['predicted_price']:.2f}")
                        with col2:
                            st.metric("Max Predicted", f"${signals['max_predicted']:.2f}")
                        with col3:
                            st.metric("Min Predicted", f"${signals['min_predicted']:.2f}")
                        with col4:
                            st.metric("Volatility", f"${signals['volatility']:.2f}")

                        # Prediction table
                        pred_df = pd.DataFrame({
                            'Date': prediction_dates,
                            'Predicted Price': future_predictions,
                            'Change from Current': [(p - current_price) / current_price * 100 for p in
                                                    future_predictions]
                        })

                        st.subheader("📅 Daily Predictions")
                        st.dataframe(
                            pred_df.style.format({
                                'Predicted Price': '${:.2f}',
                                'Change from Current': '{:.2f}%'
                            }),
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"❌ Error generating predictions: {str(e)}")
                        st.info(
                            "💡 This may be due to insufficient clean data. Try adjusting the date range or model parameters.")

            else:
                st.info("👆 Click 'Train Model' above to generate predictions")

        with tab3:
            st.subheader("📈 Technical Analysis")

            # Check if technical indicators are available
            tech_indicators = ['RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']
            available_indicators = [ind for ind in tech_indicators if ind in stock_data.columns]

            if available_indicators:
                # Create technical analysis chart
                fig_tech = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price with Bollinger Bands', 'RSI', 'MACD'),
                    row_heights=[0.5, 0.25, 0.25]
                )

                # Price with Bollinger Bands
                fig_tech.add_trace(
                    go.Scatter(x=stock_data['Date'], y=stock_data['Close'],
                               name='Close Price', line=dict(color='blue')),
                    row=1, col=1
                )

                if 'BB_upper' in stock_data.columns and 'BB_lower' in stock_data.columns:
                    fig_tech.add_trace(
                        go.Scatter(x=stock_data['Date'], y=stock_data['BB_upper'],
                                   name='Upper Band', line=dict(color='gray', dash='dash')),
                        row=1, col=1
                    )
                    fig_tech.add_trace(
                        go.Scatter(x=stock_data['Date'], y=stock_data['BB_lower'],
                                   name='Lower Band', line=dict(color='gray', dash='dash'),
                                   fill='tonexty', fillcolor='rgba(128,128,128,0.2)'),
                        row=1, col=1
                    )

                # RSI
                if 'RSI' in stock_data.columns:
                    fig_tech.add_trace(
                        go.Scatter(x=stock_data['Date'], y=stock_data['RSI'],
                                   name='RSI', line=dict(color='orange')),
                        row=2, col=1
                    )
                    # Add RSI levels
                    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # MACD
                if 'MACD' in stock_data.columns:
                    fig_tech.add_trace(
                        go.Scatter(x=stock_data['Date'], y=stock_data['MACD'],
                                   name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )

                    if 'MACD_signal' in stock_data.columns:
                        fig_tech.add_trace(
                            go.Scatter(x=stock_data['Date'], y=stock_data['MACD_signal'],
                                       name='Signal', line=dict(color='red')),
                            row=3, col=1
                        )

                # Update layout
                fig_tech.update_layout(
                    title="Enhanced Technical Indicators Analysis",
                    height=800,
                    showlegend=True
                )
                fig_tech.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig_tech.update_yaxes(title_text="RSI", row=2, col=1)
                fig_tech.update_yaxes(title_text="MACD", row=3, col=1)

                st.plotly_chart(fig_tech, use_container_width=True)

                # Technical indicators summary
                st.subheader("📊 Technical Indicators Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if 'RSI' in stock_data.columns:
                        latest_rsi = stock_data['RSI'].iloc[-1]
                        if pd.notna(latest_rsi):
                            rsi_signal = ("Overbought" if latest_rsi > 70 else
                                          "Oversold" if latest_rsi < 30 else "Neutral")
                            st.info(f"**RSI Signal:** {rsi_signal} ({latest_rsi:.2f})")
                        else:
                            st.info("**RSI:** Data not available")
                    else:
                        st.info("**RSI:** Not calculated")

                with col2:
                    if 'MACD' in stock_data.columns and 'MACD_signal' in stock_data.columns:
                        latest_macd = stock_data['MACD'].iloc[-1]
                        latest_signal = stock_data['MACD_signal'].iloc[-1]
                        if pd.notna(latest_macd) and pd.notna(latest_signal):
                            macd_signal = "Bullish" if latest_macd > latest_signal else "Bearish"
                            st.info(f"**MACD Signal:** {macd_signal}")
                        else:
                            st.info("**MACD:** Data not available")
                    else:
                        st.info("**MACD:** Not calculated")

                with col3:
                    if all(col in stock_data.columns for col in ['Close', 'BB_upper', 'BB_lower']):
                        current_price = stock_data['Close'].iloc[-1]
                        bb_upper = stock_data['BB_upper'].iloc[-1]
                        bb_lower = stock_data['BB_lower'].iloc[-1]

                        if all(pd.notna(x) for x in [current_price, bb_upper, bb_lower]):
                            if current_price > bb_upper:
                                bb_position = "Upper Band"
                            elif current_price < bb_lower:
                                bb_position = "Lower Band"
                            else:
                                bb_position = "Middle"
                            st.info(f"**Bollinger Band:** {bb_position}")
                        else:
                            st.info("**Bollinger Bands:** Data not available")
                    else:
                        st.info("**Bollinger Bands:** Not calculated")

            else:
                st.warning(
                    "⚠️ Technical indicators not available. This may be due to insufficient data or calculation errors.")

        with tab4:
            st.subheader("📉 Model Performance")

            if st.session_state.model_trained and st.session_state.model is not None:
                try:
                    # Evaluate model on test data
                    with st.spinner("Evaluating model performance..."):
                        if st.session_state.model_type == 'lstm':
                            X_train, y_train, X_test, y_test, _, _ = prepare_data_for_lstm(
                                stock_data, st.session_state.lookback_days
                            )
                        else:
                            X_train, y_train, X_test, y_test, _ = prepare_data_for_regression(
                                stock_data, st.session_state.feature_days
                            )

                        predictions, metrics = evaluate_model(
                            st.session_state.model, X_test, y_test,
                            st.session_state.scaler, st.session_state.model_type
                        )

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        rmse_value = metrics.get('RMSE', 0)
                        st.metric("RMSE", f"{rmse_value:.4f}")
                    with col2:
                        mae_value = metrics.get('MAE', 0)
                        st.metric("MAE", f"{mae_value:.4f}")
                    with col3:
                        r2_value = metrics.get('R2', 0)
                        st.metric("R² Score", f"{r2_value:.4f}")
                    with col4:
                        mse_value = metrics.get('MSE', 0)
                        st.metric("MSE", f"{mse_value:.4f}")

                    # Performance interpretation
                    if r2_value > 0.8:
                        st.success("🎯 **Excellent Model Performance** - High accuracy predictions")
                    elif r2_value > 0.6:
                        st.info("✅ **Good Model Performance** - Reliable predictions")
                    elif r2_value > 0.4:
                        st.warning("⚠️ **Moderate Model Performance** - Use predictions with caution")
                    else:
                        st.error("❌ **Poor Model Performance** - Consider using more data or different parameters")

                    # Plot actual vs predicted
                    if len(predictions) > 0 and len(stock_data) >= len(predictions):
                        fig_perf = go.Figure()

                        test_dates = stock_data['Date'].iloc[-len(predictions):].values
                        actual_prices = stock_data['Close'].iloc[-len(predictions):].values

                        fig_perf.add_trace(go.Scatter(
                            x=test_dates,
                            y=actual_prices,
                            mode='lines',
                            name='Actual',
                            line=dict(color='blue', width=2)
                        ))

                        fig_perf.add_trace(go.Scatter(
                            x=test_dates,
                            y=predictions,
                            mode='lines',
                            name='Predicted',
                            line=dict(color='red', width=2, dash='dash')
                        ))

                        fig_perf.update_layout(
                            title="Enhanced Model Performance: Actual vs Predicted",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig_perf, use_container_width=True)

                    # Backtest results
                    if st.button("🔍 Run Enhanced Backtest"):
                        with st.spinner("Running comprehensive backtest..."):
                            try:
                                backtest_results = backtest_predictions(
                                    st.session_state.model, stock_data,
                                    st.session_state.lookback_days if st.session_state.model_type == 'lstm' else st.session_state.feature_days,
                                    prediction_days, st.session_state.scaler, st.session_state.model_type
                                )

                                if len(backtest_results) > 0:
                                    st.subheader("📊 Backtest Results")
                                    st.dataframe(
                                        backtest_results.style.format({
                                            'mae': '{:.2f}',
                                            'mape': '{:.2f}%'
                                        }),
                                        use_container_width=True
                                    )

                                    # Calculate trend accuracy
                                    if 'predicted_trend' in backtest_results.columns and 'actual_trend' in backtest_results.columns:
                                        trend_accuracy = (backtest_results['predicted_trend'] ==
                                                          backtest_results['actual_trend']).mean() * 100
                                        st.success(f"🎯 **Trend Prediction Accuracy:** {trend_accuracy:.1f}%")

                                    # Average metrics
                                    avg_mae = backtest_results['mae'].mean()
                                    avg_mape = backtest_results['mape'].mean()
                                    st.info(f"📊 **Average MAE:** ${avg_mae:.2f} | **Average MAPE:** {avg_mape:.2f}%")
                                else:
                                    st.warning(
                                        "⚠️ Backtest completed but no results available. This may be due to insufficient data.")

                            except Exception as e:
                                st.error(f"❌ Error during backtesting: {str(e)}")

                except Exception as e:
                    st.error(f"❌ Error evaluating model: {str(e)}")
                    st.info("💡 Try retraining the model or using a different configuration.")

            else:
                st.info("🧠 Train a model first to see performance metrics")

        with tab5:
            st.subheader("💾 Export Enhanced Data")

            # Prepare export data
            export_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # Add technical indicators if available
            tech_cols = ['MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']
            for col in tech_cols:
                if col in stock_data.columns:
                    export_data[col] = stock_data[col]

            # Add predictions if model is trained
            if (st.session_state.model_trained and st.session_state.model is not None and
                    'future_predictions' in locals() and 'prediction_dates' in locals()):
                pred_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Predicted_Close': future_predictions,
                    'Type': 'Prediction'
                })
                export_data['Type'] = 'Historical'

                # Combine data
                export_data = pd.concat([export_data, pred_df], ignore_index=True)

            # Convert to CSV
            csv = export_data.to_csv(index=False)

            # Download button
            st.download_button(
                label="📥 Download Enhanced Data as CSV",
                data=csv,
                file_name=f"{ticker}_enhanced_stock_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download includes historical data, technical indicators, and predictions (if available)"
            )

            # Display preview
            st.subheader("👁️ Data Preview")
            st.dataframe(export_data.head(20), use_container_width=True)

            # Data statistics
            st.subheader("📈 Data Statistics")
            numeric_cols = export_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(export_data[numeric_cols].describe(), use_container_width=True)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

        # Enhanced error information
        st.info("🔧 **Troubleshooting Guide:**")
        st.info("• Verify the stock ticker symbol is correct and active")
        st.info("• Check if the selected date range has sufficient trading data")
        st.info("• Try enabling 'Auto-fix NaN Issues' in Advanced Settings")
        st.info("• Consider using a different model type (Gradient Boosting handles NaN natively)")
        st.info("• Expand the date range to include more historical data")

        # Technical details (expandable)
        with st.expander("🔍 Technical Details"):
            st.code(str(e))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>📈 Enhanced Stock Market Predictor</strong></p>
        <p>Built with ❤️ using Streamlit | Enhanced with comprehensive NaN handling | Stock data from Yahoo Finance</p>
        <p><em>⚠️ Disclaimer: This app is for educational purposes only. Enhanced predictions include uncertainty quantification. 
        Do not use for actual trading decisions without proper risk assessment.</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
