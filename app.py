"""
Streamlit application for stock market prediction.
Main interface for users to interact with the prediction system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# Import custom modules
from stock_utils import fetch_stock_data, prepare_data_for_lstm, prepare_data_for_regression
from model import StockLSTM, StockRegressor
from predict import (
    predict_future_prices, create_prediction_dates,
    evaluate_model, create_prediction_dataframe,
    generate_trading_signals, backtest_predictions
)

# Page configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        overflow-wrap: break-word;
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

# Title and description
st.title("📈 Stock Market Prediction App")
st.markdown("### Predict future stock prices using Machine Learning")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Stock selection
    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter a valid stock ticker (e.g., AAPL, GOOGL, MSFT)"
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
    st.subheader("Model Configuration")
    model_type = st.selectbox(
        "Select Model Type",
        ["LSTM (Deep Learning)", "Linear Regression", "Random Forest"],
        help="LSTM is recommended for time series prediction"
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
    with st.expander("Advanced Settings"):
        if "LSTM" in model_type:
            lookback_days = st.slider("Lookback Days", 30, 120, 60)
            epochs = st.slider("Training Epochs", 10, 100, 50)
            batch_size = st.slider("Batch Size", 16, 64, 32)
        else:
            feature_days = st.slider("Feature Days", 10, 60, 30)

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

        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
        with col2:
            change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
            change_pct = (change / stock_data['Close'].iloc[-2]) * 100
            st.metric("Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
        with col3:
            st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")
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
            st.subheader("Historical Stock Data")

            # Candlestick chart
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

            # Add moving averages
            fig.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['MA_20'],
                    name="MA 20",
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['MA_50'],
                    name="MA 50",
                    line=dict(color='red', width=1)
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
                title=f"{ticker} Stock Analysis",
                yaxis_title="Price ($)",
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show recent data
            st.subheader("Recent Data")
            st.dataframe(
                stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10),
                use_container_width=True
            )

        with tab2:
            st.subheader("Stock Price Predictions")

            if train_button:
                # Prepare data based on model type
                with st.spinner("Training model... This may take a few minutes."):
                    if "LSTM" in model_type:
                        # Prepare LSTM data
                        X_train, y_train, X_test, y_test, scaler, train_size = prepare_data_for_lstm(
                            stock_data, lookback_days
                        )

                        # Create and train LSTM model
                        model = StockLSTM(lookback_days=lookback_days)
                        history = model.train(
                            X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size
                        )

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
                        model_name = 'linear' if 'Linear' in model_type else 'random_forest'
                        model = StockRegressor(model_type=model_name)
                        model.train(X_train, y_train)

                        # Store in session state
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.model_type = 'regression'
                        st.session_state.feature_days = feature_days

                    st.session_state.model_trained = True
                    st.success("✅ Model trained successfully!")

            if st.session_state.model_trained:
                # Make predictions
                with st.spinner("Generating predictions..."):
                    if st.session_state.model_type == 'lstm':
                        # Get last sequence for prediction
                        last_sequence = stock_data['Close'].iloc[-st.session_state.lookback_days:].values
                        scaled_sequence = st.session_state.scaler.transform(
                            last_sequence.reshape(-1, 1)
                        ).flatten()
                        # Predict future prices
                        future_predictions = predict_future_prices(
                            st.session_state.model,
                            scaled_sequence,
                            st.session_state.scaler,
                            days=prediction_days,
                            model_type='lstm'
                        )
                    else:
                        # For regression models
                        last_features = stock_data[['Close', 'Volume', 'MA_10', 'RSI']].iloc[
                                        -st.session_state.feature_days:].values.flatten()
                        scaled_features = st.session_state.scaler.transform(last_features.reshape(1, -1))

                        future_predictions = predict_future_prices(
                            st.session_state.model,
                            scaled_features[0],
                            st.session_state.scaler,
                            days=prediction_days,
                            model_type='regression'
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
                        signal_color = "green" if signals['signal'] == 'BUY' else "red" if signals[
                                                                                               'signal'] == 'SELL' else "gray"
                        st.markdown(
                            f"### Trading Signal: <span style='color:{signal_color}'>{signals['signal']}</span>",
                            unsafe_allow_html=True)
                    with col2:
                        st.metric("Confidence", f"{signals['confidence']:.1f}%")
                    with col3:
                        st.metric("Expected Change", f"{signals['price_change_percent']:.2f}%")

                    # Create prediction visualization
                    fig = go.Figure()

                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=stock_data['Date'].tail(60),
                        y=stock_data['Close'].tail(60),
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
                        title=f"{ticker} Price Prediction - Next {prediction_days} Days",
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
                        'Change from Current': [(p - current_price) / current_price * 100 for p in future_predictions]
                    })

                    st.subheader("📅 Daily Predictions")
                    st.dataframe(
                        pred_df.style.format({
                            'Predicted Price': '${:.2f}',
                            'Change from Current': '{:.2f}%'
                        }),
                        use_container_width=True
                    )

        with tab3:
            st.subheader("Technical Analysis")

            # RSI Chart
            fig_rsi = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price with Bollinger Bands', 'RSI', 'MACD'),
                row_heights=[0.5, 0.25, 0.25]
            )

            # Price with Bollinger Bands
            fig_rsi.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['Close'],
                    name='Close Price',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )

            fig_rsi.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['BB_upper'],
                    name='Upper Band',
                    line=dict(color='gray', dash='dash')
                ),
                row=1, col=1
            )

            fig_rsi.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['BB_lower'],
                    name='Lower Band',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.2)'
                ),
                row=1, col=1
            )

            # RSI
            fig_rsi.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['RSI'],
                    name='RSI',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )

            # Add RSI levels
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # MACD
            fig_rsi.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )

            fig_rsi.add_trace(
                go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['MACD_signal'],
                    name='Signal',
                    line=dict(color='red')
                ),
                row=3, col=1
            )

            # Update layout
            fig_rsi.update_layout(
                title="Technical Indicators Analysis",
                height=800,
                showlegend=True
            )

            fig_rsi.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_rsi.update_yaxes(title_text="RSI", row=2, col=1)
            fig_rsi.update_yaxes(title_text="MACD", row=3, col=1)

            st.plotly_chart(fig_rsi, use_container_width=True)

            # Technical indicators summary
            st.subheader("📈 Technical Indicators Summary")

            latest_rsi = stock_data['RSI'].iloc[-1]
            latest_macd = stock_data['MACD'].iloc[-1]
            latest_signal = stock_data['MACD_signal'].iloc[-1]

            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_signal = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
                st.info(f"**RSI Signal:** {rsi_signal} ({latest_rsi:.2f})")
            with col2:
                macd_signal = "Bullish" if latest_macd > latest_signal else "Bearish"
                st.info(f"**MACD Signal:** {macd_signal}")
            with col3:
                bb_position = "Upper Band" if stock_data['Close'].iloc[-1] > stock_data['BB_upper'].iloc[
                    -1] else "Lower Band" if stock_data['Close'].iloc[-1] < stock_data['BB_lower'].iloc[
                    -1] else "Middle"
                st.info(f"**Bollinger Band:** {bb_position}")

        with tab4:
            st.subheader("Model Performance")

            if st.session_state.model_trained:
                # Evaluate model on test data
                if st.session_state.model_type == 'lstm':
                    X_train, y_train, X_test, y_test, _, _ = prepare_data_for_lstm(
                        stock_data, st.session_state.lookback_days
                    )
                else:
                    X_train, y_train, X_test, y_test, _ = prepare_data_for_regression(
                        stock_data, st.session_state.feature_days
                    )

                predictions, metrics = evaluate_model(
                    st.session_state.model,
                    X_test, y_test,
                    st.session_state.scaler,
                    st.session_state.model_type
                )

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                with col2:
                    st.metric("MAE", f"{metrics['MAE']:.4f}")
                with col3:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                with col4:
                    st.metric("MSE", f"{metrics['MSE']:.4f}")

                # Plot actual vs predicted
                fig_perf = go.Figure()

                test_dates = stock_data['Date'].iloc[-len(predictions):]
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
                    title="Model Performance: Actual vs Predicted",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_perf, use_container_width=True)

                # Backtest results
                if st.button("Run Backtest"):
                    with st.spinner("Running backtest..."):
                        backtest_results = backtest_predictions(
                            st.session_state.model,
                            stock_data,
                            st.session_state.lookback_days if st.session_state.model_type == 'lstm' else st.session_state.feature_days,
                            prediction_days,
                            st.session_state.scaler,
                            st.session_state.model_type
                        )

                        st.subheader("📊 Backtest Results")
                        st.dataframe(
                            backtest_results.style.format({
                                'mae': '{:.2f}',
                                'mape': '{:.2f}%'
                            }),
                            use_container_width=True
                        )

                        # Calculate accuracy
                        trend_accuracy = (backtest_results['predicted_trend'] == backtest_results[
                            'actual_trend']).mean() * 100
                        st.success(f"Trend Prediction Accuracy: {trend_accuracy:.1f}%")
            else:
                st.warning("Please train a model first to see performance metrics.")

        with tab5:
            st.subheader("Export Data")

            # Prepare export data
            export_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

            if st.session_state.model_trained:
                # Add predictions to export
                pred_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Predicted_Close': future_predictions
                })
                pred_df['Type'] = 'Prediction'
                export_data['Type'] = 'Historical'

                # Combine data
                export_data = pd.concat([export_data, pred_df], ignore_index=True)

            # Convert to CSV
            csv = export_data.to_csv(index=False)

            # Download button
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{ticker}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Display preview
            st.subheader("Data Preview")
            st.dataframe(export_data.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your ticker symbol and try again.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Stock data from Yahoo Finance</p>
        <p style='font-size: 0.8em; color: gray;'>
            Disclaimer: This app is for educational purposes only. 
            Do not use for actual trading decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
