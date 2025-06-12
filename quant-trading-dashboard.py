import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Streamlit configuration
# Streamlit app title
st.title("Quant Trading Dashboard: Moving Average Crossover with ML Prediction")

# Define top 10 companies (based on market cap as of 2024)
top_10 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ']

# Sidebar inputs
st.sidebar.header("Strategy Parameters")
ticker = st.sidebar.selectbox("Select Stock", top_10)
short_window = st.sidebar.slider("Short MA Window (days)", 5, 50, 20)
long_window = st.sidebar.slider("Long MA Window (days)", 20, 200, 50)
use_ml = st.sidebar.checkbox("Use ML Price Prediction", value=False)
start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2024, 12, 31))
threshold_pct = st.sidebar.slider("ML Prediction Threshold (%)", 0.0, 5.0, 1.0) if use_ml else 1.0

# DATA UNDERSTANDING:
# Download data
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.error(f"No data retrieved for {ticker}. Try a different ticker or date range.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index().drop_duplicates(subset='Date').set_index('Date')
        date_range = pd.date_range(start=start, end=end, freq='B')
        data = data.reindex(date_range).ffill()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

data = load_data(ticker, start_date, end_date)
if data is None:
    st.stop()

# Debugging: Check data structure
st.write(f"Data columns: {list(data.columns)}")
st.write(f"Data index monotonic: {data.index.is_monotonic_increasing}")
st.write(f"Data index duplicates: {data.index.duplicated().any()}")

# DATA PREPARATION:
# Calculate moving averages
data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

# ML model for next-day price prediction (Raindom Forest model)
def train_ml_model(data):
    data = data.copy()
    
    # Add features: RSI, ATR, Returns, Lagged_Returns, Volume_Change
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    data['RSI'] = calculate_rsi(data['Close'])
    data['ATR'] = calculate_atr(data)
    
    
    data['Returns'] = data['Close'].pct_change()
    data['Lagged_Returns'] = data['Returns'].shift(1)
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Create target
    data['Target'] = data['Close'].shift(-1)
    
    # Features list
    features = ['Lagged_Returns', 'Volume_Change', 'Short_MA', 'Long_MA', 'RSI', 'ATR']
    
    # Rolling window: Use last 2 years of data
    two_years_ago = data.index[-1] - pd.Timedelta(days=730)
    data = data.loc[data.index >= two_years_ago]
    
    # Drop NaN values and align
    model_data = data[features + ['Target']].dropna()
    valid_indices = model_data.index
    X = model_data[features]
    y = model_data['Target']
    
    # Debugging
    st.write(f"X length: {len(X)}, y length: {len(y)}, valid_indices length: {len(valid_indices)}")
    st.write(f"X index matches y index: {(X.index == y.index).all()}")
    
    # Train-test split (80-20)
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    test_indices = X_test.index
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # MODELLING:
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred_test = model.predict(X_test_scaled)
        # Predict on all data
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # EVALUATION:
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    metrics = {'MAE': mae, 'MSE': mse, 'R2': r2}
    
    # Debugging
    st.write(f"Predictions length: {len(predictions)}")
    
    # Create prediction series
    prediction_series = pd.Series(np.nan, index=data.index)
    prediction_series.loc[valid_indices] = predictions
    data['ML_Prediction'] = prediction_series
    
    return data, model, scaler, metrics, y_test, y_pred_test, test_indices

# DEPLOYMENT:
# Generate trading signals
def generate_signals(data, use_ml, threshold_pct):
    data = data.copy()
    data['Signal'] = 0
    data['Position'] = 0
    
    if use_ml:
        data, _, _, metrics, y_test, y_pred_test, test_indices = train_ml_model(data)
        # Shift predictions to use previous day's prediction
        data['ML_Prediction'] = data['ML_Prediction'].shift(1).fillna(method='ffill')
        valid_mask = (~data['Short_MA'].isna()) & (~data['Long_MA'].isna()) & (~data['ML_Prediction'].isna())
        # Apply threshold: Buy if predicted price > current price * (1 + threshold)
        threshold = 1 + threshold_pct / 100
        data.loc[valid_mask, 'Signal'] = np.where(
            (data.loc[valid_mask, 'Short_MA'] > data.loc[valid_mask, 'Long_MA']) & 
            (data.loc[valid_mask, 'ML_Prediction'] > data.loc[valid_mask, 'Close'] * threshold), 1, 0)
        data.loc[valid_mask, 'Signal'] = np.where(
            (data.loc[valid_mask, 'Short_MA'] < data.loc[valid_mask, 'Long_MA']) | 
            (data.loc[valid_mask, 'ML_Prediction'] < data.loc[valid_mask, 'Close'] / threshold), -1, 
            data.loc[valid_mask, 'Signal'])
    else:
        valid_mask = (~data['Short_MA'].isna()) & (~data['Long_MA'].isna())
        data.loc[valid_mask, 'Signal'] = np.where(
            data.loc[valid_mask, 'Short_MA'] > data.loc[valid_mask, 'Long_MA'], 1, 0)
        data.loc[valid_mask, 'Signal'] = np.where(
            data.loc[valid_mask, 'Short_MA'] < data.loc[valid_mask, 'Long_MA'], -1, 
            data.loc[valid_mask, 'Signal'])
        metrics, y_test, y_pred_test, test_indices = None, None, None, None
    
    data['Position'] = data['Signal'].replace(-1, 0).ffill()
    
    return data, metrics, y_test, y_pred_test, test_indices

# Backtest strategy
def backtest(data):
    data = data.copy()
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()
    data['Cumulative_Market'] = (1 + data['Returns']).cumprod()
    
    total_return = data['Cumulative_Strategy'].iloc[-1] - 1
    annualized_return = ((1 + total_return) ** (252 / len(data))) - 1
    sharpe_ratio = (data['Strategy_Returns'].mean() * 252) / (data['Strategy_Returns'].std() * np.sqrt(252)) if data['Strategy_Returns'].std() != 0 else np.nan
    max_drawdown = (data['Cumulative_Strategy'] / data['Cumulative_Strategy'].cummax() - 1).min()
    
    return data, {
        'Total Return': total_return * 100,
        'Annualized Return': annualized_return * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown * 100
    }

# Generate signals and backtest
data, ml_metrics, y_test, y_pred_test, test_indices = generate_signals(data, use_ml, threshold_pct)
data, backtest_metrics = backtest(data)

# Display backtest metrics
st.header("Backtest Results")
st.write(f"**Total Return**: {backtest_metrics['Total Return']:.2f}%")
st.write(f"**Annualized Return**: {backtest_metrics['Annualized Return']:.2f}%")
st.write(f"**Sharpe Ratio**: {backtest_metrics['Sharpe Ratio']:.2f}")
st.write(f"**Max Drawdown**: {backtest_metrics['Max Drawdown']:.2f}%")

# Display ML model evaluation metrics
if use_ml and ml_metrics is not None:
    st.header("ML Model Evaluation")
    st.write(f"**Mean Absolute Error (MAE)**: {ml_metrics['MAE']:.2f}")
    st.write(f"**Mean Squared Error (MSE)**: {ml_metrics['MSE']:.2f}")
    st.write(f"**R² Score**: {ml_metrics['R2']:.2f}")

# Plotting with Plotly
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                    subplot_titles=("Price and Moving Averages", "Trading Signals", "Portfolio Performance", "ML Predictions vs Actual"),
                    vertical_spacing=0.1)

# Plot 1: Price and Moving Averages
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['Short_MA'], name=f'Short MA ({short_window})', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['Long_MA'], name=f'Long MA ({long_window})', line=dict(color='green')), row=1, col=1)
if use_ml:
    fig.add_trace(go.Scatter(x=data.index, y=data['ML_Prediction'], name='ML Prediction', line=dict(color='purple', dash='dash')), row=1, col=1)

# Plot 2: Trading Signals
buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', 
                         marker=dict(symbol='triangle-up', size=10, color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', 
                         marker=dict(symbol='triangle-down', size=10, color='red')), row=2, col=1)

# Plot 3: Portfolio Performance
fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative_Strategy'], name='Strategy', line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative_Market'], name='Buy & Hold', line=dict(color='gray')), row=3, col=1)

# Plot 4: ML Predictions vs Actual
if use_ml and y_test is not None:
    fig.add_trace(go.Scatter(x=test_indices, y=y_test, name='Actual Price', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=test_indices, y=y_pred_test, name='Predicted Price', line=dict(color='purple', dash='dash')), row=4, col=1)

# Update layout
fig.update_layout(height=1000, showlegend=True, title_text=f"{ticker} Trading Strategy Performance")
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Price", row=2, col=1)
fig.update_yaxes(title_text="Cumulative Return", row=3, col=1)
if use_ml:
    fig.update_yaxes(title_text="Price", row=4, col=1)

# Display plot
st.plotly_chart(fig, use_container_width=True)

# Display raw data
if st.checkbox("Show Raw Data"):
    st.write(data[['Close', 'Short_MA', 'Long_MA', 'Signal', 'Position', 'Cumulative_Strategy']].tail())