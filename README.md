# Quant Trading Dashboard

## Overview
The Quant Trading Dashboard is a user-friendly, interactive web application built with Python and Streamlit. It allows beginners and enthusiasts to test a Moving Average (MA) crossover trading strategy, with an optional machine learning (ML) prediction, using historical stock data. Inspired by educational YouTube videos on data storytelling and simulation, this Minimum Viable Product (MVP) addresses the real-world problem of inaccessible trading tools by providing a free, customizable platform to learn and experiment with stock trading strategies.

## Features
- Select from top stocks (e.g., MSFT, AAPL) and adjust MA windows (5-200 days).
- Toggle ML predictions using a Random Forest Regressor with a customizable threshold (0-5%).
- Visualize price trends, trading signals, portfolio performance, and ML predictions with interactive Plotly charts.
- Backtest strategies with metrics like Total Return, Annualized Return, Sharpe Ratio, and Max Drawdown.
- View raw data and debug information for transparency.

## Usage
### Installation
1. Clone the repository: git clone https://github.com/your-username/quant-trading-dashboard.git
cd quant-trading-dashboard; Or download it to your local VSCode.
2. Run the streamlit app via terminal: streamlit run app.py
3. Notes: Ensure these libraries are installed in your python: pip install ...
- streamlit
- yfinance
- pandas
- numpy
- scikit-learn
- plotly

### Getting started
- Select a Stock: Choose from the dropdown (e.g., MSFT for Microsoft).
- Set Parameters: Adjust Short MA (default 20 days), Long MA (default 50 days), start/end dates, and ML options via the sidebar.
- Run the Dashboard: The app automatically generates results, including charts and metrics.
### Exploring features
- Without ML: Test the basic MA crossover strategy and view backtest results (e.g., 77.22% Total Return for MSFT).
- With ML: Enable ML predictions, set a threshold (e.g., 1%), and review ML evaluation metrics (e.g., MAE, R²).
- Visualizations: Interact with charts to zoom or hover over data points.
- Raw Data: Check the "Show Raw Data" box to inspect underlying numbers.

## Project Inspiration:
This project was inspired by the following YouTube videos:
- Become a Data Storyteller with Streamlit! (https://www.youtube.com/watch?v=zi9KgTJjnjc)
- Creating Interactive, Animated Reports in Streamlit with Vizzu (https://www.youtube.com/watch?v=dVCvJYfR38k)
- Realtime Streamlit Dashboard (https://www.youtube.com/watch?v=YRA90MX4XiM)
- Simulating Real-Life Processes in Python (https://www.youtube.com/watch?v=8SLk_uRRcgc)
These tutorials guided the use of Streamlit for interactivity, Plotly for visualizations, and simulation for backtesting.
