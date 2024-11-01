import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(ticker):
    # Fetch data from yfinance
    df = yf.download(ticker, period="1y", interval="1d")
    
    # Check if data was fetched successfully
    if df.empty:
        raise ValueError(f"No data available for ticker {ticker}. Please check the ticker symbol or try again later.")

    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("The dataset does not contain a 'Close' column. Cannot perform prediction.")

    # Create the required features
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    
    # Drop any rows with missing values
    df = df.dropna()

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close', 'SMA_10', 'SMA_50', 'RSI', 'MACD']])

    return df_scaled, df, scaler

# Helper functions for indicators
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short_period=12, long_period=26):
    ema_short = series.ewm(span=short_period, adjust=False).mean()
    ema_long = series.ewm(span=long_period, adjust=False).mean()
    return ema_short - ema_long
