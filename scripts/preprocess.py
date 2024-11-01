import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import talib

def preprocess_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d")
    
    if df.empty:
        raise ValueError(f"No data available for ticker {ticker}. Please check the ticker symbol or try again later.")
    if 'Close' not in df.columns:
        raise ValueError("The dataset does not contain a 'Close' column. Cannot perform prediction.")

    # Adding more indicators
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'])
    df['MACD'], _, _ = talib.MACD(df['Close'])
    df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(df['Close'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    
    # Drop rows with NaN values from indicators
    df = df.dropna()

    # Scale features
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower', 'ATR', 'STOCH_K', 'STOCH_D']])

    return df_scaled, df, scaler
