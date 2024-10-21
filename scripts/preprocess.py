import pandas as pd
import os
import yfinance as yf
import talib as ta
from sklearn.preprocessing import MinMaxScaler

# Function to fetch and preprocess the latest data
def preprocess_data(ticker):
    # Fetch historical data for the stock
    df = yf.download(ticker, period="6mo", interval="1d")
    
    # Drop rows with missing values (if any)
    df.dropna(inplace=True)

    # Add technical indicators using TA-lib
    df['EMA_10'] = ta.EMA(df['Close'], timeperiod=10)
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['STOCH_K'], df['STOCH_D'] = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)

    # Remove NaN values caused by indicators
    df.dropna(inplace=True)

    # Scale the data for ML models
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close', 'EMA_10', 'RSI_14', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal', 'STOCH_K', 'STOCH_D']])

    # Return the scaled dataframe, raw dataframe, and scaler
    return df_scaled, df, scaler

if __name__ == "__main__":
    ticker = 'TCS.NS'  # Example stock ticker
    preprocess_data(ticker)
