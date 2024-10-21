import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to train and evaluate the ML model
def train_model(df_scaled, scaler):
    # The last row is today's data, so we exclude it from the training set
    X = df_scaled[:-1, 1:]  # Use all features except the Close price
    y = df_scaled[:-1, 0]   # Target is the Close price
    
    # Today's data for prediction (latest row)
    X_today = df_scaled[-1, 1:].reshape(1, -1)  # Today's features
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    
    # Predict today's close
    predicted_close_today_scaled = model.predict(X_today)[0]

    # Inverse transform the predicted close price to its original scale
    predicted_close_today = scaler.inverse_transform([[predicted_close_today_scaled] + [0]*(X_today.shape[1])])[0][0]

    # Evaluate the model using Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    print(f"Predicted Close Price for Today: {predicted_close_today}")
    
    return predicted_close_today

if __name__ == "__main__":
    from preprocess import preprocess_data
    ticker = 'TCS.NS'
    df_scaled, df, scaler = preprocess_data(ticker)
    train_model(df_scaled, scaler)
