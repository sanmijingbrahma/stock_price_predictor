import tkinter as tk
from tkinter import messagebox
from preprocess import preprocess_data
from model import train_model

# Function to predict the close price for today
def predict_close(ticker):
    try:
        # Fetch and preprocess data
        df_scaled, df, scaler = preprocess_data(ticker)
        predicted_close_today = train_model(df_scaled, scaler)
        return round(predicted_close_today, 2)
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
        return None
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        return None

# Function to handle button click
def on_predict():
    ticker = ticker_entry.get().strip().upper()  # Make it case-insensitive
    if not ticker:
        messagebox.showerror("Error", "Please enter a stock ticker.")
        return
    
    result = predict_close(ticker)
    if result is not None:
        result_label.config(text=f"Predicted Close Price for {ticker}: {result} ")
    else:
        result_label.config(text="")

# Create the main application window
root = tk.Tk()
root.title("Stock Price Predictor")
root.geometry("400x300")  # Set a fixed size for the window
root.configure(bg='#E2F1E7')  # Set a light background color

# Create a main frame
frame = tk.Frame(root, bg='#E2F1E7', padx=20, pady=20)
frame.pack(pady=20)

# Title
title_label = tk.Label(frame, text="Stock Price Predictor", font=('Helvetica', 16, 'bold'), bg='#E2F1E7', fg='#243642')
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

# Subtitle
subtitle_label = tk.Label(frame, text="Enter Stock Ticker:", font=('Helvetica', 12), bg='#E2F1E7', fg='#243642')
subtitle_label.grid(row=1, column=0, padx=5, pady=(0, 10))

# Input Field
ticker_entry = tk.Entry(frame, font=('Helvetica', 12), width=20)
ticker_entry.grid(row=1, column=1, padx=5, pady=(0, 10))

# Predict Button
predict_button = tk.Button(frame, text="Predict", command=on_predict, bg='#387478', fg='white', font=('Helvetica', 12, 'bold'))
predict_button.grid(row=2, column=0, columnspan=2, pady=20)

# Result Label
result_label = tk.Label(frame, text="", font=('Helvetica', 12), bg='#E2F1E7', fg='#243642', wraplength=300)
result_label.grid(row=3, column=0, columnspan=2)

# Run the application
root.mainloop()
