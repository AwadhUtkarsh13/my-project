# File: visualize_predictions.py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import argparse
from normalize_stock_data import normalize_and_prepare_data
from fetch_historical_stock_data import fetch_stock_data, calculate_moving_averages

def load_and_predict(model_path, X_test):
    """
    Load the trained model and make predictions.
    
    Args:
        model_path (str): Path to the saved model file.
        X_test (np.array): Test input sequences.
    
    Returns:
        np.array: Predicted values.
    """
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        predictions = model.predict(X_test, verbose=0)
        return predictions.flatten()  # Ensure 1D array
    except Exception as e:
        raise ValueError(f"Error loading model or making predictions: {e}")

def plot_predictions(actual, predictions, ticker, scaler):
    """
    Plot actual vs. predicted prices.
    
    Args:
        actual (np.array): Actual test values (normalized).
        predictions (np.array): Predicted values (normalized).
        ticker (str): Stock ticker symbol.
        scaler (MinMaxScaler): Scaler for inverse transformation.
    """
    # Inverse transform to actual prices
    actual_prices = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    pred_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Prices", color='blue')
    plt.plot(pred_prices, label="Predicted Prices", color='red')
    plt.title(f"{ticker} - Actual vs Predicted Stock Prices")
    plt.xlabel("Time")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Visualize actual vs predicted stock prices.")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of data for training (0-1)')
    parser.add_argument('--seq_length', type=int, default=60, help='Sequence length for prediction')
    parser.add_argument('--short_window', type=int, default=50, help='Short-term SMA window')
    parser.add_argument('--long_window', type=int, default=200, help='Long-term SMA window')
    parser.add_argument('--model_path', type=str, default='stock_price_model.h5', help='Path to trained model')
    parser.add_argument('--scaler_path', type=str, default='scaler.pkl', help='Path to saved scaler')
    args = parser.parse_args()
    
    # Fetch and prepare data
    stock_data = fetch_stock_data(args.ticker)
    if stock_data is None:
        print("Failed to fetch stock data.")
        return
    
    stock_data, short_window, long_window = calculate_moving_averages(
        stock_data, args.short_window, args.long_window
    )
    
    # Normalize and prepare sequences
    try:
        X_train, y_train, X_test, y_test, scaler = normalize_and_prepare_data(
            stock_data, train_split=args.train_split, seq_length=args.seq_length
        )
    except ValueError as e:
        print(f"Error in data preparation: {e}")
        return
    
    # Load scaler (in case it differs from the one returned above)
    try:
        scaler = joblib.load(args.scaler_path)
        print("Scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return
    
    # Load model and make predictions
    try:
        predictions = load_and_predict(args.model_path, X_test)
    except ValueError as e:
        print(e)
        return
    
    # Plot results
    plot_predictions(y_test, predictions, args.ticker, scaler)

if __name__ == "__main__":
    main()