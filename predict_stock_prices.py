# File: predict_stock_prices.py
import numpy as np
import matplotlib.pyplot as plt
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

def generate_signals(predictions, actual):
    """
    Generate buy/sell signals based on predictions vs. actual values.
    
    Args:
        predictions (np.array): Predicted values.
        actual (np.array): Actual values.
    
    Returns:
        tuple: (buy_signals, sell_signals) as lists of indices.
    """
    buy_signals = np.where(predictions[1:] > actual[:-1])[0].tolist()
    sell_signals = np.where(predictions[1:] < actual[:-1])[0].tolist()
    return buy_signals, sell_signals

def plot_predictions(actual, predictions, buy_signals, sell_signals, ticker, scaler):
    """
    Plot actual vs. predicted prices with buy/sell signals.
    
    Args:
        actual (np.array): Actual test values (normalized).
        predictions (np.array): Predicted values (normalized).
        buy_signals (list): Indices of buy signals.
        sell_signals (list): Indices of sell signals.
        ticker (str): Stock ticker symbol.
        scaler (MinMaxScaler): Scaler for inverse transformation.
    """
    # Inverse transform to actual prices
    actual_prices = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    pred_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    # Plot actual vs. predicted
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
    
    # Plot with buy/sell signals
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Prices", color='blue')
    plt.scatter(buy_signals, actual_prices[buy_signals], marker='^', color='green', label="Buy Signal", alpha=1)
    plt.scatter(sell_signals, actual_prices[sell_signals], marker='v', color='red', label="Sell Signal", alpha=1)
    plt.title(f"{ticker} - Trading Strategy Based on Predictions")
    plt.xlabel("Time")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Predict stock prices and generate trading signals.")
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
    
    # Generate trading signals
    buy_signals, sell_signals = generate_signals(predictions, y_test.flatten())
    
    # Plot results
    plot_predictions(y_test, predictions, buy_signals, sell_signals, args.ticker, scaler)

if __name__ == "__main__":
    main()