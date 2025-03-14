import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import argparse
from fetch_historical_stock_data import fetch_stock_data, calculate_moving_averages, plot_stock_data  # Added plot_stock_data

def normalize_and_prepare_data(data, column='Close', train_split=0.8, seq_length=60):
    """
    Normalize stock data and prepare sequences for training/testing.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame.
        column (str): Column to normalize (default: 'Close').
        train_split (float): Fraction of data for training (0-1).
        seq_length (int): Length of sequences for prediction.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    # Validate input
    if not isinstance(data, pd.DataFrame) or column not in data.columns:
        raise ValueError(f"Invalid data or missing '{column}' column.")
    if not 0 < train_split < 1:
        raise ValueError("train_split must be between 0 and 1.")
    if len(data) <= seq_length:
        raise ValueError(f"Data length ({len(data)}) must be greater than sequence length ({seq_length}).")
    
    # Extract and reshape the target column
    prices = data[column].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    
    # Split into training and testing sets
    train_size = int(len(scaled_data) * train_split)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    # Create sequences efficiently with vectorization
    def create_sequences(data, seq_length):
        if len(data) <= seq_length:
            raise ValueError("Not enough data to create sequences.")
        n = len(data) - seq_length
        indices = np.arange(n)
        X = np.stack([data[i:i+seq_length] for i in indices], axis=0)
        y = data[seq_length:seq_length+n]
        return X, y
    
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    return X_train, y_train, X_test, y_test, scaler

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Normalize stock data and prepare sequences.")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of data for training (0-1)')
    parser.add_argument('--seq_length', type=int, default=60, help='Sequence length for prediction')
    parser.add_argument('--short_window', type=int, default=50, help='Short-term SMA window')
    parser.add_argument('--long_window', type=int, default=200, help='Long-term SMA window')
    args = parser.parse_args()
    
    # Fetch stock data
    stock_data = fetch_stock_data(args.ticker)
    if stock_data is None:
        print("Failed to fetch stock data.")
        return
    
    # Calculate moving averages
    stock_data, short_window, long_window = calculate_moving_averages(
        stock_data, args.short_window, args.long_window
    )
    
    # Normalize and prepare data
    try:
        X_train, y_train, X_test, y_test, scaler = normalize_and_prepare_data(
            stock_data, 
            column='Close', 
            train_split=args.train_split, 
            seq_length=args.seq_length
        )
        
        # Print data overview
        print("First few rows of stock data:\n", stock_data.head())
        print(f"\nTraining Samples: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Testing Samples: {X_test.shape}, Labels: {y_test.shape}")
        
        # Save the scaler
        joblib.dump(scaler, "scaler.pkl")
        print("Scaler saved successfully as 'scaler.pkl'.")
        
        # Plot the data (now properly imported)
        plot_stock_data(stock_data, args.ticker, short_window, long_window)
        
    except ValueError as e:
        print(f"Error in normalization: {e}")
        return

if __name__ == "__main__":
    main()