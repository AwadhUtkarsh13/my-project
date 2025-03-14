# File: train_stock_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import argparse
import numpy as np
from normalize_stock_data import normalize_and_prepare_data  # Import the function
from fetch_historical_stock_data import fetch_stock_data, calculate_moving_averages, plot_stock_data

def build_lstm_model(seq_length, units=50, dropout_rate=0.2):
    """
    Build and compile an LSTM model for stock price prediction.
    
    Args:
        seq_length (int): Length of input sequences.
        units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate for regularization.
    
    Returns:
        Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the LSTM model and evaluate its performance.
    
    Args:
        model (Sequential): Compiled LSTM model.
        X_train (np.array): Training input sequences.
        y_train (np.array): Training target values.
        X_test (np.array): Testing input sequences.
        y_test (np.array): Testing target values.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        Sequential: Trained model.
        dict: Training history.
    """
    # Validate input shapes
    if X_train.shape[1] != X_test.shape[1] or X_train.shape[2] != 1 or y_train.shape[1] != 1:
        raise ValueError("Invalid input shapes for LSTM model.")
    
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(X_test, y_test), verbose=1)
    
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")
    
    return model, history

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Train an LSTM model on stock data.")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of data for training (0-1)')
    parser.add_argument('--seq_length', type=int, default=60, help='Sequence length for prediction')
    parser.add_argument('--short_window', type=int, default=50, help='Short-term SMA window')
    parser.add_argument('--long_window', type=int, default=200, help='Long-term SMA window')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lstm_units', type=int, default=50, help='Number of LSTM units')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
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
    
    # Build and train the model
    try:
        model = build_lstm_model(args.seq_length, units=args.lstm_units, dropout_rate=args.dropout_rate)
        model.summary()
        
        model, history = train_model(model, X_train, y_train, X_test, y_test, 
                                   epochs=args.epochs, batch_size=args.batch_size)
        
        # Make predictions
        predictions = model.predict(X_test)
        print(f"Sample Predictions (first 5): {predictions[:5].flatten()}")
        
        # Save the model
        model.save("stock_price_model.h5")
        print("Model saved successfully as 'stock_price_model.h5'")
        
        # Optional: Plot the original data
        plot_stock_data(stock_data, args.ticker, short_window, long_window)
        
    except Exception as e:
        print(f"Error in model training or saving: {e}")
        return

if __name__ == "__main__":
    main()