import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from fetch_historical_stock_data import fetch_stock_data, calculate_moving_averages
from normalize_stock_data import normalize_and_prepare_data

# Paths to your model and scaler
MODEL_PATH = "stock_price_model.h5"
SCALER_PATH = "scaler.pkl"

def evaluate_model(ticker="AAPL", seq_length=60, train_split=0.8):
    # Load model and scaler
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return

    # Fetch and prepare data
    stock_data = fetch_stock_data(ticker, period_years=5)  # Use 5 years of data
    if stock_data is None:
        print("Failed to fetch stock data.")
        return

    stock_data, _, _ = calculate_moving_averages(stock_data)  # Optional SMAs

    # Normalize and split data
    try:
        X_train, y_train, X_test, y_test, scaler = normalize_and_prepare_data(
            stock_data, train_split=train_split, seq_length=seq_length
        )
    except ValueError as e:
        print(f"Error preparing data: {e}")
        return

    # Make predictions
    predictions = model.predict(X_test, verbose=0)

    # Inverse transform to get actual prices
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test_actual, predictions_actual)
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    r2 = r2_score(y_test_actual, predictions_actual)

    # Calculate accuracy: percentage of predictions within 5% of actual values
    tolerance = 0.05  # 5% tolerance
    within_tolerance = np.abs(predictions_actual - y_test_actual) <= tolerance * y_test_actual
    accuracy = np.mean(within_tolerance) * 100  # Convert to percentage

    # Print results
    print(f"\nEvaluation Metrics for {ticker}:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Accuracy (within {tolerance*100}% tolerance): {accuracy:.2f}%")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label="Actual Prices", color="blue")
    plt.plot(predictions_actual, label="Predicted Prices", color="red")
    plt.title(f"{ticker} - Actual vs Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_model(ticker="AAPL")  # You can change the ticker here
