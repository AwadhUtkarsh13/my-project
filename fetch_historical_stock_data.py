import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date=None, end_date=None, period_years=5):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start_date (str, optional): Start date in "YYYY-MM-DD" format.
        end_date (str, optional): End date in "YYYY-MM-DD" format.
        period_years (int): Number of years of data if start_date is not provided.
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns, or None if fetch fails.
    """
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("Ticker must be a non-empty string.")
    
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    start_date = start_date or (datetime.today() - timedelta(days=int(period_years * 365.25))).strftime("%Y-%m-%d")
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data found for ticker '{ticker}'.")
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required OHLCV columns for '{ticker}'.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_moving_averages(data, short_window=50, long_window=200):
    """
    Calculate short-term and long-term simple moving averages.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' column.
        short_window (int): Window size for short-term SMA.
        long_window (int): Window size for long-term SMA.
    
    Returns:
        pd.DataFrame: Data with added SMA columns.
    """
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain 'Close' column.")
    
    data = data.copy()
    data["SMA_short"] = data["Close"].rolling(window=short_window, min_periods=1).mean()
    data["SMA_long"] = data["Close"].rolling(window=long_window, min_periods=1).mean()
    return data, short_window, long_window  # Return window sizes for use in plotting

def plot_stock_data(data, ticker, short_window, long_window, plot_moving_averages=True):
    """
    Plot stock closing price and optionally moving averages.
    
    Args:
        data (pd.DataFrame): Stock data with 'Close' and optional SMA columns.
        ticker (str): Stock ticker symbol.
        short_window (int): Short-term SMA window size.
        long_window (int): Long-term SMA window size.
        plot_moving_averages (bool): Whether to include SMA in the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data["Close"], label="Close Price", color="blue")
    
    if plot_moving_averages and "SMA_short" in data.columns and "SMA_long" in data.columns:
        plt.plot(data["SMA_short"], label=f"Short SMA ({short_window})", color="red")
        plt.plot(data["SMA_long"], label=f"Long SMA ({long_window})", color="green")
    
    plt.title(f"{ticker} Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    stock_ticker = "AAPL"
    stock_data = fetch_stock_data(stock_ticker)
    if stock_data is None:
        return
    
    print("First few rows of data:\n", stock_data.head())
    print("\nMissing values in each column:\n", stock_data.isnull().sum())
    print("\nSummary statistics:\n", stock_data.describe())
    
    # Define window sizes here and pass them through
    short_window = 50
    long_window = 200
    stock_data, short_window, long_window = calculate_moving_averages(stock_data, short_window, long_window)
    plot_stock_data(stock_data, stock_ticker, short_window, long_window)

if __name__ == "__main__":
    main()