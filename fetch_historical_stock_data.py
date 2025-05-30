import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
import matplotlib.dates as mdates

def fallback_fetch_with_retries(ticker, start_date, end_date, retries=3, sleep_seconds=20):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=False)
            if not data.empty:
                return data
            time.sleep(sleep_seconds)
        except:
            time.sleep(sleep_seconds)
    return None

def fetch_stock_data(ticker, start_date=None, end_date=None, period_years=10, retries=3, sleep_seconds=10):
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("Ticker must be a non-empty string.")
    
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    start_date = start_date or (datetime.today() - timedelta(days=int(period_years * 365.25))).strftime("%Y-%m-%d")

    for _ in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not data.empty:
                return data
            time.sleep(sleep_seconds)
        except:
            time.sleep(sleep_seconds)

    # fallback method
    data = fallback_fetch_with_retries(ticker, start_date, end_date, retries=retries, sleep_seconds=sleep_seconds)
    return data

def calculate_moving_averages(data, short_window=50, long_window=200):
    data = data.copy()
    data["SMA_short"] = data["Close"].rolling(window=short_window, min_periods=1).mean()
    data["SMA_long"] = data["Close"].rolling(window=long_window, min_periods=1).mean()
    return data

def plot_stock_data(data, ticker, short_window, long_window):
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    ax.plot(data.index, data["Close"], label="Close Price", color="blue", linewidth=1)
    ax.plot(data.index, data["SMA_short"], label=f"Short SMA ({short_window})", color="red", linewidth=1.5)
    ax.plot(data.index, data["SMA_long"], label=f"Long SMA ({long_window})", color="green", linewidth=1.5)

    ax.set_title(f"{ticker} Stock Price with Moving Averages", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price (USD)", fontsize=12)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    ax.set_facecolor("white")
    plt.gcf().patch.set_facecolor("white")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    ax.legend(loc="upper left", fontsize=10)
    plt.show()

def load_cache(cache_file):
    try:
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index, errors='coerce')
            if data.index.isnull().any():
                raise ValueError("Index contains non-datetime values after parsing.")
        return data
    except Exception as e:
        print(f"Warning: Failed to load cache with parse_dates=True ({e}). Trying without parse_dates...")
        try:
            data = pd.read_csv(cache_file, index_col=0)
            data.index = pd.to_datetime(data.index, errors='coerce')
            if data.index.isnull().any():
                raise ValueError("Index contains non-datetime values after manual parsing.")
            return data
        except Exception as e2:
            print(f"Failed to parse dates in index after retry: {e2}")
            return data

def get_stock_data(ticker: str, period_years: int = 10, force_refresh_cache: bool = True) -> pd.DataFrame:
    """
    Fetch and cache historical stock data for use in preprocessing or modeling.
    This function is designed for import use (e.g. in normalize_stock_data.py).
    """
    cache_file = f"{ticker}.csv"

    if not force_refresh_cache and os.path.exists(cache_file):
        print(f"Loading cached data for {ticker}")
        return load_cache(cache_file)
    
    print(f"Fetching fresh data for {ticker}...")
    data = fetch_stock_data(ticker, period_years=period_years)
    if data is not None and not data.empty:
        data.to_csv(cache_file, index=True)
        return data
    else:
        raise ValueError(f"Failed to fetch data for {ticker}")

def main():
    ticker = "AAPL"
    data = get_stock_data(ticker)
    data = calculate_moving_averages(data)
    plot_stock_data(data, ticker, 50, 200)

if __name__ == "__main__":
    main()
