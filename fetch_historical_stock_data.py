import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def get_stock_data(file_path):
    try:
        data = pd.read_csv(file_path)

        # Rename columns to standard names if needed
        data.rename(columns=lambda x: x.strip().replace("Close/Last", "Close"), inplace=True)

        # Strip $ and commas, convert to float
        for col in ["Open", "High", "Low", "Close"]:
            data[col] = data[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

        # Convert Volume to numeric
        data["Volume"] = pd.to_numeric(data["Volume"], errors='coerce')

        # Convert date, handling mixed formats
        data["Date"] = pd.to_datetime(data["Date"], errors='coerce', dayfirst=False)
        data.dropna(subset=["Date"], inplace=True)

        # Set index and sort
        data.set_index("Date", inplace=True)
        data.sort_index(inplace=True)

        return data

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def calculate_moving_averages(data, short_window=50, long_window=200):
    data = data.copy()
    data["SMA_short"] = data["Close"].rolling(window=short_window, min_periods=1).mean()
    data["SMA_long"] = data["Close"].rolling(window=long_window, min_periods=1).mean()
    return data

def plot_stock_data(data, ticker, short_window, long_window):
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    ax.plot(data.index, data["Close"], label="Close Price", color="blue", linewidth=1.2)
    ax.plot(data.index, data["SMA_short"], label=f"SMA {short_window}", color="red", linewidth=1)
    ax.plot(data.index, data["SMA_long"], label=f"SMA {long_window}", color="green", linewidth=1)

    ax.set_title(f"{ticker} Stock Price with {short_window}- and {long_window}-Day Moving Averages", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    file_path = "AAPL.csv"
    ticker = "AAPL"
    short_window = 50
    long_window = 200

    data = get_stock_data(file_path)
    if data is None or data.empty:
        print("Failed to load stock data.")
        return

    data = calculate_moving_averages(data, short_window, long_window)

    print("\nData preview:")
    print(data.head())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nSummary statistics:")
    print(data.describe())

    plot_stock_data(data, ticker, short_window, long_window)

if __name__ == "__main__":
    main()
