import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import sys

# ------------------------
# Data Processing Functions
# ------------------------

def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)

        # Rename column
        df.rename(columns=lambda x: x.strip().replace("Close/Last", "Close"), inplace=True)

        # Ensure required columns
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

        df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df.dropna(subset=["Date"], inplace=True)

        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        return df

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def calculate_moving_averages(df, short_window=50, long_window=200):
    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(window=short_window, min_periods=1).mean()
    df["SMA_long"] = df["Close"].rolling(window=long_window, min_periods=1).mean()
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    df_norm = df.copy()
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])
    return df_norm

# ------------------------
# Plotting Functions
# ------------------------

def plot_stock_data(df, title, short_window=None, long_window=None, normalized=False):
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    ax.plot(df.index, df["Close"], label="Close" if not normalized else "Normalized Close", color="blue", linewidth=1.2)

    if not normalized and "SMA_short" in df and "SMA_long" in df:
        ax.plot(df.index, df["SMA_short"], label=f"SMA {short_window}", color="red", linewidth=1)
        ax.plot(df.index, df["SMA_long"], label=f"SMA {long_window}", color="green", linewidth=1)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price" if normalized else "Price (USD)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ------------------------
# Main Script
# ------------------------

def main():
    input_file = "AAPL.csv"
    output_file = "AAPL_normalized.csv"
    ticker = "AAPL"
    short_window = 50
    long_window = 200

    df = load_and_clean_data(input_file)
    if df is None or df.empty:
        sys.exit("Failed to load data.")

    df_with_sma = calculate_moving_averages(df, short_window, long_window)
    df_normalized = normalize_data(df)

    # Show info
    print("\nData preview:")
    print(df_with_sma.head())

    print("\nMissing values:")
    print(df_with_sma.isnull().sum())

    print("\nSummary statistics:")
    print(df_with_sma.describe())

    # Plot original data with SMA
    plot_stock_data(df_with_sma, f"{ticker} Price + Moving Averages", short_window, long_window)

    # Plot normalized close price
    plot_stock_data(df_normalized, f"{ticker} Normalized Close Price", normalized=True)

    # Save normalized data
    df_normalized.to_csv(output_file)
    print(f"\nNormalized data saved to: {output_file}")

if __name__ == "__main__":
    main()
