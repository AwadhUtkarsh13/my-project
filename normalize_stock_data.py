import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

def load_stock_data(file_path):
    try:
        df = pd.read_csv(file_path)

        # Standardize column names
        df.rename(columns=lambda x: x.strip().replace("Close/Last", "Close"), inplace=True)

        # Remove $ and commas, convert to float
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

        df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df.dropna(subset=["Date"], inplace=True)

        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        return df

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled

def save_normalized_data(df, output_path):
    df.to_csv(output_path)
    print(f"Normalized data saved to: {output_path}")

def plot_stock_data(df, title="Normalized Stock Data"):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Normalized Close", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    input_file = "AAPL.csv"
    output_file = "AAPL_normalized.csv"

    data = load_stock_data(input_file)
    if data is None or data.empty:
        print(f"Failed to load or process file: {input_file}")
        return

    normalized_data = normalize_data(data)
    save_normalized_data(normalized_data, output_file)

    # Plot the normalized close prices
    plot_stock_data(normalized_data)

if __name__ == "__main__":
    main()
