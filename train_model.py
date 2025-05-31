import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

from model_trainer import ModelTrainer

# --- Step 1: Load normalized data ---
data = pd.read_csv("AAPL_normalized.csv", parse_dates=["Date"], index_col="Date")
if data.isnull().any().any():
    print("Warning: Missing values detected in normalized data. Filling with forward fill.")
    data.fillna(method='ffill', inplace=True)

# --- Step 2: Create sequences for LSTM ---
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length]["Close"])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data, seq_length=seq_length)

# --- Step 3: Train-validation split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False  # No shuffling for time series
)

# --- Step 4: Train the model ---
trainer = ModelTrainer(model_dir="models")
history = trainer.train_model(X_train, y_train, X_val, y_val)

# Optional: Save training history
with open("models/training_summary.json", "w") as f:
    json.dump(history, f)

print("Training completed. Model saved as: models/stock_price_model.h5")