# File: train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from model_trainer import ModelTrainer
from sklearn.preprocessing import MinMaxScaler

# --- Step 1: Load raw CSV ---
df = pd.read_csv("AAPL.csv", parse_dates=["Date"])
df.columns = df.columns.str.strip()

# Clean 'Close' column (remove $ and spaces, convert to float)
if "Close/Last" in df.columns:
    df.rename(columns={"Close/Last": "Close"}, inplace=True)

df["Close"] = df["Close"].replace({'\$': ''}, regex=True).astype(str).str.strip()
df["Close"] = df["Close"].astype(float)

# Sort by date
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# --- Step 2: Normalize the data ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["Close"]])
df["Close"] = scaled_data

# Save scaler for future use
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# --- Step 3: Create sequences ---
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
close_values = df["Close"].values
X, y = create_sequences(close_values, seq_length)

# Reshape X to 3D: (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Step 4: Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Step 5: Train the model ---
trainer = ModelTrainer(model_dir="models")
history = trainer.train_model(X_train, y_train, X_val, y_val)

# Optional: Save training history
with open("models/training_summary.json", "w") as f:
    json.dump(history, f)

print("✅ Training completed. Model and scaler saved to 'models/'")
