from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from fetch_historical_stock_data import fetch_stock_data, calculate_moving_averages
import os

app = Flask(__name__, template_folder="templates")
app.secret_key = "your_secret_key_here"  # Replace with a secure key in production

# In-memory storage for registered users (replace with database in production)
USERS = {}  # Format: {"email": "password"}

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stock_price_model.h5")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

@app.route('/')
def home():
    if "logged_in" not in session or not session["logged_in"]:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("login.html")
    
    if request.method == 'POST':
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        if email in USERS and USERS[email] == password:
            session["logged_in"] = True
            return jsonify({"success": "Login successful"}), 200
        elif email not in USERS:
            return jsonify({"error": "User not found. Please sign up."}), 401
        else:
            return jsonify({"error": "Invalid password"}), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    
    if email in USERS:
        return jsonify({"error": "Email already registered"}), 400
    
    USERS[email] = password
    return jsonify({"success": "Signup successful"}), 201

@app.route('/predict', methods=['POST'])
def predict():
    if "logged_in" not in session or not session["logged_in"]:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    try:
        data = request.get_json()
        ticker = data.get("ticker")
        prices = data.get("prices")

        if ticker and prices:
            return jsonify({"error": "Provide either a ticker or prices, not both"}), 400

        if prices:
            if not isinstance(prices, list) or len(prices) != 60 or not all(isinstance(p, (int, float)) for p in prices):
                return jsonify({"error": "Prices must be a list of exactly 60 numbers"}), 400
            recent_prices = np.array(prices).reshape(-1, 1)
        elif ticker:
            if not isinstance(ticker, str) or not ticker:
                return jsonify({"error": "Valid ticker is required"}), 400
            
            stock_data = fetch_stock_data(ticker, period_years=1)
            if stock_data is None:
                return jsonify({"error": "Failed to fetch stock data"}), 500
            
            stock_data, _, _ = calculate_moving_averages(stock_data)
            
            if len(stock_data) < 60:
                return jsonify({"error": "Insufficient data (need at least 60 days)"}), 400
            
            recent_prices = stock_data["Close"][-60:].values.reshape(-1, 1)
        else:
            return jsonify({"error": "Must provide either a ticker or prices"}), 400

        scaled_data = scaler.transform(recent_prices)
        X = np.array([scaled_data])
        prediction = model.predict(X, verbose=0)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        response = {
            "recent_data": recent_prices.flatten().tolist(),
            "prediction": float(predicted_price)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
