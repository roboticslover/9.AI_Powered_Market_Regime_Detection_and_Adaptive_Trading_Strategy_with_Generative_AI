# AI-Powered Market Regime Detection and Adaptive Trading Strategy

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import yfinance as yf
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# Machine Learning and Deep Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def label_market_regime(row):
    """
    Label market regime based on moving average crossover
    Bull: Short-term MA > Long-term MA
    Bear: Short-term MA < Long-term MA
    Sideways: MAs are equal
    """
    if row['MA20'] > row['MA50']:
        return 'Bull'
    elif row['MA20'] < row['MA50']:
        return 'Bear'
    else:
        return 'Sideways'

def create_sequences(X, y, time_steps=5):
    """Create time series sequences for LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def get_gpt4_explanation(regime):
    """Generate market regime explanation using GPT-4"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Provide a sophisticated analysis of the current {regime} market regime in the stock market, including potential economic implications, investment strategy recommendations, and key risk factors."}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

class TradingStrategy:
    def __init__(self, regime):
        self.regime = regime

    def generate_signal(self):
        """Generate trading signal based on market regime"""
        if self.regime == 'Bull':
            return 'Buy'
        elif self.regime == 'Bear':
            return 'Sell'
        else:
            return 'Hold'

def backtest_strategy(data):
    """Backtest the trading strategy across historical data"""
    positions = []
    for i in range(len(data)):
        regime = data['Regime'].iloc[i]
        strategy = TradingStrategy(regime)
        positions.append(strategy.generate_signal())
    
    data['Position'] = positions
    position_map = {'Buy': 1, 'Sell': -1, 'Hold': 0}
    data['Position_mapped'] = data['Position'].map(position_map)

    # Calculate returns
    data['Market Return'] = data['Close'].pct_change()
    data['Strategy Return'] = data['Market Return'] * data['Position_mapped'].shift(1)

    # Calculate cumulative returns
    data['Cumulative Market Return'] = (1 + data['Market Return']).cumprod()
    data['Cumulative Strategy Return'] = (1 + data['Strategy Return'].fillna(0)).cumprod()

    return data

def main():
    # Streamlit app setup
    st.title("AI-Powered Market Regime Detection and Adaptive Trading Strategy")

    # Data Collection and Preprocessing
    ticker = "^GSPC"
    data = yf.download(ticker, start="2000-01-01", end="2023-10-01")

    # Preprocess data
    data.dropna(inplace=True)
    data.reset_index(inplace=True)

    # Display historical data
    st.subheader("Historical Data")
    st.dataframe(data.head())

    # Feature Engineering
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # RSI Calculation
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    data['RSI'] = 100 - (100 / (1 + rs))

    # Drop rows with NaN values after feature engineering
    data.dropna(inplace=True)

    # Display data with technical indicators
    st.subheader("Data with Technical Indicators")
    st.dataframe(data[['Date', 'Close', 'MA20', 'MA50', 'RSI']].head())

    # Apply market regime labeling
    data['Regime'] = data.apply(label_market_regime, axis=1)

    # Display data with market regime labels
    st.subheader("Data with Market Regime Labels")
    st.dataframe(data[['Date', 'Close', 'MA20', 'MA50', 'Regime']].head())

    # Machine Learning Model Preparation
    features = ['MA20', 'MA50', 'RSI']
    X = data[features]
    le = LabelEncoder()
    y = le.fit_transform(data['Regime'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # Evaluate Random Forest model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    # Display Random Forest model evaluation
    st.subheader("Machine Learning Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text(report)

    # Deep Learning Model (LSTM)
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare LSTM sequences
    time_steps = 5
    X_lstm, y_lstm = create_sequences(X_scaled, y, time_steps)

    # Split LSTM data
    split = int(0.8 * len(X_lstm))
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

    # Build and compile LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train LSTM model
    model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, 
              validation_data=(X_test_lstm, y_test_lstm), verbose=0)

    # Evaluate LSTM model
    loss, accuracy = model.evaluate(X_test_lstm, y_test_lstm)
    st.subheader("Deep Learning Model Evaluation")
    st.write(f"LSTM Model Accuracy: {accuracy:.2f}")

    # Get the latest market regime explanation using GPT-4
    latest_regime = data['Regime'].iloc[-1]
    explanation = get_gpt4_explanation(latest_regime)

    # Display GPT-4 explanation
    st.subheader("GPT-4 Market Regime Analysis")
    st.write(explanation or "Could not retrieve analysis.")

    # Generate trading signal
    strategy = TradingStrategy(latest_regime)
    signal = strategy.generate_signal()

    # Display trading signal
    st.subheader("Adaptive Trading Signal")
    st.write(f"Market Regime: {latest_regime}")
    st.write(f"Trading Signal: {signal}")

    # Perform backtesting
    backtested_data = backtest_strategy(data.copy())

    # Display performance
    st.subheader("Backtesting Results")
    st.line_chart(backtested_data[['Cumulative Market Return', 'Cumulative Strategy Return']])

# Run the main application
if __name__ == "__main__":
    main()