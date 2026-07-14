import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import nltk
import matplotlib.pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Download VADER
nltk.download('vader_lexicon')

# NLP Models
vader = SentimentIntensityAnalyzer()
bert_model = pipeline("sentiment-analysis")


def hybrid_sentiment(text):
    vader_score = vader.polarity_scores(text)['compound']
    bert_result = bert_model(text)[0]
    bert_score = 1 if bert_result['label'] == 'POSITIVE' else -1
    return (vader_score + bert_score) / 2

def run_analysis():
    ticker = ticker_entry.get().upper()

    if ticker == "":
        messagebox.showerror("Error", "Please enter a stock ticker!")
        return

    try:
        loading_label.config(text="⏳ Processing...")
        app.update()

        # Stock data
        df_stock = yf.download(ticker, period="6mo")

        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)

        df_stock.reset_index(inplace=True)

        # Sample news
        news_data = [
            "Company reports strong earnings growth",
            "Market crashes due to global fears",
            "Stock shows stable performance",
            "Investors optimistic about future growth",
            "Economic slowdown concerns investors"
        ]

        sentiments = [hybrid_sentiment(text) for text in news_data]
        avg_sentiment = np.mean(sentiments)

        # Add features
        df_stock['sentiment'] = avg_sentiment
        df_stock['prev_close'] = df_stock['Close'].shift(1)
        df_stock['returns'] = df_stock['Close'].pct_change()

        df_stock['prev_close'] = df_stock['prev_close'].bfill()
        df_stock['returns'].fillna(0, inplace=True)

        df = df_stock.copy()

        if len(df) < 10:
            messagebox.showerror("Error", "Not enough data!")
            return

        # Model
        X = df[['prev_close', 'sentiment', 'returns']]
        y = df['Close']

        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Output
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"RMSE: {rmse:.2f}\n\n")

        # Colored output
        for i in range(min(5, len(predictions))):
            sentiment = X_test.iloc[i]['sentiment']

            if sentiment > 0:
                msg = "📈 Positive → Price may increase"
                color = "green"
            elif sentiment < 0:
                msg = "📉 Negative → Price may decrease"
                color = "red"
            else:
                msg = "➖ Neutral → Stable"
                color = "yellow"

            result_text.insert(tk.END,
                               f"{msg} | Predicted: {predictions[i]:.2f}\n",
                               color)

        result_text.tag_config("green", foreground="#00ff00")
        result_text.tag_config("red", foreground="#ff4d4d")
        result_text.tag_config("yellow", foreground="#ffff66")

        loading_label.config(text="✅ Done!")

        # Plot
        plt.figure()
        plt.plot(y_test.values, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.legend()
        plt.title(f"{ticker} Stock Prediction")
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))



app = tk.Tk()
app.title("Hybrid NLP Stock Prediction")
app.geometry("550x450")
app.configure(bg="#9a1b12")

# Title
tk.Label(app, text="Stock Sentiment + Prediction",
         font=("Arial", 16, "bold"),
         fg="white", bg="#1e1e2f").pack(pady=10)

# Input
tk.Label(app, text="Enter Stock Ticker:",
         fg="white", bg="#1e1e2f").pack()

ticker_entry = tk.Entry(app, font=("Arial", 12))
ticker_entry.pack(pady=5)

# Loading animation label
loading_label = tk.Label(app, text="",
                         fg="yellow", bg="#1e1e2f")
loading_label.pack()

# Button
tk.Button(app, text="Run Analysis",
          command=run_analysis,
          bg="#03D631", fg="white",
          font=("Arial", 12, "bold")).pack(pady=10)

# Output box
result_text = tk.Text(app, height=15, width=65,
                      bg="#282a36", fg="#00ffcc",
                      font=("Consolas", 10))
result_text.pack(pady=10)

app.mainloop()
