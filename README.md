# 📊 Hybrid NLP Stock Prediction

A machine learning-based stock prediction application that combines **financial market data** with **Natural Language Processing (NLP)** sentiment analysis to estimate stock price movements.

The project is available in two versions:

- 🖥️ **Tkinter Desktop Application**
- 🌐 **Streamlit Web Application**

The system uses **Yahoo Finance** data, **VADER Sentiment Analysis**, **BERT Transformer**, and a **Random Forest Regressor** to predict stock closing prices.

---

# Features

- Download live stock market data using Yahoo Finance
- Hybrid sentiment analysis using:
  - VADER Sentiment Analyzer
  - BERT Transformer model
- Random Forest Regression for stock prediction
- RMSE evaluation metric
- Interactive prediction graph
- Desktop GUI using Tkinter
- Modern Web UI using Streamlit
- Adjustable model complexity (Streamlit version)

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Hugging Face Transformers
- Yahoo Finance API (yfinance)
- Matplotlib
- Tkinter
- Streamlit

---

# Project Structure

```
Hybrid-NLP-Stock-Prediction/
│
├── tkinter_app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── assets/
    └── screenshots/
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/Hybrid-NLP-Stock-Prediction.git

cd Hybrid-NLP-Stock-Prediction
```

---

## 2. Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas numpy matplotlib yfinance nltk transformers scikit-learn streamlit torch
```

---

# Running the Project

## Desktop Version (Tkinter)

```bash
python tkinter_app.py
```

---

## Web Version (Streamlit)

```bash
streamlit run streamlit_app.py
```

---

# How It Works

### Step 1

User enters a stock ticker (Example: AAPL, TSLA, MSFT).

↓

### Step 2

Historical stock data is downloaded from Yahoo Finance.

↓

### Step 3

News headlines are processed using

- VADER
- BERT

to generate a hybrid sentiment score.

↓

### Step 4

Feature engineering creates

- Previous Closing Price
- Daily Returns
- Sentiment Score

↓

### Step 5

Random Forest Regressor is trained on historical data.

↓

### Step 6

Predicted prices are generated and compared with actual prices.

↓

### Step 7

Results are displayed along with

- RMSE
- Prediction labels
- Price comparison graph

---

# Machine Learning Model

**Algorithm**

- Random Forest Regressor

**Input Features**

- Previous Close Price
- Daily Return
- Hybrid Sentiment Score

**Target**

- Closing Price

---

# Sentiment Analysis

The project combines two NLP models:

### VADER

- Lexicon-based sentiment analysis
- Fast and lightweight

### BERT

- Transformer-based deep learning model
- Context-aware sentiment prediction

### Hybrid Score

```
Hybrid Sentiment =
(VADER Score + BERT Score) / 2
```

---

# Evaluation Metric

Root Mean Squared Error (RMSE)

Lower RMSE indicates better prediction performance.

---

# Example Stock Tickers

- AAPL
- MSFT
- TSLA
- NVDA
- GOOGL
- AMZN
- META
- NFLX

---

# Future Improvements

- Real-time news integration
- Twitter/X sentiment analysis
- LSTM and GRU models
- XGBoost regression
- Prophet forecasting
- Technical indicators (RSI, MACD, Bollinger Bands)
- Candlestick charts
- Model comparison dashboard
- Portfolio recommendation system
- Stock trend classification

---

# Screenshots

## Desktop Application

(Add screenshot here)

---

## Streamlit Dashboard

(Add screenshot here)

---

# Requirements

```
pandas
numpy
matplotlib
scikit-learn
nltk
transformers
torch
yfinance
streamlit
```

---

# Notes

- Internet connection is required to download stock market data.
- The first run may take a few minutes because the BERT model is downloaded from Hugging Face.
- The application currently uses sample news headlines for sentiment analysis. Integrating real-time financial news APIs can improve prediction quality.

---

# Disclaimer

This project is intended for educational and research purposes only. Stock market predictions are inherently uncertain and should not be considered financial advice. Always conduct your own research before making investment decisions.

---

# Author

**Your Name**

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

## If you found this project helpful, consider giving it a ⭐ on GitHub!
