import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import nltk
import matplotlib.pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Stock AI Predictor", layout="wide")

# =========================
# CUSTOM UI STYLE
# =========================
st.markdown("""
<style>
.main { background-color: #0b0e14; color: #e2e8f0; }
h1 { color: #00d2ff; text-align: center; font-weight: 800; }

.stButton>button {
background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
color: white;
border: none;
font-weight: bold;
width: 100%;
height: 3rem;
margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD NLP MODELS
# =========================
nltk.download('vader_lexicon')

@st.cache_resource
def load_models():
    vader = SentimentIntensityAnalyzer()
    bert = pipeline("sentiment-analysis")
    return vader, bert

vader, bert_model = load_models()

# =========================
# SENTIMENT FUNCTION
# =========================
def hybrid_sentiment(text):
    vader_score = vader.polarity_scores(text)['compound']
    bert_result = bert_model(text)[0]
    bert_score = 1 if bert_result['label'] == 'POSITIVE' else -1
    return (vader_score + bert_score) / 2

# =========================
# TITLE
# =========================
st.title("📊 Hybrid NLP Stock Prediction")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Settings")

period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"])
n_estimators = st.sidebar.slider("Model Complexity", 50, 200, 100)

# =========================
# INPUT
# =========================
ticker = st.text_input("Enter Stock Ticker (Example: AAPL)")

run_btn = st.button("🚀 Run Analysis")

# =========================
# MAIN LOGIC
# =========================
if run_btn:

    if ticker == "":
        st.error("Please enter a stock ticker!")
    else:
        with st.spinner("Processing..."):

            try:
                # =========================
                # STOCK DATA
                # =========================
                df = yf.download(ticker, period=period)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df.reset_index(inplace=True)

                # =========================
                # SAMPLE NEWS
                # =========================
                news_data = [
                    "Company reports strong earnings growth",
                    "Market crashes due to global fears",
                    "Stock shows stable performance",
                    "Investors optimistic about future growth",
                    "Economic slowdown concerns investors"
                ]

                sentiments = [hybrid_sentiment(text) for text in news_data]
                avg_sentiment = np.mean(sentiments)

                # =========================
                # FEATURES
                # =========================
                df['sentiment'] = avg_sentiment
                df['prev_close'] = df['Close'].shift(1)
                df['returns'] = df['Close'].pct_change()

                df['prev_close'] = df['prev_close'].bfill()
                df['returns'].fillna(0, inplace=True)

                if len(df) < 10:
                    st.error("Not enough data!")
                else:
                    # =========================
                    # MODEL
                    # =========================
                    X = df[['prev_close', 'sentiment', 'returns']]
                    y = df['Close']

                    split = int(len(df) * 0.8)
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    model = RandomForestRegressor(n_estimators=n_estimators)
                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)

                    rmse = np.sqrt(mean_squared_error(y_test, predictions))

                    # =========================
                    # OUTPUT
                    # =========================
                    st.success(f"✅ RMSE: {rmse:.2f}")

                    st.subheader("📊 Predictions")

                    for i in range(min(5, len(predictions))):
                        sentiment = X_test.iloc[i]['sentiment']

                        if sentiment > 0:
                            st.markdown(f"<span style='color:lime'>📈 Positive → {predictions[i]:.2f}</span>", unsafe_allow_html=True)
                        elif sentiment < 0:
                            st.markdown(f"<span style='color:red'>📉 Negative → {predictions[i]:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color:yellow'>➖ Neutral → {predictions[i]:.2f}</span>", unsafe_allow_html=True)

                    # =========================
                    # GRAPH
                    # =========================
                    st.subheader("📈 Stock Prediction Graph")

                    fig, ax = plt.subplots()
                    ax.plot(y_test.values, label="Actual")
                    ax.plot(predictions, label="Predicted")
                    ax.legend()

                    st.pyplot(fig)

            except Exception as e:
                st.error(str(e))
