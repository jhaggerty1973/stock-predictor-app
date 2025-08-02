
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Function Definitions ---
def get_stock_data(ticker, period="1mo", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df = df.reset_index()  # Ensure Date is a column, not index
    df['Date'] = df['Date'].dt.date  # Keep only date part
    return df

def add_technical_indicators(df):
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Delta'] = df['Close'].diff()
    return df.dropna()

def get_sentiment_data():
    news_by_day = {
        "2025-07-26": ["Apple sees growth in services revenue"],
        "2025-07-29": ["Concerns over iPhone sales may affect outlook"],
        "2025-07-30": ["Apple prepares major product launch in fall"],
        "2025-08-01": ["Tech stocks rally as Fed signals pause in rate hikes"],
    }
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = []
    for date, headlines in news_by_day.items():
        scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_sentiment = sum(scores) / len(scores)
        sentiment_data.append({'Date': pd.to_datetime(date).date(), 'Sentiment': avg_sentiment})
    return pd.DataFrame(sentiment_data)

def create_features(df):
    df['Next_Close'] = df['Close'].shift(-1)
    df['Price_Up'] = (df['Next_Close'] > df['Close']).astype(int)
    return df.dropna()

def train_model(df, features):
    X = df[features]
    y = df['Price_Up']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    df['Prediction'] = model.predict(X)
    return model, df

# --- Streamlit App ---
st.set_page_config(page_title="Stock Predictor", page_icon="📈")
st.title("📈 Stock Movement Predictor with Sentiment")

ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo"], index=0)

if st.button("Run Prediction"):
    stock_df = get_stock_data(ticker, period=period)
    stock_df = add_technical_indicators(stock_df)
    sentiment_df = get_sentiment_data()

    merged_df = pd.merge(stock_df, sentiment_df, on="Date", how="left")
    merged_df['Sentiment'].fillna(0, inplace=True)
    merged_df = create_features(merged_df)

    features = ['SMA_7', 'SMA_14', 'Daily_Return', 'Price_Delta', 'Sentiment']
    model, result_df = train_model(merged_df, features)

    # Display chart
    st.subheader("Price Chart with SMA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df["Date"], y=result_df['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=result_df["Date"], y=result_df['SMA_7'], name='SMA 7'))
    fig.add_trace(go.Scatter(x=result_df["Date"], y=result_df['SMA_14'], name='SMA 14'))
    st.plotly_chart(fig)

    # Display predictions
    st.subheader("Predictions (last 5 days)")
    display_df = result_df[['Date', 'Close', 'Sentiment', 'Price_Up', 'Prediction']].tail(5)
    display_df['Signal'] = display_df['Prediction'].map({1: "📈 BUY", 0: "⚠️ HOLD"})
    st.dataframe(display_df)

    # Most recent signal
    latest = display_df.iloc[-1]
    st.markdown(f"### Today’s Signal: **{latest['Signal']}**")
