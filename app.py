from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import mplfinance as mpf
from llama_cpp import Llama
import json
import os
from dotenv import load_dotenv


# set up model
MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q8_0.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)

with open("train.json", "r", encoding="utf-8") as f:
    stock_knowledge = json.load(f)

#  time range
start_date = datetime(2025, 1, 1)
end_date = datetime.today()

# RSI
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# download stock data
def stock(ticker: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Indicators
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    df['Price_Direction'] = 0
    df.loc[df['Close'] > df['Close'].shift(1), 'Price_Direction'] = 1
    df.loc[df['Close'] < df['Close'].shift(1), 'Price_Direction'] = -1
    df['OBV_Change'] = df['Price_Direction'] * df['Volume']
    df['OBV_Change'] = df['OBV_Change'].fillna(0)
    df['OBV'] = df['OBV_Change'].cumsum()
    df['RSI'] = calculate_rsi(df['Close'], period=14)

    return df

import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo 

def get_utc_since(days_back=30):
    # Use US Eastern Time
    est = ZoneInfo("America/New_York")
    now_est = datetime.now(est)

    # Convert to UTC
    now_utc = now_est.astimezone(ZoneInfo("UTC"))

    # Subtract days and format
    since = (now_utc - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return since

from polygon import ReferenceClient
def fetch_stock_news(ticker: str, api_key: str, days_back=30, limit=20):
    since = get_utc_since(days_back=days_back)
    url = "https://api.polygon.io/v2/reference/news"
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")

    if not api_key:
        raise ValueError("Polygon API key not found. Make sure it's set in the .env file.")

    params = {
        "ticker": ticker,
        "published_utc.gte": since,
        "order": "desc",
        "sort": "published_utc",
        "limit": limit,
        "apiKey": api_key
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json().get("results", [])

    news_items = []
    for item in data:
        news_items.append({
            "date": item.get("published_utc", "")[:10],
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "url": item.get("article_url", "")
        })
    return news_items

# prompt summary for LLM 
def build_ai_prompt(ticker: str, df: pd.DataFrame, question: str) -> str:
    latest = df.iloc[-1]
    summary = (
        f"Ticker: {ticker}\n"
        f"Date: {latest.name.date()}\n"
        f"Close: {latest['Close']:.2f}\n"
        f"RSI: {latest['RSI']:.1f}\n"
        f"MACD: {latest['MACD']:.2f}\n"
        f"Signal: {latest['Signal']:.2f}\n"
        f"Histogram: {latest['Histogram']:.2f}\n"
        f"SMA(20): {latest['SMA_20']:.2f}\n"
        f"SMA(50): {latest['SMA_50']:.2f}\n"
        f"EMA(12): {latest['EMA_12']:.2f}\n"
        f"EMA(26): {latest['EMA_26']:.2f}\n"
        f"OBV: {int(latest['OBV'])}\n\n"
    )
     #fetching news
    api_key = "api"
    news = fetch_stock_news(f"{ticker}", api_key=api_key, days_back=30, limit=15)

    # Format into a prompt section
    news_section = "Recent News Headlines:\n" + "\n".join(
    f"- {item['date']} ({item['source']}): {item['title']}" for item in news
    )
    prompt = (
        "You are a smart financial stock assistant AI. You are given the latest technical indicators and recent news and informations about this stock and background knowledge to train on for stock analysis:\n\n"
        f"{summary, stock_knowledge, news_section}"
        f"User question: {question}\n"
        "Using most direct and short but thorough way to explain, provide a concise, data-driven analysis and recommendation.\n"
    )
    return prompt

# AI querry
def ask_llm(prompt: str) -> str:
    resp = llm(prompt, max_tokens=512, temperature=0.7, top_p=0.95)
    return resp["choices"][0]["text"].strip()

def main():
    print("=== Stock Analysis + AI Assistant ===")
    ticker = input("Enter a stock ticker (e.g. AAPL): ").upper()
    df = stock(ticker)

    # Print last rows of key indicators
    print(df[['Close','RSI','MACD','Signal','SMA_20','SMA_50']].tail())

    # --- Plot RSI ---
    plt.figure(figsize=(10,5))
    sns.lineplot(x=df.index, y=df['RSI'])
    plt.axhline(70, linestyle='--')
    plt.axhline(30, linestyle='--')
    plt.title(f"{ticker} RSI")
    plt.tight_layout()
    plt.show()

    # --- Plot MACD ---
    plt.figure(figsize=(10,5))
    sns.lineplot(x=df.index, y=df['MACD'], label='MACD')
    sns.lineplot(x=df.index, y=df['Signal'], label='Signal')
    plt.bar(df.index, df['Histogram'], alpha=0.3)
    plt.title(f"{ticker} MACD")
    plt.tight_layout()
    plt.show()

    # --- Fibonacci retracement on most recent increasing time range ---
    subset = df.loc["2025-07-01":"2025-07-16"].dropna() # select a time when the price is in bullish
    if subset.empty:
        raise ValueError("No data available in selected range for Fibonacci calculation.")
    maxp, minp = subset['High'].max(), subset['Low'].min()
    levels = [0, .382, .5, .618, 1]
    fibs = {f"{l:.3f}": maxp - l*(maxp-minp) for l in levels}
    print("Fibonacci levels:")
    for lvl, val in fibs.items():
        print(f" {lvl}: {val:.2f}")

    # Plot candlesticks + overlays
    fib_lines = [
        mpf.make_addplot(pd.Series(val, index=df.index), linestyle='--')
        for val in fibs.values()
    ]
    apds = [
        mpf.make_addplot(df["EMA_12"]),
        mpf.make_addplot(df["EMA_26"]),
        mpf.make_addplot(df["SMA_20"]),
        mpf.make_addplot(df["SMA_50"])
    ] + fib_lines

    mpf.plot(
        df[['Open','High','Low','Close','Volume']],
        type="candle", style="yahoo",
        addplot=apds, volume=True, title=f"{ticker} Candlestick",
        figsize=(12,6)
    )

    # --- Plot OBV ---
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['OBV'])
    plt.title(f"{ticker} On-Balance Volume")
    plt.tight_layout()
    plt.show()

    # AI Interaction 
    while True:
        question = input("\nWhat would you like to ask the AI about this stock? (or 'exit')\n> ")
        if question.lower() in ("exit","quit"):
            print("Goodbye.")
            return

        prompt = build_ai_prompt(ticker, df, question)
        print("\n--- AI PROMPT ---\n", prompt)
        answer = ask_llm(prompt)
        print("\n--- AI ANALYSIS ---\n", answer)
        

if __name__ == "__main__":
    main()
