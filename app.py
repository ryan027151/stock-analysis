from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import mplfinance as mpf

# Time window
start_date = datetime(2024, 12, 1)
end_date = datetime(2025, 7, 13)

# RSI calculation
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Download and process stock data
def stock(ticker):
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("No data returned from yfinance.")

    # Flatten MultiIndex if needed
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

df = stock("RGC")
print(df[['Close', 'RSI', 'MACD', 'EMA_12', 'EMA_26', 'SMA_20', 'SMA_50']].tail())

# === RSI ===
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='RSI')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title('RSI')
plt.ylabel('RSI')
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# === MACD ===
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='MACD', label='MACD', color='blue')
sns.lineplot(data=df, x=df.index, y='Signal', label='Signal Line', color='orange')
plt.bar(df.index, df['Histogram'], label='Histogram', color='gray', alpha=0.5, width=1.0)
plt.title('MACD Indicator')
plt.xlabel('Date')
plt.ylabel('MACD Value')
plt.legend()
plt.tight_layout()
plt.show()

# === Fibonacci Retracement ===
subset = df.loc["2025-07-01":"2025-07-13"].dropna()
if subset.empty:
    raise ValueError("No data available in selected range for Fibonacci calculation.")

maxprice = subset['Close'].max()
minprice = subset['Close'].min()
fib_levels = {
    '0.000': maxprice,
    '0.382': maxprice - 0.382*(maxprice-minprice),
    '0.500': maxprice - 0.5*(maxprice-minprice),
    '0.618': maxprice - 0.618*(maxprice-minprice),
    '1.000': minprice
}

for level, value in fib_levels.items():
    print(f"Fib retrace {level} is {value:.2f}")

# Horizontal Fib Lines
fib_lines = [
    mpf.make_addplot(pd.Series(value, index=df.index), color=color, linestyle='--', width=1.2)
    for value, color in zip(fib_levels.values(), ['black', 'purple', 'red', 'orange', 'green'])
]

# === Candlestick + EMA/SMA + Fib ===
apds = [
    mpf.make_addplot(df["EMA_12"], color="blue", width=1.0),
    mpf.make_addplot(df["EMA_26"], color="green", width=1.0),
    mpf.make_addplot(df["SMA_20"], color="orange", width=1.2),
    mpf.make_addplot(df["SMA_50"], color="red", width=1.2),
] + fib_lines

ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().astype(float)

mpf.plot(
    ohlcv,
    type="candle",
    style="yahoo",
    addplot=apds,
    volume=True,
    figsize=(14, 8),
    title="Candlestick",
    mav=(),
    tight_layout=True
)

# === OBV ===
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['OBV'], label='On-Balance Volume', color='purple')
plt.title('On-Balance Volume (OBV)')
plt.xlabel('Date')
plt.ylabel('OBV')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def analysis(rsi_value, macd_value, signal_value, histogram_value):
    buy_score = 0

    # ----- RSI Analysis -----
    if rsi_value < 30:
        print("RSI: Oversold – the asset may be undervalued and could rebound.")
        buy_score += 1
    elif 30 <= rsi_value <= 55:
        print("RSI: Strong buy signal – momentum may be shifting upward.")
        buy_score += 1.5
    elif 55 < rsi_value <= 70:
        print("RSI: Neutral to slightly overbought – proceed with caution.")
        buy_score -= 0.5
    else:  # RSI > 70
        print("RSI: Overbought – the asset may be overvalued.")
        buy_score -= 1

    # ----- MACD Analysis -----
    if macd_value > signal_value and histogram_value > 0:
        print("MACD: Bullish crossover – strong buy momentum.")
        buy_score += 1.5
    elif macd_value > signal_value and histogram_value < 0:
        print("MACD: Weak bullish crossover – watch for confirmation.")
        buy_score += 0.5
    elif macd_value < signal_value and histogram_value < 0:
        print("MACD: Bearish crossover – selling pressure increasing.")
        buy_score -= 1
    elif macd_value < signal_value and histogram_value > 0:
        print("MACD: Weak bearish crossover – could reverse.")
        buy_score -= 0.5
    else:
        print("MACD: Neutral – no strong trend signal.")
    
    return buy_score


    

    

