from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import mplfinance as mpf

end_date = datetime(year=2025, month=7, day=7)
start_date = datetime(year=2025, month=1, day=1)

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

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
    
    # Calculate SMAs
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    #MACD
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] =  df["EMA_12"] - df["EMA_26"]
    # MACD signal line (9-day EMA of MACD)
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # MACD Histogram
    df["Histogram"] = df["MACD"] - df["Signal"]

    #RSI
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    
    return df

df = stock("RGC")
print(df[['Close', 'RSI', 'MACD','EMA_12','EMA_26','SMA_20','SMA_50']].tail())
print(df.dtypes)

#draw RSI
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x=df.index, y='RSI')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title('RSI')
plt.ylabel('RSI')
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# Draw MACD
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x=df.index, y='MACD', label='MACD', color='blue')
sns.lineplot(data=df, x=df.index, y='Signal', label='Signal Line', color='orange')
plt.bar(df.index, df['Histogram'], label='Histogram', color='gray', alpha=0.5, width=1.0)
plt.title('MACD Indicator')
plt.xlabel('Date')
plt.ylabel('MACD Value')
plt.legend()
plt.tight_layout()
plt.show()

#Draw EMA and SMA
plt.figure(figsize=(14,7))

apds = [ #mpf can be used to make candle stick
    mpf.make_addplot(df["EMA_12"], color="blue",  width=1.0),
    mpf.make_addplot(df["EMA_26"], color="green", width=1.0),
    mpf.make_addplot(df["SMA_20"], color="orange", width=1.2),
    mpf.make_addplot(df["SMA_50"], color="red",   width=1.2),
]

mpf.plot( #graph candle stick; need to fix due to vauleerror of Open NOT being an int or float?
    df,
    type      = "candle",
    style     = "yahoo",      # or "charles", "nightclouds", etc.
    addplot   = apds,
    volume    = True,         # bot tom sub‑panel
    figsize   = (14, 8),
    title     =  "Graph",
    mav       = (),           # turn off mpf’s own mav so we rely on ours
    tight_layout=True
)

sns.lineplot(data=df, x=df.index, y='EMA_12', label='EMA 12', color='blue')
sns.lineplot(data=df, x=df.index, y='EMA_26', label='EMA 26', color='green')
sns.lineplot(data=df, x=df.index, y='SMA_20', label='SMA 20', color='orange')
sns.lineplot(data=df, x=df.index, y='SMA_50', label='SMA 50', color='red')

plt.title(' Moving Averages (EMA and SMA)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

