from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import mplfinance as mpf

end_date = datetime(year=2025, month=7, day=13)
start_date = datetime(year=2024, month=12, day=1)

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
    df.columns = df.columns.get_level_values(0)

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

    #OBV
    df['Price_Direction'] = 0
    df.loc[df['Close'] > df['Close'].shift(1), 'Price_Direction'] = 1
    df.loc[df['Close'] < df['Close'].shift(1), 'Price_Direction'] = -1
    df['OBV_Change'] = df['Price_Direction'] * df['Volume']
    df['OBV_Change'] = df['OBV_Change'].fillna(0)
    df['OBV'] = df['OBV_Change'].cumsum()

    #RSI
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    
    return df

df = stock("RGC")
print(df[['Close', 'RSI', 'MACD','EMA_12','EMA_26','SMA_20','SMA_50']].tail())

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

#Fib Retracement
#Time range
subset = df.loc["2025-07-01":"2025-07-13"]
#calculation
maxprice = subset['Close'].max()
minprice = subset['Close'].min()
fib_0 = maxprice
fib_382 = maxprice - 0.382*(maxprice-minprice)
fib_500 = maxprice - 0.5*(maxprice-minprice)
fib_618 = maxprice - 0.618*(maxprice-minprice)
fib_1 = minprice
print(f"Fib retrace 0.000 is {fib_0}.")
print(f"Fib retrace 0.382 is {fib_382}.")
print(f"Fib retrace 0.500 is {fib_500}.")
print(f"Fib retrace 0.618 is {fib_618}.")
print(f"Fib retrace 1.000 is {fib_1}.")

fib_0_line = pd.Series(fib_0, index=df.index)
fib_382_line = pd.Series(fib_382, index=df.index)
fib_500_line = pd.Series(fib_500, index=df.index)
fib_618_line = pd.Series(fib_618, index=df.index)
fib_1_line = pd.Series(fib_1, index=df.index)

fib_0_hl = mpf.make_addplot(fib_0_line, color='black', linestyle='--', width=1.2)
fib_382_hl = mpf.make_addplot(fib_382_line, color='purple', linestyle='--', width=1.2)
fib_500_hl = mpf.make_addplot(fib_500_line, color='red', linestyle='--', width=1.2)
fib_618_hl = mpf.make_addplot(fib_618_line, color='orange', linestyle='--', width=1.2)
fib_1_hl = mpf.make_addplot(fib_1_line, color='green', linestyle='--', width=1.2)

#Draw EMA and SMA
apds = [ #mpf can be used to make candle stick
    mpf.make_addplot(df["EMA_12"], color="blue",  width=1.0),
    mpf.make_addplot(df["EMA_26"], color="green", width=1.0),
    mpf.make_addplot(df["SMA_20"], color="orange", width=1.2),
    mpf.make_addplot(df["SMA_50"], color="red",   width=1.2),
    fib_0_hl,
    fib_382_hl,
    fib_500_hl,
    fib_618_hl,
    fib_1_hl
]
#draw candle stick stock graph
ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

ohlcv = ohlcv.dropna()  # drop rows with any NaN (important if recent dates are still forming)

ohlcv = ohlcv.astype({
    'Open': float,
    'High': float,
    'Low': float,
    'Close': float,
    'Volume': float
})
mpf.plot( #graph candle stick; need to fix due to vauleerror of Open NOT being an int or float?
    ohlcv,
    type      = "candle",
    style     = "yahoo",      # or "charles", "nightclouds", etc.
    addplot   = apds,
    volume    = True,         # bot tom sub‑panel
    figsize   = (14, 8),
    title     =  "Graph",
    mav       = (),           # turn off mpf’s own mav so we rely on ours
    tight_layout=True
)

"""sns.lineplot(data=df, x=df.index, y='EMA_12', label='EMA 12', color='blue')
sns.lineplot(data=df, x=df.index, y='EMA_26', label='EMA 26', color='green')
sns.lineplot(data=df, x=df.index, y='SMA_20', label='SMA 20', color='orange')
sns.lineplot(data=df, x=df.index, y='SMA_50', label='SMA 50', color='red')

plt.title(' Moving Averages (EMA and SMA)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()"""

#OBV
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['OBV'], label='On-Balance Volume', color='purple')
plt.title('On-Balance Volume (OBV)')
plt.xlabel('Date')
plt.ylabel('OBV')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


