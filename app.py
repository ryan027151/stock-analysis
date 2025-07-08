from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd

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
    df_single = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    
    if df_single.empty:
        raise ValueError("No data returned from yfinance.")

    # Calculate RSI based on adjusted Close (auto_adjust=True means 'Close' is adjusted)
    df_single['RSI'] = calculate_rsi(df_single['Close'], period=14)
    
    return df_single

df = stock("RGC")
print(df[['Close', 'RSI']].tail())

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
