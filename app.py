from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

end_date = datetime(year=2025, month=7, day=7)
start_date = datetime(year=2025, month=1, day=1)

def stock(ticker):
   df_single = yf.download(
      tickers=ticker,
      start=start_date,
      end=end_date,
      interval="1d",
      group_by="ticker",
      auto_adjust=True,
      progress=False
   )
   return(df_single.head(),df_single.tail())  

print(stock("NVDA"))      # See the first few rows
