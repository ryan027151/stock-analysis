import yfinance as yf

def search(x):
   ticker = x
   data = yf.download(x, start="2025-01-01", end="2025-07-01")

   start = data.head()
   end = data.tail()
   return start, end

print(search("NVDA"))
