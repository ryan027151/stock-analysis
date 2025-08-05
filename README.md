---

# Smart Stock Analyst 📈

A Python-based command‑line stock analysis tool powered by a local LLaMA 3.2.1b model and technical indicators (RSI, MACD, OBV, Fibonacci, moving averages, candlestick charts). Designed to deliver data-driven recommendations and visual insights with minimal external dependencies.

---

## 🚀 Features

* **Download historical stock data** using `yfinance`.
* Calculate key indicators:

  * **RSI** (14-day)
  * **MACD** (12/26 EMA + Signal line + Histogram)
  * **Moving averages**: SMA‑20, SMA‑50
  * **OBV** (On‑Balance Volume)
  * **Fibonacci retracement levels**
* **Visualize**:

  * RSI and MACD charts via Matplotlib & Seaborn
  * On-Balance Volume plot
  * Candlestick charts with overlaid EMAs, SMAs, and Fibonacci levels using `mplfinance`
* **LLM-powered analysis**:

  * Loads foundational stock knowledge (`stock_knowledge.json`)
  * Feeds real-time indicator data + stock background into LLaMA
  * Generates concise data-driven insights and recommendations

---

## 🧠 Architecture Overview

1. **Data retrieval & indicator computation** in `stock()`
2. **Visual presentation** of RSI, MACD, OBV, and candlestick charts
3. **LLM-based financial assistant**:

   * Builds a prompt including latest technical metrics and contextual domain knowledge
   * Uses your local `Llama-3.2-1B-Instruct-Q8_0.gguf` model to analyze user questions
   * However, the AI model being used are small and simple, hence it may not provide deep thought process and require extra training (can be added in train.json)
4. **Interactive loop**:

   * Users can ask questions about the current stock until they type `exit`

---

## ⚙️ Installation & Requirements

```bash
git clone <repo_url>
cd <repo_directory>

# Install dependencies
pip install -r requirements.txt
```

Key requirements:

* **Python 3.10+**
* `yfinance`, `pandas`, `seaborn`, `mplfinance`, `llama_cpp`
* **Local LLaMA model file**: `Llama-3.2-1B-Instruct-Q8_0.gguf` in `./models/`

---

## 🧪 Usage Example

```bash
python app.py
```

Sample workflow:

* Enter a stock ticker (e.g. `AAPL`)
* View recent closing prices and indicators
* Visualize RSI, MACD, OBV, and candlestick charts (with overlays)
* Enter questions to the AI, such as:

  ```
  Why are RSI and MACD showing bullish signals?
  ```

The AI analyzes indicators and returns a clear recommendation.

---

## 📂 Project Structure

```
.
├── app.py                    # Main script
├── requirements.txt          # Python package dependencies
├── stock_knowledge.json      # Background finance concepts for LLaMA
├── models/
│   └── Llama‑3.2‑1B‑Instruct‑Q8_0.gguf  # LLaMA model file
```

---

## 📌 Technical Details

* **Stock indicators**:

  * RSI: computed via standard `.diff()` and rolling averages
  * MACD: EMA(12) – EMA(26) and signal line (EMA of MACD), with histogram
  * OBV: cumulative volume flow based on daily price movement
  * Fibonacci: calculated from most recent bullish trend window
* **Visualization**:

  * Seaborn/Matplotlib for RSI, MACD, OBV
  * `mplfinance` for candlestick charts with moving averages and Fibonacci overlays
* **LLM prompt design**:

  * Includes latest indicator values and full domain knowledge context
  * LLAma generates actionable insights and assessments

---

Here is an **add-on section for your GitHub README** documenting the new **news-fetching feature** using the Polygon API:

---

## 🗞️ Real-Time Stock News Integration

This project now includes functionality to **fetch recent news headlines** related to a selected stock using the [Polygon.io News API](https://polygon.io/docs/rest/stocks/news).

### 🔍 Purpose

The latest news articles (from the past 30 days by default) are retrieved and summarized into plain text, which can then be **fed into a local AI model** (such as LLaMA) to enhance its market insight and decision-making accuracy.

### 📦 Features

* Fetches article titles, publication dates, sources, and links.
* Filters news by stock ticker.
* Adjustable date range (default: 30 days).
* Designed to enrich the AI model’s pre-context before analysis.

### 🧩 Example Usage

```python
from news import fetch_stock_news_polygon

api_key = "YOUR_POLYGON_API_KEY"
ticker = "AAPL"  # Change to your target stock

news = fetch_stock_news_polygon(ticker, api_key)
for item in news:
    print(f"{item['date']} - {item['source']}: {item['title']}")
```

### 📄 Output Sample

```
2025-07-10 - CNBC: Apple’s iPhone 17 may ditch buttons, analysts predict
2025-07-09 - Reuters: Apple faces EU scrutiny over iOS App Store rules
2025-07-05 - Bloomberg: Apple Vision Pro pre-orders exceed expectations
```

### 🧠 Feeding to AI

You can convert the news into a prompt-friendly format:

```python
prompt_section = "Recent news for analysis:\n" + "\n".join(
    f"- {n['date']} ({n['source']}): {n['title']}" for n in news
)
# Inject `prompt_section` into your LLM context before starting analysis.
```

---

### ⚙️ Requirements

* Polygon.io API key (free or paid)
* Internet connection
* Python 3.10+ (uses `zoneinfo` for timezone handling)

Install dependencies (if you haven't yet):

```bash
pip install requests
pip install polygon
```

---

Let me know if you'd like me to auto-generate the full `news.py` module or expand this section with visual examples.


## 📈 Extending the Tool

You can easily expand with:

* **More indicators**: Bollinger Bands, stochastic oscillator, VWAP
* **Buy/sell signal scoring logic**
* **Export charts or logs**, or integrate into a GUI
* **Run automated backtesting or batch analysis**

---

## ⚖️ License & Contributions

Feel free to adapt and extend.
If you’d like to integrate new indicators or improve the AI prompt logic, contributions are welcome!

---

## ✅ Summary

This repository delivers a comprehensive, interactive stock analysis tool that blends Python-based technical analysis with an LLaMA-powered intelligent assistant. Load the model, ask questions, visualize trends, and receive actionable insights—all offline and customizable.


