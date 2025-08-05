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

Let me know if you’d like help writing unit tests, packaging, or refining the AI’s style guide for financial advice.
