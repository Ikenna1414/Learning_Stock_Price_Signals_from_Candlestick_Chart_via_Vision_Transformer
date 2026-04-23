# Learning_Stock_Price_Signals_from_Candlestick_Chart_via_Vision_Transformer
Research Paper Replication - Alpha signal extraction from candle stick charts using Vision Tranformers


#  Stock Price Reconstruction & Candlestick Data Pipeline

## Overview
In this section of the project, a scalable data pipeline is built to transform raw CRSP stock data into model-ready, split-adjusted OHLC (Open, High, Low, Close) price series. These outputs are designed for the downstream machine learning tasks, including **candlestick image generation** and **Vision Transformer (ViT) modeling** for stock return prediction.

The primary objective is to ensure that all price data used for modeling reflects **true economic behavior**, free from distortions caused by stock splits and dividends.

---

## Investigation of Incorporation of Dividends and Stock Splits
After investigation via plotting percentage return, percentage change in prices and percentage change in shares outstanding, we were able to determine that dividends and stock splits were not accounted for in the data. Hence, the raw CRSP data presents a critical inconsistency:

- **Prices (PRC, ASKHI, BIDLO)** are **NOT adjusted** for stock splits or dividends  
- **Returns (RET)** are **already adjusted**

### Key Insight

> Stock splits and dividends are embedded in returns, but not in price fields.

This leads to:
- ❌ Discontinuous price series (artificial jumps)  
- ❌ Misleading candlestick charts  
- ❌ Models learning noise instead of signal  

---

## Solution

We reconstruct a **fully adjusted price series** using returns - e method employed by the original authors of the paper

Instead of relying on raw prices, we:
1. Rebuild price paths using returns  
2. Ensure continuity across time  
3. Generate adjusted OHLC values for candlestick charting  

### Adjusted Price Formulas

adjusted_close_t = adjusted_close_{t-1} * (1 + RET_t)

adjusted_open_t = adjusted_close_{t-1}

adjusted_high_t = ASKHI_t * (adjusted_close_t / PRC_t)

adjusted_low_t = BIDLO_t * (adjusted_close_t / PRC_t)
---

## ⚙️ Pipeline Architecture

### 1. Chunked Data Processing

- Dataset size: ~54 million rows  
- Processed using:

```python
chunksize = 100_000
