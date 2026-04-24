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

`adjusted_close_t = adjusted_close_{t-1} * (1 + RET_t)`

`adjusted_open_t = adjusted_close_{t-1}`

`adjusted_high_t = ASKHI_t * (adjusted_close_t / PRC_t)`

`adjusted_low_t = BIDLO_t * (adjusted_close_t / PRC_t)`

### Initialization
- For each PERMNO, the first adjusted_close is initialized using the observed PRC
- Subsequent values are computed recursively using returns

---

## ⚙️ Pipeline Architecture

### 1. Chunked Data Processing

- Dataset size: ~54 million rows  
- Processed using:

```python
chunksize = 100_000
```

## Candlestick Image Generation and Labeling

### Overview

This stage converts the reconstructed OHLC price data into **candlestick chart images** and assigns labels based on future returns. The output is a large-scale image dataset suitable for training deep learning models such as Vision Transformers (ViT).

---

### Data Input

- Input file: adjusted OHLC dataset (`*_adjusted_full.csv`)
- Processed in chunks to handle large scale data:
  

- Key fields used:
  - adjusted_open
  - adjusted_high
  - adjusted_low
  - adjusted_close
  - VOL (volume)

---

### Sliding Window Construction

Each image is generated using a rolling window:

- Lookback window: 25 trading days  
- Prediction horizon: 20 trading days  

For each stock (PERMNO), a rolling buffer is maintained:
```window size = lookback + horizon = 45
```

From this:
- First 25 days → used to generate the image  
- Day 25 → reference point for labeling  
- Day 45 → used to compute future return  

---

### Label Definition

The label is based on the future return:
```forward_return = (close_future - close_now) / close_now
```

Label:
- `up` if forward_return > 0  
- `down` otherwise  

---

### Dataset Split

Images are split based on time:

- **Train/Validation:** 1993–2000  
- **Test:** 2001 onward  

This ensures no look-ahead bias.

---

### Image Construction

Each 25-day window is converted into a **224 × 224 RGB image**.

#### Layout

- Top section: price chart (candlesticks + moving average)
- Bottom section: volume bars

#### Components

1. **Candlesticks**
   - Green: close > open  
   - Red: close < open  
   - Gray: neutral  

2. **Wicks**
   - Represent high–low range  

3. **Moving Average (MA20)**
   - 20-day moving average of adjusted_close  
   - Plotted as a blue line  

4. **Volume**
   - Scaled and plotted in lower section  

---

### Scaling

- Prices are normalized per window using min-max scaling
- Volume is scaled relative to maximum volume in the window

This ensures:
- consistent image representation  
- invariance to absolute price levels  

---

### Image Storage

Images are saved as PNG files with naming format:
```PERMNO_YYYYMMDD.png
```

Directory structure:
```
image_dataset/
train_val/
up/
down/
test/
up/
down/
```

---

### State Management

To handle large datasets and interruptions:

- Uses a rolling buffer (`deque`) per stock
- Maintains state across chunks
- Saves progress to a JSON file


This allows:
- resuming execution without restarting  
- tracking generated vs skipped samples  

---

### Data Filtering

Images are skipped if:

- Missing values in required fields  
- Invalid price ranges  
- Zero or missing volume  
- Invalid future return computation  

---

### Output Statistics (Example)

- Millions of images generated  
- Significant number of skipped samples due to data quality constraints  
- Balanced classification labels (up/down)

---

### Summary

This pipeline transforms structured financial time series data into a **large-scale labeled image dataset**, preserving:

- price dynamics  
- temporal structure  
- volume information  

The result is a dataset suitable for deep learning models to learn patterns from candlestick charts.
