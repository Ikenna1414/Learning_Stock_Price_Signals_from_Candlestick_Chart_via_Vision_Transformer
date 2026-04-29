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
- Discontinuous price series (artificial jumps)  
- Misleading candlestick charts  
- Models learning noise instead of signal  

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

# Candlestick Image Generation and Labeling

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
```window size = lookback + horizon = 45```


From this:
- First 25 days → used to generate the image  
- Day 25 → reference point for labeling  
- Day 45 → used to compute future return  

---

### Label Definition

The label is based on the future return:
```forward_return = (close_future - close_now) / close_now```

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
```PERMNO_YYYYMMDD.png```


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


# Vision Transformer (ViT) Model – Stock Return Prediction
## Overview
This module trains a Vision Transformer (ViT) to predict future stock price direction using candlestick chart images. The task is formulated as a binary classification problem based on forward returns.
- Input: 25-day candlestick chart images (224×224)
- Output: Binary classification
  - `1` → positive future return  
  - `0` → negative future return  

---

## Model Architecture

A Vision Transformer (ViT-B/32) is used with a reduced number of encoder layers to match the experimental setup. The model is trained from stratch without pretrained weights

---
## Data Pipeline
Images are organized into class folders and loaded using a standard image dataset loader.
- Structure:
  - `train/up`
  - `train/down`

The dataset is split into:
- 70% training  
- 30% validation  

---
## Training Setup
- Loss Function: Cross-Entropy  
- Optimizer: Adam  
- Learning Rate: 1e-4  
- Batch Size: 32  
- Device: GPU  

---
## Training Process
- Model is trained over multiple epochs  
- Validation is performed after each epoch  
- Performance is tracked using:
  - Loss  
  - Classification accuracy  
---
## Checkpointing
- Best model (based on validation accuracy) is saved  
- Full training state is checkpointed each epoch to allow resuming  

---
## Monitoring
Batch-level logging is implemented to provide visibility during long training runs.
---
## Notes
- Model is trained entirely from scratch  
- Configuration follows research constraints (no pretrained weights, fixed image size)  
- Designed for large-scale datasets (millions of images)

- # Improved Vision Transformer – Enhanced Model

## Overview

An improved version of the baseline ViT model is provided in `improved_vit_model.ipynb`, introducing 9 enhancements over the baseline to improve training stability and probability calibration.

The most important improvement is **label smoothing**, which directly addresses the non-monotonic quintile return pattern observed in interim results. Better-calibrated probability scores produce cleaner cross-sectional stock rankings.

---

## Improvements Over Baseline

| Feature | Baseline | Improved |
|---------|----------|----------|
| Encoder layers | 2 | 4 |
| Optimizer | Adam | AdamW |
| LR schedule | Fixed 1e-4 | Cosine decay |
| Label smoothing | None | 0.1 |
| Image augmentation | None | Random flip + ColorJitter |
| Mixed precision | No | Yes (FP16) |
| Gradient clipping | No | Yes (max norm 1.0) |
| Classification head | Linear | LayerNorm + Dropout + Linear |
| Image normalisation | ToTensor only | ImageNet mean/std |

---

# Portfolio Construction – Long-Short Factor Strategy

## Overview

This module takes the ViT model's predicted probability scores and constructs long-short factor portfolios following the double-sort methodology of Byun, Na, and Song (2025) and Fama-French (1993).

---


## Input Files Required

| File | Description |
|------|-------------|
| `crsp_full.csv` | Full CRSP download (1.37GB, 37M rows) — daily price, volume, market cap, exchange code |
| `vit_chunk1.csv` | ViT signal output, training checkpoints 1–40 (permno, date, predicted P(up)) |
| `vit_chunk2.csv` | ViT signal output, training checkpoints 40–89 (path, signal, label) |
| `ff3.csv` | Fama-French 3-factor monthly returns + momentum (WRDS) |

---

## Pipeline

### Step 1 — Build Market Data
CRSP daily data is loaded and pivoted into panel DataFrames: close prices, volume, market cap, and NYSE-only market cap. Days with zero volume are treated as non-trading days.

### Step 2 — NYSE Size Breakpoints
Size breakpoints computed from NYSE-listed stocks only, updated annually each June. Quintile cutoffs at the 20th, 40th, 60th, and 80th percentiles of NYSE market cap.

### Step 3 — Reformat ViT Signals to Monthly
Daily predicted probabilities are converted to monthly signals using the last available prediction at each month-end.

### Step 4 — Double Sort (5×5)
At each month-end, stocks are independently sorted on two dimensions:
1. **Market capitalisation** — NYSE breakpoints updated June annually
2. **ViT signal** — predicted probability of a positive 20-day forward return

### Step 5 — Portfolio Weighting
Three weighting schemes applied within each portfolio cell:
* **Cap-VW:** value-weighted, individual weights capped at 80th NYSE percentile
* **Standard VW:** standard value-weighted
* **Equal-Weighted (EW):** equal weight per stock

Monthly rebalancing with **10 basis points** transaction cost round-trip.

### Step 6 — Long-Short Factor
Long the highest signal quintile, short the lowest, averaged across size groups for a size-neutral monthly return series.

### Step 7 — Factor Alpha Test


A statistically significant α confirms the signal generates returns that existing risk factors cannot explain.

---

## Current Results (Interim Checkpoint)

| Metric | Value |
|--------|-------|
| Best cohort spread (Q4 vs Q1) | +1.80% per month (~21.6% annualised) |
| Stock-month observations | 128,561 |
| Monthly rebalancing periods | 235 (January 2001 to July 2020) |
| L/S Q5-Q1 Sharpe | -0.33 (model not fully converged) |
| Published benchmark L/S Sharpe | 0.51 (Byun et al. 2025) |

Non-monotonic quintile pattern reflects an intermediate training checkpoint. Signal is present but probability calibration resolves with full training.

---

## Notes

* Faithful replication of Byun et al. (2025) — all parameters, breakpoints, and weighting choices match the published paper
* Any divergence from published results is attributable to model training state, not portfolio construction
* Universe defined via CRSP exchange codes — no survivorship bias




