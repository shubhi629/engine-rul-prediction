# engine-rul-prediction
# Engine Remaining Useful Life (RUL) Prediction

Predicting *how much longer* an aircraft engine will last — before it needs maintenance or fails — using deep learning on time-series sensor data.

## What & Why

In aviation and manufacturing, unplanned equipment failure is expensive and dangerous. Knowing when a machine is *likely* to fail lets engineers schedule maintenance proactively, reducing downtime and preventing accidents.

This project builds an end-to-end ML pipeline that takes raw sensor readings from aircraft engines and predicts the **Remaining Useful Life (RUL)** — the number of cycles left before failure.

## Dataset

**NASA CMAPSS FD001** — a standard benchmark for predictive maintenance research.

- 100 aircraft engines, each run to failure
- 21 sensor readings per cycle (temperature, pressure, fan speed, etc.)
- ~20,000 rows of time-series data
  
## Approach

### 1. Feature Engineering
Rather than feeding raw sensor values directly, a **sliding window of 30 cycles** is applied to each engine's history. For every window, 5 statistical features are extracted per sensor:

| Feature | Description |
|--------|-------------|
| `mean` | Average sensor value over the window |
| `std` | Variability / noise level |
| `min` / `max` | Extreme values |
| `trend` | Linear slope — is the sensor degrading? |

This gives the model a richer, context-aware view of each engine's health trajectory.

### 2. Feature Selection
With 21 sensors × 5 stats = 105 features, not all are equally useful. A **Random Forest** was trained on the training set to rank feature importance. The **top 20 features** were selected — `sensor_3_mean` alone accounted for 46.6% of predictive importance.

### 3. Model Architecture
A **3-layer stacked LSTM** network was built using TensorFlow/Keras:

```
LSTM(128) → BatchNorm → Dropout
LSTM(64)  → BatchNorm → Dropout
LSTM(32)  → BatchNorm → Dropout
Dense(64) → Dense(32) → Dense(1)
```

LSTMs are well-suited here because engine degradation is a **sequential process** — what happened 20 cycles ago matters for predicting what happens next.

### 4. Training Setup
- **Split:** 70% train / 15% validation / 15% test (split by engine, not by row — prevents data leakage)
- **Scaler:** StandardScaler fit on train only, applied to val/test
- **RUL Cap:** 125 cycles (engines beyond this are treated as healthy — standard CMAPSS practice)
- **Callbacks:** EarlyStopping + ReduceLROnPlateau to prevent overfitting

---

## Results

| Metric | Value |
|--------|-------|
| R² Score | **0.8759** |
| MAE | **9.90 cycles** |
| RMSE | **14.45 cycles** |

The model explains ~88% of the variance in remaining engine life, with an average prediction error of under 10 cycles.

---

## Tech Stack

- **Python** — NumPy, Pandas
- **Scikit-learn** — RandomForest (feature selection), StandardScaler
- **TensorFlow / Keras** — LSTM model
- **Matplotlib / Seaborn** — Visualizations
- **Google Colab** — Training environment

---

## Project Structure

```
engine-rul-prediction/
│
├── RUL_final.ipynb       # Full pipeline: EDA → features → model → evaluation
├── README.md
└── .gitignore
```

---

## Key Takeaways

- Engine-level train/test splitting is critical — row-level splitting leaks future data
- `sensor_3_mean` is by far the strongest degradation signal in FD001
- Trend features (slope of sensor readings) add meaningful predictive value beyond raw stats
- LSTM outperforms simpler regressors on this sequential degradation task
