# Energy Consumption Forecasting

> Time series forecasting of hourly energy consumption using ARIMA, Prophet, and LSTM — benchmarked on 16 years of PJM East grid data.

---

## Project Overview

This project explores classical and deep learning approaches to energy demand forecasting on the PJM Interconnection East (PJME) hourly dataset. The pipeline covers end-to-end machine learning workflow: exploratory analysis, preprocessing, three distinct modeling strategies, and a rigorous side-by-side comparison.

The goal is to evaluate how well each model family captures the complex temporal patterns in energy consumption — including long-term trends, multi-scale seasonality, and anomalies — and to identify practical trade-offs between interpretability, accuracy, and computational cost.

---

## Dataset

| Property | Details |
|---|---|
| Source | [PJM Hourly Energy Consumption — Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) |
| Region | PJM East (PJME) — US Mid-Atlantic grid |
| Coverage | January 2002 – August 2018 |
| Records | ~145,000 hourly observations |
| Target Variable | `PJME_MW` — energy consumption in megawatts |

---

## Project Structure

```
energy-consumption-forecasting/
├── data/
│   └── PJME_hourly.csv           # Raw hourly energy consumption data
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb   # Trend, seasonality, stationarity, ACF/PACF
│   ├── preprocessing.ipynb               # Cleaning, feature engineering, train/test split
│   ├── arima_model.ipynb                 # ARIMA grid search, diagnostics, forecasting
│   ├── prophet_model.ipynb               # Prophet baseline + tuned model with holidays
│   ├── lstm_model.ipynb                  # PyTorch LSTM training, evaluation, forecasting
│   └── model_comparison.ipynb            # Cross-model metrics, visual benchmarking
│
├── models/
│   └── lstm_model.pth            # Saved PyTorch LSTM weights
│
├── results/
│   ├── arima_metrics.csv         # MAE, RMSE, MAPE for ARIMA
│   ├── arima_results.csv         # ARIMA forecast vs actuals
│   ├── prophet_metrics.csv       # MAE, RMSE, MAPE for Prophet
│   ├── prophet_results.csv       # Prophet forecast vs actuals
│   ├── lstm_metrics.csv          # MAE, RMSE, MAPE for LSTM
│   └── lstm_results.csv          # LSTM forecast vs actuals
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis
- Long-term trend decomposition (2002–2018)
- Seasonal pattern analysis: hourly, daily, weekly, and monthly cycles
- Outlier detection via IQR and visual inspection
- Stationarity testing using the Augmented Dickey-Fuller (ADF) test
- Autocorrelation (ACF) and partial autocorrelation (PACF) analysis

### 2. Preprocessing
- Missing timestamp detection and interpolation
- Outlier capping using the IQR method
- Feature engineering:
  - Cyclical encoding of time features (hour, day-of-week, month) via sine/cosine transforms
  - Lag features for autoregressive context
- Train/test split: pre-2018 data for training, 2018 for out-of-sample evaluation

### 3. ARIMA
- Daily resampled series to reduce noise and computational load
- Exhaustive grid search over `(p, d, q)` hyperparameter space
- Residual diagnostics: Ljung-Box test, QQ plot, residual ACF
- Forecast on held-out 2018 period

### 4. Prophet
- Baseline model with default settings
- Tuned model with:
  - Multiplicative seasonality mode
  - US public holiday effects
  - Custom Fourier orders for weekly and yearly seasonality
- Uncertainty interval analysis

### 5. LSTM (PyTorch)
- 2-layer LSTM network with dropout regularization
- Sliding window approach: 30-day lookback window
- MinMaxScaler normalization on training data
- Early stopping to prevent overfitting
- Training on GPU (if available), evaluation on CPU-compatible checkpoints

### 6. Model Comparison
- Unified evaluation on the 2018 test period
- Metrics: MAE, RMSE, MAPE
- Visualizations:
  - Bar chart comparison across all metrics
  - Forecast overlay plots (actual vs. predicted)
  - Monthly error breakdown
  - Residual distribution analysis

---

## Results

> Metrics computed on the 2018 out-of-sample test set (daily resolution for ARIMA and Prophet; windowed daily for LSTM).

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| ARIMA(3,1,3) | 5,837 MW | 6,654 MW | 19.99% |
| Prophet (Tuned) | 2,461 MW | 3,328 MW | 7.41% |
| LSTM (PyTorch) | 1,484 MW | 1,952 MW | 4.75% |

---

## Key Findings

- **Seasonality dominates**: Hourly and weekly seasonal patterns account for the majority of variance in energy consumption, making seasonality-aware models critical.
- **Prophet's holiday modeling** provides measurable improvement over the baseline, particularly around US public holidays where demand drops sharply.
- **LSTM captures non-linear dynamics** that statistical models miss, but requires careful tuning and significantly more computational resources.
- **ARIMA is competitive** on daily-aggregated data despite its simplicity, confirming that aggregation smooths out much of the complexity that deep learning would otherwise be needed to model.
- **Multiplicative seasonality** outperforms additive in Prophet, consistent with the proportional nature of consumption fluctuations.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Statistical Modeling | `statsmodels` (ARIMA, ADF, ACF/PACF) |
| Probabilistic Forecasting | `prophet` |
| Deep Learning | `torch` (PyTorch) |
| ML Utilities | `scikit-learn` (scaling, metrics) |
| Environment | Python 3.10+, Jupyter Notebook |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/energy-consumption-forecasting.git
cd energy-consumption-forecasting
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `PJME_hourly.csv` from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) and place it in the `data/` directory.

### 5. Run notebooks in order
```
1. exploratory_data_analysis.ipynb
2. preprocessing.ipynb
3. arima_model.ipynb
4. prophet_model.ipynb
5. lstm_model.ipynb
6. model_comparison.ipynb
```

---

## Future Improvements

- **Higher resolution modeling**: Re-run Prophet and LSTM on raw hourly data (rather than daily aggregates for ARIMA) to capture intraday demand spikes
- **SARIMA / SARIMAX**: Extend ARIMA with explicit seasonal order and exogenous variables (temperature, calendar features)
- **Transformer-based forecasting**: Experiment with Temporal Fusion Transformer (TFT) or PatchTST for long-sequence forecasting
- **Ensemble approach**: Combine Prophet's uncertainty estimates with LSTM point forecasts for a hybrid model
- **Interactive dashboard**: Build a Streamlit app for real-time forecast visualization and model comparison
- **Exogenous variables**: Incorporate weather data (temperature, humidity) as covariates to improve accuracy

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Developer

<table>
  <tr>
    <td align="center">
      <b>ILGINLI</b><br/>
    </td>
  </tr>
</table>

---

<p align="center">
  <sub>Built with Python · PJM Hourly Energy Consumption Dataset · 2002–2018</sub>
</p>
