# 🚲 Predicting Bike Sharing Demand Using Machine Learning and Temporal Features

> **Author:** Thammishetti Venkat Sai Prathap  
> **Programme:** MSc Data Science — University of Hertfordshire

---

## 📌 Overview

This project predicts hourly bike rental demand using the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset), which contains 17,379 hourly records from Capital Bikeshare in Washington D.C. (2011–2012).

Three modelling approaches are benchmarked — two ensemble ML models and one classical time-series model — evaluated on a strict chronological train/test split to simulate real-world forecasting conditions.

---

## 🗂️ Project Structure

```
├── Predicting_Bike_Sharing_Demand_Using_Machine_Learning_and_Temporal_Features.ipynb
├── hour.csv               # UCI Bike Sharing Dataset (hourly)
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| File | `hour.csv` |
| Records | 17,379 hourly observations |
| Time span | January 2011 – December 2012 |
| Target | `cnt` — combined casual + registered rentals per hour |

Key predictors include hour of day, day of week, month, season, weather condition, temperature, humidity, wind speed, and holiday/working day flags.

---

## 🔍 Exploratory Data Analysis

- **Rental distribution** — right-skewed; moderate demand in most hours with rare high-count spikes during commuter peaks
- **Temporal patterns** — bimodal daily profile (morning peak ~08:00, evening peak ~17:00–18:00); summer months (Jun–Sep) drive peak usage; year-on-year growth from ~145 to ~234 mean hourly rentals
- **Key correlations** — temperature is the strongest positive predictor (r ≈ 0.40); humidity has a moderate negative relationship (r ≈ −0.32)
- **Stationarity** — ADF test confirms the series is stationary (p < 0.05), so no differencing is required for SARIMAX
- **Autocorrelation** — ACF/PACF plots reveal strong lag-1, lag-24, and lag-168 dependencies, informing XGBoost lag feature design

---

## ⚙️ Feature Engineering

| Technique | Detail |
|---|---|
| **Cyclical encoding** | Sin/cos transformation of `hr` (period 24) and `mnth` (period 12) so model respects circularity (e.g. 23:00 ≈ 00:00) |
| **Lag features** | `lag_1`, `lag_24`, `lag_168` — previous hour, same hour yesterday, same hour last week |
| **One-hot encoding** | `season`, `weathersit`, `weekday` with `drop_first=True` to avoid dummy-variable trap |
| **Chronological split** | Strict 80/20 split on sorted timestamps — no future data leaks into training |

---

## 🤖 Models

### 1. Random Forest Regressor
- Ensemble of decision trees trained on cyclical and calendar features
- Baseline trained with default hyperparameters; tuned version uses `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)`

### 2. XGBoost Regressor
- Gradient boosting with lag features (`lag_1`, `lag_24`, `lag_168`) that directly capture temporal autocorrelation
- Tuned via `RandomizedSearchCV` over learning rate, depth, subsample ratios, and regularisation terms

### 3. SARIMAX
- Classical statistical reference model: `SARIMAX(1,0,1)(1,0,1)[24]`
- Exogenous regressors include cyclical encodings and meteorological variables
- Tuned via `auto_arima` (AIC-guided stepwise search on a 3,000-row subset) then re-fitted on the full training set

---

## 📏 Evaluation Metrics

Each model is assessed using four metrics on the held-out test set:

- **RMSE** — Root Mean Squared Error (penalises large errors)
- **MAE** — Mean Absolute Error (interpretable in rental units)
- **R²** — Proportion of variance explained
- **MAPE** — Mean Absolute Percentage Error

---

## 🛠️ Installation & Setup

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone (https://github.com/Saiprathap22/Predicting-Bike-Sharing-Demand-Using-Machine-Learning-and-Temporal-Features/tree/main)
cd Predicting Bike Sharing Demand Using Machine Learning and Temporal Features. 

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels pmdarima
```

**Download the dataset** from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) and place `hour.csv` in the project root.

**Run the notebook:**
```bash
jupyter notebook "Predicting_Bike_Sharing_Demand_Using_Machine_Learning_and_Temporal_Features.ipynb"
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Random Forest, metrics, `TimeSeriesSplit`, `RandomizedSearchCV` |
| `xgboost` | Gradient boosting |
| `statsmodels` | SARIMAX, ADF test, ACF/PACF |
| `pmdarima` | `auto_arima` for SARIMAX order selection |

---

## 📄 License

This project was developed for academic purposes as part of the MSc Data Science programme at the University of Hertfordshire. Please credit the author if you reuse or adapt this work.

---

## 🙏 Acknowledgements

- Dataset: [Fanaee-T, H. & Gama, J. (2013)](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) — UCI Machine Learning Repository
- Capital Bikeshare, Washington D.C., for the original ride data
