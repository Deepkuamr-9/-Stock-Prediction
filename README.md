# 📈 Stock Price Prediction System

A **machine learning pipeline** for predicting next-day stock returns using **technical indicators**.
This project is designed for **educational purposes** and runs smoothly in **Google Colab**.

> ⚠️ *Not financial advice. For learning & research only.*

---

## 🚀 Features

* Downloads historical data via **Yahoo Finance (`yfinance`)**
* Extracts technical features:

  * Returns (1d, 5d, 10d, rolling mean/std)
  * SMA/EMA
  * RSI, MACD, Bollinger Bands
  * Volume Z-scores & changes
  * Lagged returns
* Models supported:

  * Gradient Boosting Regressor
  * Random Forest Regressor
* Evaluates:

  * **MAE**, **RMSE**, **Directional Accuracy**
  * Baselines: zero return & lag-1 return
* Visualizations:

  * Predicted vs Actual returns
  * Rebased price path
  * Naive long/cash backtest equity curve
* Quick **inference function** for next-day predictions

---

## 🛠️ Installation

Clone the repo and install requirements:

```bash
git clone https://github.com/Deepkuamr-9/-Stock-Prediction.git
cd -Stock-Prediction
pip install -r requirements.txt
```

Main dependencies:

```
pandas
numpy
matplotlib
yfinance
scikit-learn
```

---

## 📊 Usage

Run the pipeline in **Colab** or locally:

```bash
python stock_prediction.py
```

Example output:

```
Model: GradientBoostingRegressor
- MAE: 0.012345
- RMSE: 0.023456
- Directional Acc.: 54.32%

Baselines:
- Predict 0:     MAE=0.0129, RMSE=0.0241, DA=48.76%
- Predict lag-1: MAE=0.0121, RMSE=0.0235, DA=52.14%
```

Charts generated:

* Next-day return (actual vs predicted)
* Rebased price path
* Strategy equity curve

---

## 🔮 Quick Inference

Use the helper function to get a **next-day prediction**:

```python
from stock_prediction import predict_next_day_return

res = predict_next_day_return("TSLA")
print(res)
```

Example:

```
{
  'ticker': 'TSLA',
  'last_close': 204.15,
  'predicted_next_day_return': 0.0031,
  'predicted_next_day_close': 204.78
}
---


## ✅ Next Steps

* Add **hyperparameter tuning** (GridSearchCV / Optuna)
* Try **deep learning models** (LSTM, GRU, Transformers)
* Add **sentiment features** (news, Twitter, etc.)
* Improve backtest with **transaction costs & shorting**

