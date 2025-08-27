# ==========================================
# Stock Price Prediction System (Colab-ready)
# ==========================================
# - Predicts next-day return from technical features
# - Evaluates MAE/RMSE, directional accuracy, baselines
# - Plots predictions and a simple long/cash backtest
# - Includes a quick inference function
#
# Note: This is educational; not investment advice.


import warnings, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- Configuration ----------------
TICKER = "META"      # change to e.g., "RELIANCE.NS" for NSE symbols
PERIOD = "10y"       # "5y", "10y", etc.
TEST_SIZE = 0.2      # last 20% as test
RANDOM_STATE = 42

# ---------------- Helpers: Indicators ----------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, window=20, n_std=2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / (mid.replace(0, np.nan))
    return upper, mid, lower, width

# ---------------- Data loading ----------------
def load_data(ticker=TICKER, period=PERIOD):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data received for {ticker}. Check the symbol or period.")
    df = df.rename(columns=str.title)  # ensure 'Close','Open','High','Low','Volume'
    return df

# ---------------- Feature engineering ----------------
def build_features(df):
    out = df.copy()

    # Basic returns
    out["ret_1"]  = np.log(out["Close"] / out["Close"].shift(1))
    out["ret_5"]  = np.log(out["Close"] / out["Close"].shift(5))
    out["ret_10"] = np.log(out["Close"] / out["Close"].shift(10))

    # Rolling stats on returns (momentum/volatility)
    out["ret_1_mean_5"]  = out["ret_1"].rolling(5).mean()
    out["ret_1_std_5"]   = out["ret_1"].rolling(5).std()
    out["ret_1_mean_20"] = out["ret_1"].rolling(20).mean()
    out["ret_1_std_20"]  = out["ret_1"].rolling(20).std()

    # Price-based indicators
    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["ema_10"] = ema(out["Close"], 10)
    out["ema_20"] = ema(out["Close"], 20)

    out["rsi_14"] = rsi(out["Close"], 14)
    macd_line, sig_line, hist = macd(out["Close"])
    out["macd"] = macd_line
    out["macd_signal"] = sig_line
    out["macd_hist"] = hist

    upper, mid, lower, bb_width = bollinger_bands(out["Close"], 20, 2)
    out["bb_width"] = bb_width
    out["bb_pos"] = (out["Close"] - lower) / (upper - lower)  # 0 at lower band, 1 at upper

    # Volume features
    out["vol_z_20"] = (out["Volume"] - out["Volume"].rolling(20).mean()) / (out["Volume"].rolling(20).std())
    out["vol_chg_1"] = out["Volume"].pct_change()

    # Lags of returns (autocorrelation)
    for k in [1, 2, 3]:
        out[f"ret_1_lag{k}"] = out["ret_1"].shift(k)

    # Target: next-day return
    out["target_ret_1_ahead"] = out["ret_1"].shift(-1)

    # Drop initial NaNs
    out = out.dropna().copy()
    return out

# ---------------- Dataset split ----------------
def train_test_split_time(df, test_size=TEST_SIZE):
    n = len(df)
    n_test = int(n * test_size)
    test_idx = df.index[-n_test:]
    train_idx = df.index[:-n_test]
    return train_idx, test_idx

# ---------------- Modeling & evaluation ----------------
def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    # Directional accuracy
    da = (np.sign(y_true) == np.sign(y_pred)).mean()
    return {"MAE": mae, "RMSE": rmse, "Directional_Acc": da}

def baselines(df_feat, train_idx, test_idx):
    # Baseline 1: predict 0 return
    y_test = df_feat.loc[test_idx, "target_ret_1_ahead"].values
    y_pred_zero = np.zeros_like(y_test)
    m0 = evaluate_predictions(y_test, y_pred_zero)

    # Baseline 2: predict previous day's return (lag-1)
    y_pred_lag1 = df_feat.loc[test_idx, "ret_1"].values
    m1 = evaluate_predictions(y_test, y_pred_lag1)
    return m0, m1

def plot_predictions(test_close, y_test, y_pred, ticker):
    # Plot returns
    plt.figure(figsize=(12, 4))
    plt.plot(y_test, label="Actual next-day return")
    plt.plot(y_pred, label="Predicted next-day return", alpha=0.8)
    plt.axhline(0, color='gray', lw=1)
    plt.title(f"{ticker} — Next-day return: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Convert predicted/actual returns to a price path
    # Start from first test close, apply cumulative returns
    p0 = test_close[0]
    act_price = p0 * np.exp(np.cumsum(y_test))
    pred_price = p0 * np.exp(np.cumsum(y_pred))

    plt.figure(figsize=(12, 4))
    plt.plot(act_price, label="Actual (rebased)")
    plt.plot(pred_price, label="Predicted (rebased)", alpha=0.8)
    plt.title(f"{ticker} — Rebased price path from predicted vs actual returns")
    plt.legend()
    plt.tight_layout()
    plt.show()

def simple_backtest(y_test, y_pred, costs_bps=0.0):
    # Long if predicted return > 0, else cash
    signals = (y_pred > 0).astype(int)
    rets = y_test * signals

    # Apply simple transaction cost when position changes
    if costs_bps > 0:
        changes = np.abs(np.diff(signals, prepend=0))
        rets -= changes * (costs_bps / 10000.0)

    equity = (1 + rets).cumprod()
    cagr = equity[-1] ** (252 / len(equity)) - 1  # assuming ~252 trading days/yr
    vol_ann = np.std(rets) * np.sqrt(252)
    sharpe = (np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252)

    plt.figure(figsize=(12, 4))
    plt.plot(equity, label="Equity curve (long/cash)")
    plt.title("Simple strategy: long if predicted return > 0")
    plt.legend(); plt.tight_layout(); plt.show()

    return {"CAGR": cagr, "Vol_Ann": vol_ann, "Sharpe": sharpe}

# ---------------- Run pipeline ----------------
df = load_data(TICKER, PERIOD)
df_feat = build_features(df)

features = [
    "ret_1","ret_5","ret_10",
    "ret_1_mean_5","ret_1_std_5","ret_1_mean_20","ret_1_std_20",
    "sma_10","sma_20","ema_10","ema_20",
    "rsi_14","macd","macd_signal","macd_hist",
    "bb_width","bb_pos","vol_z_20","vol_chg_1",
    "ret_1_lag1","ret_1_lag2","ret_1_lag3"
]

X = df_feat[features].values
y = df_feat["target_ret_1_ahead"].values

train_idx, test_idx = train_test_split_time(df_feat, TEST_SIZE)

X_train, y_train = X[df_feat.index.get_indexer(train_idx)], y[df_feat.index.get_indexer(train_idx)]
X_test,  y_test  = X[df_feat.index.get_indexer(test_idx)],  y[df_feat.index.get_indexer(test_idx)]

# Models to try (choose one)
gbm = GradientBoostingRegressor(random_state=RANDOM_STATE)
rf  = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)

model = gbm  # switch to rf if you prefer
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
metrics = evaluate_predictions(y_test, y_pred)
m0, m1 = baselines(df_feat, train_idx, test_idx)

print(f"Model: {model.__class__.__name__}")
print(f"- MAE:              {metrics['MAE']:.6f}")
print(f"- RMSE:             {metrics['RMSE']:.6f}")
print(f"- Directional Acc.: {metrics['Directional_Acc']*100:.2f}%")
print("\nBaselines:")
print(f"- Predict 0:        MAE={m0['MAE']:.6f}, RMSE={m0['RMSE']:.6f}, DA={m0['Directional_Acc']*100:.2f}%")
print(f"- Predict lag-1:    MAE={m1['MAE']:.6f}, RMSE={m1['RMSE']:.6f}, DA={m1['Directional_Acc']*100:.2f}%")

# Visuals
test_close = df_feat.loc[test_idx, "Close"].values
plot_predictions(test_close, y_test, y_pred, TICKER)

bt = simple_backtest(y_test, y_pred, costs_bps=0.0)
print("\nBacktest (naive long/cash, no costs):")
print(f"- CAGR:   {bt['CAGR']*100:.2f}%")
print(f"- VolAnn: {bt['Vol_Ann']*100:.2f}%")
print(f"- Sharpe: {bt['Sharpe']:.2f}")

# ---------------- Quick inference ----------------
def predict_next_day_return(ticker=TICKER, period="3y", use_model=model, feature_list=features):
    df2 = load_data(ticker, period)
    df2f = build_features(df2)
    X2 = df2f[feature_list].values
    last_row = X2[-1].reshape(1, -1)
    pred_ret = float(use_model.predict(last_row)[0])
    last_close = float(df2f["Close"].iloc[-1])
    pred_price = last_close * math.exp(pred_ret)
    return {
        "ticker": ticker,
        "last_close": last_close,
        "predicted_next_day_return": pred_ret,
        "predicted_next_day_close": pred_price
    }

res = predict_next_day_return(TICKER)
print(f"\nNext-day prediction for {res['ticker']}:")
print(f"- Last close:                 {res['last_close']:.2f}")
print(f"- Predicted next-day return:  {res['predicted_next_day_return']:.4%}")
print(f"- Predicted next-day close:   {res['predicted_next_day_close']:.2f}")