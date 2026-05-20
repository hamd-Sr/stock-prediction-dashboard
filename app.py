"""Intraday Stock Prediction Dashboard

Run:
    pip install streamlit yfinance pandas numpy scikit-learn plotly
    streamlit run intraday_stock_dashboard.py

Notes:
- Educational use only. Not financial advice.
- Yahoo Finance intraday data is limited and can be delayed.
- The app benchmarks a few lightweight models and picks the best one on a time-based validation split.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Intraday Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Intraday Stock Prediction Dashboard")
st.caption(
    "A Streamlit dashboard for intraday market visualization and next-bar direction prediction."
)


# -----------------------------
# Utilities
# -----------------------------

def normalize_ticker(symbol: str, market: str) -> str:
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    suffix = ".NS" if market == "NSE" else ".BO"
    return f"{symbol}{suffix}"


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance columns to Open/High/Low/Close/Volume."""
    out = df.copy()

    def find_col(target: str):
        target_l = target.lower()
        for col in out.columns:
            col_s = str(col).replace(" ", "_").lower()
            if col_s == target_l:
                return col
        for col in out.columns:
            col_s = str(col).replace(" ", "_").lower()
            if col_s.endswith(f"_{target_l}") or col_s.startswith(f"{target_l}_") or target_l in col_s:
                return col
        return None

    rename_map = {}
    for canonical in ["Open", "High", "Low", "Close", "Volume"]:
        source = find_col(canonical)
        if source is not None and str(source) != canonical:
            rename_map[source] = canonical

    if rename_map:
        out = out.rename(columns=rename_map)

    cleaned = []
    for c in out.columns:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if p not in ("", None)]
            cleaned.append("_".join(parts) if parts else str(c))
        else:
            cleaned.append(str(c).replace(" ", "_"))
    out.columns = cleaned
    return out


@st.cache_data(ttl=300)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = standardize_ohlcv_columns(df)
    df = df.dropna(how="all")
    return df


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.copy()

    if "Close" not in data.columns:
        raise ValueError("The downloaded data does not contain a Close column.")

    close = data["Close"]
    volume = data["Volume"] if "Volume" in data.columns else pd.Series(index=data.index, dtype=float)

    # Price/return features
    data["ret_1"] = close.pct_change()
    data["log_ret_1"] = np.log(close / close.shift(1))
    data["ret_3"] = close.pct_change(3)
    data["ret_5"] = close.pct_change(5)

    # Trend features
    for w in (5, 10, 20, 50):
        data[f"sma_{w}"] = close.rolling(w).mean()
        data[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        data[f"vol_{w}"] = data["ret_1"].rolling(w).std()
        data[f"mom_{w}"] = close - close.shift(w)
        data[f"pct_from_sma_{w}"] = close / data[f"sma_{w}"] - 1

    # Momentum indicators
    data["rsi_14"] = rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # Volatility / range
    data["hl_range"] = (data["High"] - data["Low"]) / close
    data["oc_range"] = (data["Open"] - close) / close
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data["bb_width"] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid
    data["bb_z"] = (close - bb_mid) / bb_std

    # Volume features
    if "Volume" in data.columns:
        data["vol_chg"] = volume.pct_change()
        data["vol_sma_20"] = volume.rolling(20).mean()
        data["vol_z"] = (volume - data["vol_sma_20"]) / volume.rolling(20).std()
        data["price_vol"] = data["ret_1"] * data["vol_chg"].fillna(0)

    # Calendar / session features
    idx = pd.DatetimeIndex(data.index)
    data["dow"] = idx.dayofweek
    data["hour"] = idx.hour
    data["minute"] = idx.minute
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["sin_minute"] = np.sin(2 * np.pi * data["minute"] / 60)
    data["cos_minute"] = np.cos(2 * np.pi * data["minute"] / 60)

    # Prediction target: whether next close is higher than current close
    data["target"] = (close.shift(-1) > close).astype(int)

    # Drop rows with incomplete features or unknown target
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    y = data.pop("target")
    return data, y


@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    f1: float
    roc_auc: float


def train_best_model(X: pd.DataFrame, y: pd.Series) -> Tuple[ModelResult, pd.DataFrame]:
    split = max(int(len(X) * 0.8), 50)
    split = min(split, len(X) - 1)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    candidates: Dict[str, object] = {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_iter=250,
            max_depth=4,
            min_samples_leaf=25,
            l2_regularization=0.1,
            random_state=42,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
    }

    scored: list[ModelResult] = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        roc = roc_auc_score(y_test, proba) if proba is not None and len(np.unique(y_test)) > 1 else np.nan
        scored.append(ModelResult(name=name, model=model, accuracy=acc, f1=f1, roc_auc=roc))

    scored = sorted(scored, key=lambda r: (r.accuracy, r.f1), reverse=True)
    best = scored[0]

    eval_df = pd.DataFrame(
        {
            "model": [r.name for r in scored],
            "accuracy": [r.accuracy for r in scored],
            "f1": [r.f1 for r in scored],
            "roc_auc": [r.roc_auc for r in scored],
        }
    )
    eval_df["accuracy"] = eval_df["accuracy"].round(4)
    eval_df["f1"] = eval_df["f1"].round(4)
    eval_df["roc_auc"] = eval_df["roc_auc"].round(4)
    return best, eval_df


def make_candlestick(df: pd.DataFrame, ma_window: int = 20) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"].rolling(ma_window).mean(),
            mode="lines",
            name=f"MA {ma_window}",
        )
    )
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )
    return fig


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    market = st.selectbox("Market", ["NSE", "BSE"], index=0)
    ticker_input = st.text_input("Ticker", value="RELIANCE")
    period = st.selectbox("History window", ["5d", "7d", "30d", "60d", "3mo"], index=3)
    interval = st.selectbox("Candles", ["1m", "2m", "5m", "15m", "30m", "60m"], index=2)
    threshold = st.slider("Signal confidence threshold", 0.50, 0.75, 0.55, 0.01)
    refresh = st.button("Refresh data")

    st.markdown("---")
    st.write("Default model: **HistGradientBoostingClassifier**")
    st.caption("The app also benchmarks Logistic Regression and keeps the better validation result.")


symbol = normalize_ticker(ticker_input, market)

if refresh:
    st.cache_data.clear()

st.subheader(f"Live view for {symbol}")

raw = load_data(symbol, period, interval)
if raw.empty:
    st.error(
        "No data returned. Check the ticker, market suffix, period, or interval. "
        "For intraday data, Yahoo Finance may limit how much history is available."
    )
    st.stop()

if len(raw) < 120:
    st.warning("Very small dataset returned. Predictions may be unstable.")

# Use last rows only for display if the dataset is huge
plot_df = raw.tail(300).copy()
fig = make_candlestick(plot_df)
st.plotly_chart(fig, use_container_width=True)

# Feature engineering and model training
try:
    feat_df, y = add_features(raw)
except Exception as exc:
    st.error(f"Feature engineering failed: {exc}")
    st.stop()

if len(feat_df) < 120:
    st.error("Not enough rows after feature engineering to train a reliable model.")
    st.stop()

best_model, leaderboard = train_best_model(feat_df, y)

# Current prediction from the latest row
latest_X = feat_df.iloc[[-1]]
prob_up = float(best_model.model.predict_proba(latest_X)[0, 1]) if hasattr(best_model.model, "predict_proba") else 0.0
if prob_up >= threshold:
    signal = "BUY"
elif prob_up <= 1 - threshold:
    signal = "SELL"
else:
    signal = "HOLD"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{feat_df['Close'].iloc[-1]:.2f}")
c2.metric("Up probability", f"{prob_up:.2%}")
c3.metric("Signal", signal)
c4.metric("Best model", best_model.name)

st.markdown("### Model leaderboard")
st.dataframe(leaderboard, use_container_width=True, hide_index=True)

# Backtest on the hold-out split
split = max(int(len(feat_df) * 0.8), 50)
split = min(split, len(feat_df) - 1)
X_train, X_test = feat_df.iloc[:split], feat_df.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = best_model.model
pred_test = model.predict(X_test)
proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.full(len(X_test), 0.5)

results = pd.DataFrame(index=X_test.index)
results["close"] = feat_df.loc[X_test.index, "Close"]
results["future_return"] = feat_df["Close"].shift(-1).loc[X_test.index] / feat_df.loc[X_test.index, "Close"] - 1
results["prob_up"] = proba_test
results["position"] = np.where(results["prob_up"] >= threshold, 1, np.where(results["prob_up"] <= 1 - threshold, -1, 0))
results["strategy_return"] = results["position"] * results["future_return"]
results["buy_hold"] = results["future_return"]
results = results.dropna()
results["strategy_curve"] = (1 + results["strategy_return"]).cumprod()
results["buy_hold_curve"] = (1 + results["buy_hold"]).cumprod()

acc = accuracy_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test, zero_division=0)
roc = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else np.nan

m1, m2, m3 = st.columns(3)
m1.metric("Validation accuracy", f"{acc:.2%}")
m2.metric("Validation F1", f"{f1:.2%}")
m3.metric("Validation ROC-AUC", "N/A" if np.isnan(roc) else f"{roc:.2%}")

st.markdown("### Confusion matrix")
cm = confusion_matrix(y_test, pred_test)
st.write(pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"]))

st.markdown("### Strategy curve vs buy-and-hold")
curve = go.Figure()
curve.add_trace(go.Scatter(x=results.index, y=results["strategy_curve"], mode="lines", name="Strategy"))
curve.add_trace(go.Scatter(x=results.index, y=results["buy_hold_curve"], mode="lines", name="Buy & Hold"))
curve.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(curve, use_container_width=True)

st.markdown("### Recent predictions")
show_cols = ["Close", "prob_up", "position", "strategy_return", "future_return"]
st.dataframe(results[show_cols].tail(20).round(4), use_container_width=True)

with st.expander("Model report"):
    st.text(classification_report(y_test, pred_test, zero_division=0))

st.info(
    "This dashboard is for research and education. Intraday market forecasting is noisy, and any signal should be validated with out-of-sample testing, transaction costs, and risk controls before real use."
)
