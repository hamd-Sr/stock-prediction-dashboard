"""Intraday Stock Prediction Dashboard

Run locally:
    pip install streamlit yfinance pandas numpy scikit-learn plotly
    streamlit run app.py

Notes:
- Educational use only. Not financial advice.
- Yahoo Finance intraday data is limited and may be delayed.
- This app benchmarks lightweight models and picks the best one on a time-based split.
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
st.caption("A Streamlit dashboard for intraday market visualization and next-bar direction prediction.")


# -----------------------------
# Utilities
# -----------------------------
def normalize_ticker(symbol: str, market: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        return "RELIANCE.NS"
    if "." in symbol:
        return symbol
    suffix = ".NS" if market == "NSE" else ".BO"
    return f"{symbol}{suffix}"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        flat_cols = []
        for col in out.columns.to_flat_index():
            parts = [str(part) for part in col if part not in ("", None)]
            flat_cols.append("_".join(parts) if parts else str(col))
        out.columns = flat_cols
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance columns to exact Open/High/Low/Close/Volume names."""
    out = _flatten_columns(df)

    normalized = {str(c).replace(" ", "_").lower(): c for c in out.columns}

    def find_col(target: str):
        target_l = target.lower()
        exact = normalized.get(target_l)
        if exact is not None:
            return exact
        for norm_name, original in normalized.items():
            if norm_name.endswith(f"_{target_l}") or norm_name.startswith(f"{target_l}_") or target_l in norm_name:
                return original
        return None

    rename_map = {}
    for canonical in ["Open", "High", "Low", "Close", "Volume"]:
        source = find_col(canonical)
        if source is not None and source != canonical:
            rename_map[source] = canonical

    out = out.rename(columns=rename_map)
    out.columns = [str(c).replace(" ", "_") for c in out.columns]

    # Fallback: if the columns are still weird, map first five positionally.
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing and len(out.columns) >= 5:
        pos_map = {}
        first_five = list(out.columns[:5])
        for i, canonical in enumerate(required):
            if canonical not in out.columns and i < len(first_five):
                pos_map[first_five[i]] = canonical
        out = out.rename(columns=pos_map)

    return out


def get_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_ohlcv_columns(df)
    needed = ["Open", "High", "Low", "Close"]
    if not all(c in out.columns for c in needed):
        raise ValueError(f"Could not standardize OHLC columns. Found: {list(out.columns)}")

    if "Volume" not in out.columns:
        out["Volume"] = np.nan

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
    data = get_price_frame(df).copy()

    close = data["Close"]
    volume = data["Volume"]

    data["ret_1"] = close.pct_change()
    data["log_ret_1"] = np.log(close / close.shift(1))
    data["ret_3"] = close.pct_change(3)
    data["ret_5"] = close.pct_change(5)

    for w in (5, 10, 20, 50):
        data[f"sma_{w}"] = close.rolling(w).mean()
        data[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        data[f"vol_{w}"] = data["ret_1"].rolling(w).std()
        data[f"mom_{w}"] = close - close.shift(w)
        data[f"pct_from_sma_{w}"] = close / data[f"sma_{w}"] - 1

    data["rsi_14"] = rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    data["hl_range"] = (data["High"] - data["Low"]) / close
    data["oc_range"] = (data["Open"] - close) / close
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data["bb_width"] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid
    data["bb_z"] = (close - bb_mid) / bb_std

    data["vol_chg"] = volume.pct_change()
    data["vol_sma_20"] = volume.rolling(20).mean()
    data["vol_z"] = (volume - data["vol_sma_20"]) / volume.rolling(20).std()
    data["price_vol"] = data["ret_1"] * data["vol_chg"].fillna(0)

    idx = pd.DatetimeIndex(data.index)
    data["dow"] = idx.dayofweek
    data["hour"] = idx.hour
    data["minute"] = idx.minute
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["sin_minute"] = np.sin(2 * np.pi * data["minute"] / 60)
    data["cos_minute"] = np.cos(2 * np.pi * data["minute"] / 60)

    data["target"] = (close.shift(-1) > close).astype(int)

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
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = np.full(len(X_test), 0.5)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        roc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
        scored.append(ModelResult(name=name, model=model, accuracy=acc, f1=f1, roc_auc=roc))

    scored = sorted(scored, key=lambda r: (r.accuracy, r.f1), reverse=True)
    eval_df = pd.DataFrame(
        {
            "model": [r.name for r in scored],
            "accuracy": [round(r.accuracy, 4) for r in scored],
            "f1": [round(r.f1, 4) for r in scored],
            "roc_auc": [None if np.isnan(r.roc_auc) else round(r.roc_auc, 4) for r in scored],
        }
    )
    return scored[0], eval_df


def make_candlestick(df: pd.DataFrame, ma_window: int = 20) -> go.Figure:
    price = get_price_frame(df)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=price.index,
            open=price["Open"],
            high=price["High"],
            low=price["Low"],
            close=price["Close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price["Close"].rolling(ma_window).mean(),
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
        "Yahoo Finance may limit intraday history."
    )
    st.stop()

if len(raw) < 120:
    st.warning("Very small dataset returned. Predictions may be unstable.")

try:
    fig = make_candlestick(raw.tail(300).copy())
except Exception as exc:
    st.error(f"Could not build price chart: {exc}")
    st.write("Columns received from Yahoo:")
    st.write(list(raw.columns))
    st.stop()

st.plotly_chart(fig, use_container_width=True)

try:
    feat_df, y = add_features(raw)
except Exception as exc:
    st.error(f"Feature engineering failed: {exc}")
    st.write("Columns received from Yahoo:")
    st.write(list(raw.columns))
    st.stop()

if len(feat_df) < 120:
    st.error("Not enough rows after feature engineering to train a reliable model.")
    st.stop()

best_model, leaderboard = train_best_model(feat_df, y)

latest_X = feat_df.iloc[[-1]]
if hasattr(best_model.model, "predict_proba"):
    prob_up = float(best_model.model.predict_proba(latest_X)[0, 1])
else:
    prob_up = 0.5

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

split = max(int(len(feat_df) * 0.8), 50)
split = min(split, len(feat_df) - 1)

X_train, X_test = feat_df.iloc[:split], feat_df.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = best_model.model
pred_test = model.predict(X_test)
proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.full(len(X_test), 0.5)

results = pd.DataFrame(index=X_test.index)
results["Close"] = feat_df.loc[X_test.index, "Close"]
results["future_return"] = feat_df["Close"].shift(-1).loc[X_test.index] / feat_df.loc[X_test.index, "Close"] - 1
results["prob_up"] = proba_test
results["position"] = np.where(
    results["prob_up"] >= threshold,
    1,
    np.where(results["prob_up"] <= 1 - threshold, -1, 0)
)
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
    "This dashboard is for research and education. Intraday market forecasting is noisy, and any signal should be validated "
    "with out-of-sample testing, transaction costs, and risk controls before real use."
)
