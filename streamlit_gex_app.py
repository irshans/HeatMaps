import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta, datetime
from scipy.stats import norm
import time
import random

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

CONTRACT_SIZE = 100
FALLBACK_IV = 0.30
RISK_FREE_RATE = 0.042

# -------------------------
# Holiday & Date Logic
# -------------------------
def get_actual_trading_day(date_str):
    dt = pd.to_datetime(date_str)
    holidays = [
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
        '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
        '2026-01-01'
    ]
    while dt.weekday() > 4:
        dt -= timedelta(days=1)
    while dt.strftime('%Y-%m-%d') in holidays:
        dt -= timedelta(days=1)
        while dt.weekday() > 4:
            dt -= timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

# -------------------------
# Black-Scholes Math
# -------------------------
def get_greeks(S, K, r, sigma, T, option_type):
    if T <= 0: # Handle 0DTE decay
        T = 0.00001 
    if sigma <= 0:
        sigma = FALLBACK_IV
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    return gamma, delta

# -------------------------
# Enhanced Data Fetcher
# -------------------------
@st.cache_data(ttl=300)
def fetch_data_safe(ticker, max_exp):
    # Handle Index Tickers
    if ticker.upper() == "SPX":
        # Check both AM (^SPX) and PM (^SPXW) weekly symbols
        ticker = "^SPX" 
    elif ticker.upper() in ["NDX", "RUT"] and not ticker.startswith("^"):
        ticker = f"^{ticker.upper()}"

    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="2d")
        if hist.empty: return None, None
        S = hist["Close"].iloc[-1]

        all_exps = stock.options
        if not all_exps: return S, None
        
        # Ensure we grab the first available (usually 0DTE or 1DTE)
        target_exps = all_exps[:max_exp]

        dfs = []
        progress_text = st.empty()
        prog_bar = st.progress(0)

        for i, exp in enumerate(target_exps):
            progress_text.text(f"Fetching {ticker} expiry: {exp}")
            chain = stock.option_chain(exp)
            t_date = get_actual_trading_day(exp)
            calls = chain.calls.assign(option_type="call", expiration=t_date)
            puts = chain.puts.assign(option_type="put", expiration=t_date)
            dfs.append(pd.concat([calls, puts], ignore_index=True))
            prog_bar.progress((i + 1) / len(target_exps))

        progress_text.empty()
        prog_bar.empty()
        return S, pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None, None

# -------------------------
# GEX & DEX Processing
# -------------------------
def process_exposure(df, S, s_range, model_type):
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    now = pd.Timestamp.now().normalize() + pd.Timedelta(hours=16)
    df["expiry_dt"] = pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16)
    df["T"] = (df["expiry_dt"] - now).dt.total_seconds() / (365 * 24 * 3600)
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)]

    res = []
    for _, row in df.iterrows():
        K, T = float(row["strike"]), max(float(row["T"]), 0.00001)
        liq = row.get("openInterest", 0) or row.get("volume", 0)
        if liq <= 0: continue
        iv = row["impliedVolatility"] if (0.05 < row["impliedVolatility"] < 4.0) else FALLBACK_IV

        gamma, delta = get_greeks(S, K, RISK_FREE_RATE, iv, T, row["option_type"])
        
        # Model Logic
        if model_type == "Dealer Short All (Absolute Stress)":
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * liq
        else:
            gex = (-gamma if row["option_type"] == "call" else gamma) * S**2 * 0.01 * CONTRACT_SIZE * liq

        dex = -delta * S * CONTRACT_SIZE * liq
        res.append({"strike": K, "expiry": row["expiration"], "gex": gex, "dex": dex})

    return pd.DataFrame(res)

# -------------------------
# Visualizations
# -------------------------
def render_plots(df, ticker, S, mode):
    if df.empty: return None, None
    val_col = mode.lower()
    agg = df.groupby('strike')[val_col].sum().sort_index()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum', fill_value=0).sort_index(ascending=False)

    x_labs, y_labs = pivot.columns.tolist(), pivot.index.tolist()
    
    # Tooltip and Annotation Formatting
    h_text = []
    for i, strike in enumerate(y_labs):
        row = []
        for j, val in enumerate(pivot.values[i]):
            prefix = "-" if val < 0 else ""
            v_abs = abs(val)
            f_val = f"{prefix}${v_abs/1e6:,.2f}M" if v_abs >= 1e6 else f"{prefix}${v_abs/1e3:,.1f}K"
            row.append(f"Strike: ${strike:,.0f}<br>Expiry: {x_labs[j]}<br>{mode}: {f_val}")
        h_text.append(row)

    # Plotly Heatmap
    z_scaled = np.sign(pivot.values) * (np.abs(pivot.values) ** (1.0 / 2.0))
    fig_h = go.Figure(data=go.Heatmap(z=z_scaled, x=x_labs, y=y_labs, text=h_text, hoverinfo="text", colorscale='Viridis', zmid=0))

    # Add Cell Annotations
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = pivot.values[i, j]
            if abs(val) < 500: continue
            prefix = "-" if val < 0 else ""
            txt = f"{prefix}${abs(val)/1e3:,.0f}K"
            fig_h.add_annotation(x=exp, y=strike, text=txt, showarrow=False, font=dict(color="white", size=12), xref="x", yref="y")

    fig_h.update_layout(title=f"{ticker} {mode} Exposure | Spot: ${S:,.2f}", template="plotly_dark", height=800)
    fig_h.add_hline(y=S, line_dash="solid", line_color="yellow", annotation_text=f"SPOT: ${S:,.2f}")

    return fig_h, None

def main():
    st.title("📈 GEX / DEX Pro (0DTE Enabled)")
    with st.sidebar:
        ticker = st.text_input("Ticker", "SPY").upper()
        mode = st.radio("Metric", ["GEX", "DEX"])
        model_type = st.selectbox("Dealer Model", ["Dealer Short All (Absolute Stress)", "Short Calls / Long Puts"])
        max_exp = st.slider("Max Expirations", 1, 10, 5)
        s_range = st.slider("Strike Range", 5, 100, 20)
        run = st.button("Calculate", type="primary")

    if run:
        S, raw_df = fetch_data_safe(ticker, max_exp)
        if S and raw_df is not None:
            processed = process_exposure(raw_df, S, s_range, model_type)
            t_gex = processed["gex"].sum() / 1e9
            prefix = "-" if t_gex < 0 else ""
            st.metric(f"Net Dealer {mode}", f"{prefix}${abs(t_gex):,.2f}B")
            h_fig, _ = render_plots(processed, ticker, S, mode)
            st.plotly_chart(h_fig, use_container_width=True)

if __name__ == "__main__":
    main()