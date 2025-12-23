import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta
from scipy.stats import norm
import requests
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
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    return gamma, delta

# -------------------------
# Data Fetcher (Resilient)
# -------------------------
@st.cache_data(ttl=300)
def fetch_data_safe(ticker, max_exp):
    if ticker.upper() in ["SPX", "NDX", "RUT"] and not ticker.startswith("^"):
        ticker = f"^{ticker.upper()}"

    stock = yf.Ticker(ticker)

    try:
        hist = stock.history(period="5d")
        if hist.empty:
            return None, None
        S = hist["Close"].iloc[-1]

        all_exps = stock.options
        if not all_exps:
            return S, None
        target_exps = all_exps[:max_exp]

        dfs = []
        progress_text = st.empty()
        prog_bar = st.progress(0)

        for i, exp in enumerate(target_exps):
            progress_text.text(f"Fetching expiry: {exp} ({i+1}/{len(target_exps)})")
            success = False
            for attempt in range(4):
                try:
                    time.sleep(random.uniform(0.5, 1.0))
                    chain = stock.option_chain(exp)
                    t_date = get_actual_trading_day(exp)
                    calls = chain.calls.assign(option_type="call", expiration=t_date)
                    puts = chain.puts.assign(option_type="put", expiration=t_date)
                    dfs.append(pd.concat([calls, puts], ignore_index=True))
                    success = True
                    break
                except Exception:
                    time.sleep(2)

            prog_bar.progress((i + 1) / len(target_exps))

        progress_text.empty()
        prog_bar.empty()

        if not dfs:
            return S, None
        return S, pd.concat(dfs, ignore_index=True)

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None, None

# -------------------------
# GEX & DEX Processing
# -------------------------
def process_exposure(df, S, s_range, model_type):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Aligning time to Market Close (4 PM)
    now = pd.Timestamp.now().normalize() + pd.Timedelta(hours=16)
    df["expiry_dt"] = pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16)
    df["T"] = (df["expiry_dt"] - now).dt.total_seconds() / (365 * 24 * 3600)

    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range) & (df["T"] > 0)]

    res = []
    for _, row in df.iterrows():
        K, T = float(row["strike"]), max(float(row["T"]), 0.00001)
        liq = (row.get("openInterest") or 0) if (row.get("openInterest") or 0) > 0 else (row.get("volume") or 0)
        
        if liq <= 0: continue

        iv = row["impliedVolatility"]
        if not (0.05 < iv < 4.0): iv = FALLBACK_IV

        gamma, delta = get_greeks(S, K, RISK_FREE_RATE, iv, T, row["option_type"])

        # MODEL SELECTION
        if model_type == "Dealer Short All (Absolute Stress)":
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * liq
        else:
            # Short Calls / Long Puts (Claude's Logic)
            if row["option_type"] == "call":
                gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * liq
            else:
                gex = gamma * S**2 * 0.01 * CONTRACT_SIZE * liq

        dex = -delta * S * CONTRACT_SIZE * liq

        res.append({"strike": K, "expiry": row["expiration"], "gex": gex, "dex": dex})

    return pd.DataFrame(res)

# -------------------------
# Visualizations
# -------------------------
def render_plots(df, ticker, S, mode, boost):
    if df.empty: return None, None

    val_col = mode.lower()
    agg = df.groupby('strike')[val_col].sum().sort_index()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum', fill_value=0).sort_index(ascending=False)

    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** (1.0 / boost))

    colorscale = 'RdBu' if mode == "DEX" else [[0, '#32005A'], [0.3, '#BF00FF'], [0.5, '#000000'], [0.7, '#FFB400'], [1, '#FFFFB4']]
    
    fig_h = go.Figure(data=go.Heatmap(
        z=z_scaled, x=pivot.columns, y=pivot.index,
        colorscale=colorscale, zmid=0, colorbar=dict(title=mode)
    ))

    fig_h.update_layout(
        title=f"{ticker} {mode} Exposure Map", template="plotly_dark", height=750,
        xaxis=dict(type='category', title="Expiration"), yaxis=dict(title="Strike")
    )
    fig_h.add_hline(y=S, line_dash="dash", line_color="cyan", annotation_text=f"Spot: {S:.2f}")

    fig_b = go.Figure(go.Bar(x=agg.index, y=agg.values, marker_color=['#FF00FF' if v < 0 else '#FFFF00' for v in agg.values]))
    fig_b.update_layout(title=f"Total {mode} by Strike", template="plotly_dark", height=400)

    return fig_h, fig_b

# -------------------------
# Main App Interface
# -------------------------
def main():
    st.title("📈 GEX / DEX Pro - Dealer Exposure")
    
    with st.sidebar:
        st.header("Control Panel")
        ticker = st.text_input("Ticker", "SPY").upper().strip()
        mode = st.radio("Metric", ["GEX", "DEX"])
        model_type = st.selectbox("Dealer Model", ["Dealer Short All (Absolute Stress)", "Short Calls / Long Puts"])
        
        max_exp = st.slider("Max Expirations", 1, 15, 6)
        
        # SLIDER CUSTOMIZATION: Min 10, Max 100, Default 25
        # Step is 1 to allow 25, but UI will show intervals naturally
        s_range = st.slider("Strike Range ± Spot", 10, 100, 25, step=1)
        
        boost = st.slider("Heatmap Contrast Boost", 1.0, 5.0, 2.5)
        run = st.button("Calculate Exposure", type="primary")

    if run:
        with st.spinner(f"Analyzing {ticker} flow..."):
            S, raw_df = fetch_data_safe(ticker, max_exp)

        if S and not raw_df.empty:
            st.success(f"{ticker} Trading at ${S:.2f}")
            processed = process_exposure(raw_df, S, s_range, model_type)
            
            if not processed.empty:
                t_gex = processed["gex"].sum() / 1e9
                t_dex = processed["dex"].sum() / 1e9
                
                c1, c2 = st.columns(2)
                c1.metric("Net Dealer GEX", f"${t_gex:.2f}B")
                c2.metric("Net Dealer DEX", f"${t_dex:.2f}B")

                h_fig, b_fig = render_plots(processed, ticker, S, mode, boost)
                
                # FIXED: Replacement of use_container_width with width="container"
                st.plotly_chart(h_fig, width="container")
                st.plotly_chart(b_fig, width="container")
            else:
                st.warning("No liquidity found in that range.")
        else:
            st.error("Fetch failed. Please check ticker or try again later.")

if __name__ == "__main__":
    main()