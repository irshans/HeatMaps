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

st.set_page_config(page_title="GEX Pro - Stable", page_icon="📊", layout="wide")

CONTRACT_SIZE = 100
FALLBACK_IV = 0.25

# -------------------------
# Fail-Safe Session
# -------------------------
@st.cache_resource
def get_global_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    })
    return session

# -------------------------
# Holiday Rollback
# -------------------------
def get_actual_trading_day(date_str):
    dt = pd.to_datetime(date_str)
    holidays = ['2025-12-25', '2026-01-01', '2026-01-19'] # Update as needed
    while dt.weekday() > 4: dt -= timedelta(days=1)
    while dt.strftime('%Y-%m-%d') in holidays:
        dt -= timedelta(days=1)
        while dt.weekday() > 4: dt -= timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

# -------------------------
# Math
# -------------------------
def get_greeks(S, K, r, sigma, T, option_type):
    if T <= 0 or sigma <= 0: return 0.0, 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    return gamma, delta

# -------------------------
# The Resilient Fetcher
# -------------------------
@st.cache_data(ttl=300)
def fetch_data_safe(ticker, max_exp):
    # Auto-prefix indices
    if ticker.upper() in ["SPX", "NDX", "RUT"] and not ticker.startswith("^"):
        ticker = f"^{ticker}"
    
    session = get_global_session()
    stock = yf.Ticker(ticker, session=session)
    
    try:
        # 1. Fetch Spot
        hist = stock.history(period="2d") # Fetch 2 days to ensure we get 'last'
        if hist.empty: return None, None
        S = hist["Close"].iloc[-1]
        
        # 2. Fetch Expirations
        all_exps = stock.options
        if not all_exps: return S, None
        target_exps = all_exps[:max_exp]
        
        dfs = []
        progress_text = st.empty()
        prog_bar = st.progress(0)
        
        for i, exp in enumerate(target_exps):
            progress_text.text(f"Fetching Expiry: {exp} ({i+1}/{len(target_exps)})")
            
            # Retry logic for each individual chain
            success = False
            for attempt in range(3):
                try:
                    time.sleep(random.uniform(0.3, 0.7)) # Be polite to Yahoo
                    chain = stock.option_chain(exp)
                    t_date = get_actual_trading_day(exp)
                    c = chain.calls.assign(option_type="call", expiration=t_date)
                    p = chain.puts.assign(option_type="put", expiration=t_date)
                    dfs.append(pd.concat([c, p], ignore_index=True))
                    success = True
                    break
                except Exception:
                    time.sleep(1) # Wait longer on fail
            
            prog_bar.progress((i + 1) / len(target_exps))
            if not success:
                st.warning(f"Skipped {exp} due to connection timeout.")

        progress_text.empty()
        prog_bar.empty()
        
        if not dfs: return S, None
        return S, pd.concat(dfs, ignore_index=True)

    except Exception as e:
        st.error(f"Fetcher encountered an error: {e}")
        return None, None

# -------------------------
# GEX Calculation
# -------------------------
def process_exposure(df, S, s_range):
    df = df.copy()
    now = pd.Timestamp.now(tz='US/Eastern').replace(tzinfo=None)
    df["expiry_dt"] = pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16)
    df["T"] = (df["expiry_dt"] - now).dt.total_seconds() / (365*24*3600)
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range) & (df["T"] > -0.0001)]
    
    res = []
    for _, row in df.iterrows():
        K, T = float(row["strike"]), max(float(row["T"]), 0.00001)
        liq = (row.get("openInterest") or 0) if (row.get("openInterest") or 0) > 0 else (row.get("volume") or 0)
        if liq <= 0: continue
        
        iv = row.get("impliedVolatility") if row.get("impliedVolatility") and row.get("impliedVolatility") > 0 else FALLBACK_IV
        gamma, delta = get_greeks(S, K, 0.045, iv, T, row["option_type"])
        
        gex = (gamma * S**2) * 0.01 * CONTRACT_SIZE * liq
        dex = (delta * S) * CONTRACT_SIZE * liq
        if row["option_type"] == "put": gex *= -1
            
        res.append({"strike": K, "expiry": row["expiration"], "gex": gex, "dex": dex})
    return pd.DataFrame(res)

# -------------------------
# Visuals
# -------------------------
def render_plots(df, ticker, S, mode, boost):
    val_col = 'gex' if mode == "Gamma" else 'dex'
    # Aggregation
    agg = df.groupby('strike')[val_col].sum().sort_index()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum', fill_value=0).sort_index(ascending=False)
    
    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** (1.0 / boost))
    x_labs, y_labs = pivot.columns.tolist(), pivot.index.tolist()
    
    # Hover Text Generation
    h_text = []
    for i, strike in enumerate(y_labs):
        row = [f"Expiry: {ex}<br>Strike: {strike}<br>Net {mode}: ${z_raw[i, j]:,.0f}" for j, ex in enumerate(x_labs)]
        h_text.append(row)

    # Heatmap
    fig_h = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labs, y=y_labs, text=h_text, hoverinfo="text",
        colorscale='RdBu' if mode == "Delta" else [[0,'#32005A'],[0.25,'#BF00FF'],[0.5,'#000000'],[0.75,'#FFB400'],[1,'#FFFFB4']],
        zmid=0
    ))
    
    # Annotations for top/bottom 10% of values to keep it clean
    threshold = np.max(np.abs(z_raw)) * 0.15
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < threshold: continue
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            fig_h.add_annotation(x=exp, y=strike, text=txt, showarrow=False, font=dict(color="white", size=9))

    fig_h.update_layout(title=f"{ticker} {mode} Heatmap", template="plotly_dark", height=700, xaxis={'type': 'category'})
    fig_h.add_hline(y=S, line_dash="dash", line_color="cyan", annotation_text=f"SPOT: {S:.2f}")

    # Bar
    fig_b = go.Figure(go.Bar(x=agg.index, y=agg.values, marker_color=['#BF00FF' if x < 0 else '#FFD700' for x in agg.values]))
    fig_b.update_layout(title=f"Total {mode} by Strike", template="plotly_dark", height=400)
    
    return fig_h, fig_b

# -------------------------
# App Interface
# -------------------------
def main():
    st.sidebar.header("GEX Analytics Settings")
    ticker = st.sidebar.text_input("Ticker (TSLA, SPY, SPX)", "TSLA").upper()
    mode = st.sidebar.radio("View Mode", ["Gamma", "Delta"])
    max_exp = st.sidebar.slider("Expirations to fetch", 1, 10, 5)
    s_range = st.sidebar.slider("Strike Range +/-", 5, 200, 40)
    boost = st.sidebar.slider("Color Boost", 1.0, 5.0, 3.0)
    
    if st.sidebar.button("Run Analytics", type="primary"):
        S, raw_df = fetch_data_safe(ticker, max_exp)
        
        if S and raw_df is not None:
            processed = process_exposure(raw_df, S, s_range)
            if not processed.empty:
                st.metric(f"{ticker} Current Price", f"${S:.2f}")
                h, b = render_plots(processed, ticker, S, mode, boost)
                st.plotly_chart(h, use_container_width=True)
                st.plotly_chart(b, use_container_width=True)
            else:
                st.warning("No option volume/OI found in this strike range.")
        else:
            st.error("Connection blocked by Yahoo. Please wait 2-3 minutes, try a VPN, or reduce the number of expirations.")

if __name__ == "__main__":
    main()