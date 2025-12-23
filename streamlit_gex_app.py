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
FALLBACK_IV = 0.30
RISK_FREE_RATE = 0.042

# -------------------------
# Holiday Rollback
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
# Black-Scholes Greeks
# -------------------------
def get_greeks(S, K, r, sigma, T, option_type):
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    return gamma, delta

# -------------------------
# Data Fetcher
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
                    time.sleep(random.uniform(0.4, 0.9))
                    chain = stock.option_chain(exp)
                    t_date = get_actual_trading_day(exp)
                    calls = chain.calls.assign(option_type="call", expiration=t_date)
                    puts = chain.puts.assign(option_type="put", expiration=t_date)
                    dfs.append(pd.concat([calls, puts], ignore_index=True))
                    success = True
                    break
                except Exception:
                    time.sleep(1.5)

            prog_bar.progress((i + 1) / len(target_exps))
            if not success:
                st.warning(f"Failed to fetch {exp} after retries.")

        progress_text.empty()
        prog_bar.empty()

        if not dfs:
            return S, None
        return S, pd.concat(dfs, ignore_index=True)

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None, None

# -------------------------
# GEX & DEX Calculation (Dealer Positioning)
# -------------------------
def process_exposure(df, S, s_range):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Updated to handle timezone-naive comparison safely
    now = pd.Timestamp.now().normalize() + pd.Timedelta(hours=16)

    # Convert expiration to datetime and set to 4 PM (Market Close)
    df["expiry_dt"] = pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16)
    
    # Calculate Time to Expiry (T) in years
    df["T"] = (df["expiry_dt"] - now).dt.total_seconds() / (365 * 24 * 3600)

    # Filter by Strike Range and Expiry
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range) & (df["T"] > 0)]

    res = []
    for _, row in df.iterrows():
        K = float(row["strike"])
        T = max(float(row["T"]), 0.00001)
        oi = row.get("openInterest") or 0
        vol = row.get("volume") or 0
        liq = oi if oi > 0 else vol
        if liq <= 0:
            continue

        iv = row["impliedVolatility"]
        if not (0.05 < iv < 4.0):
            iv = FALLBACK_IV

        gamma, delta = get_greeks(S, K, RISK_FREE_RATE, iv, T, row["option_type"])

        # Claude's Logic: Short Calls (Negative) vs Long Puts (Positive)
        if row["option_type"] == "call":
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * liq  # Dealer Short Call
            dex = -delta * S * CONTRACT_SIZE * liq
        else:
            gex = gamma * S**2 * 0.01 * CONTRACT_SIZE * liq   # Dealer Long Put
            dex = -delta * S * CONTRACT_SIZE * liq

        res.append({
            "strike": K,
            "expiry": row["expiration"],
            "gex": gex,
            "dex": dex
        })

    return pd.DataFrame(res)

# -------------------------
# Visualizations
# -------------------------
def render_plots(df, ticker, S, mode, boost):
    if df.empty:
        return None, None

    val_col = 'gex' if mode == "GEX" else 'dex'
    display_name = mode

    agg = df.groupby('strike')[val_col].sum().sort_index()

    pivot = df.pivot_table(
        index='strike',
        columns='expiry',
        values=val_col,
        aggfunc='sum',
        fill_value=0
    ).sort_index(ascending=False)

    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** (1.0 / boost))

    x_labs = pivot.columns.tolist()
    y_labs = pivot.index.tolist()

    h_text = []
    for i, strike in enumerate(y_labs):
        row = []
        for j, ex in enumerate(x_labs):
            val = z_raw[i, j]
            formatted = f"${val/1e6:.2f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            row.append(f"Expiry: {ex}<br>Strike: {strike}<br>{display_name}: {formatted}")
        h_text.append(row)

    colorscale = 'RdBu' if mode == "DEX" else [[0, '#32005A'], [0.3, '#BF00FF'], [0.5, '#000000'], [0.7, '#FFB400'], [1, '#FFFFB4']]
    fig_h = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labs, y=y_labs, text=h_text, hoverinfo="text",
        colorscale=colorscale, zmid=0, colorbar=dict(title=display_name)
    ))

    # Add dynamic annotations for high-exposure strikes
    threshold = np.percentile(np.abs(z_raw), 85) if z_raw.size > 0 else 0
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < threshold:
                continue
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val/1e3:.0f}K"
            fig_h.add_annotation(x=exp, y=strike, text=txt, showarrow=False,
                                 font=dict(color="white" if abs(z_scaled[i,j]) > np.max(np.abs(z_scaled))*0.5 else "black", size=9))

    fig_h.update_layout(
        title=f"{ticker} {display_name} Exposure Heatmap (Dealer Positioning)",
        template="plotly_dark",
        height=800,
        xaxis=dict(type='category', title="Expiration"),
        yaxis=dict(title="Strike")
    )
    fig_h.add_hline(y=S, line_dash="dash", line_color="cyan", annotation_text=f"Spot: {S:.2f}")

    colors = ['#FF00FF' if v < 0 else '#FFFF00' for v in agg.values]
    fig_b = go.Figure(go.Bar(x=agg.index, y=agg.values, marker_color=colors))
    fig_b.update_layout(
        title=f"Total {display_name} Exposure by Strike",
        template="plotly_dark",
        height=450,
        xaxis_title="Strike",
        yaxis_title=f"{display_name} Exposure ($)"
    )

    return fig_h, fig_b

# -------------------------
# Main App
# -------------------------
def main():
    st.title("📈 GEX / DEX Pro - Dealer Exposure Analyzer")
    st.markdown("*Industry-standard calculation assuming dealers are short options*")

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", "SPY").upper().strip()
        mode = st.radio("Exposure Type", ["GEX", "DEX"], help="GEX = Gamma Exposure | DEX = Delta Exposure")
        max_exp = st.slider("Max Expirations", 1, 12, 6)
        
        # --- UPDATED SLIDER LOGIC ---
        is_index = ticker.lstrip("^") in ["SPX", "NDX", "RUT"]
        default_range = 25 # Set default to 25 as requested
        
        # Minimum 10, Maximum 100, Step 10 (creates the dots/intervals), Default 25
        # Note: Since step is 10, 25 isn't technically on a "dot". 
        # I will set step to 1 to allow 25, or set default to 20/30 to snap to dots.
        # Keeping step=1 so you can hit 25 exactly.
        s_range = st.slider("Strike Range ± Spot", min_value=10, max_value=100, value=25, step=1)
        
        boost = st.slider("Heatmap Color Boost", 1.0, 5.0, 2.5)

        run = st.button("Run Analysis", type="primary")

    if run:
        with st.spinner("Fetching option chains..."):
            S, raw_df = fetch_data_safe(ticker, max_exp)

        if S is None:
            st.error("Unable to fetch data.")
            return

        if raw_df is None or raw_df.empty:
            st.warning("No option expirations found.")
            return

        st.success(f"{ticker} Spot Price: **${S:.2f}**")
        processed = process_exposure(raw_df, S, s_range)

        if processed.empty:
            st.warning("No data found in range.")
            return

        total_gex = processed["gex"].sum()
        total_dex = processed["dex"].sum() / 1e9

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Dealer GEX", f"${total_gex/1e9:.2f}B")
        with col2:
            st.metric("Total Dealer DEX", f"${total_dex:.2f}B")

        heatmap_fig, bar_fig = render_plots(processed, ticker, S, mode, boost)

        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        if bar_fig:
            st.plotly_chart(bar_fig, use_container_width=True)

if __name__ == "__main__":
    main()