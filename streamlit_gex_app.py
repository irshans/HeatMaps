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
    if T <= 0:
        T = 0.00001 
    if sigma <= 0:
        sigma = FALLBACK_IV
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
    
    now = pd.Timestamp.now(tz='US/Eastern').normalize() + pd.Timedelta(hours=16)
    expiry_dates = pd.to_datetime(df["expiration"])
    expiry_dt_localized = expiry_dates.dt.tz_localize('US/Eastern')
    df["expiry_dt"] = expiry_dt_localized + pd.Timedelta(hours=16)
    df["T"] = (df["expiry_dt"] - now).dt.total_seconds() / (365 * 24 * 3600)
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range) & (df["T"] > -0.0001)]

    res = []
    for _, row in df.iterrows():
        K, T = float(row["strike"]), max(float(row["T"]), 0.00001)
        oi = row.get("openInterest") or 0
        vol = row.get("volume") or 0
        liq = oi if oi > 0 else vol
        if liq <= 0: continue

        iv = row["impliedVolatility"]
        if not (0.05 < iv < 4.0): iv = FALLBACK_IV

        gamma, delta = get_greeks(S, K, RISK_FREE_RATE, iv, T, row["option_type"])

        if model_type == "Dealer Short All (Absolute Stress)":
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * liq
        else:
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * liq if row["option_type"] == "call" else gamma * S**2 * 0.01 * CONTRACT_SIZE * liq

        dex = -delta * S * CONTRACT_SIZE * liq
        res.append({"strike": K, "expiry": row["expiration"], "gex": gex, "dex": dex})

    return pd.DataFrame(res)

# -------------------------
# Visualizations
# -------------------------
def render_plots(df, ticker, S, mode):
    if df.empty: 
        return None, None

    val_col = mode.lower()
    agg = df.groupby('strike')[val_col].sum().sort_index()
    pivot = df.pivot_table(
        index='strike', 
        columns='expiry', 
        values=val_col, 
        aggfunc='sum', 
        fill_value=0
    ).sort_index(ascending=False)

    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** (1.0 / 2.0))

    x_labs = pivot.columns.tolist()
    y_labs = pivot.index.tolist()

    # Find closest strike to spot
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    h_text = []
    for i, strike in enumerate(y_labs):
        row = []
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            prefix = "-" if val < 0 else ""
            v_abs = abs(val)
            formatted = f"{prefix}${v_abs/1e6:,.2f}M" if v_abs >= 1e6 else f"{prefix}${v_abs/1e3:,.1f}K"
            row.append(f"Strike: ${strike:,.0f}<br>Expiry: {exp}<br>{mode}: {formatted}")
        h_text.append(row)

    colorscale = [
        [0.0, '#2d0052'], [0.2, '#1a1f4d'], [0.35, '#1e3a5f'], [0.5, '#1f4d4d'], 
        [0.65, '#2d5a3d'], [0.8, '#6b7c1f'], [0.9, '#d4a017'], [1.0, '#ffd700']
    ]
    
    fig_h = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labs, y=y_labs, text=h_text, hoverinfo="text",
        colorscale=colorscale, zmid=0, showscale=True
    ))

    max_abs_val = np.max(np.abs(z_raw))
    
    # Cell annotations
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            prefix = "-" if val < 0 else ""
            v_abs = abs(val)
            txt = f"{prefix}${v_abs/1e3:,.0f}K"
            if abs(val) == max_abs_val: txt += " ⭐"
            cell_val = z_scaled[i, j]
            z_norm = (cell_val - z_scaled.min()) / (z_scaled.max() - z_scaled.min()) if z_scaled.max() != z_scaled.min() else 0.5
            text_color = "black" if z_norm > 0.55 else "white"
            fig_h.add_annotation(
                x=exp, y=strike, text=txt, showarrow=False,
                font=dict(color=text_color, size=12, family="Arial"), 
                xref="x", yref="y"
            )

    timestamp = datetime.now().strftime("%H:%M:%S")
    
    fig_h.update_layout(
        title=f"{ticker} {mode} Exposure Map | Last Update: {timestamp} | Spot: ${S:,.2f}",
        template="plotly_dark", height=900,
        xaxis=dict(type='category', side='top', tickfont=dict(size=12)),
        yaxis=dict(
            title="Strike", 
            tickfont=dict(size=12, color="#cccccc"),
            autorange=True,
            tickmode='array', 
            tickvals=y_labs, 
            ticktext=[f"{s:,.0f}" for s in y_labs]
        ),
        margin=dict(l=80, r=120, t=100, b=40)
    )

    # === RELIABLE SPOT STRIKE HIGHLIGHT ===
    # Calculate padding based on strike spacing
    if len(y_labs) > 1:
        padding = (y_labs[1] - y_labs[0]) * 0.45
    else:
        padding = 10

    # Strong red bar on the left behind the strike label
    fig_h.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=-0.08, x1=0.0,  # Extends slightly left of plot area
        y0=closest_strike - padding,
        y1=closest_strike + padding,
        fillcolor="#dc143c",  # Crimson red
        line=dict(width=0),
        layer="below"
    )

    # Bold white text overlay for the spot strike label
    fig_h.add_annotation(
        xref="paper", yref="y",
        x=-0.02,  # Positioned just right of the red bar
        y=closest_strike,
        text=f"<b>{closest_strike:,.0f}</b>",
        showarrow=False,
        font=dict(color="white", size=14, family="Arial"),
        align="right"
    )

    # Bar chart
    fig_b = go.Figure(go.Bar(x=agg.index, y=agg.values, 
                             marker_color=['#2563eb' if v < 0 else '#fbbf24' for v in agg.values]))
    fig_b.update_layout(title=f"Net {mode} by Strike", template="plotly_dark", height=400,
                        xaxis=dict(title="Strike", tickformat=",d"), 
                        yaxis=dict(title="Exposure ($)", tickformat="$,.2s"))
    fig_b.add_vline(x=S, line_dash="dash", line_color="yellow")

    return fig_h, fig_b

# -------------------------
# Main App
# -------------------------
def main():
    st.title("📈 GEX / DEX Pro")
    
    with st.sidebar:
        st.header("Control Panel")
        ticker = st.text_input("Ticker", "SPX").upper().strip()
        mode = st.radio("Metric", ["GEX", "DEX"])
        model_type = st.selectbox("Dealer Model", ["Dealer Short All (Absolute Stress)", "Short Calls / Long Puts"])
        max_exp = st.slider("Max Expirations", 1, 15, 5)
        s_range = st.slider("Strike Range ± Spot", 5, 100, 40, step=1)
        run = st.button("Calculate Exposure", type="primary")

    if run:
        with st.spinner(f"Analyzing {ticker}..."):
            S, raw_df = fetch_data_safe(ticker, max_exp)

        if S and raw_df is not None and not raw_df.empty:
            processed = process_exposure(raw_df, S, s_range, model_type)
            
            if not processed.empty:
                t_gex = processed["gex"].sum() / 1e9
                t_dex = processed["dex"].sum() / 1e9
                
                prefix_g = "-" if t_gex < 0 else ""
                prefix_d = "-" if t_dex < 0 else ""
                
                c1, c2 = st.columns(2)
                c1.metric("Net Dealer GEX", f"{prefix_g}${abs(t_gex):,.2f}B")
                c2.metric("Net Dealer DEX", f"{prefix_d}${abs(t_dex):,.2f}B")

                h_fig, b_fig = render_plots(processed, ticker, S, mode)
                
                if h_fig: st.plotly_chart(h_fig, use_container_width=True)
                if b_fig: st.plotly_chart(b_fig, use_container_width=True)
            else:
                st.warning("No data found in range.")
        else:
            st.error("Fetch failed. Please check ticker.")

if __name__ == "__main__":
    main()