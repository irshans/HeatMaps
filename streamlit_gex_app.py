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
    now = pd.Timestamp.now().normalize() + pd.Timedelta(hours=16)
    df["expiry_dt"] = pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16)
    df["T"] = (df["expiry_dt"] - now).dt.total_seconds() / (365 * 24 * 3600)

    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range) & (df["T"] > 0)]

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
    # Removed boost parameter - using raw scaled values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** (1.0 / 2.0))

    # Create hover text with actual values
    x_labs = pivot.columns.tolist()
    y_labs = pivot.index.tolist()
    h_text = []
    for i, strike in enumerate(y_labs):
        row = []
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            formatted = f"${val/1e6:.2f}M" if abs(val) >= 1e6 else f"${val/1e3:.1f}K"
            row.append(f"Strike: ${strike:.0f}<br>Expiry: {exp}<br>{mode}: {formatted}")
        h_text.append(row)

    # Color scheme - dark purple to bright yellow through dark blues/greens
    colorscale = [
        [0.0, '#2d0052'],    # Dark purple (most negative)
        [0.2, '#1a1f4d'],    # Dark blue-purple
        [0.35, '#1e3a5f'],   # Dark blue
        [0.5, '#1f4d4d'],    # Dark teal/green (zero)
        [0.65, '#2d5a3d'],   # Dark green
        [0.8, '#6b7c1f'],    # Olive
        [0.9, '#d4a017'],    # Bright gold
        [1.0, '#ffd700']     # Bright yellow (most positive)
    ]
    
    fig_h = go.Figure(data=go.Heatmap(
        z=z_scaled, 
        x=x_labs, 
        y=y_labs,
        text=h_text,
        hoverinfo="text",
        colorscale=colorscale, 
        zmid=0, 
        colorbar=dict(
            title=mode,
            tickmode="linear",
            tick0=z_scaled.min(),
            dtick=(z_scaled.max() - z_scaled.min()) / 5
        ),
        hovertemplate='%{text}<extra></extra>'
    ))

    # Find the biggest absolute value for star annotation
    max_abs_val = np.max(np.abs(z_raw))
    max_position = None
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            if abs(z_raw[i, j]) == max_abs_val:
                max_position = (i, j)
                break
        if max_position:
            break
    
    # Add annotations for ALL cells (like the reference image)
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 1000:  # Skip very small values
                continue
            
            # Format text - always display in thousands with K
            txt = f"${val/1e3:.1f}K"
            
            # Add star to the biggest absolute value
            if max_position and i == max_position[0] and j == max_position[1]:
                txt += " ⭐"
            
            # Determine text color based on background intensity
            # Lighter colors (positive values) = black text
            # Darker colors (negative values) = white text
            cell_val = z_scaled[i, j]
            z_normalized = (cell_val - z_scaled.min()) / (z_scaled.max() - z_scaled.min()) if z_scaled.max() != z_scaled.min() else 0.5
            
            # Use black text for lighter colors (threshold at 0.6)
            if z_normalized > 0.6:
                text_color = "black"
            else:
                text_color = "white"
            
            fig_h.add_annotation(
                x=exp, 
                y=strike, 
                text=txt, 
                showarrow=False,
                font=dict(color=text_color, size=8),
                xref="x",
                yref="y"
            )

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    fig_h.update_layout(
        title=f"SPY {mode} - {timestamp} - Current Price: ${S:.2f}",
        template="plotly_dark",
        height=900,
        xaxis=dict(
            type='category', 
            title="",
            tickfont=dict(size=10),
            side='top'  # Move dates to top like reference
        ),
        yaxis=dict(
            title="Strike",
            tickfont=dict(size=8),
            autorange=True,
            tickmode='array',  # Force all strikes to show
            tickvals=y_labs,  # Use all strike values
            ticktext=[f"{s:.0f}" for s in y_labs]  # Format as integers
        ),
        font=dict(size=10),
        margin=dict(l=80, r=120, t=80, b=40)
    )
    
    # Add horizontal line at current price
    fig_h.add_hline(
        y=S, 
        line_dash="solid", 
        line_color="yellow", 
        line_width=2,
        annotation_text=f"Current: ${S:.2f}",
        annotation_position="right"
    )

    # Bar chart
    colors = ['#2563eb' if v < 0 else '#fbbf24' for v in agg.values]
    fig_b = go.Figure(go.Bar(
        x=agg.index, 
        y=agg.values, 
        marker_color=colors,
        hovertemplate='Strike: $%{x:.0f}<br>Total: $%{y:.2s}<extra></extra>'
    ))
    
    fig_b.update_layout(
        title=f"Total {mode} by Strike", 
        template="plotly_dark", 
        height=400,
        xaxis_title="Strike Price",
        yaxis_title=f"{mode} Exposure ($)"
    )
    
    fig_b.add_vline(
        x=S, 
        line_dash="dash", 
        line_color="yellow",
        annotation_text=f"Spot: ${S:.2f}"
    )

    return fig_h, fig_b

# -------------------------
# Main App
# -------------------------
def main():
    st.title("📈 GEX / DEX Pro")
    
    with st.sidebar:
        st.header("Control Panel")
        ticker = st.text_input("Ticker", "SPY").upper().strip()
        mode = st.radio("Metric", ["GEX", "DEX"])
        model_type = st.selectbox("Dealer Model", ["Dealer Short All (Absolute Stress)", "Short Calls / Long Puts"])
        max_exp = st.slider("Max Expirations", 1, 15, 6)
        s_range = st.slider("Strike Range ± Spot", 10, 100, 25, step=1)
        run = st.button("Calculate Exposure", type="primary")

    if run:
        with st.spinner(f"Analyzing {ticker} flow..."):
            S, raw_df = fetch_data_safe(ticker, max_exp)

        if S and raw_df is not None and not raw_df.empty:
            st.success(f"{ticker} Trading at ${S:.2f}")
            processed = process_exposure(raw_df, S, s_range, model_type)
            
            if not processed.empty:
                t_gex = processed["gex"].sum() / 1e9
                t_dex = processed["dex"].sum() / 1e9
                
                c1, c2 = st.columns(2)
                c1.metric("Net Dealer GEX", f"${t_gex:.2f}B")
                c2.metric("Net Dealer DEX", f"${t_dex:.2f}B")

                h_fig, b_fig = render_plots(processed, ticker, S, mode)
                
                if h_fig:
                    st.plotly_chart(h_fig, width="stretch")
                if b_fig:
                    st.plotly_chart(b_fig, width="stretch")
            else:
                st.warning("No liquidity found in that range.")
        else:
            st.error("Fetch failed. Please check ticker or try again later.")

if __name__ == "__main__":
    main()