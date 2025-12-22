import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta

st.set_page_config(page_title="GEX Analytics Pro", page_icon="📊", layout="wide")

CONTRACT_SIZE = 100
FALLBACK_IV = 0.25

# -------------------------
# Holiday & Weekend Logic
# -------------------------
def get_actual_trading_day(date_str):
    dt = pd.to_datetime(date_str)
    holidays = ['2024-12-25', '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25']
    while dt.weekday() > 4:
        dt -= timedelta(days=1)
    while dt.strftime('%Y-%m-%d') in holidays:
        dt -= timedelta(days=1)
        while dt.weekday() > 4:
            dt -= timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

# -------------------------
# Math Engine
# -------------------------
def norm_pdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def bs_gamma(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

@st.cache_data(ttl=300)
def fetch_options_yahoo(ticker, max_expirations):
    stock = yf.Ticker(ticker)
    try:
        exp_list = stock.options[:max_expirations]
    except: return pd.DataFrame()
    dfs = []
    for raw_exp in exp_list:
        try:
            chain = stock.option_chain(raw_exp)
            trading_date = get_actual_trading_day(raw_exp)
            calls = chain.calls.assign(option_type="call", expiration=trading_date)
            puts = chain.puts.assign(option_type="put", expiration=trading_date)
            dfs.append(pd.concat([calls, puts], ignore_index=True))
        except: continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def compute_gex(df, S, strike_range):
    df = df.copy()
    now = pd.Timestamp.now(tz='US/Eastern').replace(tzinfo=None)
    df["expiry_dt"] = pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16)
    df["T"] = (df["expiry_dt"] - now).dt.total_seconds() / (365*24*3600)
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range) & (df["T"] > -0.0001)]
    
    out = []
    for _, row in df.iterrows():
        K, T = float(row["strike"]), max(float(row["T"]), 0.00001)
        oi, vol = float(row.get("openInterest") or 0), float(row.get("volume") or 0)
        liquidity = oi if oi > 0 else vol
        if liquidity <= 0: continue 
        sigma = row.get("impliedVolatility") if row.get("impliedVolatility") and row.get("impliedVolatility") > 0 else FALLBACK_IV
        gamma = bs_gamma(S, K, 0.045, 0.0, sigma, T)
        dollar_gex = (gamma * S**2) * CONTRACT_SIZE * liquidity * 0.01
        if row["option_type"] == "put": dollar_gex *= -1
        out.append({"strike": K, "expiry": row["expiration"], "gex_total": dollar_gex})
    return pd.DataFrame(out)

# -------------------------
# Plotting & Summary
# -------------------------
def plot_analysis(gex_df, ticker, S, sensitivity):
    strike_agg = gex_df.groupby('strike')['gex_total'].sum().sort_index()
    
    flip_price = S
    for i in range(1, len(strike_agg)):
        if np.sign(strike_agg.values[i-1]) != np.sign(strike_agg.values[i]):
            flip_price = strike_agg.index[i]
            break
            
    top_call_wall = strike_agg.idxmax()
    top_put_wall = strike_agg.idxmin()
    
    # Pivot the data
    pivot = gex_df.pivot_table(index='strike', columns='expiry', values='gex_total', aggfunc='sum', fill_value=0).sort_index(ascending=False)
    
    # Raw values for hover and scaling
    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** (1.0 / sensitivity))
    x_labels, y_labels = pivot.columns, pivot.index
    
    # Identify Star Cell
    max_idx = np.unravel_index(np.argmax(np.abs(z_raw)), z_raw.shape) if z_raw.size > 0 else (0,0)

    # 1. HEATMAP with FIX for Hover
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_scaled, 
        x=x_labels, 
        y=y_labels,
        # We must wrap z_raw in an extra list or ensure it is a 2D array
        customdata=z_raw,
        hovertemplate=(
            "<b>Expiry:</b> %{x}<br>" +
            "<b>Strike:</b> %{y}<br>" +
            "<b>Net GEX:</b> $%{customdata:,.0f}<br>" +
            "<extra></extra>"
        ),
        colorscale=[[0, '#32005A'], [0.25, '#BF00FF'], [0.48, '#001E00'], [0.5, '#00C800'], [0.52, '#001E00'], [0.75, '#FFB400'], [1, '#FFFFB4']],
        zmid=0, showscale=True
    ))

    # Text Annotations
    annotations = []
    for i, strike in enumerate(y_labels):
        for j, expiry in enumerate(x_labels):
            val = z_raw[i, j]
            if val == 0: continue
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            if (i, j) == max_idx: txt = f"★ {txt}"
            color = "black" if (np.abs(z_scaled[i,j]) > np.max(np.abs(z_scaled)) * 0.6) else "white"
            annotations.append(dict(x=expiry, y=strike, text=txt, showarrow=False, font=dict(color=color, size=10, family="Arial")))

    fig_heat.add_shape(type="rect", xref="x", yref="y", x0=max_idx[1]-0.5, x1=max_idx[1]+0.5, y0=y_labels[max_idx[0]]-0.5, y1=y_labels[max_idx[0]]+0.5,
                      line=dict(color="Cyan", width=3), fillcolor="rgba(0, 255, 255, 0.15)")
    
    fig_heat.update_layout(title=f"<b>{ticker}</b> Gamma Heatmap", template="plotly_dark", height=700, annotations=annotations, xaxis={'type': 'category'})
    fig_heat.add_hline(y=S, line_dash="dash", line_color="white", annotation_text=f"SPOT: {S:.2f}")

    # 2. BAR CHART
    fig_bar = go.Figure(go.Bar(x=strike_agg.index, y=strike_agg.values, marker_color=['#BF00FF' if x < 0 else '#FFD700' for x in strike_agg.values]))
    fig_bar.add_vline(x=S, line_color="white", line_dash="dash", annotation_text="SPOT")
    fig_bar.add_vline(x=flip_price, line_color="cyan", line_width=3, annotation_text=f"FLIP")
    fig_bar.update_layout(title="Aggregate Gamma Profile", template="plotly_dark", height=400, xaxis_title="Strike", yaxis_title="Net GEX ($)")

    return fig_heat, fig_bar, flip_price, top_call_wall, top_put_wall

# -------------------------
# UI Main
# -------------------------
def main():
    with st.sidebar:
        st.header("GEX Controls")
        ticker = st.text_input("Ticker Symbol", "SPY").upper()
        max_exp = st.slider("Expirations", 1, 15, 8)
        s_range = st.slider("Strike Range ($)", 10, 300, 60)
        boost = st.slider("Color Sensitivity", 1.0, 5.0, 3.0)
        run = st.button("Analyze GEX", type="primary")

    if run:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            S = hist["Close"].iloc[-1]
            with st.spinner(f"Computing levels for {ticker}..."):
                raw = fetch_options_yahoo(ticker, max_exp)
                if not raw.empty:
                    gex_data = compute_gex(raw, S, s_range)
                    if not gex_data.empty:
                        h_map, b_chart, flip, c_wall, p_wall = plot_analysis(gex_data, ticker, S, boost)
                        
                        st.subheader(f"📊 {ticker} Market Context")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Spot Price", f"${S:.2f}")
                        c2.metric("Gamma Flip", f"${flip:.0f}", delta=f"{S-flip:.2f}", delta_color="inverse")
                        c3.metric("Call Wall", f"${c_wall:.0f}")
                        c4.metric("Put Wall", f"${p_wall:.0f}")
                        st.markdown("---")
                        
                        st.plotly_chart(h_map, use_container_width=True)
                        st.plotly_chart(b_chart, use_container_width=True)
                    else: st.warning("Range too narrow. No data found.")
                else: st.error("Failed to fetch options.")
        else: st.error("Ticker not found.")

if __name__ == "__main__":
    main()