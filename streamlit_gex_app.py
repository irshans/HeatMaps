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
    # Market holidays
    holidays = ['2024-12-25', '2025-01-01', '2025-07-04', '2025-12-25']
    while dt.weekday() > 4: dt -= timedelta(days=1)
    while dt.strftime('%Y-%m-%d') in holidays:
        dt -= timedelta(days=1)
        while dt.weekday() > 4: dt -= timedelta(days=1)
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
    now = pd.Timestamp.now()
    df["T"] = (pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16) - now).dt.total_seconds() / (365*24*3600)
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range) & (df["T"] > 0)]
    out = []
    for _, row in df.iterrows():
        K, T = float(row["strike"]), float(row["T"])
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
# Plotting with Star & Pop-out
# -------------------------
def plot_analysis(gex_df, ticker, S, sensitivity):
    pivot = gex_df.pivot_table(index='strike', columns='expiry', values='gex_total', aggfunc='sum', fill_value=0).sort_index(ascending=False)
    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** (1.0 / sensitivity))
    x_labels, y_labels = pivot.columns, pivot.index
    
    # Locate the Star Cell (Highest Absolute GEX)
    max_idx = np.unravel_index(np.argmax(np.abs(z_raw)), z_raw.shape) if z_raw.size > 0 else (0,0)
    star_strike = y_labels[max_idx[0]]
    star_expiry = x_labels[max_idx[1]]

    annotations = []
    for i, strike in enumerate(y_labels):
        for j, expiry in enumerate(x_labels):
            val = z_raw[i, j]
            if val == 0: continue
            
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            
            # Add the STAR to the highest value text
            if (i, j) == max_idx:
                txt = f"★ {txt}"
            
            color = "black" if (np.abs(z_scaled[i,j]) > np.max(np.abs(z_scaled)) * 0.6) else "white"
            
            annotations.append(dict(
                x=expiry, y=strike, text=txt, 
                showarrow=False, 
                font=dict(color=color, size=11, family="Arial Black")
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labels, y=y_labels,
        colorscale=[
            [0.0, '#32005A'], [0.25, '#BF00FF'], [0.48, '#001E00'], 
            [0.5, '#00C800'], [0.52, '#001E00'], [0.75, '#FFB400'], [1.0, '#FFFFB4']
        ],
        zmid=0, showscale=True
    ))

    # ADD CYAN POP-OUT BOX FOR THE STAR
    fig.add_shape(
        type="rect",
        xref="x", yref="y",
        x0=max_idx[1] - 0.5, x1=max_idx[1] + 0.5,
        y0=star_strike - 0.5, y1=star_strike + 0.5,
        line=dict(color="Cyan", width=4),
        fillcolor="rgba(0, 255, 255, 0.2)"
    )

    fig.update_layout(
        title=f"<b>{ticker}</b> Gamma Heatmap | ★ = Max Exposure", 
        template="plotly_dark", 
        height=850, 
        annotations=annotations,
        xaxis={'type': 'category'} 
    )
    
    fig.add_hline(y=S, line_dash="dash", line_color="white", 
                  annotation_text=f"SPOT: {S:.2f}", annotation_position="top right")
    
    return fig

# -------------------------
# Streamlit Main
# -------------------------
def main():
    with st.sidebar:
        st.header("GEX Analytics")
        ticker = st.text_input("Ticker Symbol", "SPY").upper()
        max_exp = st.slider("Expirations", 1, 15, 8)
        s_range = st.slider("Strike Range ($)", 10, 300, 60)
        boost = st.slider("Color Sensitivity", 1.0, 5.0, 3.0)
        run = st.button("Run Calculation", type="primary")

    if run:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            S = hist["Close"].iloc[-1]
            with st.spinner(f"Analyzing {ticker}..."):
                raw = fetch_options_yahoo(ticker, max_exp)
                if not raw.empty:
                    gex_data = compute_gex(raw, S, s_range)
                    if not gex_data.empty:
                        st.plotly_chart(plot_analysis(gex_data, ticker, S, boost), use_container_width=True)
                    else: st.warning("Increase strike range to find data.")
                else: st.error("No options data returned.")
        else: st.error("Ticker not found.")

if __name__ == "__main__":
    main()