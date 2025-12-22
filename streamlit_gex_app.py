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
    """
    Ensures the date is a valid trading day. 
    Rolls back weekends AND provides a mechanism to skip holidays if they 
    fall on weekdays (e.g., Christmas/New Years).
    """
    dt = pd.to_datetime(date_str)
    
    # List of major US Market Holidays (2024-2025 simplified)
    # You can add specific dates here if Yahoo returns them
    holidays = [
        '2024-12-25', '2025-01-01', '2025-07-04', '2025-12-25'
    ]
    
    # 1. Roll back weekends
    while dt.weekday() > 4: # 5=Sat, 6=Sun
        dt -= timedelta(days=1)
    
    # 2. Roll back if it lands on a known holiday
    while dt.strftime('%Y-%m-%d') in holidays:
        dt -= timedelta(days=1)
        # Re-check weekend after rolling back from a holiday
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
            # Use our new robust trading day logic
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
        oi = float(row.get("openInterest") or 0)
        vol = float(row.get("volume") or 0)
        liquidity = oi if oi > 0 else vol
        if liquidity <= 0: continue 
        
        sigma = row.get("impliedVolatility") if row.get("impliedVolatility") and row.get("impliedVolatility") > 0 else FALLBACK_IV
        gamma = bs_gamma(S, K, 0.045, 0.0, sigma, T)
        dollar_gex = (gamma * S**2) * CONTRACT_SIZE * liquidity * 0.01
        
        if row["option_type"] == "put": dollar_gex *= -1
        out.append({"strike": K, "expiry": row["expiration"], "gex_total": dollar_gex})
    return pd.DataFrame(out)

# -------------------------
# Plotting with Power Scaling
# -------------------------
def plot_analysis(gex_df, ticker, S, sensitivity):
    pivot = gex_df.pivot_table(index='strike', columns='expiry', values='gex_total', aggfunc='sum', fill_value=0).sort_index(ascending=False)
    
    z_raw = pivot.values
    exponent = 1.0 / sensitivity
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** exponent)

    x_labels = pivot.columns
    y_labels = pivot.index
    
    max_val_scaled = np.max(np.abs(z_scaled)) if z_scaled.size > 0 else 1
    max_idx = np.unravel_index(np.argmax(np.abs(z_raw)), z_raw.shape) if z_raw.size > 0 else (0,0)

    annotations = []
    for i, strike in enumerate(y_labels):
        for j, expiry in enumerate(x_labels):
            val = z_raw[i, j]
            if val == 0: continue
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            if (i, j) == max_idx: txt += " ★"
            color = "black" if (np.abs(z_scaled[i,j]) > max_val_scaled * 0.6) else "white"
            annotations.append(dict(x=expiry, y=strike, text=txt, showarrow=False, font=dict(color=color, size=10)))

    custom_colorscale = [
        [0.0, 'rgb(50, 0, 90)'],      # Deep Dark Purple
        [0.25, 'rgb(180, 0, 255)'],   # Electric Purple (Short Gamma)
        [0.48, 'rgb(0, 30, 0)'],      # Dark Green Edge
        [0.5, 'rgb(0, 200, 0)'],      # Pure Green (Neutral)
        [0.52, 'rgb(0, 30, 0)'],      # Dark Green Edge
        [0.75, 'rgb(255, 180, 0)'],   # Bright Gold (Long Gamma)
        [1.0, 'rgb(255, 255, 180)']   # Neon Yellow
    ]

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labels, y=y_labels,
        colorscale=custom_colorscale, zmid=0, showscale=True
    ))
    
    fig_heat.update_layout(
        title=f"{ticker} Gamma Map (Holiday/Weekend Corrected)", 
        template="plotly_dark", height=850, annotations=annotations,
        xaxis={'type': 'category'}
    )
    fig_heat.add_hline(y=S, line_dash="dash", line_color="cyan", annotation_text="SPOT")

    return fig_heat

def main():
    with st.sidebar:
        st.header("GEX Control")
        ticker = st.text_input("Ticker", "SPY").upper()
        max_exp = st.slider("Expirations", 1, 20, 10)
        s_range = st.slider("Strike Range +/-", 5, 200, 50)
        sensitivity = st.slider("Color Boost", 1.0, 5.0, 3.0)
        run = st.button("Calculate GEX", type="primary")

    if run:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            S = hist["Close"].iloc[-1]
            raw = fetch_options_yahoo(ticker, max_exp)
            if not raw.empty:
                gex_data = compute_gex(raw, S, s_range)
                if not gex_data.empty:
                    st.plotly_chart(plot_analysis(gex_data, ticker, S, sensitivity), use_container_width=True)
                else: st.warning("No data found for this range.")
            else: st.error("No options data.")
        else: st.error("Ticker not found.")

if __name__ == "__main__":
    main()