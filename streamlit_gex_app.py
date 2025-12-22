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
# Improved Date Logic
# -------------------------
def fix_expiration_date(date_str):
    dt = pd.to_datetime(date_str)
    if dt.weekday() == 6: # Sunday
        dt = dt - timedelta(days=2)
    elif dt.weekday() == 5: # Saturday
        dt = dt - timedelta(days=1)
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
    for exp in exp_list:
        try:
            chain = stock.option_chain(exp)
            trading_date = fix_expiration_date(exp)
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
    
    # APPLY POWER SCALING to the Z-values for the heatmap colors only
    # This stretches the mid-range values while keeping the real values for text/labels
    z_raw = pivot.values
    # We use sign(z) * |z|^(1/sensitivity) to squash the extremes and pump the lows
    exponent = 1.0 / sensitivity
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** exponent)

    x_labels = pivot.columns
    y_labels = pivot.index
    
    # For annotations, we still use the raw values
    max_val = np.max(np.abs(z_raw)) if z_raw.size > 0 else 1
    max_idx = np.unravel_index(np.argmax(np.abs(z_raw)), z_raw.shape) if z_raw.size > 0 else (0,0)

    annotations = []
    for i, strike in enumerate(y_labels):
        for j, expiry in enumerate(x_labels):
            val = z_raw[i, j]
            if val == 0: continue
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            if (i, j) == max_idx: txt += " ★"
            # Text contrast logic
            color = "black" if (val > 0 and np.abs(z_scaled[i,j]) > np.max(np.abs(z_scaled))*0.5) else "white"
            annotations.append(dict(x=expiry, y=strike, text=txt, showarrow=False, font=dict(color=color, size=10)))

    # Enhanced Diverging Colorscale
    custom_colorscale = [
        [0.0, 'rgb(40, 0, 80)'],      # Deep Purple
        [0.2, 'rgb(180, 0, 255)'],    # Electric Purple
        [0.45, 'rgb(0, 40, 0)'],      # Dark Green
        [0.5, 'rgb(0, 200, 0)'],      # Neutral Green
        [0.55, 'rgb(0, 40, 0)'],      # Dark Green
        [0.8, 'rgb(255, 200, 0)'],    # Gold
        [1.0, 'rgb(255, 255, 150)']    # Neon Yellow
    ]

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labels, y=y_labels,
        colorscale=custom_colorscale, zmid=0, showscale=True,
        colorbar=dict(title="Scaled GEX intensity")
    ))
    
    fig_heat.update_layout(
        title=f"{ticker} Net GEX Heatmap (Non-Linear Scale)", 
        template="plotly_dark", height=850, annotations=annotations
    )
    fig_heat.add_hline(y=S, line_dash="dash", line_color="cyan", annotation_text="SPOT")

    # Bar Chart (Raw Values)
    strike_agg = gex_df.groupby('strike')['gex_total'].sum().sort_index()
    fig_bar = go.Figure(go.Bar(
        x=strike_agg.index, y=strike_agg.values,
        marker_color=['#B400FF' if x < 0 else '#FFD700' for x in strike_agg.values]
    ))
    fig_bar.add_vline(x=S, line_color="cyan", line_dash="dash", annotation_text="SPOT")
    fig_bar.update_layout(title="Total Gamma Profile (Raw Value)", template="plotly_dark", height=400)

    return fig_heat, fig_bar

def main():
    with st.sidebar:
        st.header("Control Panel")
        ticker = st.text_input("Ticker", "TSLA").upper()
        max_exp = st.slider("Expirations", 1, 15, 5)
        s_range = st.slider("Strike Range +/-", 5, 100, 40)
        
        st.markdown("---")
        st.subheader("Visualization Scale")
        # Higher sensitivity = more vibrant colors for small numbers
        sensitivity = st.slider("Color Boost (Power Scale)", 1.0, 5.0, 2.5, step=0.5, 
                                help="Increase this to make smaller GEX levels more visible.")
        
        run = st.button("Update Analysis", type="primary")

    if run:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            S = hist["Close"].iloc[-1]
            raw = fetch_options_yahoo(ticker, max_exp)
            if not raw.empty:
                gex_data = compute_gex(raw, S, s_range)
                if not gex_data.empty:
                    h_map, b_chart = plot_analysis(gex_data, ticker, S, sensitivity)
                    st.plotly_chart(h_map, use_container_width=True)
                    st.plotly_chart(b_chart, use_container_width=True)
                else: st.warning("No data found.")
            else: st.error("No options data.")
        else: st.error("Ticker not found.")

if __name__ == "__main__":
    main()