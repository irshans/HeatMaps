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
    """Corrects Sunday/Saturday dates to Friday, but ignores mid-week dailies (0DTE)."""
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
            # Apply smarter date fix
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
        # SPY FIX: Fallback to volume if Open Interest is 0/Missing
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
# Plotting
# -------------------------
def plot_analysis(gex_df, ticker, S):
    pivot = gex_df.pivot_table(index='strike', columns='expiry', values='gex_total', aggfunc='sum', fill_value=0).sort_index(ascending=False)
    z = pivot.values
    
    # 1. Labels, Star, and Dynamic Text Color
    abs_z = np.abs(z)
    max_val = np.max(abs_z) if abs_z.size > 0 else 1
    max_idx = np.unravel_index(np.argmax(abs_z), z.shape) if z.size > 0 else (0,0)

    text_matrix = []
    font_colors = []
    for i in range(len(pivot.index)):
        row_text, row_color = [], []
        for j in range(len(pivot.columns)):
            val = z[i, j]
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            if (i, j) == max_idx and abs(val) > 0: txt += " ★"
            row_text.append(txt)
            
            # Contrast Logic: Black text for bright yellow, White for dark purple/green
            if val > (max_val * 0.25): # Positive and bright
                row_color.append("black")
            else:
                row_color.append("white")
        text_matrix.append(row_text)
        font_colors.append(row_color)

    # 2. Refined High-Contrast Color Scale
    custom_colorscale = [
        [0.0, 'rgb(30, 0, 50)'],    # Deep Purple (Short Gamma)
        [0.3, 'rgb(120, 60, 210)'], # Medium Purple
        [0.48, 'rgb(0, 35, 0)'],    # Dark Green Boundary
        [0.5, 'rgb(0, 140, 0)'],    # Pure Green (Neutral/Zero)
        [0.52, 'rgb(0, 35, 0)'],    # Dark Green Boundary
        [0.8, 'rgb(190, 160, 0)'],  # Darker Gold
        [1.0, 'rgb(255, 215, 0)']   # Bright Yellow (Long Gamma)
    ]

    fig_heat = go.Figure(data=go.Heatmap(
        z=z, x=pivot.columns, y=pivot.index,
        text=text_matrix, texttemplate="%{text}",
        textfont={"size": 11, "color": font_colors},
        colorscale=custom_colorscale, zmid=0, showscale=True
    ))
    fig_heat.add_hline(y=S, line_dash="dash", line_color="cyan", annotation_text=f"SPOT: {S:.2f}")
    fig_heat.update_layout(title=f"{ticker} Net GEX Map", template="plotly_dark", height=750)

    # 3. Bar Chart with Gamma Flip
    strike_agg = gex_df.groupby('strike')['gex_total'].sum().sort_index()
    
    # Calculate Flip Point (Simplified zero-crossing)
    flip_price = S
    for i in range(1, len(strike_agg)):
        if np.sign(strike_agg.values[i-1]) != np.sign(strike_agg.values[i]):
            flip_price = strike_agg.index[i]
            break

    fig_bar = go.Figure(go.Bar(
        x=strike_agg.index, y=strike_agg.values,
        marker_color=['#9B59B6' if x < 0 else '#F1C40F' for x in strike_agg.values]
    ))
    fig_bar.add_vline(x=S, line_color="cyan", line_dash="dash", annotation_text="SPOT")
    fig_bar.add_vline(x=flip_price, line_color="orange", line_width=4, annotation_text="GAMMA FLIP")
    fig_bar.update_layout(title="Total Gamma Profile & Volatility Flip Point", template="plotly_dark", height=450)

    return fig_heat, fig_bar

# -------------------------
# Main Application
# -------------------------
def main():
    with st.sidebar:
        st.header("Control Panel")
        ticker = st.text_input("Ticker (TSLA, SPY, etc.)", "TSLA").upper()
        max_exp = st.slider("Expiration Cycles", 1, 15, 5)
        s_range = st.slider("Strike Range +/-", 5, 100, 30)
        run = st.button("Calculate GEX", type="primary")

    if run:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            st.error("Ticker not found.")
            return
            
        S = hist["Close"].iloc[-1]
        raw = fetch_options_yahoo(ticker, max_exp)
        
        if not raw.empty:
            gex_data = compute_gex(raw, S, s_range)
            if not gex_data.empty:
                h_map, b_chart = plot_analysis(gex_data, ticker, S)
                st.plotly_chart(h_map, use_container_width=True)
                st.plotly_chart(b_chart, use_container_width=True)
            else:
                st.warning("No GEX data for this range. Try widening the strike range.")
        else:
            st.error("Failed to retrieve options data.")

if __name__ == "__main__":
    main()