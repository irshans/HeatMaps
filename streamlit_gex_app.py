import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta
from scipy.stats import norm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="GEX & DEX Pro", page_icon="📊", layout="wide")

CONTRACT_SIZE = 100
FALLBACK_IV = 0.25

# -------------------------
# Session Setup for Rate Limits
# -------------------------
def get_session():
    session = requests.Session()
    # Headers make the request look like a standard browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    retries = Retry(
        total=5,
        backoff_factor=1, # Wait 1s, 2s, 4s, 8s between retries
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# -------------------------
# Holiday & Weekend Logic
# -------------------------
def get_actual_trading_day(date_str):
    dt = pd.to_datetime(date_str)
    holidays = ['2024-12-25', '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25']
    while dt.weekday() > 4: dt -= timedelta(days=1)
    while dt.strftime('%Y-%m-%d') in holidays:
        dt -= timedelta(days=1)
        while dt.weekday() > 4: dt -= timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

# -------------------------
# Math Engine
# -------------------------
def get_greeks(S, K, r, sigma, T, option_type):
    if T <= 0 or sigma <= 0: return 0.0, 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    return gamma, delta

@st.cache_data(ttl=600)
def fetch_options_data(ticker, max_expirations):
    # Handle Index tickers automatically
    if ticker in ["SPX", "NDX", "RUT"]:
        ticker = f"^{ticker}"
    
    session = get_session()
    stock = yf.Ticker(ticker, session=session)
    
    try:
        # Fetch current price
        hist = stock.history(period="1d")
        if hist.empty: return None, pd.DataFrame()
        S = hist["Close"].iloc[-1]
        
        # Fetch Expirations
        exp_list = stock.options[:max_expirations]
        dfs = []
        
        # Progress bar for visual feedback
        progress_bar = st.progress(0)
        for i, raw_exp in enumerate(exp_list):
            try:
                chain = stock.option_chain(raw_exp)
                trading_date = get_actual_trading_day(raw_exp)
                calls = chain.calls.assign(option_type="call", expiration=trading_date)
                puts = chain.puts.assign(option_type="put", expiration=trading_date)
                dfs.append(pd.concat([calls, puts], ignore_index=True))
                progress_bar.progress((i + 1) / len(exp_list))
            except Exception:
                continue
        
        if not dfs: return S, pd.DataFrame()
        return S, pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None, pd.DataFrame()

def compute_exposure(df, S, strike_range):
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
        gamma, delta = get_greeks(S, K, 0.045, sigma, T, row["option_type"])
        
        dollar_gex = (gamma * S**2) * 0.01 * CONTRACT_SIZE * liquidity
        dollar_dex = (delta * S) * CONTRACT_SIZE * liquidity
        
        if row["option_type"] == "put": dollar_gex *= -1
            
        out.append({"strike": K, "expiry": row["expiration"], "gex": dollar_gex, "dex": dollar_dex})
    return pd.DataFrame(out)

# -------------------------
# Plotting (Fail-Safe)
# -------------------------
def plot_analysis(df, ticker, S, sensitivity, mode):
    val_col = 'gex' if mode == "Gamma" else 'dex'
    strike_agg = df.groupby('strike')[val_col].sum().sort_index()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum', fill_value=0).sort_index(ascending=False)
    
    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** (1.0 / sensitivity))
    x_labels, y_labels = pivot.columns.tolist(), pivot.index.tolist()
    
    hover_text = []
    for i, strike in enumerate(y_labels):
        row_text = []
        for j, expiry in enumerate(x_labels):
            val = z_raw[i, j]
            row_text.append(f"Expiry: {expiry}<br>Strike: {strike}<br>Net {mode}: ${val:,.0f}")
        hover_text.append(row_text)

    max_idx = np.unravel_index(np.argmax(np.abs(z_raw)), z_raw.shape) if z_raw.size > 0 else (0,0)

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labels, y=y_labels, text=hover_text, hoverinfo="text",
        colorscale='RdBu' if mode == "Delta" else [[0, '#32005A'], [0.25, '#BF00FF'], [0.48, '#001E00'], [0.5, '#00C800'], [0.52, '#001E00'], [0.75, '#FFB400'], [1, '#FFFFB4']],
        zmid=0
    ))

    # Static annotations for key cells
    for i, strike in enumerate(y_labels):
        for j, expiry in enumerate(x_labels):
            val = z_raw[i, j]
            if abs(val) < (np.max(np.abs(z_raw)) * 0.1): continue # Hide tiny values
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            if (i, j) == max_idx: txt = f"★ {txt}"
            color = "black" if (np.abs(z_scaled[i,j]) > np.max(np.abs(z_scaled)) * 0.6) else "white"
            fig_heat.add_annotation(x=expiry, y=strike, text=txt, showarrow=False, font=dict(color=color, size=10, family="Arial"))

    fig_heat.update_layout(title=f"<b>{ticker}</b> {mode} Analysis", template="plotly_dark", height=700, xaxis={'type': 'category'})
    fig_heat.add_hline(y=S, line_dash="dash", line_color="white", annotation_text=f"SPOT: {S:.2f}")
    
    fig_bar = go.Figure(go.Bar(x=strike_agg.index, y=strike_agg.values, marker_color=['#BF00FF' if x < 0 else '#FFD700' for x in strike_agg.values]))
    fig_bar.update_layout(title=f"Aggregate {mode} Profile", template="plotly_dark", height=400)

    return fig_heat, fig_bar

# -------------------------
# UI
# -------------------------
def main():
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker (e.g. SPX, TSLA)", "SPX").upper()
        mode = st.radio("Mode", ["Gamma", "Delta"])
        max_exp = st.slider("Expirations", 1, 15, 5) # Reduced default to save rate limits
        s_range = st.slider("Strike Range", 10, 500, 100)
        boost = st.slider("Sensitivity", 1.0, 5.0, 3.0)
        run = st.button("Run Analysis", type="primary")

    if run:
        S, raw = fetch_options_data(ticker, max_exp)
        if S and not raw.empty:
            data = compute_exposure(raw, S, s_range)
            if not data.empty:
                h_map, b_chart = plot_analysis(data, ticker, S, boost, mode)
                st.subheader(f"📊 {ticker} Market Context")
                st.metric("Spot Price", f"${S:.2f}")
                st.plotly_chart(h_map, use_container_width=True)
                st.plotly_chart(b_chart, use_container_width=True)
            else:
                st.warning("No data found in that strike range.")
        else:
            st.error("Could not fetch data. Try reducing 'Expirations' or wait 1 minute.")

if __name__ == "__main__":
    main()