import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta

# --- Page Setup ---
st.set_page_config(page_title="GEX Heatmap Pro", page_icon="📊", layout="wide")

CONTRACT_SIZE = 100
FALLBACK_IV = 0.25

# -------------------------
# Helpers & Math
# -------------------------
def fix_expiration_date(date_str):
    """Corrects yfinance Sunday/Saturday dates to Fridays for trading accuracy."""
    dt = pd.to_datetime(date_str)
    if dt.weekday() == 6: # Sunday
        dt = dt - timedelta(days=2)
    elif dt.weekday() == 5: # Saturday
        dt = dt - timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

def norm_pdf(x): 
    return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def bs_gamma(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

# -------------------------
# Data Fetching
# -------------------------
@st.cache_data(ttl=300)
def fetch_options_yahoo(ticker, max_expirations):
    stock = yf.Ticker(ticker)
    try:
        exp_list = stock.options[:max_expirations]
    except:
        return pd.DataFrame()
    
    dfs = []
    for exp in exp_list:
        try:
            chain = stock.option_chain(exp)
            # Apply Sunday -> Friday fix
            trading_date = fix_expiration_date(exp)
            calls = chain.calls.assign(option_type="call", expiration=trading_date)
            puts = chain.puts.assign(option_type="put", expiration=trading_date)
            dfs.append(pd.concat([calls, puts], ignore_index=True))
        except:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# -------------------------
# Gamma Calculation
# -------------------------
def compute_gex(df, S, strike_range):
    df = df.copy()
    now = pd.Timestamp.now()
    
    # Calculate Time to Expiry (T)
    df["T"] = (pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=16) - now).dt.total_seconds() / (365*24*3600)
    
    # Filter for Strike Range & valid T
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range) & (df["T"] > 0)]

    out = []
    for _, row in df.iterrows():
        K, T, oi = float(row["strike"]), float(row["T"]), float(row.get("openInterest") or 0)
        # Use provided IV or fallback
        sigma = row.get("impliedVolatility") if row.get("impliedVolatility") and row.get("impliedVolatility") > 0 else FALLBACK_IV
        
        gamma = bs_gamma(S, K, 0.045, 0.0, sigma, T) # Assumes 4.5% Risk Free Rate
        dollar_gex = (gamma * S**2) * CONTRACT_SIZE * oi * 0.01
        
        if row["option_type"] == "put":
            dollar_gex *= -1
            
        out.append({"strike": K, "expiry": row["expiration"], "gex_total": dollar_gex, "oi": oi})
    return pd.DataFrame(out)

# -------------------------
# Plotting
# -------------------------
def plot_analysis(gex_df, ticker, S):
    # Pivot Data
    pivot = gex_df.pivot_table(index='strike', columns='expiry', values='gex_total', aggfunc='sum', fill_value=0).sort_index(ascending=False)
    z = pivot.values
    
    # Identify Max Absolute GEX for the Star
    abs_z = np.abs(z)
    max_idx = np.unravel_index(np.argmax(abs_z), z.shape) if z.size > 0 else (0,0)

    # Generate labels
    text_matrix = []
    for i in range(len(pivot.index)):
        row_text = []
        for j in range(len(pivot.columns)):
            val = z[i, j]
            txt = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            if (i, j) == max_idx and abs(val) > 0: txt += " ★"
            row_text.append(txt)
        text_matrix.append(row_text)

    # NARROW-BAND SCALE: Purple (Neg) | Green (Zero) | Yellow (Pos)
    custom_colorscale = [
        [0.0, 'rgb(48, 0, 77)'],    # Deep Purple
        [0.4, 'rgb(180, 140, 255)'],# Light Purple
        [0.48, 'rgb(0, 50, 0)'],    # Start Green
        [0.5, 'rgb(0, 200, 0)'],    # Bright Green (Exactly Zero)
        [0.52, 'rgb(0, 50, 0)'],    # End Green
        [0.6, 'rgb(255, 255, 150)'],# Light Yellow
        [1.0, 'rgb(255, 215, 0)']    # Deep Yellow
    ]

    # Heatmap
    fig_heat = go.Figure(data=go.Heatmap(
        z=z, x=pivot.columns, y=pivot.index,
        text=text_matrix, texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        colorscale=custom_colorscale, zmid=0, showscale=True
    ))
    
    fig_heat.add_hline(y=S, line_dash="dash", line_color="white", annotation_text=f"Spot: {S:.2f}", annotation_position="bottom right")
    fig_heat.update_layout(title=f"Net Gamma Exposure: {ticker}", xaxis_title="Expiration", yaxis_title="Strike", height=800, template="plotly_dark")

    # Aggregate Bar Chart
    strike_agg = gex_df.groupby('strike')['gex_total'].sum().reset_index()
    fig_bar = go.Figure(go.Bar(
        x=strike_agg['strike'], y=strike_agg['gex_total'],
        marker_color=['#9575CD' if x < 0 else '#FFD700' for x in strike_agg['gex_total']]
    ))
    fig_bar.add_vline(x=S, line_dash="dash", line_color="red")
    fig_bar.update_layout(title="Total GEX by Strike (All Expirations)", template="plotly_dark")

    return fig_heat, fig_bar

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("📊 Options Gamma Analytics")
    
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker Symbol", "TSLA").upper()
        max_exp = st.slider("Max Expirations", 1, 10, 5)
        s_range = st.slider("Strike Range ($)", 5, 50, 30)
        btn = st.button("Generate Heatmap", type="primary")

    if btn:
        with st.spinner(f"Analyzing {ticker}..."):
            stock_obj = yf.Ticker(ticker)
            hist = stock_obj.history(period="1d")
            if hist.empty:
                st.error("Invalid Ticker.")
                return
            
            S = hist["Close"].iloc[-1]
            raw_data = fetch_options_yahoo(ticker, max_exp)
            
            if raw_data.empty:
                st.error("No options data available.")
                return

            gex_data = compute_gex(raw_data, S, s_range)
            
            if gex_data.empty:
                st.warning("No data found within the selected strike/date range.")
                return

            # Display Stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Total GEX", f"${gex_data['gex_total'].sum()/1e6:.2f}M")
            c2.metric("Call GEX", f"${gex_data[gex_data['gex_total']>0]['gex_total'].sum()/1e6:.2f}M")
            c3.metric("Put GEX", f"${gex_data[gex_data['gex_total']<0]['gex_total'].sum()/1e6:.2f}M")

            fig_h, fig_b = plot_analysis(gex_data, ticker, S)
            st.plotly_chart(fig_h, use_container_width=True)
            st.plotly_chart(fig_b, use_container_width=True)

if __name__ == "__main__":
    main()