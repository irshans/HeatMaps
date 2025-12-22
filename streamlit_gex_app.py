import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Gamma Exposure Heatmap",
    page_icon="📊",
    layout="wide"
)

CONTRACT_SIZE = 100
FALLBACK_IV = 0.25

# -------------------------
# Black-Scholes & Greeks
# -------------------------
def norm_pdf(x):
    return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def norm_cdf(x):
    return 0.5*(1 + math.erf(x/math.sqrt(2)))

def bs_gamma(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

def bs_price(option_type, S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type == 'call':
        return math.exp(-q*T)*S*norm_cdf(d1) - math.exp(-r*T)*K*norm_cdf(d2)
    else:
        return math.exp(-r*T)*K*norm_cdf(-d2) - math.exp(-q*T)*S*norm_cdf(-d1)

def implied_vol_bisect(option_type, market_price, S, K, r, q, T, tol=1e-4, max_iter=80):
    low, high = 1e-4, 5.0
    try:
        if not (bs_price(option_type, S, K, r, q, low, T) <= market_price <= bs_price(option_type, S, K, r, q, high, T)):
            return None
    except:
        return None
    for _ in range(max_iter):
        mid = 0.5*(low+high)
        p_mid = bs_price(option_type, S, K, r, q, mid, T)
        if abs(p_mid - market_price) < tol:
            return mid
        if p_mid < market_price:
            low = mid
        else:
            high = mid
    return 0.5*(low+high)

# -------------------------
# Fetch options
# -------------------------
@st.cache_data(ttl=300)
def fetch_options_yahoo(ticker, max_expirations=5):
    stock = yf.Ticker(ticker)
    try:
        expirations = stock.options[:max_expirations]
    except Exception as e:
        st.warning(f"Error fetching expirations: {e}")
        return pd.DataFrame()
    dfs = []
    for exp in expirations:
        try:
            chain = stock.option_chain(exp)
            calls = chain.calls.assign(option_type="call", expiration=exp)
            puts = chain.puts.assign(option_type="put", expiration=exp)
            dfs.append(pd.concat([calls, puts], ignore_index=True))
        except Exception as e:
            st.warning(f"Error fetching {exp}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# -------------------------
# Compute gamma exposure
# -------------------------
def compute_gex(df, S, r=0.0, q=0.0, fallback_iv=FALLBACK_IV, strike_range=30):
    df = df.copy()
    # Fill missing prices with best available
    df["mid_price"] = df[["bid", "ask", "lastPrice"]].bfill(axis=1).iloc[:, 0]
    
    # Time logic fix: Use UTC to avoid local timezone offsets and add market close buffer
    df["expiration_dt"] = pd.to_datetime(df["expiration"])
    now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    
    # Use 21:00 UTC (approx 4PM ET) to represent market close
    df["expiration_dt_with_time"] = df["expiration_dt"] + pd.Timedelta(hours=21)
    df["T"] = (df["expiration_dt_with_time"] - now).dt.total_seconds() / (365*24*3600)
    
    # Filter for strike range and positive time to expiry
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range)]
    df = df[df["T"] > 0]

    out_rows = []
    for _, row in df.iterrows():
        K = float(row["strike"])
        T = float(row["T"])
        oi = float(row.get("openInterest") or 0.0)
        opt_type = row.get("option_type", "call").lower()
        mid = row.get("mid_price")

        b, a = row.get("bid"), row.get("ask")
        market_price = 0.5*(b+a) if (pd.notnull(b) and pd.notnull(a) and b > 0) else mid
        
        sigma = implied_vol_bisect(opt_type, market_price, S, K, r, q, T)
        if sigma is None or sigma <= 0:
            sigma = fallback_iv

        gamma = bs_gamma(S, K, r, q, sigma, T)
        raw_dollar_gamma = gamma * S**2
        gex_total = raw_dollar_gamma * CONTRACT_SIZE * oi * 0.01
        
        if opt_type == "put":
            gex_total *= -1

        out_rows.append({
            "strike": K,
            "expiry": row["expiration"],
            "option_type": opt_type,
            "open_interest": oi,
            "sigma": sigma,
            "gex_total": gex_total,
            "premium": mid if mid is not None else 0.0
        })
    return pd.DataFrame(out_rows)

# -------------------------
# Heatmap Plotting
# -------------------------
def plot_heatmap(gex_df, ticker, S):
    pivot = gex_df.pivot_table(
        index='strike',
        columns='expiry',
        values='gex_total',
        aggfunc='sum',
        fill_value=0
    ).sort_index(ascending=False)

    z = pivot.values
    x = list(pivot.columns)
    y = list(pivot.index)

    # Create text matrix for display
    text_matrix = [[f"${v:,.0f}" for v in row] for row in z]

    colorscale = [
        [0.0, 'rgb(75,0,130)'],   # deep purple negative
        [0.5, 'rgb(20,20,20)'],    # dark neutral
        [1.0, 'rgb(255,215,0)']    # yellow positive
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=colorscale,
        zmid=0,
        showscale=True
    ))

    fig.add_hline(y=S, line_dash="dash", line_color="red", annotation_text="Spot")
    
    fig.update_layout(
        title=f"{ticker} Net GEX Heatmap",
        xaxis_title="Expiration Date",
        yaxis_title="Strike Price",
        height=800,
        template="plotly_dark"
    )
    return fig

# -------------------------
# Main App
# -------------------------
def main():
    st.title("📊 Gamma Exposure Heatmap")

    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
    
    # UPDATED SLIDERS
    max_expirations = st.sidebar.slider("Number of Expirations", 1, 10, 5)
    strike_range = st.sidebar.slider("Strike Range ($)", 5, 50, 30)

    if st.sidebar.button("🚀 Generate Heatmap", type="primary"):
        try:
            with st.spinner(f"Loading {ticker} data..."):
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if hist.empty:
                    st.error("Could not fetch stock price. Check ticker symbol.")
                    return
                
                S = hist["Close"].iloc[-1]
                st.metric("Current Spot Price", f"${S:.2f}")

                df = fetch_options_yahoo(ticker, max_expirations=max_expirations)
                if df.empty:
                    st.error("No options data found.")
                    return

                gex = compute_gex(df, S, strike_range=strike_range)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total GEX", f"${gex['gex_total'].sum():,.0f}")
                col2.metric("Call GEX", f"${gex[gex['option_type']=='call']['gex_total'].sum():,.0f}")
                col3.metric("Put GEX", f"${gex[gex['option_type']=='put']['gex_total'].sum():,.0f}")

                fig = plot_heatmap(gex, ticker, S)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing data: {e}")

if __name__ == "__main__":
    main()