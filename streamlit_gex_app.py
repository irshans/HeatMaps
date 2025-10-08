"""
Streamlit Gamma Exposure Heatmap App

Installation:
    pip install streamlit yfinance pandas numpy plotly

Run locally:
    streamlit run streamlit_gex_app.py

Deploy to Streamlit Cloud (free):
    1. Push this file to GitHub
    2. Go to share.streamlit.io
    3. Connect your GitHub repo
    4. Deploy!
"""

import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# Page config
st.set_page_config(
    page_title="Gamma Exposure Heatmap",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# CONFIG
# -------------------------
CONTRACT_SIZE = 100
FALLBACK_IV = 0.25

# -------------------------
# Black-Scholes & Greeks
# -------------------------
def norm_pdf(x):
    return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def norm_cdf(x):
    return 0.5*(1 + math.erf(x/math.sqrt(2)))

def bs_price(option_type, S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type == 'call':
        return math.exp(-q*T)*S*norm_cdf(d1) - math.exp(-r*T)*K*norm_cdf(d2)
    else:
        return math.exp(-r*T)*K*norm_cdf(-d2) - math.exp(-q*T)*S*norm_cdf(-d1)

def bs_gamma(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

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
# Fetch options from Yahoo Finance
# -------------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_options_yahoo(ticker, max_expirations=2):
    stock = yf.Ticker(ticker)
    expirations = stock.options[:max_expirations]
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
    options = pd.concat(dfs, ignore_index=True)
    return options

# -------------------------
# Compute gamma exposure
# -------------------------
def compute_gex(df, S, r=0.0, q=0.0, fallback_iv=FALLBACK_IV, strike_range=20):
    df = df.copy()
    df["mid_price"] = df[["bid", "ask", "lastPrice"]].bfill(axis=1).iloc[:, 0]
    
    # Use the raw expiration date string from Yahoo Finance - no manipulation
    df["expiry_display"] = df["expiration"]
    
    # Parse for time calculations only
    df["expiration_dt"] = pd.to_datetime(df["expiration"])

    # Only keep strikes within +/- strike_range of spot
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range)]

    # Compute time to expiration
    now = pd.Timestamp.now(tz=None)
    df["expiration_dt_with_time"] = df["expiration_dt"] + pd.Timedelta(hours=16)
    df["T"] = (df["expiration_dt_with_time"] - now).dt.total_seconds() / (365*24*3600)
    df = df[df["T"] > 1/3650.0]

    out_rows = []
    for _, row in df.iterrows():
        K = float(row["strike"])
        expiry = row["expiry_display"]
        T = float(row["T"])
        oi = float(row.get("openInterest") or 0.0)
        opt_type = row.get("option_type", "call").lower()
        mid = row.get("mid_price")

        # implied vol
        sigma = None
        if mid is not None:
            b, a = row.get("bid"), row.get("ask")
            market_price = 0.5*(b+a) if b is not None and a is not None else mid
            sigma = implied_vol_bisect(opt_type, market_price, S, K, r, q, T)
        if sigma is None or sigma <= 0:
            sigma = fallback_iv

        gamma = bs_gamma(S, K, r, q, sigma, T)
        raw_dollar_gamma = gamma * S**2
        gex_total = raw_dollar_gamma * CONTRACT_SIZE * oi * 0.01

        out_rows.append({
            "strike": K,
            "expiry": expiry,
            "option_type": opt_type,
            "open_interest": oi,
            "sigma": sigma,
            "gamma": gamma,
            "raw_dollar_gamma": raw_dollar_gamma,
            "gex_total": gex_total,
            "premium": mid if mid is not None else 0.0
        })
    return pd.DataFrame(out_rows)

# -------------------------
# Plot heatmap
# -------------------------
def plot_heatmap(gex_df, ticker, S):
    # Sort data
    gex_df = gex_df.sort_values("expiry")
    pivot_gex = gex_df.pivot_table(
        index="strike",
        columns="expiry",
        values="gex_total",
        aggfunc="sum",
        fill_value=0
    )
    pivot_gex = pivot_gex.sort_index(ascending=False)

    y_strikes = list(pivot_gex.index)
    x_exps = list(pivot_gex.columns)

    # Text annotations
    text_annotations = []
    for i, strike in enumerate(y_strikes):
        row_text = []
        for gex in pivot_gex.values[i]:
            if gex == 0:
                row_text.append("")
            else:
                sign = "-" if strike <= S else ""
                row_text.append(f"${sign}{abs(gex):,.0f}")
        text_annotations.append(row_text)

    # Signed GEX (negative below ATM)
    z_data = pivot_gex.values.copy()
    z_signed = np.zeros_like(z_data)
    for i, strike in enumerate(y_strikes):
        z_signed[i, :] = -z_data[i, :] if strike <= S else z_data[i, :]

    # Normalize per expiration column
    z_normalized = np.zeros_like(z_signed)
    for col_idx in range(z_signed.shape[1]):
        col_data = z_signed[:, col_idx]
        max_abs = np.abs(col_data).max()
        z_normalized[:, col_idx] = col_data / max_abs if max_abs > 0 else 0

    # --- DARKER MUTED COLOR PALETTE ---
    custom_colorscale = [
        [0.00, 'rgb(40, 10, 60)'],     # Deep purple (strongest negative)
        [0.15, 'rgb(20, 40, 80)'],     # Dark navy blue
        [0.35, 'rgb(25, 70, 85)'],     # Muted teal
        [0.50, 'rgb(30, 90, 80)'],     # Deep blue-green neutral
        [0.65, 'rgb(45, 110, 80)'],    # Forest green
        [0.85, 'rgb(80, 130, 70)'],    # Muted green
        [1.00, 'rgb(180, 180, 100)']   # Soft gold (highest positive)
    ]

    # Plotly Heatmap
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_normalized,
        x=x_exps,
        y=y_strikes,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        colorscale=custom_colorscale,
        zmid=0,
        showscale=True,
        colorbar=dict(
            title="Relative<br>GEX",
            ticktext=["Strong Negative", "Neutral", "Strong Positive"],
            tickvals=[-1, 0, 1]
        ),
        customdata=z_data,
        hovertemplate="Expiry: %{x}<br>Strike: %{y}<br>GEX: %{customdata:,.0f}<extra></extra>"
    ))

    # Highlight ATM
    fig.add_hline(
        y=S,
        line_dash="dash",
        line_color="white",
        annotation_text="ATM",
        annotation_position="top right"
    )

    # Layout
    fig.update_layout(
        title=f"{ticker} - Gamma Exposure Heatmap",
        xaxis_title="Expiry",
        yaxis_title="Strike",
        xaxis=dict(type="category"),
        yaxis=dict(
            tickmode="array",
            tickvals=y_strikes,
            ticktext=[str(s) for s in y_strikes]
        ),
        autosize=True,
        height=800,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )
    return fig

# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    st.title("📊 Gamma Exposure Heatmap")
    st.markdown("Visualize option gamma exposure across strikes and expirations")
    
    # Sidebar inputs
    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
    max_expirations = st.sidebar.slider("Max Expirations", 1, 10, 5)
    strike_range = st.sidebar.slider("Strike Range ($)", 5, 100, 50, step=5)
    
    # Color legend
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Color Legend")
    st.sidebar.markdown("**Above ATM:** Blue → Yellow (Higher GEX)")
    st.sidebar.markdown("**Below ATM:** Blue → Purple (Higher GEX)")
    
    # Generate button
    if st.sidebar.button("🚀 Generate Heatmap", type="primary"):
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                # Get underlying price
                stock = yf.Ticker(ticker)
                S = stock.history(period="1d")["Close"].iloc[-1]
                
                st.info(f"**Current Price:** ${S:.2f}")
                
                # Fetch options
                df = fetch_options_yahoo(ticker, max_expirations=max_expirations)
                
                if df.empty:
                    st.error(f"No options data found for {ticker}")
                    return
                
                # Compute gamma exposure
                gex = compute_gex(df, S, fallback_iv=FALLBACK_IV, strike_range=strike_range)
                
                if gex.empty:
                    st.error("No gamma exposure computed.")
                    return
                
                # Aggregate and plot
                agg = gex.groupby(["strike", "expiry"]).agg({
                    "gex_total": "sum",
                    "premium": "mean"
                }).reset_index()
                
                fig = plot_heatmap(agg, ticker, S)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total GEX", f"${agg['gex_total'].sum():,.0f}")
                with col2:
                    st.metric("Max Strike", f"${agg['strike'].max():.0f}")
                with col3:
                    st.metric("Min Strike", f"${agg['strike'].min():.0f}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Enter a ticker symbol and click 'Generate Heatmap' to get started!")
        
        # Show example
        st.markdown("### Example Output")
        st.markdown("This tool will display a heatmap showing gamma exposure across different strikes and expiration dates.")

if __name__ == "__main__":
    main()