import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(
    page_title="Gamma Exposure Analytics",
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
    df["mid_price"] = df[["bid", "ask", "lastPrice"]].bfill(axis=1).iloc[:, 0]
    
    df["expiration_dt"] = pd.to_datetime(df["expiration"])
    now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    df["expiration_dt_with_time"] = df["expiration_dt"] + pd.Timedelta(hours=21)
    df["T"] = (df["expiration_dt_with_time"] - now).dt.total_seconds() / (365*24*3600)
    
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range)]
    df = df[df["T"] > 0]

    out_rows = []
    for _, row in df.iterrows():
        K, T = float(row["strike"]), float(row["T"])
        oi = float(row.get("openInterest") or 0.0)
        opt_type = row.get("option_type", "call").lower()
        mid = row.get("mid_price")
        b, a = row.get("bid"), row.get("ask")
        market_price = 0.5*(b+a) if (pd.notnull(b) and pd.notnull(a) and b > 0) else mid
        
        sigma = implied_vol_bisect(opt_type, market_price, S, K, r, q, T)
        if sigma is None or sigma <= 0:
            sigma = fallback_iv

        gamma = bs_gamma(S, K, r, q, sigma, T)
        gex_total = (gamma * S**2) * CONTRACT_SIZE * oi * 0.01
        if opt_type == "put": gex_total *= -1

        out_rows.append({"strike": K, "expiry": row["expiration"], "option_type": opt_type, "gex_total": gex_total})
    return pd.DataFrame(out_rows)

# -------------------------
# Plotting
# -------------------------
def plot_charts(gex_df, ticker, S):
    pivot = gex_df.pivot_table(index='strike', columns='expiry', values='gex_total', aggfunc='sum', fill_value=0).sort_index(ascending=False)
    
    z = pivot.values
    expiries = list(pivot.columns)
    strikes = list(pivot.index)

    # Star logic
    abs_z = np.abs(z)
    max_idx = np.unravel_index(np.argmax(abs_z), z.shape)

    text_matrix = []
    for i in range(len(strikes)):
        row_text = []
        for j in range(len(expiries)):
            val = z[i, j]
            val_str = f"${val/1e6:.1f}M" if abs(val) >= 1e6 else f"${val:,.0f}"
            if (i, j) == max_idx:
                val_str += " ★"
            row_text.append(val_str)
        text_matrix.append(row_text)

    # CUSTOM COLOR SCHEME: Purple (Negative) -> Green (Middle/Near Zero) -> Yellow (High Positive)
    # We use a 5-point scale to ensure Green is the "neutral" zone
    custom_colorscale = [
        [0.0, 'rgb(48, 0, 77)'],    # Dark Purple (Extreme Negative)
        [0.25, 'rgb(149, 117, 205)'], # Light Purple (Low Negative)
        [0.5, 'rgb(0, 100, 0)'],    # Dark Green (Zero/Middle)
        [0.75, 'rgb(144, 238, 144)'], # Light Green (Low Positive)
        [1.0, 'rgb(255, 215, 0)']    # Yellow (High Positive)
    ]

    fig_heat = go.Figure(data=go.Heatmap(
        z=z, x=expiries, y=strikes,
        text=text_matrix, texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        colorscale=custom_colorscale, 
        zmid=0, # Forces the 0.5 color (Dark Green) to be at 0 GEX
        showscale=True
    ))
    
    fig_heat.add_hline(y=S, line_dash="dash", line_color="white", annotation_text="Spot")
    fig_heat.update_layout(title=f"{ticker} Gamma Exposure Heatmap", height=800, template="plotly_dark")

    # Bar Chart
    strike_agg = gex_df.groupby('strike')['gex_total'].sum().reset_index()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=strike_agg['strike'], y=strike_agg['gex_total'],
        marker_color=['#9575CD' if x < 0 else '#90EE90' for x in strike_agg['gex_total']]
    ))
    fig_bar.add_vline(x=S, line_dash="dash", line_color="red", annotation_text="Spot")
    fig_bar.update_layout(title=f"Total Gamma Concentration by Strike ({ticker})", xaxis_title="Strike", yaxis_title="Net GEX ($)", template="plotly_dark")

    return fig_heat, fig_bar

def main():
    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
    max_expirations = st.sidebar.slider("Number of Expirations", 1, 10, 5)
    strike_range = st.sidebar.slider("Strike Range ($)", 5, 50, 30)

    if st.sidebar.button("🚀 Generate Analysis", type="primary"):
        with st.spinner("Processing..."):
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                st.error("Ticker not found."); return
            
            S = hist["Close"].iloc[-1]
            df = fetch_options_yahoo(ticker, max_expirations)
            gex = compute_gex(df, S, strike_range=strike_range)

            st.header(f"Gamma Analysis: {ticker} @ ${S:.2f}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Net GEX", f"${gex['gex_total'].sum():,.0f}")
            c2.metric("Call GEX", f"${gex[gex['gex_total']>0]['gex_total'].sum():,.0f}")
            c3.metric("Put GEX", f"${gex[gex['gex_total']<0]['gex_total'].sum():,.0f}")

            fig_heat, fig_bar = plot_charts(gex, ticker, S)
            st.plotly_chart(fig_heat, use_container_width=True)
            st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()