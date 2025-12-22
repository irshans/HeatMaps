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
        if not (bs_price(option_type, S, K, r, q, low, T)
                <= market_price
                <= bs_price(option_type, S, K, r, q, high, T)):
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
@st.cache_data(ttl=300)
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
    return pd.concat(dfs, ignore_index=True)

# -------------------------
# Compute gamma exposure
# -------------------------
def compute_gex(df, S, r=0.0, q=0.0, fallback_iv=FALLBACK_IV, strike_range=20):
    df = df.copy()
    df["mid_price"] = df[["bid", "ask", "lastPrice"]].bfill(axis=1).iloc[:, 0]

    df["expiry_display"] = df["expiration"]
    df["expiration_dt"] = pd.to_datetime(df["expiration"])

    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range)]

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

        # Use Yahoo IV if available
        sigma = row.get("impliedVolatility")
        if not sigma or sigma <= 0:
            if mid is not None:
                b, a = row.get("bid"), row.get("ask")
                market_price = 0.5*(b+a) if b is not None and a is not None else mid
                sigma = implied_vol_bisect(opt_type, market_price, S, K, r, q, T)

        if sigma is None or sigma <= 0:
            sigma = fallback_iv

        gamma = bs_gamma(S, K, r, q, sigma, T)
        raw_dollar_gamma = gamma * S**2

        # 🔑 SIGN FIX: calls positive, puts negative
        sign = 1 if opt_type == "call" else -1
        gex_total = sign * raw_dollar_gamma * CONTRACT_SIZE * oi * 0.01

        out_rows.append({
            "strike": K,
            "expiry": expiry,
            "option_type": opt_type,
            "open_interest": oi,
            "sigma": sigma,
            "gamma": gamma,
            "gex_total": gex_total,
            "premium": mid if mid is not None else 0.0
        })

    return pd.DataFrame(out_rows)

# -------------------------
# Plot heatmap
# -------------------------
def plot_heatmap(gex_df, ticker, S):
    gex_df = gex_df.sort_values("expiry")

    pivot_gex = gex_df.pivot_table(
        index="strike",
        columns="expiry",
        values="gex_total",
        aggfunc="sum",
        fill_value=0
    ).sort_index(ascending=False)

    y_strikes = list(pivot_gex.index)
    x_exps = list(pivot_gex.columns)

    z_data = pivot_gex.values

    # Normalize per expiry (column)
    z_normalized = np.zeros_like(z_data)
    for col in range(z_data.shape[1]):
        max_abs = np.abs(z_data[:, col]).max()
        if max_abs > 0:
            z_normalized[:, col] = z_data[:, col] / max_abs

    text_annotations = [
        ["" if v == 0 else f"${v:,.0f}" for v in row]
        for row in z_data
    ]

    fig = go.Figure(go.Heatmap(
        z=z_normalized,
        x=x_exps,
        y=y_strikes,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        colorscale=[
            [0.0, "rgb(75, 0, 130)"],     # Strong negative (puts)
            [0.45, "rgb(30, 60, 114)"],
            [0.5, "rgb(200, 200, 200)"],
            [0.75, "rgb(255, 215, 0)"],
            [1.0, "rgb(255, 140, 0)"]     # Strong positive (calls)
        ],
        zmid=0,
        customdata=z_data,
        hovertemplate="Expiry: %{x}<br>Strike: %{y}<br>Net GEX: %{customdata:,.0f}<extra></extra>",
        colorbar=dict(title="Relative Net GEX")
    ))

    fig.add_hline(
        y=S,
        line_color="red",
        line_width=1,
        annotation_text="ATM",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"{ticker} – Net Gamma Exposure Heatmap",
        xaxis_title="Expiry",
        yaxis_title="Strike",
        height=800
    )

    return fig

# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    st.title("📊 Gamma Exposure Heatmap")

    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Ticker", "TSLA").upper()
    max_expirations = st.sidebar.slider("Max Expirations", 1, 10, 5)
    strike_range = st.sidebar.slider("Strike Range ($)", 5, 100, 50, step=5)

    if st.sidebar.button("🚀 Generate Heatmap", type="primary"):
        with st.spinner("Fetching data..."):
            stock = yf.Ticker(ticker)
            S = stock.history(period="1d")["Close"].iloc[-1]

            df = fetch_options_yahoo(ticker, max_expirations)
            gex = compute_gex(df, S, strike_range=strike_range)

            agg = gex.groupby(["strike", "expiry"]).agg(
                gex_total=("gex_total", "sum")
            ).reset_index()

            fig = plot_heatmap(agg, ticker, S)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
