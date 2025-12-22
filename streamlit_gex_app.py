import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import time

# -------------------------
# CONFIG
# -------------------------
CONTRACT_SIZE = 100
FALLBACK_IV = 0.25
MAX_EXPIRATIONS_HARD_CAP = 10
MAX_STRIKES_AROUND_ATM = 20  # updated to 20

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
# Fetch options from Yahoo Finance with retry
# -------------------------
@st.cache_data(ttl=300)
def fetch_options_yahoo_safe(ticker, max_expirations=5, max_retries=3, delay=2):
    stock = yf.Ticker(ticker)
    try:
        expirations = stock.options[:max_expirations]
    except Exception as e:
        st.warning(f"Unable to get expirations for {ticker}: {e}")
        return pd.DataFrame()

    dfs = []
    for exp in expirations:
        retries = 0
        while retries < max_retries:
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls.assign(option_type="call", expiration=exp)
                puts = chain.puts.assign(option_type="put", expiration=exp)
                dfs.append(pd.concat([calls, puts], ignore_index=True))
                break
            except Exception as e:
                retries += 1
                time.sleep(delay)
                if retries >= max_retries:
                    st.warning(f"Failed to fetch {exp} after {max_retries} retries: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# -------------------------
# Compute gamma exposure
# -------------------------
def compute_gex(df, S, r=0.0, q=0.0, fallback_iv=FALLBACK_IV, strike_range=20):
    df = df.copy()
    df["mid_price"] = df[["bid","ask","lastPrice"]].bfill(axis=1).iloc[:,0]
    df["expiration_dt"] = pd.to_datetime(df["expiration"])
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range)]
    now = pd.Timestamp.now(tz=None)
    df["T"] = ((df["expiration_dt"] + pd.Timedelta(hours=16)) - now).dt.total_seconds() / (365*24*3600)
    df = df[df["T"] > 1/3650.0]

    rows = []
    for _, row in df.iterrows():
        K = float(row["strike"])
        T = float(row["T"])
        oi = float(row.get("openInterest") or 0.0)
        opt_type = row.get("option_type", "call").lower()
        mid = row.get("mid_price")

        sigma = None
        if mid is not None:
            b, a = row.get("bid"), row.get("ask")
            market_price = 0.5*(b+a) if b is not None and a is not None else mid
            sigma = implied_vol_bisect(opt_type, market_price, S, K, r, q, T)
        if sigma is None or sigma <= 0:
            sigma = fallback_iv

        gamma = bs_gamma(S, K, r, q, sigma, T)
        raw_dollar_gamma = gamma * S**2
        sign = 1 if opt_type == "call" else -1
        gex_total = raw_dollar_gamma * CONTRACT_SIZE * oi * 0.01 * sign

        rows.append({
            "strike": K,
            "expiry": row["expiration"],
            "option_type": opt_type,
            "open_interest": oi,
            "sigma": sigma,
            "gamma": gamma,
            "raw_dollar_gamma": raw_dollar_gamma,
            "gex_total": gex_total,
            "premium": mid if mid is not None else 0.0
        })
    return pd.DataFrame(rows)

# -------------------------
# Plot heatmap with color thresholds + star
# -------------------------
def plot_heatmap(gex_df, ticker, S, max_strikes=MAX_STRIKES_AROUND_ATM):
    strikes_sorted = sorted(gex_df['strike'].unique())
    closest_idx = np.abs(np.array(strikes_sorted)-S).argmin()
    start = max(0, closest_idx - max_strikes)
    end = min(len(strikes_sorted), closest_idx + max_strikes + 1)
    strikes_to_use = strikes_sorted[start:end]
    df_plot = gex_df[gex_df['strike'].isin(strikes_to_use)]

    pivot = df_plot.pivot_table(
        index='strike',
        columns='expiry',
        values='gex_total',
        aggfunc='sum',
        fill_value=0
    ).sort_index(ascending=False)

    strikes = list(pivot.index)
    expiries = list(pivot.columns)
    z = pivot.values

    # Identify max absolute GEX
    max_idx = np.unravel_index(np.argmax(np.abs(z)), z.shape)

    # Prepare text matrix with star on max absolute GEX
    text_matrix = []
    for i, row in enumerate(z):
        row_text = []
        for j, val in enumerate(row):
            val_str = f"${val:,.0f}"
            if (i, j) == max_idx:
                val_str += " ★"
            row_text.append(val_str)
        text_matrix.append(row_text)

    # Build custom color scale
    colors = []
    for row in z:
        color_row = []
        for val in row:
            if val <= -1_000_000:
                color_row.append("rgb(75,0,130)")  # deep purple
            elif val >= 1_000_000:
                color_row.append("rgb(255,215,0)")  # gold/yellow
            else:
                color_row.append("rgb(0,128,0)")  # green
        colors.append(color_row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=expiries,
        y=strikes,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate="Expiry: %{x}<br>Strike: %{y}<br>GEX: %{z:,.0f}<extra></extra>",
        showscale=True,
        colorscale=None,
        zmin=np.min(z),
        zmax=np.max(z),
        xgap=1,
        ygap=1,
        zsmooth=False,
        colors=colors
    ))

    # ATM line
    fig.add_hline(
        y=S,
        line_color='red',
        line_width=1,
        annotation_text='ATM',
        annotation_position='top right'
    )

    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        title=f"{ticker} – Net Gamma Exposure Heatmap",
        xaxis_title='Expiry',
        yaxis_title='Strike',
        height=800
    )
    return fig

# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    st.title("📊 Gamma Exposure Heatmap")
    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
    max_expirations = st.sidebar.slider("Max Expirations", 1, MAX_EXPIRATIONS_HARD_CAP, 5)
    strike_range = st.sidebar.slider("Strike Range ($)", 5, 100, 50, step=5)

    if st.sidebar.button("🚀 Generate Heatmap"):
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                stock = yf.Ticker(ticker)
                S = stock.history(period="1d")["Close"].iloc[-1]
                st.info(f"**Current Price:** ${S:.2f}")

                df = fetch_options_yahoo_safe(ticker, max_expirations=max_expirations)
                if df.empty:
                    st.error("⚠️ Could not fetch any options data. Yahoo may be rate-limiting requests.")
                    return

                gex = compute_gex(df, S, strike_range=strike_range)
                if gex.empty:
                    st.error("No gamma exposure computed.")
                    return

                fig = plot_heatmap(gex, ticker, S)
                st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👈 Enter a ticker symbol and click 'Generate Heatmap' to get started!")

if __name__ == "__main__":
    main()
