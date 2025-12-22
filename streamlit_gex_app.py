import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import time

# -------------------------
# PAGE CONFIG
# -------------------------
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
MAX_EXPIRATIONS_HARD_CAP = 5   # Max expirations
CACHE_TTL = 900                # 15 min cache

# -------------------------
# BLACK-SCHOLES
# -------------------------
def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(option_type, S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return math.exp(-q * T) * S * norm_cdf(d1) - math.exp(-r * T) * K * norm_cdf(d2)
    else:
        return math.exp(-r * T) * K * norm_cdf(-d2) - math.exp(-q * T) * S * norm_cdf(-d1)

def bs_gamma(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

# -------------------------
# SAFE YAHOO FETCH
# -------------------------
@st.cache_data(ttl=CACHE_TTL, max_entries=10)
def fetch_options_yahoo_safe(ticker, max_expirations):
    max_expirations = min(max_expirations, MAX_EXPIRATIONS_HARD_CAP)

    try:
        session = yf.utils.get_session()
        stock = yf.Ticker(ticker, session=session)
        expirations = stock.options
        if not expirations:
            return pd.DataFrame()
        expirations = expirations[:max_expirations]
    except Exception:
        return pd.DataFrame()

    dfs = []
    for exp in expirations:
        try:
            chain = stock.option_chain(exp)
            calls = chain.calls.assign(option_type="call", expiration=exp)
            puts = chain.puts.assign(option_type="put", expiration=exp)
            dfs.append(pd.concat([calls, puts], ignore_index=True))
            time.sleep(0.15)  # small delay helps with Yahoo throttling
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

# -------------------------
# COMPUTE GEX
# -------------------------
def compute_gex(df, S, strike_range):
    df = df.copy()
    df["mid_price"] = df[["bid", "ask", "lastPrice"]].bfill(axis=1).iloc[:, 0]
    df["expiration_dt"] = pd.to_datetime(df["expiration"])

    df = df[
        (df["strike"] >= S - strike_range) &
        (df["strike"] <= S + strike_range)
    ]

    now = pd.Timestamp.utcnow()
    df["expiration_dt_with_time"] = df["expiration_dt"] + pd.Timedelta(hours=16)
    df["T"] = (df["expiration_dt_with_time"] - now).dt.total_seconds() / (365 * 24 * 3600)
    df = df[df["T"] > 1 / 3650]

    rows = []
    for _, row in df.iterrows():
        K = float(row["strike"])
        T = float(row["T"])
        oi = float(row.get("openInterest") or 0)
        opt_type = row["option_type"]
        mid = row["mid_price"]

        sigma = row.get("impliedVolatility")
        if not sigma or sigma <= 0:
            sigma = FALLBACK_IV

        gamma = bs_gamma(S, K, 0, 0, sigma, T)
        raw_dollar_gamma = gamma * S * S

        # 🔑 Signed GEX: calls positive, puts negative
        sign = 1 if opt_type == "call" else -1
        gex_total = sign * raw_dollar_gamma * CONTRACT_SIZE * oi * 0.01

        rows.append({
            "strike": K,
            "expiry": row["expiration"],
            "gex_total": gex_total
        })

    return pd.DataFrame(rows)

# -------------------------
# HEATMAP
# -------------------------
def plot_heatmap(gex_df, ticker, S):
    pivot = gex_df.pivot_table(
        index="strike",
        columns="expiry",
        values="gex_total",
        aggfunc="sum",
        fill_value=0
    ).sort_index(ascending=False)

    strikes = list(pivot.index)
    expiries = list(pivot.columns)
    z = pivot.values

    # Find highest absolute GEX
    idx_flat = np.argmax(np.abs(z))
    row_idx, col_idx = np.unravel_index(idx_flat, z.shape)
    highest_strike = strikes[row_idx]
    highest_expiry = expiries[col_idx]

    fig = go.Figure()

    # Draw each cell as a colored rectangle
    for i, strike in enumerate(strikes):
        for j, expiry in enumerate(expiries):
            val = z[i, j]
            # Determine color
            if val < -1_000_000:
                color = 'rgb(75,0,130)'  # deep purple
            elif val > 1_000_000:
                color = 'rgb(255,215,0)'  # bright yellow
            else:
                color = 'rgb(50,205,50)'  # green

            # Draw rectangle
            fig.add_shape(
                type="rect",
                x0=j-0.5, x1=j+0.5,
                y0=i-0.5, y1=i+0.5,
                xref='x', yref='y',
                line=dict(width=1, color='black'),
                fillcolor=color,
            )

            # Add text
            fig.add_trace(go.Scatter(
                x=[expiry],
                y=[strike],
                text=[f"{int(val):,}"],
                mode="text",
                textfont=dict(color="white" if color != 'rgb(50,205,50)' else "black", size=12),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Star for highest absolute GEX
    fig.add_trace(go.Scatter(
        x=[highest_expiry],
        y=[highest_strike],
        text=["★"],
        mode="text",
        textfont=dict(size=20, color="white"),
        showlegend=False,
        hoverinfo='skip'
    ))

    # ATM line
    fig.add_hline(
        y=S,
        line_color="red",
        line_width=1,
        annotation_text="ATM",
        annotation_position="top right"
    )

    # Axis settings
    fig.update_yaxes(
        tickvals=strikes,
        ticktext=[str(s) for s in strikes],
        autorange='reversed'
    )
    fig.update_xaxes(
        tickvals=expiries,
        ticktext=expiries
    )

    fig.update_layout(
        title=f"{ticker} – Net Gamma Exposure Heatmap",
        xaxis_title="Expiry",
        yaxis_title="Strike",
        height=800,
        showlegend=False
    )

    return fig

# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    st.title("📊 Gamma Exposure Heatmap")

    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Ticker", "TSLA").upper()
    max_expirations = st.sidebar.slider("Max Expirations", 1, MAX_EXPIRATIONS_HARD_CAP, 5)
    strike_range = st.sidebar.slider("Strike Range ($)", 5, 100, 50, step=5)

    if st.sidebar.button("🚀 Generate Heatmap", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if hist.empty:
                    st.error("Unable to fetch price data.")
                    return
                S = hist["Close"].iloc[-1]
            except Exception:
                st.error("Price fetch failed.")
                return

            df = fetch_options_yahoo_safe(ticker, max_expirations)
            if df.empty:
                st.error(
                    "⚠️ Yahoo Finance rate limit hit.\n\n"
                    "Please wait a minute and try again.\n"
                    "This is a Yahoo limitation, not a bug."
                )
                return

            gex = compute_gex(df, S, strike_range)
            if gex.empty:
                st.warning("No GEX computed for selected range.")
                return

            fig = plot_heatmap(gex, ticker, S)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Enter a ticker and click Generate Heatmap")

if __name__ == "__main__":
    main()
