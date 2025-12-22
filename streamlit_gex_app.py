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
@st.cache_data(ttl=300)
def fetch_options_yahoo(ticker, max_expirations=2):
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
    options = pd.concat(dfs, ignore_index=True)
    return options

# -------------------------
# Compute gamma exposure
# -------------------------
def compute_gex(df, S, r=0.0, q=0.0, fallback_iv=FALLBACK_IV, strike_range=20):
    df = df.copy()
    df["mid_price"] = df[["bid", "ask", "lastPrice"]].bfill(axis=1).iloc[:, 0]
    df["expiration_dt"] = pd.to_datetime(df["expiration"])
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range)]
    now = pd.Timestamp.now(tz=None)
    df["expiration_dt_with_time"] = df["expiration_dt"] + pd.Timedelta(hours=16)
    df["T"] = (df["expiration_dt_with_time"] - now).dt.total_seconds() / (365*24*3600)
    df = df[df["T"] > 1/3650.0]
    out_rows = []
    for _, row in df.iterrows():
        K = float(row["strike"])
        expiry = row["expiration"]
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
        # Sign logic: calls positive, puts negative
        gex_total = raw_dollar_gamma * CONTRACT_SIZE * oi * 0.01
        if opt_type == "put":
            gex_total *= -1
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
# Heatmap
# -------------------------
def plot_heatmap(gex_df, ticker, S, max_strikes=20):
    gex_df = gex_df.copy()
    gex_df['expiration_dt'] = pd.to_datetime(gex_df['expiry'])
    gex_df['expiration_dt'] = gex_df['expiration_dt'] + pd.offsets.Week(weekday=4)
    gex_df['expiry_display'] = gex_df['expiration_dt'].dt.strftime('%Y-%m-%d')

    strikes_sorted = sorted(gex_df['strike'].unique())
    atm_idx = np.abs(np.array(strikes_sorted) - S).argmin()
    start = max(0, atm_idx - max_strikes)
    end = min(len(strikes_sorted), atm_idx + max_strikes + 1)
    strikes_to_use = strikes_sorted[start:end]
    df_plot = gex_df[gex_df['strike'].isin(strikes_to_use)]

    pivot = df_plot.pivot_table(
        index='strike',
        columns='expiry_display',
        values='gex_total',
        aggfunc='sum',
        fill_value=0
    ).sort_index(ascending=False)

    strikes = list(pivot.index)
    expiries = list(pivot.columns)
    z = pivot.values

    max_idx = np.unravel_index(np.argmax(np.abs(z)), z.shape)

    text_matrix = []
    for i, row in enumerate(z):
        row_text = []
        for j, val in enumerate(row):
            val_str = f"${val:,.0f}"
            if (i, j) == max_idx:
                val_str += " ★"
            row_text.append(val_str)
        text_matrix.append(row_text)

    colorscale = [
        [0.0, 'rgb(75,0,130)'],      # deep purple negative
        [0.33, 'rgb(0,128,0)'],      # green small
        [0.66, 'rgb(0,128,0)'],      # green small
        [1.0, 'rgb(255,215,0)']      # yellow positive
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=expiries,
        y=strikes,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate="Expiry: %{x}<br>Strike: %{y}<br>GEX: %{z:,.0f}<extra></extra>",
        showscale=True,
        colorscale=colorscale,
        zmin=z.min(),
        zmax=z.max()
    ))

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
# Streamlit App
# -------------------------
def main():
    st.title("📊 Gamma Exposure Heatmap")
    st.markdown("Visualize option gamma exposure across strikes and expirations")

    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
    max_expirations = st.sidebar.slider("Max Expirations", 1, 10, 5)
    strike_range = st.sidebar.slider("Strike Range ($)", 5, 100, 20, step=5)

    if st.sidebar.button("🚀 Generate Heatmap", type="primary"):
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                stock = yf.Ticker(ticker)
                S = stock.history(period="1d")["Close"].iloc[-1]
                st.info(f"**Current Price:** ${S:.2f}")

                df = fetch_options_yahoo(ticker, max_expirations=max_expirations)
                if df.empty:
                    st.error(f"No options data found for {ticker}")
                    return

                gex = compute_gex(df, S, fallback_iv=FALLBACK_IV, strike_range=strike_range)
                if gex.empty:
                    st.error("No gamma exposure computed.")
                    return

                agg = gex.groupby(["strike", "expiry"]).agg({
                    "gex_total": "sum",
                    "premium": "mean"
                }).reset_index()

                call_oi = gex[gex['option_type'] == 'call']['open_interest'].sum()
                put_oi = gex[gex['option_type'] == 'put']['open_interest'].sum()
                pc_ratio = put_oi / call_oi if call_oi > 0 else 0
                call_gex = gex[gex['option_type'] == 'call']['gex_total'].sum()
                put_gex = gex[gex['option_type'] == 'put']['gex_total'].sum()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total GEX", f"${agg['gex_total'].sum():,.0f}")
                with col2:
                    pc_color = "🔴" if pc_ratio > 1 else "🟢"
                    st.metric("Put/Call Ratio", f"{pc_color} {pc_ratio:.2f}")
                with col3:
                    st.metric("Call GEX", f"${call_gex:,.0f}")
                with col4:
                    st.metric("Put GEX", f"${put_gex:,.0f}")

                fig = plot_heatmap(agg, ticker, S, max_strikes=20)
                st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Enter a ticker symbol and click 'Generate Heatmap' to get started!")
        st.markdown("### Example Output")
        st.markdown("This tool displays a heatmap showing gamma exposure across different strikes and expiration dates.")

if __name__ == "__main__":
    main()
