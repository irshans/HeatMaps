import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import pytz
from scipy.stats import norm

# --- APP CONFIG ---
st.set_page_config(page_title="GEX & VANEX Pro", page_icon="📊", layout="wide")

st.markdown("""
<style>
* { font-family: 'Arial', sans-serif !important; }
.block-container { padding-top: 24px; padding-bottom: 8px; }
[data-testid="stMetricValue"] { font-size: 20px !important; }
hr { margin: 15px 0 !important; }
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- SECRETS ---
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"

CUSTOM_COLORSCALE = [
    [0.00, '#050018'], [0.25, '#56117a'],
    [0.50, '#224B8B'], [0.65, '#39B481'],
    [0.80, '#A8D42A'], [1.00, '#F1F50C']
]

# -------------------------
# QUANT PARAMETERS
# -------------------------
RISK_FREE = 0.045
RISK_FREE_RATE = 0.045

# -------------------------
# FETCH
# -------------------------
def tradier_get(endpoint, params):
    headers = {
        "Authorization": f"Bearer {TRADIER_TOKEN}",
        "Accept": "application/json"
    }
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Tradier error: {e}")
    return None


@st.cache_data(ttl=3600)
def fetch_data(ticker, max_exp):
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data:
        return None, None

    q = quote_data['quotes']['quote']
    S = float(q['last']) if isinstance(q, dict) else float(q[0]['last'])

    exp_data = tradier_get(
        "markets/options/expirations",
        {"symbol": ticker, "includeAllRoots": "true"}
    )
    if not exp_data:
        return S, None

    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list):
        all_exps = [all_exps]

    dfs = []
    prog = st.progress(0, text="Fetching chain…")

    for i, exp in enumerate(sorted(all_exps)[:max_exp]):
        chain = tradier_get(
            "markets/options/chains",
            {"symbol": ticker, "expiration": exp, "greeks": "true"}
        )

        if chain and chain['options'] and chain['options']['option']:
            opts = chain['options']['option']
            dfs.append(
                pd.DataFrame(opts) if isinstance(opts, list)
                else pd.DataFrame([opts])
            )

        prog.progress((i + 1) / max_exp)

    prog.empty()

    return S, pd.concat(dfs, ignore_index=True) if dfs else None


# -------------------------
# BLACK–SCHOLES RECOMPUTE
# -------------------------
def bs_time_to_exp(expiry):
    today = pd.Timestamp.now()
    exp = pd.to_datetime(expiry)
    days = max((exp - today).days, 0)
    return days / 365.0


def bs_gamma(S, K, T, iv):
    if T <= 0 or iv <= 0:
        return 0.0

    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    return norm.pdf(d1) / (S * iv * np.sqrt(T))


def bs_charm(S, K, T, iv):
    if T <= 0 or iv <= 0:
        return 0.0

    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)

    term1 = -norm.pdf(d1) * (2 * RISK_FREE * T - d2 * iv * np.sqrt(T))
    term2 = 2 * T * (S * iv * np.sqrt(T))**2

    return term1 / term2


def dealer_delta_weight(days):
    if days <= 1:
        return 2.2
    if days <= 7:
        return 1.6
    if days <= 30:
        return 1.25
    return 1.0


# -------------------------
# EXPOSURE PROCESSING – ENHANCED
# -------------------------
def process_exposure(df, S, s_range):
    if df is None or df.empty:
        st.warning("Input dataframe empty")
        return pd.DataFrame()

    df = df.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)]

    # Expiry weights
    expiries_dom = df.groupby("expiration_date")["open_interest"].sum()
    total_oi_all = expiries_dom.sum()

    expiry_weights = {
        exp: oi / total_oi_all if total_oi_all > 0 else 1.0
        for exp, oi in expiries_dom.items()
    }

    res = []
    today = pd.Timestamp.now()

    for _, row in df.iterrows():
        g = row.get("greeks") or {}

        gamma_p = float(g.get("gamma") or 0)
        delta_p = float(g.get("delta") or 0)
        vega_p = float(g.get("vega") or 0)

        iv = float(g.get("smv_vol") or g.get("mid_iv") or 0)
        if iv > 1:
            iv /= 100.0
        iv = max(iv, 0.05)

        K = float(row["strike"])
        expiry = row["expiration_date"]
        T = max((pd.to_datetime(expiry) - today).days, 0) / 365.0
        days = max((pd.to_datetime(expiry) - today).days, 0)

        # BS Greeks
        gamma_bs = bs_gamma(S, K, T, iv)

        side = 1 if row["option_type"].lower() == "call" else -1
        oi = int(row.get("open_interest", 0) or 0)

        weight = expiry_weights.get(expiry, 1.0)
        weight *= dealer_delta_weight(days)

        gex = side * gamma_bs * (S**2) * 0.01 * 100 * oi * weight

        # VANEX
        vanna = vega_p * delta_p / (S * iv)
        vanex = -vanna * S * 100 * oi * weight

        # 0DTE boost
        if days <= 1:
            gex *= 2.5
            vanex *= 2.0

        charm = bs_charm(S, K, T, iv)
        gex_charm = gex + side * charm * 100 * oi * weight

        if not np.isfinite(gex):
            continue

        res.append({
            "strike": K,
            "expiry": expiry,
            "gex": gex,
            "vanex": vanex,
            "gex_charm": gex_charm,
            "gamma": gamma_bs * side * oi,
            "oi": oi,
            "days": days,
            "weight": weight
        })

    return pd.DataFrame(res)


def find_gamma_flip(df):
    if df.empty:
        return None
    sums = df.groupby("strike")["gex"].sum().sort_index()
    for i in range(len(sums) - 1):
        if sums.iloc[i] * sums.iloc[i + 1] < 0:
            return sums.index[i]
    return None


# -------------------------
# RENDER
# -------------------------
def render_heatmap(df, ticker, S, mode, flip_strike):
    if df.empty:
        return None

    pivot = df.pivot_table(index="strike", columns="expiry", values="gex", aggfunc="sum")
    pivot = pivot.sort_index(ascending=False)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=CUSTOM_COLORSCALE,
            zmid=0
        )
    )

    fig.add_vline(x=str(S), line_dash="dot")

    if flip_strike:
        fig.add_annotation(
            y=flip_strike,
            x=pivot.columns[0],
            text="Gamma Flip"
        )

    return fig


# -------------------------
# MAIN PAGE
# -------------------------
def main():
    st.title("Dealer Aware GEX & VANEX")

    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])

    ticker = c1.text_input("Ticker", value="SPX").upper().strip()
    max_exp = c2.number_input("Expiries", 1, 15, 5)
    s_range = c3.number_input("Strike ±", 5, 500, 80)

    if c4.button("🔄 Refresh Data"):
        st.cache_data.clear()

    if st.button("RUN STUDY"):
        S, raw_df = fetch_data(ticker, max_exp)

        if raw_df is None:
            st.warning("No data fetched")
            return

        df = process_exposure(raw_df, S, s_range)

        flip = find_gamma_flip(df)
        st.write("Regime", regime_probabilities(df, S))

        fig_gex = render_heatmap(df, ticker, S, "GEX", flip)
        fig_vex = render_heatmap(df, ticker, S, "VEX", flip)

        if fig_gex:
            st.plotly_chart(fig_gex, key="gex_chart_key")

        if fig_vex:
            st.plotly_chart(fig_vex, key="vex_chart_key")

        st.dataframe(df.head(20), key="diag_table")


def regime_probabilities(df, S):
    if df.empty:
        return {}
    flip = find_gamma_flip(df)
    net = df["gex"].sum()

    if flip:
        dist = (S - flip) / S
        p_above = norm.cdf(dist * 4)
        return {"aboveFlip": p_above, "belowFlip": 1 - p_above}

    return {"stable": norm.cdf(net / 1e6)}


if __name__ == "__main__":
    main()
