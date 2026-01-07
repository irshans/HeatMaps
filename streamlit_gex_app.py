import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz
from scipy.stats import norm

st.set_page_config(page_title="GEX & VANEX Pro", page_icon="📊", layout="wide")

# --- SECRETS ---
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"

CUSTOM_COLORSCALE = [
    [0.00, '#050018'], [0.10, '#260446'], [0.25, '#56117a'],
    [0.40, '#6E298A'], [0.49, '#783F8F'], [0.50, '#224B8B'],
    [0.52, '#32A7A7'], [0.65, '#39B481'], [0.80, '#A8D42A'],
    [0.92, '#FFDF4A'], [1.00, '#F1F50C']
]

# -------------------------
# BLACK–SCHOLES CORE
# -------------------------

RISK_FREE = 0.045

def bs_time_to_exp(expiry):
    now = pd.Timestamp.now()
    exp = pd.to_datetime(expiry)
    return max((exp - now).days, 0) / 365.0


def bs_gamma(S, K, T, iv, option_type):
    if T <= 0 or iv <= 0:
        return 0.0

    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
    return gamma


def bs_charm(S, K, T, iv):
    """Time derivative of gamma – CHARM"""
    if T <= 0 or iv <= 0:
        return 0.0

    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)

    term1 = -norm.pdf(d1) * (2 * RISK_FREE * T - d2 * iv * np.sqrt(T))
    term2 = 2 * T * (S * iv * np.sqrt(T))**2

    return term1 / term2


def dealer_delta_weight(exp_type, T):
    """Extra weight for 0DTE products"""
    days = T * 365

    if days <= 1:
        return 2.2
    if days <= 7:
        return 1.6
    if days <= 30:
        return 1.25
    return 1.0

# -------------------------
# PROCESSING
# -------------------------

def process_exposure_enhanced(df, S, s_range):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)]

    expiries_dom = df.groupby("expiration_date")["open_interest"].sum()
    total_oi_all = expiries_dom.sum()

    expiry_weights = {}
    for exp, oi in expiries_dom.items():
        w = oi / total_oi_all if total_oi_all > 0 else 1.0
        w *= dealer_delta_weight("spx" if "SPX" in exp.upper() else "other", oi)
        expiry_weights[exp] = w

    records = []

    for _, row in df.iterrows():
        g = row.get("greeks") or {}

        iv = float(g.get("smv_vol") or g.get("mid_iv") or 0)
        if iv > 1:
            iv /= 100.0
        iv = max(iv, 0.05)

        K = float(row["strike"])
        T = bs_time_to_exp(row["expiration_date"])

        gamma = bs_gamma(S, K, T, iv, row["option_type"])
        charm = bs_charm(S, K, T, iv)

        oi = int(row.get("open_interest", 0) or 0)
        delta = float(g.get("delta") or 0)

        side = 1 if row["option_type"].lower() == "call" else -1
        w_exp = expiry_weights.get(row["expiration_date"], 1.0)

        # --- DOLLAR GAMMA RECOMPUTED ---
        gex_dollar = side * gamma * (S**2) * 0.01 * 100 * oi * w_exp

        # --- DEALER VANNA ---
        vanna = (vega := float(g.get("vega") or 0)) * delta / (S * iv)
        vanex_dealer = -vanna * S * 100 * oi * w_exp

        # --- 0DTE INTRADAY FLOW SENSITIVITY ---
        if T * 365 <= 1:
            gex_intraday = gex_dollar * 2.5
            vanex_intraday = vanex_dealer * 2.0
        else:
            gex_intraday = gex_dollar
            vanex_intraday = vanex_dealer

        # --- CHARM ADJUSTMENT ---
        charm_impact = side * charm * 100 * oi * w_exp
        gex_charm_adj = gex_dollar + charm_impact

        records.append({
            "strike": K,
            "expiry": row["expiration_date"],
            "gex": gex_dollar,
            "gex_charm": gex_charm_adj,
            "vanex": vanex_intraday,
            "dex": -side * delta * 100 * oi * w_exp,
            "gamma_bs": gamma * side * oi,
            "charm": charm_impact,
            "oi": oi,
            "weight": w_exp
        })

    out = pd.DataFrame(records)

    # --- VANNA WALL INFERENCE ---
    vanna_walls = out.groupby("strike")["vanex"].sum()
    top_walls = vanna_walls.reindex(sorted(vanna_walls.index)).nlargest(3)

    st.info(f"🧱 Top 1–3 Dealer Vanna Walls: {list(top_walls.index)}")

    # --- REGIME PROBABILITIES ---
    net_gex = out["gex"].sum()
    flip = find_gamma_flip(out)

    regime_probs = {}

    if flip:
        dist = (S - flip) / S
        p_above = norm.cdf(dist * 4)
        regime_probs = {
            "aboveFlip": round(p_above, 3),
            "belowFlip": round(1 - p_above, 3)
        }
    else:
        p_stable = norm.cdf(net_gex / 1e6)
        regime_probs = {
            "stable": round(p_stable, 3),
            "trend": round(1 - p_stable, 3)
        }

    st.write("Regime Probabilities", regime_probs)

    return out


def find_gamma_flip(df):
    if df.empty:
        return None

    strike_sums = df.groupby("strike")["gex"].sum().sort_index()

    for i in range(len(strike_sums) - 1):
        if strike_sums.iloc[i] * strike_sums.iloc[i + 1] < 0:
            return strike_sums.index[i]

    return None

# -------------------------
# RENDERING (YOUR STUDY)
# -------------------------

def render_heatmap(df, ticker, S, mode, flip_strike):
    pivot = df.pivot_table(index="strike", columns="expiry", values="gex", aggfunc="sum")
    pivot = pivot.sort_index(ascending=False)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=CUSTOM_COLORSCALE,
        zmid=0
    ))

    return fig

# -------------------------
# MAIN
# -------------------------

def main():
    c1, c2, c3 = st.columns([1.5, 1, 1])
    ticker = c1.text_input("Ticker", value="SPX").upper().strip()
    max_exp = c2.number_input("Expiries", 1, 15, 5)
    s_range = c3.number_input("Strike ±", 5, 500, 80)

    if st.button("🔄 Run Enhanced"):
        S, raw_df = fetch_data(ticker, max_exp)
        df = process_exposure_enhanced(raw_df, S, s_range)

        st.plotly_chart(render_heatmap(df, ticker, S, "GEX", None))


if __name__ == "__main__":
    main()
