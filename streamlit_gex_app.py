import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from scipy.stats import norm
import time
import random

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

# Compact UI styling
st.markdown(
    """
    <style>
    /* Increase top padding so title is visible and not cut off */
    .block-container { padding-top: 24px; padding-bottom: 8px; }

    /* Buttons */
    button[kind="primary"], .stButton>button {
        padding:4px 8px !important;
        font-size:12px !important;
        height:30px !important;
    }

    /* Inputs, selects, number inputs */
    input[type="text"], input[type="number"], select {
        padding:6px 8px !important;
        font-size:12px !important;
        height:28px !important;
    }

    /* Radio, selectbox height & font */
    div[role="radiogroup"] label, .stSelectbox, .stRadio {
        font-size:12px !important;
    }

    /* Sliders compact */
    .stSlider > div, .stNumberInput > div {
        font-size:12px !important;
        height:34px !important;
    }

    /* Reduce margins for columns */
    .css-1lcbmhc.e1tzin5v0 { gap: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

CONTRACT_SIZE = 100
FALLBACK_IV = 0.30
RISK_FREE_RATE = 0.042
SECONDS_IN_YEAR = 365 * 24 * 3600

# -------------------------
# Holiday & Date Logic
# -------------------------
def get_actual_trading_day(date_str):
    dt = pd.to_datetime(date_str).date()
    holidays = {
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
        '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
        '2026-01-01'
    }
    while dt.weekday() > 4 or dt.strftime('%Y-%m-%d') in holidays:
        dt = dt - timedelta(days=1)
    return dt.strftime('%Y-%m-%d')

# -------------------------
# Black-Scholes Math
# -------------------------
def get_greeks(S, K, r, sigma, T, option_type):
    """
    Return (gamma, delta, vega).
    - vega returned is the standard BS vega (dPrice / dVol) per 1.0 vol unit.
    """
    MIN_T_YEARS = 60 / SECONDS_IN_YEAR
    T_eff = max(T, MIN_T_YEARS)
    sig_eff = max(sigma, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sig_eff**2) * T_eff) / (sig_eff * math.sqrt(T_eff))
    gamma = norm.pdf(d1) / (S * sig_eff * math.sqrt(T_eff))
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    vega = S * norm.pdf(d1) * math.sqrt(T_eff)
    return gamma, delta, vega

# -------------------------
# Data Fetcher
# -------------------------
@st.cache_data(ttl=300)
def fetch_data_safe(ticker, max_exp):
    if ticker.upper() in ["SPX", "NDX", "RUT"] and not ticker.startswith("^"):
        ticker = f"^{ticker.upper()}"

    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="5d")
        if hist.empty:
            return None, None
        S = float(hist["Close"].iloc[-1])

        all_exps = stock.options
        if not all_exps:
            return S, None

        target_exps = all_exps[:max_exp]
        dfs = []
        progress_text = st.empty()
        prog_bar = st.progress(0)

        for i, exp in enumerate(target_exps):
            progress_text.text(f"Fetching {ticker} expiry: {exp} ({i+1}/{len(target_exps)})")
            for attempt in range(3):
                try:
                    time.sleep(random.uniform(0.25, 0.6))
                    chain = stock.option_chain(exp)
                    t_date = get_actual_trading_day(exp)
                    calls = chain.calls.assign(option_type="call", expiration=t_date)
                    puts = chain.puts.assign(option_type="put", expiration=t_date)
                    dfs.append(pd.concat([calls, puts], ignore_index=True))
                    break
                except Exception:
                    time.sleep(1)
            prog_bar.progress((i + 1) / len(target_exps))

        progress_text.empty()
        prog_bar.empty()

        if not dfs:
            return S, None
        options_df = pd.concat(dfs, ignore_index=True)
        return S, options_df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None, None

# -------------------------
# GEX & VEX Processing
# -------------------------
def process_exposure(df, S, s_range):
    """
    Computes exposures assuming the dealer model is:
      Short Calls / Long Puts (fixed).
    Dealer is short calls (dealer_pos = -1) and long puts (dealer_pos = +1).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    eastern = ZoneInfo("US/Eastern")
    now_eastern = datetime.now(tz=eastern)
    today = now_eastern.date()

    df["exp_dt_obj"] = pd.to_datetime(df["expiration"]).dt.date
    df["exp_date_only"] = df["exp_dt_obj"]

    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range) & (df["exp_date_only"] >= today)]

    res = []
    for _, row in df.iterrows():
        try:
            K = float(row["strike"])
        except Exception:
            continue

        exp_date = pd.to_datetime(row["expiration"]).date()
        exp_close = datetime(exp_date.year, exp_date.month, exp_date.day, 16, 0, tzinfo=eastern)
        seconds_left = (exp_close - now_eastern).total_seconds()
        if seconds_left <= 0:
            continue

        MIN_SECONDS = 60
        seconds_for_T = max(seconds_left, MIN_SECONDS)
        T = seconds_for_T / SECONDS_IN_YEAR

        oi = row.get("openInterest") or 0
        vol = row.get("volume") or 0
        liq = max(oi, vol)
        if liq <= 0:
            continue

        iv = row.get("impliedVolatility")
        if iv is None or pd.isna(iv) or not (0.02 < float(iv) < 4.0):
            iv = FALLBACK_IV
        else:
            iv = float(iv)

        gamma, delta, vega = get_greeks(S, K, RISK_FREE_RATE, iv, T, row["option_type"])

        # Fixed dealer model: Short Calls / Long Puts
        dealer_pos = -1 if row["option_type"] == "call" else +1

        # GEX: dollars per 1% spot move
        gex = dealer_pos * gamma * (S ** 2) * 0.01 * CONTRACT_SIZE * liq
        # VEX: dollars per 1% IV move (vega per 1.0 * 0.01)
        vex = dealer_pos * vega * 0.01 * CONTRACT_SIZE * liq

        res.append({"strike": K, "expiry": row["expiration"], "gex": gex, "vex": vex})

    if not res:
        return pd.DataFrame()
    return pd.DataFrame(res)

# -------------------------
# Visualizations
# -------------------------
def render_plot(df, ticker, S, mode):
    if df.empty:
        return None

    val_col = mode.lower()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum', fill_value=0).sort_index(ascending=False)

    z_raw = pivot.values
    x_labs = pivot.columns.tolist()
    y_labs = pivot.index.tolist()
    if not y_labs:
        return None

    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    # Build hover text
    h_text = []
    for i, strike in enumerate(y_labs):
        row = []
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            prefix = "-" if val < 0 else ""
            v_abs = abs(val)
            if v_abs >= 1e6:
                formatted = f"{prefix}${v_abs/1e6:,.2f}M"
            elif v_abs >= 1e3:
                formatted = f"{prefix}${v_abs/1e3:,.1f}K"
            else:
                formatted = f"{prefix}${v_abs:,.0f}"
            note = " (per 1% IV)" if mode == "VEX" else ""
            row.append(f"Strike: ${strike:,.0f}<br>Expiry: {exp}<br>{mode}: {formatted}{note}")
        h_text.append(row)

    max_abs = np.max(np.abs(z_raw)) if z_raw.size else 1.0
    if max_abs == 0:
        max_abs = 1.0
    zmin = -max_abs
    zmax = max_abs

    colorscale = [
        [0.00, "#221557"],
        [0.10, '#260446'],
        [0.25, '#56117a'],
        [0.40, '#6E298A'],
        [0.49, '#783F8F'],
        [0.50, '#224B8B'],
        [0.52, '#32A7A7'],
        [0.65, '#39B481'],
        [0.80, '#A8D42A'],
        [0.92, '#FFDF4A'],
        [1.00, '#F1F50C']
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs, text=h_text, hoverinfo="text",
        colorscale=colorscale, zmin=zmin, zmax=zmax, zmid=0, showscale=True,
        colorbar=dict(title=dict(text=f"{mode} ($)"), tickformat=",.0s")
    ))

    # FIXED: Text color based purely on sign
    max_abs_val = np.max(np.abs(z_raw)) if z_raw.size else 0
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500:
                continue
            prefix = "-" if val < 0 else ""
            txt = f"{prefix}${abs(val)/1e3:,.0f}K"
            if abs(val) == max_abs_val and max_abs_val > 0:
                txt += " ⭐"

            text_color = "black" if val >= 0 else "white"  # ← This is the fix

            fig.add_annotation(
                x=exp, y=strike, text=txt, showarrow=False,
                font=dict(color=text_color, size=11, family="Arial"),
                xref="x", yref="y"
            )

    # Spot strike highlight (unchanged)
    sorted_strikes = sorted(y_labs)
    strike_diffs = np.diff(sorted_strikes) if len(sorted_strikes) > 1 else np.array([sorted_strikes[0] * 0.05])
    padding = (strike_diffs[0] * 0.45) if len(strike_diffs) > 0 else 2.5

    fig.add_shape(type="rect", xref="paper", yref="y", x0=-0.08, x1=1.0, 
                  y0=closest_strike - padding, y1=closest_strike + padding, 
                  fillcolor="rgba(255, 51, 51, 0.25)", line=dict(width=0), layer="below")

    fig.update_layout(
        title=f"{ticker} {mode} | Spot: ${S:,.2f}",
        template="plotly_dark", 
        height=700, 
        xaxis=dict(type='category', side='top', tickfont=dict(size=11)), 
        yaxis=dict(title="Strike", tickfont=dict(size=11), autorange=True, 
                   tickmode='array', tickvals=y_labs, 
                   ticktext=[f"<b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs]), 
        margin=dict(l=70, r=50, t=80, b=30)
    )

    return fig
# -------------------------
# Main App
# -------------------------
def main():
    st.markdown(
        "<div style='text-align:center; margin-top:6px;'><h2 style='font-size:18px; margin:10px 0 6px 0; font-weight:600;'>📈 GEX / VEX Pro</h2></div>",
        unsafe_allow_html=True,
    )

    # Compact toolbar
    col1, col2, col3, col4 = st.columns([1.5, 0.8, 0.8, 0.6])
    with col1:
        ticker = st.text_input("Ticker", "SPY", key="ticker_compact").upper().strip()
    
    # Set defaults based on ticker
    is_spx = ticker in ["SPX", "^SPX", "SPXW", "^SPXW"]
    default_exp = 5
    default_range = 80 if is_spx else 25
    
    with col2:
        max_exp = st.number_input("Max Exp", min_value=1, max_value=15, value=default_exp, step=1, key="maxexp_compact")
    with col3:
        s_range = st.number_input("Strike ±", min_value=5, max_value=200, value=default_range, step=5, key="srange_compact")
    with col4:
        run = st.button("Run", type="primary", key="run_compact")

    if run:
        with st.spinner(f"Analyzing {ticker}..."):
            S, raw_df = fetch_data_safe(ticker, int(max_exp))

        if S and raw_df is not None and not raw_df.empty:
            processed = process_exposure(raw_df, S, s_range)

            if not processed.empty:
                t_gex = processed["gex"].sum() / 1e9
                t_vex = processed["vex"].sum() / 1e9

                p_g = "-" if t_gex < 0 else ""
                p_v = "-" if t_vex < 0 else ""

                c1, c2 = st.columns(2)
                c1.metric("Net Dealer GEX", f"{p_g}${abs(t_gex):,.2f}B")
                c2.metric("Net Dealer VEX (per 1% IV)", f"{p_v}${abs(t_vex):,.2f}B")

                st.markdown("---")
                
                # Render both GEX and VEX side by side
                col_gex, col_vex = st.columns(2)
                
                with col_gex:
                    st.markdown("### GEX Exposure")
                    gex_fig = render_plot(processed, ticker, S, "GEX")
                    if gex_fig:
                        st.plotly_chart(gex_fig, width="stretch")
                    else:
                        st.warning("No GEX data to display")
                
                with col_vex:
                    st.markdown("### VEX Exposure")
                    vex_fig = render_plot(processed, ticker, S, "VEX")
                    if vex_fig:
                        st.plotly_chart(vex_fig, width="stretch")
                    else:
                        st.warning("No VEX data to display")
                        
            else:
                st.warning("No data found in range. Check if market is open or broaden strike range.")
        else:
            st.error("Fetch failed. Try manually entering ^SPXW for SPX 0DTE.")

if __name__ == "__main__":
    main()