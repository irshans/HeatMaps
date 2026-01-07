import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from scipy.stats import norm
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- APP CONFIG ---
st.set_page_config(page_title="GEX & VEX Pro", page_icon="📊", layout="wide")

# --- SECRETS ---
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"
RISK_FREE_RATE = 0.046
CUSTOM_COLORSCALE = [[0.0, '#FF3333'], [0.45, '#121212'], [0.5, '#121212'], [0.55, '#121212'], [1.0, '#00FF7F']]

# --- UTILS ---
def get_market_holidays():
    cal = USFederalHolidayCalendar()
    return cal.holidays(start='2024-01-01', end='2026-12-31')

MARKET_HOLIDAYS = get_market_holidays()

def format_thousands(val):
    """Formats value in thousands with commas. Returns $0 if 0."""
    if val == 0 or np.isnan(val):
        return "$0"
    return f'{val/1000:,.0f}'

# --- FETCHING ---
def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=10)
        return r.json() if r.status_code == 200 else None
    except: return None

@st.cache_data(ttl=300)
def fetch_data(ticker, max_exp):
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data: return None, None
    q = quote_data['quotes']['quote']
    S = float(q['last']) if isinstance(q, dict) else float(q[0]['last'])

    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data: return S, None
    
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list): all_exps = [all_exps]
    
    # PRE-FILTER WEEKENDS
    valid_exps = []
    for e in all_exps:
        dt = pd.to_datetime(e)
        if dt.dayofweek < 5 and dt not in MARKET_HOLIDAYS:
            valid_exps.append(e)
            
    exps_to_fetch = sorted(valid_exps)[:max_exp]
    dfs = []
    prog = st.progress(0, text="Fetching...")
    for i, exp in enumerate(exps_to_fetch):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and chain.get('options') and chain['options'].get('option'):
            opts = chain['options']['option']
            dfs.append(pd.DataFrame(opts if isinstance(opts, list) else [opts]))
        prog.progress((i + 1) / len(exps_to_fetch))
    prog.empty()
    return S, pd.concat(dfs, ignore_index=True) if dfs else (S, None)

# --- CALCULATIONS ---
def process_exposure(df, S, strikes_count):
    if df is None or df.empty: return pd.DataFrame()
    
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0)
    
    # Filter to exactly the chosen number of strikes around spot
    unique_strikes = np.array(sorted(df['strike'].unique()))
    idx = np.abs(unique_strikes - S).argmin()
    half = strikes_count // 2
    selected_strikes = unique_strikes[max(0, idx-half) : min(len(unique_strikes), idx+half)]
    df = df[df['strike'].isin(selected_strikes)].copy()

    # Greeks
    df['iv'] = pd.to_numeric(df.get('greeks', {}).apply(lambda x: x.get('mid_iv') if isinstance(x, dict) else 0.2), errors='coerce').fillna(0.2)
    df.loc[df['iv'] > 1.0, 'iv'] /= 100.0
    
    df['expiry_dt'] = pd.to_datetime(df['expiration_date'])
    df['T'] = np.maximum((df['expiry_dt'] - pd.Timestamp.now()).dt.total_seconds() / (365*24*3600), 1e-5)
    
    d1 = (np.log(S / df['strike']) + (RISK_FREE_RATE + 0.5 * df['iv']**2) * df['T']) / (df['iv'] * np.sqrt(df['T']))
    df['gamma_bs'] = norm.pdf(d1) / (S * df['iv'] * np.sqrt(df['T']))
    df['vanna_bs'] = -norm.pdf(d1) * ((d1 - df['iv'] * np.sqrt(df['T'])) / df['iv'])
    
    df['side'] = np.where(df['option_type'].str.lower() == 'call', 1, -1)
    df['gex'] = df['side'] * (df['gamma_bs'] * (S**2) * 0.01) * df['open_interest'] * 100
    df['vex'] = -1 * df['vanna_bs'] * S * 100 * df['open_interest']
    
    return df

# -------------------------
# RENDER
# -------------------------
def render_heatmap(df, S, metric, title):
    if df.empty: return None

    # 1. Pivot
    pivot = df.pivot_table(index='strike', columns='expiration_date', values=metric, aggfunc='sum')
    
    # 2. STRICT WEEKEND REMOVAL (Drop columns if date is Sat/Sun)
    cols_to_keep = [c for c in pivot.columns if pd.to_datetime(c).dayofweek < 5]
    pivot = pivot[cols_to_keep]
    
    # 3. Fill missing values with 0 so they display as $0
    pivot = pivot.fillna(0).sort_index(ascending=False)
    
    text_values = pivot.applymap(format_thousands).values
    max_val = np.percentile(np.abs(pivot.values), 98) if np.any(pivot.values) else 1

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        text=text_values, texttemplate="%{text}", textfont={"size": 9},
        colorscale=CUSTOM_COLORSCALE, zmid=0, zmin=-max_val, zmax=max_val
    ))

    fig.add_hline(y=S, line_dash="dash", line_color="white", annotation_text=f"Spot: {S:.2f}")
    fig.update_layout(title=title, height=800, template="plotly_dark")
    return fig

def main():
    st.title("GEX & VEX: Weekend-Free Surface")
    
    with st.sidebar:
        ticker = st.text_input("Ticker", "SPX").upper()
        strike_depth = st.select_slider("Strikes", options=[25, 40, 50, 80, 100, 150], value=80 if ticker=="SPX" else 40)
        max_exp = st.number_input("Expiries", 1, 15, 5)
        run = st.button("Calculate", type="primary")

    if run:
        S, raw_df = fetch_data(ticker, max_exp)
        if raw_df is not None:
            df = process_exposure(raw_df, S, strike_depth)
            
            t1, t2, t3 = st.tabs(["Gamma (GEX)", "Vanna (VEX)", "Diagnostics"])
            with t1:
                st.plotly_chart(render_heatmap(df, S, 'gex', f"{ticker} Gamma ($'000s)"), use_container_width=True)
            with t2:
                st.plotly_chart(render_heatmap(df, S, 'vex', f"{ticker} Vanna ($'000s)"), use_container_width=True)
            with t3:
                st.dataframe(df[['expiration_date', 'strike', 'option_type', 'gex', 'vex', 'open_interest', 'iv']].sort_values('gex'), use_container_width=True)

if __name__ == "__main__":
    main()