import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from scipy.stats import norm
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# --- APP CONFIG ---
st.set_page_config(page_title="GEX & VEX Pro", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    div[data-testid="stDataFrame"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- SECRETS ---
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"

# --- CONSTANTS ---
RISK_FREE_RATE = 0.046
# Custom colorscale: Red (Neg) -> Black (Neutral) -> Green (Pos)
CUSTOM_COLORSCALE = [
    [0.0, '#FF3333'], [0.45, '#121212'],
    [0.5, '#121212'], [0.55, '#121212'],
    [1.0, '#00FF7F']
]

# --- UTILS: HOLIDAYS & FORMATTING ---
def get_market_calendar():
    cal = USFederalHolidayCalendar()
    # Get holidays for current and next year to be safe
    years = [pd.Timestamp.now().year, pd.Timestamp.now().year + 1]
    holidays = cal.holidays(start=f'{years[0]}-01-01', end=f'{years[1]}-12-31')
    return holidays

MARKET_HOLIDAYS = get_market_calendar()

def is_market_day(date_str):
    """Returns False if date is weekend or holiday."""
    dt = pd.to_datetime(date_str)
    # Check weekend (5=Sat, 6=Sun)
    if dt.dayofweek >= 5:
        return False
    # Check Federal Holidays
    if dt in MARKET_HOLIDAYS:
        return False
    return True

def format_currency(val):
    """Formats large numbers for heatmap text (1.2B, 50M)."""
    abs_val = abs(val)
    if abs_val >= 1e9:
        return f'{val/1e9:.1f}B'
    elif abs_val >= 1e6:
        return f'{val/1e6:.0f}M'
    elif abs_val >= 1e3:
        return f'{val/1e3:.0f}K'
    return ""

# --- FETCHING ---
def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=300)
def fetch_data(ticker, max_exp):
    # 1. Spot Price
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data or 'quotes' not in quote_data:
        return None, None
    
    q = quote_data['quotes']['quote']
    S = float(q['last']) if isinstance(q, dict) else float(q[0]['last'])

    # 2. Expirations
    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data or 'expirations' not in exp_data:
        return S, None
    
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list):
        all_exps = [all_exps]
    
    # FILTER: Remove Weekends and Holidays immediately
    valid_exps = [e for e in all_exps if is_market_day(e)]
    exps_to_fetch = sorted(valid_exps)[:max_exp]
    
    # 3. Fetch Chains
    dfs = []
    prog_bar = st.progress(0, text="Fetching Option Chains...")
    
    for i, exp in enumerate(exps_to_fetch):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and chain.get('options') and chain['options'].get('option'):
            opts = chain['options']['option']
            data = opts if isinstance(opts, list) else [opts]
            dfs.append(pd.DataFrame(data))
        prog_bar.progress((i + 1) / len(exps_to_fetch))
    
    prog_bar.empty()
    
    if not dfs: return S, None
    return S, pd.concat(dfs, ignore_index=True)

# --- CALCULATIONS ---
def calculate_greeks_vectorized(df, S):
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0)
    
    # Parse IV
    df['iv'] = pd.to_numeric(df.get('greeks', {}).apply(lambda x: x.get('mid_iv') if isinstance(x, dict) else np.nan), errors='coerce')
    df['iv'] = df['iv'].fillna(0.2).clip(upper=5.0) # Clip crazy IVs
    mask_high = df['iv'] > 1.0 # Fix percentage scaling if API returns 20.0 instead of 0.2
    df.loc[mask_high, 'iv'] = df.loc[mask_high, 'iv'] / 100.0

    # Time
    df['expiry_dt'] = pd.to_datetime(df['expiration_date'])
    now = pd.Timestamp.now()
    df['days_to_exp'] = (df['expiry_dt'] - now).dt.total_seconds() / (24 * 3600)
    df['days_to_exp'] = df['days_to_exp'].clip(lower=0)
    
    # Avoid 0 division
    df['T'] = np.maximum(df['days_to_exp'] / 365.0, 1e-5)

    # BS
    K = df['strike'].values
    T = df['T'].values
    sigma = df['iv'].values
    r = RISK_FREE_RATE

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    df['gamma_bs'] = pdf_d1 / (S * sigma * np.sqrt(T))
    df['vanna_bs'] = -pdf_d1 * (d2 / sigma) 
    
    df['side'] = np.where(df['option_type'].str.lower() == 'call', 1, -1)
    return df

def process_exposure(df, S, strikes_count):
    if df.empty: return pd.DataFrame()

    # 1. Filter Strikes dynamically based on Count
    # We want 'strikes_count' number of strikes centered around S
    # Get unique strikes
    unique_strikes = sorted(df['strike'].unique().astype(float))
    
    # Find index closest to S
    closest_idx = min(range(len(unique_strikes)), key=lambda i: abs(unique_strikes[i] - S))
    
    # Determine bounds
    half_window = strikes_count // 2
    start_idx = max(0, closest_idx - half_window)
    end_idx = min(len(unique_strikes), closest_idx + half_window)
    
    allowed_strikes = unique_strikes[start_idx:end_idx]
    
    # Filter DF
    df = df[df['strike'].isin(allowed_strikes)].copy()
    
    # 2. Compute
    df = calculate_greeks_vectorized(df, S)
    
    # 3. Exposure
    # Weight near-dated options higher (dealer gamma urgency)
    df['time_weight'] = np.select(
        [df['days_to_exp'] <= 1, df['days_to_exp'] <= 5], 
        [2.0, 1.5], default=1.0
    )

    # GEX ($)
    df['gex_raw'] = df['gamma_bs'] * (S**2) * 0.01 
    df['gex'] = df['side'] * df['gex_raw'] * df['open_interest'] * 100 * df['time_weight']

    # VEX ($) - Vanna Exposure
    df['vex'] = -1 * df['vanna_bs'] * S * 100 * df['open_interest'] * df['time_weight']

    return df[['expiration_date', 'strike', 'gex', 'vex', 'open_interest', 'days_to_exp']]

def find_gamma_flip(df):
    if df.empty: return None
    strikes = df.groupby('strike')['gex'].sum().sort_index()
    for i in range(len(strikes) - 1):
        if strikes.iloc[i] < 0 and strikes.iloc[i+1] > 0:
            return strikes.index[i+1]
    return None

# -------------------------
# VISUALIZATION
# -------------------------
def render_heatmap(df, ticker, S, metric, title, flip_strike=None):
    if df.empty: return None

    # Pivot Data
    pivot = df.pivot_table(index='strike', columns='expiration_date', values=metric, aggfunc='sum').fillna(0)
    pivot = pivot.sort_index(ascending=False)
    
    # Format Text for Cells (e.g. "1.2B")
    text_values = pivot.applymap(format_currency).values

    # Colorscale Limit (Robust)
    max_val = np.percentile(np.abs(pivot.values), 98)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        text=text_values,        # <--- Display Values
        texttemplate="%{text}",  # <--- Format
        textfont={"size": 10},   # <--- Size
        colorscale=CUSTOM_COLORSCALE,
        zmid=0,
        zmin=-max_val,
        zmax=max_val,
        colorbar=dict(title="Exposure ($)")
    ))

    # Spot Price Line
    fig.add_hline(y=S, line_dash="dash", line_color="white", annotation_text=f"Spot: {S:.2f}")

    if flip_strike and metric == 'gex':
        fig.add_hline(y=flip_strike, line_dash="dot", line_color="yellow", annotation_text="Flip")

    fig.update_layout(
        title=f"{ticker} {title}",
        height=700, # Taller for visibility
        xaxis_title="Expiration",
        yaxis_title="Strike",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# -------------------------
# MAIN
# -------------------------
def main():
    st.title("Dealer Flow: GEX & VEX Pro")

    # --- SIDEBAR / TOP SETTINGS ---
    with st.expander("Configuration", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        
        ticker = c1.text_input("Ticker", "SPX").upper()
        
        # Default strike count based on ticker
        default_strikes = 80 if ticker == "SPX" else 40
        
        # SLIDER FOR STRIKES
        strike_depth = c3.select_slider(
            "Number of Strikes to Display",
            options=[25, 40, 50, 80, 100, 150],
            value=default_strikes
        )
        
        max_exp = c2.number_input("Expiries", min_value=1, max_value=15, value=5)
        
        btn = st.button("Run Analysis", type="primary")

    if btn:
        S, raw_df = fetch_data(ticker, max_exp)
        
        if raw_df is None or raw_df.empty:
            st.error("No data found.")
            return

        # PROCESS with STRIKE COUNT
        df = process_exposure(raw_df, S, strike_depth)
        
        # METRICS
        total_gex = df['gex'].sum()
        total_vex = df['vex'].sum()
        flip = find_gamma_flip(df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spot Price", f"{S:,.2f}")
        m2.metric("Net GEX", format_currency(total_gex))
        m3.metric("Net VEX", format_currency(total_vex))
        m4.metric("Gamma Flip", f"{flip}" if flip else "-")

        # CHARTS
        tab1, tab2 = st.tabs(["🔥 Gamma Exposure", "🌊 Vanna Exposure"])
        
        with tab1:
            fig = render_heatmap(df, ticker, S, 'gex', "Gamma Exposure (GEX)", flip)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            fig = render_heatmap(df, ticker, S, 'vex', "Vanna Exposure (VEX)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()