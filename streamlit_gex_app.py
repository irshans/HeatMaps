import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from scipy.stats import norm

# --- APP CONFIG ---
st.set_page_config(page_title="GEX & VEX Pro", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
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

# Divergent colorscale: Red (Negative) -> Blue (Neutral) -> Green (Positive)
# Adjusted for financial heatmaps
CUSTOM_COLORSCALE = [
    [0.0, '#FF3333'],  # Negative Gamma/Vanna (Red)
    [0.4, '#191925'],  # Near Zero (Dark)
    [0.5, '#191925'],  # Zero
    [0.6, '#191925'],  # Near Zero (Dark)
    [1.0, '#00FF7F']   # Positive Gamma/Vanna (Green)
]

# --- CONSTANTS ---
RISK_FREE_RATE = 0.046  # 4.6% roughly
TODAY = pd.Timestamp.now()

# --- FETCHING ---
def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None

@st.cache_data(ttl=300)
def fetch_data(ticker, max_exp):
    # 1. Get Quote for Spot Price
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data or 'quotes' not in quote_data:
        return None, None
    
    q = quote_data['quotes']['quote']
    S = float(q['last']) if isinstance(q, dict) else float(q[0]['last'])

    # 2. Get Expirations
    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data or 'expirations' not in exp_data:
        return S, None
    
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list):
        all_exps = [all_exps]
    
    # 3. Get Chains (Loop)
    dfs = []
    exps_to_fetch = sorted(all_exps)[:max_exp]
    
    prog_bar = st.progress(0, text="Fetching Option Chains...")
    
    for i, exp in enumerate(exps_to_fetch):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and chain.get('options') and chain['options'].get('option'):
            opts = chain['options']['option']
            # Ensure list format even if single option returned
            data = opts if isinstance(opts, list) else [opts]
            dfs.append(pd.DataFrame(data))
        
        prog_bar.progress((i + 1) / len(exps_to_fetch))
    
    prog_bar.empty()
    
    if not dfs:
        return S, None
        
    full_df = pd.concat(dfs, ignore_index=True)
    return S, full_df

# --- VECTORIZED BLACK-SCHOLES ---
def calculate_greeks_vectorized(df, S):
    """
    Performs vectorized Black-Scholes calculations on the entire DataFrame at once.
    """
    # 1. Prepare Inputs
    # Ensure types
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0)
    
    # Get IV (normalize 0-100 scale to 0.0-1.0)
    # Prefer 'smv_vol' (Surface IV) -> 'mid_iv' -> fallback 0.5
    df['iv'] = pd.to_numeric(df.get('greeks', {}).apply(lambda x: x.get('smv_vol') if isinstance(x, dict) else np.nan), errors='coerce')
    df['iv'] = df['iv'].fillna(pd.to_numeric(df.get('greeks', {}).apply(lambda x: x.get('mid_iv') if isinstance(x, dict) else np.nan), errors='coerce'))
    df['iv'] = df['iv'].fillna(0.20) # Fallback IV
    
    # Fix IV scale (if > 1 assume it's percentage 20.0 -> 0.20)
    mask_high_iv = df['iv'] > 1.0
    df.loc[mask_high_iv, 'iv'] = df.loc[mask_high_iv, 'iv'] / 100.0
    df['iv'] = df['iv'].clip(lower=0.01) # Avoid div/0

    # Time to Expiry (Annualized)
    # Use 1e-6 for 0DTE to avoid division by zero while keeping gamma high
    df['expiry_dt'] = pd.to_datetime(df['expiration_date'])
    now = pd.Timestamp.now()
    
    # Calculate days, clip negative to 0
    df['days_to_exp'] = (df['expiry_dt'] - now).dt.total_seconds() / (24 * 3600)
    df['days_to_exp'] = df['days_to_exp'].clip(lower=0) 
    
    # T in years. For < 1 day, use a small epsilon
    df['T'] = df['days_to_exp'] / 365.0
    df.loc[df['T'] < 1e-5, 'T'] = 1e-5

    # 2. Black-Scholes Terms
    K = df['strike'].values
    T = df['T'].values
    sigma = df['iv'].values
    r = RISK_FREE_RATE

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    
    # 3. Greeks Calculation
    # Gamma: N'(d1) / (S * sigma * sqrt(T))
    df['gamma_bs'] = pdf_d1 / (S * sigma * np.sqrt(T))
    
    # Vanna: -N'(d1) * d2 / sigma  (Change in Delta per unit Vol)
    # Note: Traditional Vanna is dDelta/dVol.
    df['vanna_bs'] = -pdf_d1 * (d2 / sigma)
    
    # Charm: -N'(d1) * [2rT - d2*sigma*sqrt(T)] / (2T * sigma * sqrt(T))
    # Simplified decay approx for speed: Delta decay
    # We will use the provided BS charm logic logic
    term1 = -pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T))
    term2 = 2 * T * (S * sigma * np.sqrt(T))
    df['charm_bs'] = term1 / term2
    
    # Option Side (Call=1, Put=-1)
    df['side'] = np.where(df['option_type'].str.lower() == 'call', 1, -1)

    return df

def process_exposure(df, S, s_range):
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter Strike Range first to speed up calc
    df = df[(df['strike'] >= S - s_range) & (df['strike'] <= S + s_range)].copy()
    
    # Compute BS Greeks
    df = calculate_greeks_vectorized(df, S)
    
    # --- EXPOSURE LOGIC ---
    # Weighting Logic (Dealer dominance in near term)
    conditions = [
        df['days_to_exp'] <= 1,
        df['days_to_exp'] <= 7,
        df['days_to_exp'] <= 30
    ]
    values = [2.2, 1.6, 1.25]
    df['time_weight'] = np.select(conditions, values, default=1.0)

    # 0DTE Boost logic for display
    df['boost_factor'] = np.where(df['days_to_exp'] <= 1, 2.5, 1.0)

    # GEX Calculation
    # Standard: Gamma * S^2 * 0.01 (Dollar Gamma per 1% move) * OI * Contract(100)
    # Adjusted for Put/Call side
    # Calls: Dealers are Short Gamma (if long cust) -> Market Maker GEX is Negative? 
    # Convention: 
    #   Customer Long Call -> Dealer Short Call -> Dealer Long Gamma (Positive GEX)
    #   Customer Long Put  -> Dealer Short Put  -> Dealer Short Gamma (Negative GEX)
    # The code below follows the "Spot Gamma" convention where Calls = +GEX, Puts = -GEX
    
    df['gex_raw'] = df['gamma_bs'] * (S**2) * 0.01
    df['gex'] = df['side'] * df['gex_raw'] * df['open_interest'] * 100 * df['time_weight'] * df['boost_factor']

    # VEX (Vanna Exposure) Calculation
    # Vanna: Change in Delta per 1% vol change
    # Formula: Vanna * S * OI * 100 (Adjusted to dollar notional equivalent)
    # Convention: Dealers sell calls -> Short Vanna. Dealers sell puts -> Long Vanna.
    # We apply the side multiplier similarly.
    # Note: Vanna is usually negative for calls (vol up -> delta down? No, vol up -> delta up for OTM calls).
    
    df['vex'] = -1 * df['vanna_bs'] * S * 100 * df['open_interest'] * df['time_weight']
    
    # Charm Exposure
    df['cex'] = df['side'] * df['charm_bs'] * 100 * df['open_interest'] * df['time_weight']

    # Cleanup
    return df[['expiration_date', 'strike', 'option_type', 'gex', 'vex', 'cex', 'open_interest', 'days_to_exp', 'iv']]

def find_gamma_flip(df):
    """Finds the strike where total GEX flips from negative to positive."""
    if df.empty: return None
    
    strikes = df.groupby('strike')['gex'].sum().sort_index()
    
    # Scan for sign change
    for i in range(len(strikes) - 1):
        if strikes.iloc[i] < 0 and strikes.iloc[i+1] > 0:
            return strikes.index[i+1] # approximate
    return None

# -------------------------
# VISUALIZATION
# -------------------------
def render_heatmap(df, ticker, S, metric, title, flip_strike=None):
    if df.empty: return None

    # Pivot: Index=Strike, Columns=Expiry
    pivot = df.pivot_table(index='strike', columns='expiration_date', values=metric, aggfunc='sum')
    pivot = pivot.sort_index(ascending=False) # High strikes on top
    
    # Sort columns by date
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Dynamic limit for colorscale to avoid outliers washing out the chart
    max_val = np.percentile(np.abs(pivot.values), 95)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=CUSTOM_COLORSCALE,
        zmid=0,
        zmin=-max_val,
        zmax=max_val,
        colorbar=dict(title="Exposure ($)")
    ))

    # Add Spot Price Line (Horizontal)
    fig.add_hline(
        y=S, 
        line_dash="dash", 
        line_color="white", 
        annotation_text=f"Spot: {S:.2f}",
        annotation_position="bottom right"
    )

    if flip_strike and metric == 'gex':
        fig.add_hline(
            y=flip_strike,
            line_dash="dot",
            line_color="yellow",
            annotation_text=f"Flip: {flip_strike}",
            annotation_position="top left"
        )

    fig.update_layout(
        title=f"{ticker} {title} Surface",
        height=600,
        xaxis_title="Expiration",
        yaxis_title="Strike",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def render_profile(df, S, flip):
    # Aggregate GEX by Strike
    gex_profile = df.groupby('strike')['gex'].sum()
    
    fig = go.Figure()
    
    # GEX Bars
    fig.add_trace(go.Bar(
        x=gex_profile.index, 
        y=gex_profile.values,
        marker_color=np.where(gex_profile.values < 0, '#FF3333', '#00FF7F'),
        name='Net GEX'
    ))
    
    fig.add_vline(x=S, line_dash="dash", line_color="white", annotation_text="Spot")
    if flip:
        fig.add_vline(x=flip, line_dash="dot", line_color="yellow", annotation_text="Flip")

    fig.update_layout(
        title="Net GEX by Strike (All Expiries)",
        template="plotly_dark",
        xaxis_title="Strike",
        yaxis_title="Net GEX ($)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# -------------------------
# MAIN
# -------------------------
def main():
    st.title("Dealer Flow: GEX & Vanna Analysis")
    
    # Controls
    with st.expander("Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        ticker = c1.text_input("Ticker", "SPX").upper()
        max_exp = c2.slider("Expiries to Fetch", 1, 20, 5)
        s_range = c3.number_input("Strike Window", value=150)
        refresh = c4.button("Run Analysis", type="primary")

    if refresh:
        # Fetch
        S, raw_df = fetch_data(ticker, max_exp)
        
        if raw_df is None or raw_df.empty:
            st.error("No data found. Check Ticker or API Token.")
            return

        # Process
        df = process_exposure(raw_df, S, s_range)
        
        # Metrics
        total_gex = df['gex'].sum() / 10**9 # In Billions
        total_vex = df['vex'].sum() / 10**6 # In Millions
        flip = find_gamma_flip(df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spot Price", f"${S:,.2f}")
        m2.metric("Net GEX", f"${total_gex:.2f} B", delta_color="normal")
        m3.metric("Net VEX", f"${total_vex:.2f} M")
        m4.metric("Gamma Flip", f"{flip}" if flip else "N/A")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["🔥 GEX Heatmap", "🌊 Vanna Heatmap", "📋 Diagnostics"])
        
        with tab1:
            fig_gex = render_heatmap(df, ticker, S, 'gex', "Gamma Exposure (GEX)", flip)
            if fig_gex: st.plotly_chart(fig_gex, use_container_width=True)
            
            fig_prof = render_profile(df, S, flip)
            st.plotly_chart(fig_prof, use_container_width=True)

        with tab2:
            fig_vex = render_heatmap(df, ticker, S, 'vex', "Vanna Exposure (VEX)")
            if fig_vex: st.plotly_chart(fig_vex, use_container_width=True)

        with tab3:
            st.dataframe(df.sort_values('gex', ascending=False).head(50), use_container_width=True)

if __name__ == "__main__":
    main()