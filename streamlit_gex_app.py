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
    return f'${val/1000:,.0f}'

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
    
    # PRE-FILTER WEEKENDS AND HOLIDAYS
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
    return S, pd.concat(dfs, ignore_index=True) if dfs else None

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
    
    # Calculate d1 and d2 for Black-Scholes Greeks
    d1 = (np.log(S / df['strike']) + (RISK_FREE_RATE + 0.5 * df['iv']**2) * df['T']) / (df['iv'] * np.sqrt(df['T']))
    d2 = d1 - df['iv'] * np.sqrt(df['T'])
    
    # Gamma (same for calls and puts, always positive)
    df['gamma_bs'] = norm.pdf(d1) / (S * df['iv'] * np.sqrt(df['T']))
    
    # Vanna (corrected formula): dGamma/dVol
    # Vanna = -φ(d1) * d2 / (S * σ * √T)
    df['vanna_bs'] = -norm.pdf(d1) * d2 / (S * df['iv'] * np.sqrt(df['T']))
    
    # GEX from DEALER perspective (dealers short options to customers)
    # Negative GEX = dealers short gamma (destabilizing, need to hedge dynamically)
    # Positive GEX = dealers long gamma (stabilizing)
    # Gamma is always positive, so we flip sign for dealer perspective
    df['gex'] = -1 * df['gamma_bs'] * (S**2) * 0.01 * df['open_interest'] * 100
    
    # VEX from DEALER perspective  
    # Represents change in delta per 1 vol point change
    # Scaled by 1% spot move for consistency with GEX
    df['vex'] = -1 * df['vanna_bs'] * S * 0.01 * df['open_interest'] * 100
    
    return df

# -------------------------
# RENDER
# -------------------------
def render_heatmap(df, S, metric, title):
    if df.empty: return None

    # 1. Filter to strikes within 20 above and 20 below spot
    unique_strikes = sorted(df['strike'].unique())
    idx = np.argmin(np.abs(np.array(unique_strikes) - S))
    strikes_to_show = unique_strikes[max(0, idx-20):min(len(unique_strikes), idx+21)]
    df_filtered = df[df['strike'].isin(strikes_to_show)].copy()
    
    # 2. Pivot
    pivot = df_filtered.pivot_table(index='strike', columns='expiration_date', values=metric, aggfunc='sum')
    
    # 3. STRICT WEEKEND REMOVAL - Drop weekend/holiday columns entirely
    weekday_cols = []
    for c in pivot.columns:
        dt = pd.to_datetime(c)
        if dt.dayofweek < 5 and dt not in MARKET_HOLIDAYS:
            weekday_cols.append(c)
    
    if not weekday_cols:
        return None
        
    pivot = pivot[weekday_cols]
    
    # 4. Fill missing values with 0
    pivot = pivot.fillna(0).sort_index(ascending=False)
    
    # 4. Find max absolute value cell for star marker
    abs_vals = np.abs(pivot.values)
    if abs_vals.size > 0:
        max_idx = np.unravel_index(np.argmax(abs_vals), abs_vals.shape)
    else:
        max_idx = None
    
    # 5. Create text with star on max value
    text_values = []
    for i in range(len(pivot.index)):
        row_text = []
        for j in range(len(pivot.columns)):
            val_str = format_thousands(pivot.iloc[i, j])
            if max_idx and (i, j) == max_idx:
                val_str += " ★"
            row_text.append(val_str)
        text_values.append(row_text)
    
    max_val = np.percentile(abs_vals, 98) if np.any(pivot.values) else 1

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, 
        x=pivot.columns, 
        y=pivot.index,
        text=text_values, 
        texttemplate="%{text}", 
        textfont={"size": 9},
        colorscale=CUSTOM_COLORSCALE, 
        zmid=0, 
        zmin=-max_val, 
        zmax=max_val
    ))

    # Add spot marker with arrow
    fig.add_annotation(
        x=-0.08, y=S, xref="paper", yref="y",
        text="➤", showarrow=False,
        font=dict(size=20, color="yellow"),
        xanchor="right"
    )
    
    fig.add_hline(y=S, line_dash="dash", line_color="yellow", line_width=1, opacity=0.5)
    
    fig.update_layout(
        title=title, 
        height=800, 
        template="plotly_dark",
        yaxis=dict(
            title="Strike",
            tickmode='linear',
            tick0=pivot.index.min(),
            dtick=max(1, (pivot.index.max() - pivot.index.min()) / 40)  # Show all strikes
        )
    )
    return fig

def render_gex_concentration(df, S):
    """Render horizontal bar chart of GEX concentration by strike"""
    if df.empty: return None
    
    # Filter to strikes within 20 above and 20 below spot
    unique_strikes = sorted(df['strike'].unique())
    idx = np.argmin(np.abs(np.array(unique_strikes) - S))
    strikes_to_show = unique_strikes[max(0, idx-20):min(len(unique_strikes), idx+21)]
    df_filtered = df[df['strike'].isin(strikes_to_show)].copy()
    
    # Aggregate GEX by strike
    gex_by_strike = df_filtered.groupby('strike')['gex'].sum().sort_index(ascending=False)
    
    # Calculate key levels
    abs_gex = gex_by_strike.abs()
    
    # Find walls and floor/ceiling
    positive_gex = gex_by_strike[gex_by_strike > 0]
    negative_gex = gex_by_strike[gex_by_strike < 0]
    
    call_wall = positive_gex.idxmax() if len(positive_gex) > 0 else None
    put_wall = negative_gex.idxmin() if len(negative_gex) > 0 else None
    
    # Floor = highest strike with negative GEX, Ceiling = lowest strike with positive GEX
    floor = negative_gex.index.max() if len(negative_gex) > 0 else None
    ceiling = positive_gex.index.min() if len(positive_gex) > 0 else None
    
    # Create color array
    colors = ['#00FF7F' if v > 0 else '#FF3333' for v in gex_by_strike.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=gex_by_strike.index,
        x=gex_by_strike.values / 1000,  # Convert to thousands
        orientation='h',
        marker=dict(color=colors),
        text=[format_thousands(v) for v in gex_by_strike.values],
        textposition='outside',
        hovertemplate='Strike: %{y}<br>GEX: %{text}<extra></extra>'
    ))
    
    # Add annotations for key levels
    annotations = []
    if call_wall:
        annotations.append(dict(
            x=gex_by_strike[call_wall]/1000, y=call_wall,
            text="Call Wall", showarrow=True, arrowhead=2,
            ax=40, ay=0, font=dict(color='#00FF7F', size=12)
        ))
    if put_wall:
        annotations.append(dict(
            x=gex_by_strike[put_wall]/1000, y=put_wall,
            text="Put Wall", showarrow=True, arrowhead=2,
            ax=-40, ay=0, font=dict(color='#FF3333', size=12)
        ))
    if floor:
        annotations.append(dict(
            x=0, y=floor,
            text="Floor", showarrow=True, arrowhead=2,
            ax=-30, ay=-20, font=dict(color='white', size=10)
        ))
    if ceiling:
        annotations.append(dict(
            x=0, y=ceiling,
            text="Ceiling", showarrow=True, arrowhead=2,
            ax=-30, ay=20, font=dict(color='white', size=10)
        ))
    
    fig.update_layout(
        title="GEX Concentration by Strike",
        xaxis_title="GEX (

def render_diagnostics(df, S):
    """Render calls and puts side by side with net calculations"""
    if df.empty: return None
    
    # Filter to strikes within 20 above and 20 below spot
    unique_strikes = sorted(df['strike'].unique())
    idx = np.argmin(np.abs(np.array(unique_strikes) - S))
    strikes_to_show = unique_strikes[max(0, idx-20):min(len(unique_strikes), idx+21)]
    df_filtered = df[df['strike'].isin(strikes_to_show)].copy()
    
    # Separate calls and puts
    calls = df_filtered[df_filtered['option_type'].str.lower() == 'call'].copy()
    puts = df_filtered[df_filtered['option_type'].str.lower() == 'put'].copy()
    
    # Calculate net values
    net_gex = df_filtered.groupby('strike')[['gex', 'vex']].sum().sort_index(ascending=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📞 Calls")
        if not calls.empty:
            calls_display = calls[['expiration_date', 'strike', 'gex', 'vex', 'open_interest', 'iv']].sort_values('strike', ascending=False)
            calls_display['gex'] = calls_display['gex'].apply(lambda x: f"{x/1000:,.0f}K")
            calls_display['vex'] = calls_display['vex'].apply(lambda x: f"{x/1000:,.0f}K")
            st.dataframe(calls_display, use_container_width=True, height=400)
        else:
            st.info("No call data")
    
    with col2:
        st.subheader("📉 Puts")
        if not puts.empty:
            puts_display = puts[['expiration_date', 'strike', 'gex', 'vex', 'open_interest', 'iv']].sort_values('strike', ascending=False)
            puts_display['gex'] = puts_display['gex'].apply(lambda x: f"{x/1000:,.0f}K")
            puts_display['vex'] = puts_display['vex'].apply(lambda x: f"{x/1000:,.0f}K")
            st.dataframe(puts_display, use_container_width=True, height=400)
        else:
            st.info("No put data")
    
    with col3:
        st.subheader("🎯 Net by Strike")
        net_display = net_gex.copy()
        net_display['gex_fmt'] = net_display['gex'].apply(lambda x: f"{x/1000:,.0f}K")
        net_display['vex_fmt'] = net_display['vex'].apply(lambda x: f"{x/1000:,.0f}K")
        st.dataframe(
            net_display[['gex_fmt', 'vex_fmt']].rename(columns={'gex_fmt': 'Net GEX', 'vex_fmt': 'Net VEX'}),
            use_container_width=True,
            height=400
        )

def main():
    st.title("GEX & VEX: Weekend-Free Surface")
    
    # Add explanation
    with st.expander("ℹ️ Understanding GEX & VEX"):
        st.markdown("""
        **Dealer Gamma Exposure (GEX)**
        - **Negative GEX**: Dealers are short gamma → must buy rallies and sell dips (destabilizing)
        - **Positive GEX**: Dealers are long gamma → sell rallies and buy dips (stabilizing)
        - **Zero GEX**: Gamma neutral, potential pivot point
        
        **Dealer Vanna Exposure (VEX)**
        - **Positive VEX**: Rising volatility causes dealers to buy spot
        - **Negative VEX**: Rising volatility causes dealers to sell spot
        
        **Key Levels**
        - **Call Wall**: Largest positive GEX strike (resistance)
        - **Put Wall**: Largest negative GEX strike (support)
        - **Ceiling**: Lowest strike with positive GEX
        - **Floor**: Highest strike with negative GEX
        """)
    
    with st.sidebar:
        ticker = st.text_input("Ticker", "SPX").upper()
        strike_depth = st.select_slider("Strikes", options=[25, 40, 50, 80, 100, 150], value=80 if ticker=="SPX" else 40)
        max_exp = st.number_input("Expiries", 1, 15, 5)
        run = st.button("Calculate", type="primary")

    if run:
        S, raw_df = fetch_data(ticker, max_exp)
        if raw_df is not None and S is not None:
            df = process_exposure(raw_df, S, strike_depth)
            
            if not df.empty:
                t1, t2, t3, t4 = st.tabs(["Gamma (GEX)", "Vanna (VEX)", "GEX Concentration", "Diagnostics"])
                
                with t1:
                    fig = render_heatmap(df, S, 'gex', f"{ticker} Gamma ($'000s)")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with t2:
                    fig = render_heatmap(df, S, 'vex', f"{ticker} Vanna ($'000s)")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with t3:
                    fig = render_gex_concentration(df, S)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with t4:
                    render_diagnostics(df, S)
            else:
                st.error("No data available after processing")
        else:
            st.error("Failed to fetch data")

if __name__ == "__main__":
    main()
000s)",
        yaxis_title="Strike",
        height=700,
        template="plotly_dark",
        showlegend=False,
        annotations=annotations,
        yaxis=dict(
            tickmode='array',
            tickvals=gex_by_strike.index,
            ticktext=[f"{strike:.0f}" for strike in gex_by_strike.index]
        )
    )
    
    return fig

def render_diagnostics(df):
    """Render calls and puts side by side with net calculations"""
    if df.empty: return None
    
    # Separate calls and puts
    calls = df[df['option_type'].str.lower() == 'call'].copy()
    puts = df[df['option_type'].str.lower() == 'put'].copy()
    
    # Calculate net values
    net_gex = df.groupby('strike')[['gex', 'vex']].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📞 Calls")
        if not calls.empty:
            calls_display = calls[['expiration_date', 'strike', 'gex', 'vex', 'open_interest', 'iv']].sort_values('gex', ascending=False)
            calls_display['gex'] = calls_display['gex'].apply(lambda x: f"{x/1000:,.0f}K")
            calls_display['vex'] = calls_display['vex'].apply(lambda x: f"{x/1000:,.0f}K")
            st.dataframe(calls_display, use_container_width=True, height=400)
        else:
            st.info("No call data")
    
    with col2:
        st.subheader("📉 Puts")
        if not puts.empty:
            puts_display = puts[['expiration_date', 'strike', 'gex', 'vex', 'open_interest', 'iv']].sort_values('gex')
            puts_display['gex'] = puts_display['gex'].apply(lambda x: f"{x/1000:,.0f}K")
            puts_display['vex'] = puts_display['vex'].apply(lambda x: f"{x/1000:,.0f}K")
            st.dataframe(puts_display, use_container_width=True, height=400)
        else:
            st.info("No put data")
    
    with col3:
        st.subheader("🎯 Net by Strike")
        net_display = net_gex.copy()
        net_display['gex_fmt'] = net_display['gex'].apply(lambda x: f"{x/1000:,.0f}K")
        net_display['vex_fmt'] = net_display['vex'].apply(lambda x: f"{x/1000:,.0f}K")
        st.dataframe(
            net_display[['gex_fmt', 'vex_fmt']].rename(columns={'gex_fmt': 'Net GEX', 'vex_fmt': 'Net VEX'}),
            use_container_width=True,
            height=400
        )

def main():
    st.title("GEX & VEX: Weekend-Free Surface")
    
    # Add explanation
    with st.expander("ℹ️ Understanding GEX & VEX"):
        st.markdown("""
        **Dealer Gamma Exposure (GEX)**
        - **Negative GEX**: Dealers are short gamma → must buy rallies and sell dips (destabilizing)
        - **Positive GEX**: Dealers are long gamma → sell rallies and buy dips (stabilizing)
        - **Zero GEX**: Gamma neutral, potential pivot point
        
        **Dealer Vanna Exposure (VEX)**
        - **Positive VEX**: Rising volatility causes dealers to buy spot
        - **Negative VEX**: Rising volatility causes dealers to sell spot
        
        **Key Levels**
        - **Call Wall**: Largest positive GEX strike (resistance)
        - **Put Wall**: Largest negative GEX strike (support)
        - **Ceiling**: Lowest strike with positive GEX
        - **Floor**: Highest strike with negative GEX
        """)
    
    with st.sidebar:
        ticker = st.text_input("Ticker", "SPX").upper()
        strike_depth = st.select_slider("Strikes", options=[25, 40, 50, 80, 100, 150], value=80 if ticker=="SPX" else 40)
        max_exp = st.number_input("Expiries", 1, 15, 5)
        run = st.button("Calculate", type="primary")

    if run:
        S, raw_df = fetch_data(ticker, max_exp)
        if raw_df is not None and S is not None:
            df = process_exposure(raw_df, S, strike_depth)
            
            if not df.empty:
                t1, t2, t3, t4 = st.tabs(["Gamma (GEX)", "Vanna (VEX)", "GEX Concentration", "Diagnostics"])
                
                with t1:
                    fig = render_heatmap(df, S, 'gex', f"{ticker} Gamma ($'000s)")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with t2:
                    fig = render_heatmap(df, S, 'vex', f"{ticker} Vanna ($'000s)")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with t3:
                    fig = render_gex_concentration(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with t4:
                    render_diagnostics(df)
            else:
                st.error("No data available after processing")
        else:
            st.error("Failed to fetch data")

if __name__ == "__main__":
    main()