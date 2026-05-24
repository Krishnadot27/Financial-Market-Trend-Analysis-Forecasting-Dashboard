"""
app/streamlit_app.py
─────────────────────
NSE Stock Prediction Dashboard — Production Streamlit App

Run with:
    streamlit run app/streamlit_app.py

Features:
  - Live NSE price data via yfinance
  - Real-time predictions from trained XGBoost model
  - Interactive candlestick + Bollinger Bands chart
  - Backtest performance panel
  - Feature importance visualisation
  - SHAP explainability (if shap installed)
  - Multi-ticker support
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.utils import load_config, ensure_dirs, project_path
from src.prediction_store import PredictionStore
from src.watchlist import WatchlistManager
from src.broker_angel import get_broker, ManualHoldingsManager
from src.feature_engineering import FeatureEngineer
from src.backtesting import Backtester
from src.evaluation import ModelEvaluator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Alpha — Stock Prediction",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════
       FONTS
    ═══════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

    /* ═══════════════════════════════════════════════════
       DESIGN TOKENS
    ═══════════════════════════════════════════════════ */
    :root {
        --bg-base:        #070B14;
        --bg-surface:     #0D1321;
        --bg-card:        #111827;
        --bg-card-hover:  #141E2F;
        --border:         #1E293B;
        --border-subtle:  #162032;
        --text-primary:   #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted:     #475569;
        --accent-blue:    #3B82F6;
        --accent-indigo:  #6366F1;
        --accent-cyan:    #06B6D4;
        --accent-purple:  #8B5CF6;
        --green:          #22C55E;
        --red:            #EF4444;
        --amber:          #F59E0B;
        --grad-primary:   linear-gradient(135deg, #1D4ED8 0%, #6366F1 50%, #8B5CF6 100%);
        --grad-surface:   linear-gradient(145deg, #0F172A 0%, #0D1321 100%);
        --grad-green:     linear-gradient(135deg, #052e16 0%, #14532d 100%);
        --grad-red:       linear-gradient(135deg, #3B0009 0%, #7f1d1d 100%);
        --glow-blue:      0 0 24px rgba(59,130,246,0.18);
        --glow-green:     0 0 20px rgba(34,197,94,0.15);
        --glow-red:       0 0 20px rgba(239,68,68,0.15);
        --shadow-card:    0 4px 24px rgba(0,0,0,0.45), 0 1px 4px rgba(0,0,0,0.3);
        --radius-sm:      8px;
        --radius-md:      12px;
        --radius-lg:      16px;
        --radius-xl:      20px;
        --transition:     0.28s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ═══════════════════════════════════════════════════
       GLOBAL RESET & BASE
    ═══════════════════════════════════════════════════ */
    html, body, [class*="css"], .stApp {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background: var(--bg-base);
        background-image:
            radial-gradient(ellipse 80% 50% at 20% -10%, rgba(59,130,246,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(99,102,241,0.05) 0%, transparent 60%);
        color: var(--text-primary);
    }

    /* Fade-in on load */
    .main .block-container {
        animation: fadeInUp 0.5s ease both;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ═══════════════════════════════════════════════════
       SCROLLBAR
    ═══════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ═══════════════════════════════════════════════════
       SIDEBAR
    ═══════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface);
        border-right: 1px solid var(--border-subtle);
        box-shadow: 4px 0 32px rgba(0,0,0,0.35);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    /* Sidebar labels */
    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    /* Sidebar selectbox / radio wrappers */
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stRadio > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        transition: border-color var(--transition);
    }

    section[data-testid="stSidebar"] .stSelectbox > div:hover,
    section[data-testid="stSidebar"] .stRadio > div:hover {
        border-color: var(--accent-blue);
    }

    /* Sidebar sliders */
    section[data-testid="stSidebar"] .stSlider > div {
        padding: 4px 0;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: var(--grad-primary) !important;
        box-shadow: var(--glow-blue);
    }

    /* Sidebar divider */
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-subtle);
        margin: 16px 0;
    }

    /* ═══════════════════════════════════════════════════
       TYPOGRAPHY
    ═══════════════════════════════════════════════════ */
    h1, h2, h3, h4 {
        font-family: 'Syne', sans-serif;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    h1 { font-size: 2rem;   font-weight: 800; }
    h2 { font-size: 1.5rem; font-weight: 700; }
    h3 { font-size: 1.2rem; font-weight: 700; }
    h4 { font-size: 1rem;   font-weight: 600; }

    p, li { color: var(--text-secondary); line-height: 1.65; }

    .mono { font-family: 'DM Mono', monospace; }
    .pos  { color: var(--green); font-weight: 600; }
    .neg  { color: var(--red);   font-weight: 600; }
    .neutral { color: var(--text-secondary); }

    /* ═══════════════════════════════════════════════════
       DIVIDERS
    ═══════════════════════════════════════════════════ */
    hr {
        border: none;
        border-top: 1px solid var(--border-subtle);
        margin: 24px 0;
    }

    /* ═══════════════════════════════════════════════════
       METRIC CARDS (st.metric)
    ═══════════════════════════════════════════════════ */
    [data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 18px 20px;
        box-shadow: var(--shadow-card);
        transition: transform var(--transition), border-color var(--transition), box-shadow var(--transition);
        position: relative;
        overflow: hidden;
    }

    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--grad-primary);
        opacity: 0.6;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(59,130,246,0.35);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5), var(--glow-blue);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.09em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.76rem !important;
    }

    /* ═══════════════════════════════════════════════════
       PREDICTION BADGES
    ═══════════════════════════════════════════════════ */
    .pred-up {
        background: var(--grad-green);
        border: 1px solid rgba(34,197,94,0.4);
        border-radius: var(--radius-md);
        padding: 24px 28px;
        text-align: center;
        box-shadow: var(--glow-green), var(--shadow-card);
        transition: transform var(--transition), box-shadow var(--transition);
    }

    .pred-up:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 36px rgba(34,197,94,0.22), var(--shadow-card);
    }

    .pred-down {
        background: var(--grad-red);
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: var(--radius-md);
        padding: 24px 28px;
        text-align: center;
        box-shadow: var(--glow-red), var(--shadow-card);
        transition: transform var(--transition), box-shadow var(--transition);
    }

    .pred-down:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 36px rgba(239,68,68,0.22), var(--shadow-card);
    }

    .pred-label {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .pred-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.76rem;
        color: var(--text-secondary);
        margin-top: 6px;
        letter-spacing: 0.04em;
    }

    /* ═══════════════════════════════════════════════════
       BUTTONS
    ═══════════════════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg, #1D4ED8 0%, #4F46E5 100%);
        color: #fff !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.03em !important;
        padding: 8px 20px !important;
        box-shadow: 0 2px 12px rgba(79,70,229,0.3) !important;
        transition: all var(--transition) !important;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.08), transparent);
        opacity: 0;
        transition: opacity var(--transition);
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 24px rgba(79,70,229,0.5), 0 0 0 1px rgba(99,102,241,0.5) !important;
        border-color: rgba(99,102,241,0.7) !important;
    }

    .stButton > button:hover::after { opacity: 1; }

    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(79,70,229,0.3) !important;
    }

    /* Download button variant */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #065F46 0%, #047857 100%) !important;
        border-color: rgba(34,197,94,0.35) !important;
        box-shadow: 0 2px 12px rgba(34,197,94,0.2) !important;
    }

    .stDownloadButton > button:hover {
        box-shadow: 0 4px 24px rgba(34,197,94,0.35), 0 0 0 1px rgba(34,197,94,0.45) !important;
        border-color: rgba(34,197,94,0.6) !important;
    }

    /* ═══════════════════════════════════════════════════
       INPUTS & SELECTS
    ═══════════════════════════════════════════════════ */
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        transition: border-color var(--transition), box-shadow var(--transition) !important;
    }

    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within,
    .stSelectbox > div > div:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }

    /* ═══════════════════════════════════════════════════
       TABS
    ═══════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 5px;
        gap: 4px;
        box-shadow: var(--shadow-card);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--radius-sm);
        color: var(--text-muted);
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 8px 18px;
        border: none;
        transition: all var(--transition);
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary);
        background: rgba(255,255,255,0.04);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E3A5F 0%, #1E293B 100%) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(59,130,246,0.25) !important;
    }

    /* Tab panel */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }

    /* ═══════════════════════════════════════════════════
       EXPANDER
    ═══════════════════════════════════════════════════ */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        transition: background var(--transition), border-color var(--transition) !important;
    }

    .streamlit-expanderHeader:hover {
        background: var(--bg-card-hover) !important;
        border-color: rgba(59,130,246,0.3) !important;
        color: var(--text-primary) !important;
    }

    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
    }

    /* ═══════════════════════════════════════════════════
       DATAFRAMES / TABLES
    ═══════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }

    [data-testid="stDataFrame"] iframe {
        border-radius: var(--radius-md);
    }

    /* ═══════════════════════════════════════════════════
       ALERTS / INFO / SUCCESS / WARNING
    ═══════════════════════════════════════════════════ */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border-left-width: 3px !important;
        font-size: 0.85rem !important;
    }

    [data-baseweb="notification"][kind="info"] {
        background: rgba(59,130,246,0.08) !important;
        border-color: var(--accent-blue) !important;
    }

    [data-baseweb="notification"][kind="positive"] {
        background: rgba(34,197,94,0.08) !important;
        border-color: var(--green) !important;
    }

    [data-baseweb="notification"][kind="warning"] {
        background: rgba(245,158,11,0.08) !important;
        border-color: var(--amber) !important;
    }

    [data-baseweb="notification"][kind="negative"] {
        background: rgba(239,68,68,0.08) !important;
        border-color: var(--red) !important;
    }

    /* ═══════════════════════════════════════════════════
       SPINNER
    ═══════════════════════════════════════════════════ */
    .stSpinner > div {
        border-top-color: var(--accent-blue) !important;
    }

    /* ═══════════════════════════════════════════════════
       PLOTLY CHARTS — transparent wrappers
    ═══════════════════════════════════════════════════ */
    .stPlotlyChart {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-card);
        transition: border-color var(--transition), box-shadow var(--transition);
    }

    .stPlotlyChart:hover {
        border-color: rgba(59,130,246,0.2);
        box-shadow: 0 8px 40px rgba(0,0,0,0.5), var(--glow-blue);
    }

    /* ═══════════════════════════════════════════════════
       CAPTION & SMALL TEXT
    ═══════════════════════════════════════════════════ */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 0.72rem !important;
        color: var(--text-muted) !important;
        line-height: 1.5 !important;
    }

    /* ═══════════════════════════════════════════════════
       RADIO BUTTONS
    ═══════════════════════════════════════════════════ */
    .stRadio [data-baseweb="radio"] {
        gap: 8px;
    }

    .stRadio label {
        font-size: 0.82rem !important;
        color: var(--text-secondary) !important;
    }

    /* ═══════════════════════════════════════════════════
       GLASSMORPHISM UTILITY CLASSES
       (used inside st.markdown HTML wrappers)
    ═══════════════════════════════════════════════════ */

    /* Glass card base */
    .glass-card {
        background: rgba(17,24,39,0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(59,130,246,0.15), inset 0 1px 0 rgba(255,255,255,0.07);
        border-color: rgba(59,130,246,0.2);
    }

    /* Stat mini card */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px 20px;
        text-align: center;
        box-shadow: var(--shadow-card);
        transition: transform var(--transition), border-color var(--transition), box-shadow var(--transition);
        position: relative;
        overflow: hidden;
    }

    .stat-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    }

    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99,102,241,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }

    .stat-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 6px;
    }

    .stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    /* Info row (key-value) */
    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid rgba(30,41,59,0.6);
    }

    .info-row:last-child { border-bottom: none; }

    .info-key {
        font-size: 0.78rem;
        color: var(--text-muted);
    }

    .info-val {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-primary);
    }

    /* Badge pill */
    .badge {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        padding: 2px 10px;
        border-radius: 99px;
    }

    /* Indicator card (regime, intraday) */
    .ind-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px;
        box-shadow: var(--shadow-card);
        height: 100%;
        transition: border-color var(--transition), box-shadow var(--transition);
    }

    .ind-card:hover {
        border-color: rgba(59,130,246,0.25);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    /* Section headings inside main content */
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* Market status pill */
    .market-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 99px;
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-family: 'DM Mono', monospace;
    }

    /* Empty state container */
    .empty-state {
        text-align: center;
        padding: 64px 32px;
        color: var(--text-muted);
        background: var(--bg-card);
        border: 1px dashed var(--border);
        border-radius: var(--radius-lg);
    }

    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
        opacity: 0.5;
    }

    .empty-state-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }

    .empty-state-sub {
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    /* Confluence badge (intraday) */
    .conf-badge {
        border-radius: var(--radius-md);
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 20px;
        border-width: 2px;
        border-style: solid;
        box-shadow: var(--shadow-card);
    }

    /* Watchlist stock card */
    .stock-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px;
        margin-bottom: 4px;
        transition: transform var(--transition), border-color var(--transition), box-shadow var(--transition);
    }

    .stock-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59,130,246,0.25);
        box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    }

    /* Intraday signal cards */
    .signal-card {
        border-radius: var(--radius-md);
        padding: 16px;
        border-style: solid;
        border-width: 1px;
        border-left-width: 4px;
        box-shadow: var(--shadow-card);
        transition: transform var(--transition), box-shadow var(--transition);
    }

    .signal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    }

    /* Tooltip-style footer */
    .footer-note {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        color: var(--text-muted);
        line-height: 1.9;
        padding: 12px 16px;
        background: rgba(7,11,20,0.5);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
    }

    /* Code blocks */
    code, .stCode {
        background: rgba(15,23,42,0.8) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.8rem !important;
        color: var(--accent-cyan) !important;
    }

    /* Stagger animation for cards */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Subtle pulse for alerts */
    @keyframes subtlePulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(245,158,11,0); }
        50%       { box-shadow: 0 0 0 4px rgba(245,158,11,0.1); }
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Config & caching
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_store():
    """Cached PredictionStore instance — shared across all sessions."""
    return PredictionStore()

@st.cache_resource
def get_watchlist():
    return WatchlistManager()

@st.cache_data(ttl=300)   # Refresh every 5 minutes
def fetch_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance with caching."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df.dropna()


class _ModelWrapper:
    """
    Unified wrapper for both v1 (XGBoost pipeline) and v2 (ensemble) models.
    Handles feature alignment automatically.
    """
    def __init__(self, pipeline, features: list, metrics: dict = None, version: str = "v1"):
        self._pipeline            = pipeline
        self.feature_names_in_    = features
        self.metrics              = metrics or {}
        self.version              = version

    def _align(self, X):
        """Align X to the exact features the model expects."""
        import pandas as pd
        needed    = self.feature_names_in_
        available = [f for f in needed if f in X.columns]
        aligned   = pd.DataFrame(0.0, index=X.index, columns=needed)
        if available:
            aligned[available] = X[available].values
        return aligned

    def predict(self, X):
        return self._pipeline.predict(self._align(X))

    def predict_proba(self, X):
        try:
            return self._pipeline.predict_proba(self._align(X))
        except Exception:
            return None


@st.cache_resource
def load_model(model_path: str):
    """
    Smart model loader — tries v2 ensemble first, falls back to v1 XGBoost.
    Always returns a _ModelWrapper with .predict() and feature alignment.
    """
    safe_ticker = Path(model_path).stem.replace("xgboost_classification_", "")                                         .replace("xgboost_regression_", "")
    task = "classification" if "classification" in model_path else "regression"

    # Try v2 ensemble model first
    v2_path = str(model_path).replace(
        f"xgboost_{task}_", f"ensemble_{task}_"
    )
    for path in [v2_path, model_path]:
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict):
                pipeline = obj.get("pipeline")
                features = obj.get("feature_names", [])
                metrics  = {k: v for k, v in obj.items()
                            if k not in ("pipeline","feature_names","task","train_time","params")}
                version  = "v2" if "ensemble" in str(path) else "v1"
                if pipeline is not None and features:
                    return _ModelWrapper(pipeline, features, metrics, version)
                elif pipeline is not None:
                    # v1 without feature list
                    try:
                        est = pipeline.named_steps.get("model", pipeline)
                        feats = list(getattr(est, "feature_names_in_", []))
                    except Exception:
                        feats = []
                    return _ModelWrapper(pipeline, feats, {}, "v1")
            # Raw pipeline (no dict wrapper)
            elif obj is not None:
                try:
                    est   = obj.named_steps.get("model", obj)
                    feats = list(getattr(est, "feature_names_in_", []))
                except Exception:
                    feats = []
                return _ModelWrapper(obj, feats, {}, "v1")
        except Exception:
            continue
    return None


@st.cache_data(ttl=300)
def build_features(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch prices + run feature engineering pipeline."""
    cfg = get_config()
    df  = fetch_price_data(ticker, period)
    if len(df) < 220:
        return pd.DataFrame()
    fe      = FeatureEngineer(df, config=cfg)
    df_feat = fe.build()
    return df_feat


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(cfg: dict) -> tuple:
    with st.sidebar:
        st.markdown("""
        <div style='padding: 12px 0 28px'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
                <div style='width:34px;height:34px;border-radius:8px;
                            background:linear-gradient(135deg,#1D4ED8,#6366F1);
                            display:flex;align-items:center;justify-content:center;
                            box-shadow:0 4px 14px rgba(79,70,229,0.45);flex-shrink:0'>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>
                        <polyline points="16 7 22 7 22 13"/>
                    </svg>
                </div>
                <div>
                    <div style='font-family:Syne,sans-serif;font-size:1.2rem;
                                font-weight:800;color:#F1F5F9;letter-spacing:-0.03em;line-height:1.1'>
                        NSE Alpha
                    </div>
                    <div style='font-family:DM Mono,monospace;font-size:0.6rem;
                                color:#475569;letter-spacing:0.07em;text-transform:uppercase;margin-top:1px'>
                        ML Prediction Engine
                    </div>
                </div>
            </div>
            <div style='height:1px;background:linear-gradient(90deg,rgba(59,130,246,0.7),rgba(99,102,241,0.5),transparent);
                        margin-top:4px'></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Select Ticker**")
        tickers = cfg["data"]["tickers"]
        ticker  = st.selectbox("", tickers, label_visibility="collapsed")

        st.markdown("**Data Period**")
        period = st.selectbox("", ["1y", "2y", "3y", "5y"],
                              index=1, label_visibility="collapsed")

        st.markdown("**Model Task**")
        task = st.radio("", ["regression", "classification"],
                        label_visibility="collapsed")

        st.divider()

        st.markdown("**Backtest Settings**")
        txn_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.05) / 100
        slippage = st.slider("Slippage (%)", 0.0, 0.2, 0.05, 0.01) / 100
        threshold = st.slider("Signal Threshold", -0.5, 0.5, 0.0, 0.01)

        st.divider()
        st.markdown("""
        <div class='footer-note'>
            Data: Yahoo Finance / NSE<br>
            Model: XGBoost + Ensemble<br>
            Features: 125 engineered<br>
            <span style='color:#334155'>Not financial advice. Research only.</span>
        </div>
        """, unsafe_allow_html=True)

    return ticker, period, task, txn_cost, slippage, threshold


# ══════════════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════════════

def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick with Bollinger Bands overlay."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="Price",
        increasing_line_color="#22C55E",
        decreasing_line_color="#EF4444",
        increasing_fillcolor="#22C55E",
        decreasing_fillcolor="#EF4444",
    ), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"],
            name="BB Upper", line=dict(color="#3B82F6", width=1, dash="dot"),
            opacity=0.6,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"],
            name="BB Lower", line=dict(color="#3B82F6", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.06)",
            opacity=0.6,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Mid"],
            name="BB Mid", line=dict(color="#64748B", width=1),
            opacity=0.5,
        ), row=1, col=1)

    # EMA 20
    if "EMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_20"],
            name="EMA 20", line=dict(color="#F59E0B", width=1.5),
        ), row=1, col=1)

    # Volume
    colors = ["#22C55E" if c >= o else "#EF4444"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors,
        opacity=0.6, showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Price & Volume", font=dict(size=15, color="#F8FAFC")),
        height=520,
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0F1729",
        font=dict(color="#94A3B8"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1E293B", showline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1E293B", showline=False)
    return fig


def backtest_chart(results: pd.DataFrame, ticker: str) -> go.Figure:
    """Backtest performance chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=("Cumulative Returns", "Daily Signal"),
    )

    fig.add_trace(go.Scatter(
        x=results.index, y=results["strat_cum"],
        name="ML Strategy",
        line=dict(color="#3B82F6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results.index, y=results["bh_cum"],
        name="Buy & Hold",
        line=dict(color="#94A3B8", width=1.5, dash="dash"),
    ), row=1, col=1)

    # Drawdown
    roll_max = results["strat_cum"].cummax()
    dd = (results["strat_cum"] - roll_max) / roll_max
    fig.add_trace(go.Scatter(
        x=results.index, y=dd,
        name="Drawdown", fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="rgba(239,68,68,0.4)", width=0.5),
        showlegend=False,
    ), row=1, col=1)

    # Signal bars
    colors = ["#22C55E" if s > 0 else "#EF4444" for s in results["signal"]]
    fig.add_trace(go.Bar(
        x=results.index, y=results["signal"],
        marker_color=colors, name="Signal", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=480,
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0F1729",
        font=dict(color="#94A3B8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1E293B")
    fig.update_yaxes(showgrid=True, gridcolor="#1E293B")
    return fig


def feature_importance_chart(importance: pd.Series) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    fig = go.Figure(go.Bar(
        x=importance.values[::-1],
        y=importance.index[::-1],
        orientation="h",
        marker=dict(
            color=importance.values[::-1],
            colorscale=[[0, "#1E3A5F"], [0.5, "#2563EB"], [1, "#60A5FA"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        title=dict(text="Top Feature Importances", font=dict(color="#F8FAFC", size=14)),
        height=420,
        paper_bgcolor="#0A0E1A",
        plot_bgcolor="#0F1729",
        font=dict(color="#94A3B8"),
        margin=dict(l=0, r=20, t=40, b=0),
        xaxis=dict(showgrid=True, gridcolor="#1E293B"),
        yaxis=dict(showgrid=False),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main app
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = get_config()

    # Sidebar
    ticker, period, task, txn_cost, slippage, threshold = render_sidebar(cfg)
    safe_ticker = ticker.replace(".", "_")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='margin-bottom:24px;padding-bottom:20px;border-bottom:1px solid #162032'>
        <div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap'>
            <h1 style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
                       color:#F1F5F9;margin:0;letter-spacing:-0.04em;
                       background:linear-gradient(135deg,#F1F5F9 30%,#94A3B8 100%);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       background-clip:text'>
                {ticker}
            </h1>
            <span style='font-size:0.82rem;color:#334155;font-weight:500;
                         font-family:DM Mono,monospace;letter-spacing:0.04em;
                         background:var(--bg-card);border:1px solid #1E293B;
                         padding:3px 10px;border-radius:99px'>NSE INDIA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker} data..."):
        df_feat = build_features(ticker, period)

    if df_feat.empty:
        st.error("Not enough data. Try a different ticker or longer period.")
        return

    df_raw = fetch_price_data(ticker, period)

    # Latest price info
    last_price  = float(df_raw["Close"].iloc[-1])
    prev_price  = float(df_raw["Close"].iloc[-2])
    price_chg   = last_price - prev_price
    price_pct   = price_chg / prev_price * 100
    last_date   = df_raw.index[-1].strftime("%d %b %Y")

    # ── Top metrics bar ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Last Close", f"₹{last_price:,.2f}",
                  f"{price_chg:+.2f} ({price_pct:+.2f}%)")
    with c2:
        high_52 = float(df_raw["High"].rolling(252).max().iloc[-1])
        st.metric("52W High", f"₹{high_52:,.2f}")
    with c3:
        low_52 = float(df_raw["Low"].rolling(252).min().iloc[-1])
        st.metric("52W Low", f"₹{low_52:,.2f}")
    with c4:
        avg_vol = df_raw["Volume"].rolling(20).mean().iloc[-1]
        st.metric("Avg Volume (20D)", f"{avg_vol/1e6:.1f}M")
    with c5:
        rsi_val = df_feat["RSI_14"].iloc[-1] if "RSI_14" in df_feat.columns else 0
        rsi_label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
        st.metric("RSI (14)", f"{rsi_val:.1f}", rsi_label)

    st.divider()

    # ── Regime detection ──────────────────────────────────────────────────────
    try:
        from src.regime_detection import RegimeDetector
        rd          = RegimeDetector()
        regime_info = rd.detect(df_feat)
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            st.markdown(
                f"<div class='ind-card' style='border-left:3px solid {regime_info['color']};'>"
                f"<div class='stat-label'>MARKET REGIME</div>"
                f"<div style='font-size:1.25rem;font-weight:700;color:{regime_info['color']};font-family:Syne,sans-serif;margin:6px 0'>"
                f"{regime_info['regime']}</div>"
                f"<div style='font-size:0.72rem;color:#64748B;font-family:DM Mono,monospace'>"
                f"Confidence: <span style='color:{regime_info['color']}'>{regime_info['confidence']:.0%}</span></div></div>",
                unsafe_allow_html=True,
            )
        with r_col2:
            bull_score = regime_info['score'].get('BULL', 0)
            bear_score = regime_info['score'].get('BEAR', 0)
            side_score = regime_info['score'].get('SIDEWAYS', 0)
            st.markdown(
                f"<div class='ind-card'>"
                f"<div class='stat-label' style='margin-bottom:10px'>REGIME SCORES</div>"
                f"<div style='display:flex;flex-direction:column;gap:6px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-size:0.75rem;color:#64748B'>Bull</span>"
                f"<span style='font-size:0.82rem;font-weight:600;color:#22C55E;font-family:DM Mono,monospace'>{bull_score:.0%}</span></div>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-size:0.75rem;color:#64748B'>Bear</span>"
                f"<span style='font-size:0.82rem;font-weight:600;color:#EF4444;font-family:DM Mono,monospace'>{bear_score:.0%}</span></div>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-size:0.75rem;color:#64748B'>Sideways</span>"
                f"<span style='font-size:0.82rem;font-weight:600;color:#F59E0B;font-family:DM Mono,monospace'>{side_score:.0%}</span></div>"
                f"</div>"
                f"<div style='font-size:0.68rem;color:#334155;margin-top:8px;line-height:1.4'>"
                f"{regime_info['description'][:80]}...</div></div>",
                unsafe_allow_html=True,
            )
        with r_col3:
            sigs     = regime_info.get('signals', {})
            sig_html = "".join(
                f"<div class='info-row'>"
                f"<span class='info-key'>{k}</span>"
                f"<span class='info-val'>{v}</span></div>"
                for k, v in list(sigs.items())[:4]
            )
            st.markdown(
                f"<div class='ind-card'>"
                f"<div class='stat-label' style='margin-bottom:10px'>REGIME SIGNALS</div>"
                f"{sig_html}</div>",
                unsafe_allow_html=True,
            )
        st.divider()
    except Exception:
        pass

    # ── Prediction panel ───────────────────────────────────────────────────────
    # Try v2 ensemble first, fall back to v1 XGBoost
    model_path    = f"models/xgboost_{task}_{safe_ticker}.pkl"
    v2_model_path = f"models/ensemble_{task}_{safe_ticker}.pkl"
    model = load_model(v2_model_path) or load_model(model_path)

    # Show model version badge
    if model is not None:
        version     = getattr(model, "version", "v1")
        ver_color   = "#6366F1" if version == "v2" else "#475569"
        ver_label   = "v2 Ensemble" if version == "v2" else "v1 XGBoost"
        metrics_kv  = getattr(model, "metrics", {})
        badge_extra = ""
        if metrics_kv:
            if task == "classification":
                acc = metrics_kv.get("test_accuracy", metrics_kv.get("accuracy"))
                auc = metrics_kv.get("test_auc",      metrics_kv.get("auc"))
                if acc: badge_extra += f"  Acc: {acc:.1%}"
                if auc: badge_extra += f"  AUC: {auc:.3f}"
            else:
                r2  = metrics_kv.get("test_r2", metrics_kv.get("r2"))
                if r2: badge_extra += f"  R²: {r2:.3f}"
        nfeat = len(getattr(model, "feature_names_in_", []))
        feat_str = f"  {nfeat} features" if nfeat else ""
        st.markdown(
            f"<div style='margin-bottom:8px'>"
            f"<span style='background:{ver_color};color:#fff;font-size:0.72rem;"
            f"padding:3px 10px;border-radius:99px;font-weight:600'>{ver_label}</span>"
            f"<span style='color:#475569;font-size:0.72rem;margin-left:8px'>"
            f"{feat_str}{badge_extra}</span></div>",
            unsafe_allow_html=True,
        )

    pred_col, chart_col = st.columns([1, 3])

    with pred_col:
        st.markdown("#### Tomorrow's Prediction")

        if model is None:
            st.warning(
                f"No trained model found.\n\n"
                f"Run:\n```\npython scripts/run_model_training.py\n```"
            )
        else:
            feature_cols = FeatureEngineer.get_feature_cols(df_feat)
            X_latest     = df_feat[feature_cols].iloc[[-1]]

            try:
                pred = float(model.predict(X_latest)[0])

                if task == "regression":
                    direction = "UP ▲" if pred > threshold else "DOWN ▼"
                    css_class = "pred-up" if pred > threshold else "pred-down"
                    color     = "#22C55E" if pred > threshold else "#EF4444"
                    pred_display = f"{pred*100:+.3f}%"
                    label_text   = "Expected Return"
                else:
                    direction = "UP ▲" if pred >= 0.5 else "DOWN ▼"
                    css_class = "pred-up" if pred >= 0.5 else "pred-down"
                    color     = "#22C55E" if pred >= 0.5 else "#EF4444"
                    pred_display = f"{pred*100:.1f}%"
                    label_text   = "P(Up)"

                st.markdown(f"""
                <div class="{css_class}">
                    <div class="pred-label" style="color:{color}">
                        {direction}
                    </div>
                    <div style="font-family: DM Mono, monospace;
                                font-size: 1.3rem; color: {color};
                                margin-top: 6px; font-weight: 500">
                        {pred_display}
                    </div>
                    <div class="pred-sub">{label_text}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Auto-save prediction to store ───────────────────────────
                try:
                    store       = get_store()
                    regime_data = {}
                    try:
                        from src.regime_detection import RegimeDetector
                        regime_data = RegimeDetector().detect(df_feat)
                    except Exception:
                        pass
                    dir_clean = "UP" if "UP" in direction else "DOWN"
                    store.save({
                        "ticker":        ticker,
                        "direction":     dir_clean,
                        "confidence":    abs(pred) if task == "regression" else abs(pred - 0.5) * 2,
                        "raw_score":     pred,
                        "price_at_pred": float(df_feat["Close"].iloc[-1]),
                        "timeframe":     "daily",
                        "model_version": getattr(model, "version", "v1"),
                        "task":          task,
                        "regime":        regime_data.get("regime", None),
                        "notes":         f"Dashboard prediction",
                    })
                except Exception:
                    pass   # Never block the UI for storage errors

                # Key indicators
                st.markdown(
                    "<div style='margin-top:16px'>"
                    "<div class='stat-label' style='margin-bottom:8px'>KEY INDICATORS</div>"
                    "<div class='ind-card' style='padding:12px 16px'>",
                    unsafe_allow_html=True,
                )
                indicators = {
                    "RSI (14)":     f"{df_feat['RSI_14'].iloc[-1]:.1f}" if "RSI_14" in df_feat.columns else "N/A",
                    "MACD":         f"{df_feat['MACD'].iloc[-1]:.3f}" if "MACD" in df_feat.columns else "N/A",
                    "BB %B":        f"{df_feat['BB_PctB'].iloc[-1]:.2f}" if "BB_PctB" in df_feat.columns else "N/A",
                    "ATR (14)":     f"₹{df_feat['ATR'].iloc[-1]:.2f}" if "ATR" in df_feat.columns else "N/A",
                    "Vol Ratio":    f"{df_feat['Volume_ratio'].iloc[-1]:.2f}x" if "Volume_ratio" in df_feat.columns else "N/A",
                }
                rows_html = "".join(
                    f"<div class='info-row'>"
                    f"<span class='info-key'>{k}</span>"
                    f"<span class='info-val'>{v}</span></div>"
                    for k, v in indicators.items()
                )
                st.markdown(rows_html + "</div></div>", unsafe_allow_html=True)

                # Multi-day outlook
                st.markdown(
                    "<div style='margin-top:16px'>"
                    "<div class='stat-label' style='margin-bottom:8px'>MULTI-DAY OUTLOOK</div>"
                    "<div class='ind-card' style='padding:12px 16px'>",
                    unsafe_allow_html=True,
                )
                for horizon, target_col in [("3-Day", "Target_Dir_3d"), ("5-Day", "Target_Dir_5d")]:
                    md_model_path = f"models/xgboost_classification_{safe_ticker}_{horizon.replace('-','d').lower()}.pkl"
                    md_model      = load_model(md_model_path)
                    if md_model is not None:
                        try:
                            md_pred = float(md_model.predict(X_latest)[0])
                            md_dir  = "UP ▲" if md_pred > 0.5 else "DOWN ▼"
                            md_col  = "#22C55E" if md_pred > 0.5 else "#EF4444"
                        except Exception:
                            md_dir = "—"; md_col = "#64748B"
                    else:
                        # Fallback: use regression signal direction
                        md_dir = "UP ▲" if pred > 0 else "DOWN ▼"
                        md_col = "#22C55E" if pred > 0 else "#EF4444"
                    st.markdown(
                        f"<div class='info-row'>"
                        f"<span class='info-key'>{horizon}</span>"
                        f"<span style='color:{md_col};font-size:0.78rem;font-weight:600;"
                        f"font-family:DM Mono,monospace'>{md_dir}</span></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with chart_col:
        # Show last 180 days for the chart
        df_chart = df_feat.iloc[-180:] if len(df_feat) > 180 else df_feat
        st.plotly_chart(
            candlestick_chart(df_chart, ticker),
            use_container_width=True,
        )

    # ── Tabs: Backtest | Features | Data ──────────────────────────────────────
    st.divider()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Backtest", "Features", "Data", "Intraday", "History", "Watchlist", "Portfolio"])

    # ── Tab 1: Backtest ────────────────────────────────────────────────────────
    with tab1:
        if model is None:
            st.info("Train a model first to see backtest results.")
        else:
            with st.spinner("Running backtest..."):
                try:
                    feature_cols = FeatureEngineer.get_feature_cols(df_feat)
                    X            = df_feat[feature_cols]
                    prices       = df_feat["Close"]
                    signals      = pd.Series(
                        model.predict(X), index=X.index, name="signal"
                    )
                    bt      = Backtester(prices, signals, txn_cost, slippage, threshold)
                    results = bt.run()
                    ev      = ModelEvaluator()

                    # Metrics row
                    strat_sharpe = ev.sharpe_ratio(results["strat_return"])
                    bh_sharpe    = ev.sharpe_ratio(results["bh_return"])
                    strat_ret    = float(results["strat_cum"].iloc[-1] - 1) * 100
                    bh_ret       = float(results["bh_cum"].iloc[-1] - 1) * 100
                    strat_mdd    = float(ev.max_drawdown(results["strat_cum"])) * 100
                    n_trades     = int(results["trade"].sum())
                    alpha        = strat_ret - bh_ret

                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Total Return",    f"{strat_ret:+.1f}%",
                              f"vs B&H {bh_ret:+.1f}%")
                    m2.metric("Sharpe Ratio",    f"{strat_sharpe:.3f}",
                              f"B&H {bh_sharpe:.3f}")
                    m3.metric("Max Drawdown",    f"{strat_mdd:.1f}%")
                    m4.metric("Alpha",           f"{alpha:+.1f}%")
                    m5.metric("Total Trades",    f"{n_trades}")
                    m6.metric("Long %",          f"{results['position_lag'].mean()*100:.0f}%")

                    # Chart
                    st.plotly_chart(
                        backtest_chart(results, ticker),
                        use_container_width=True,
                    )

                    # Monthly returns
                    st.markdown("#### Monthly Returns (%)")
                    monthly = (
                        (1 + results["strat_return"])
                        .resample("ME").prod() - 1
                    ) * 100
                    table = monthly.groupby([
                        monthly.index.year, monthly.index.month
                    ]).first().unstack(level=1)
                    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                                   "Jul","Aug","Sep","Oct","Nov","Dec"]
                    table.columns = [month_names[m-1] for m in table.columns]
                    st.dataframe(
                        table.style
                            .background_gradient(cmap="RdYlGn", axis=None)
                            .format("{:.1f}%", na_rep="—"),
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Backtest error: {e}")

    # ── Tab 2: Features ────────────────────────────────────────────────────────
    with tab2:
        if model is None:
            st.info("Train a model to see feature importances.")
        else:
            try:
                estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
                if hasattr(estimator, "feature_importances_"):
                    feature_cols = FeatureEngineer.get_feature_cols(df_feat)
                    imp = pd.Series(
                        estimator.feature_importances_,
                        index=feature_cols,
                    ).sort_values(ascending=False).head(20)

                    col_a, col_b = st.columns([3, 2])
                    with col_a:
                        st.plotly_chart(
                            feature_importance_chart(imp),
                            use_container_width=True,
                        )
                    with col_b:
                        st.markdown("#### Feature Values (Latest)")
                        latest = df_feat[feature_cols].iloc[-1]
                        display = latest[imp.index].reset_index()
                        display.columns = ["Feature", "Value"]
                        display["Importance"] = imp.values
                        display["Value"] = display["Value"].round(4)
                        display["Importance"] = display["Importance"].round(4)
                        st.dataframe(display, use_container_width=True, height=380)
            except Exception as e:
                st.error(f"Feature importance error: {e}")

        # Feature correlation heatmap (top 15)
        st.markdown("#### Feature Correlations (Top 15 by Variance)")
        try:
            feature_cols = FeatureEngineer.get_feature_cols(df_feat)
            top_var = (
                df_feat[feature_cols]
                .var()
                .sort_values(ascending=False)
                .head(15)
                .index.tolist()
            )
            corr = df_feat[top_var].corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmid=0,
                showscale=True,
            ))
            fig_corr.update_layout(
                height=420,
                paper_bgcolor="#0A0E1A",
                plot_bgcolor="#0F1729",
                font=dict(color="#94A3B8", size=10),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.caption(f"Correlation chart unavailable: {e}")

    # ── Tab 3: Raw data ────────────────────────────────────────────────────────
    with tab3:
        st.markdown(f"#### Raw OHLCV — {ticker} (last 60 days)")
        st.dataframe(
            df_raw.tail(60).sort_index(ascending=False)
                  .style.format({
                      "Open": "₹{:.2f}", "High": "₹{:.2f}",
                      "Low": "₹{:.2f}", "Close": "₹{:.2f}",
                      "Volume": "{:,.0f}",
                  }),
            use_container_width=True,
            height=420,
        )

        st.markdown("#### Feature Matrix (last 10 rows)")
        feature_cols = FeatureEngineer.get_feature_cols(df_feat)
        st.dataframe(
            df_feat[feature_cols].tail(10)
                  .style.format("{:.4f}"),
            use_container_width=True,
        )

    with tab4:
        st.markdown("### Intraday Prediction")
        st.caption("Multi-timeframe analysis: 5min · 15min · 1hr | Price targets + Stop Loss")

        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_intraday = st.button("Scan Now", use_container_width=True)
        with col_info:
            from src.intraday import NSE_OPEN, NSE_CLOSE
            from datetime import datetime as _dt
            now = _dt.now().time()
            is_open = NSE_OPEN <= now <= NSE_CLOSE
            status_color = "#22C55E" if is_open else "#EF4444"
            status_dot   = "●"
            market_status = f'{status_dot} Market Open' if is_open else f'{status_dot} Market Closed (using last session)'
            st.markdown(
                f"<div style='padding:8px 16px;background:#111827;border-radius:99px;"
                f"border:1px solid #1E293B;color:#94A3B8;font-size:0.78rem;"
                f"display:inline-flex;align-items:center;gap:8px;font-family:DM Mono,monospace'>"
                f"<span style='color:{status_color};font-size:0.6rem'>{status_dot}</span>"
                f"<span>{market_status.lstrip('● ')}</span>"
                f"<span style='color:#334155'>|</span>"
                f"<span style='color:#475569'>NSE 9:15 – 15:30 IST</span></div>",
                unsafe_allow_html=True,
            )

        if run_intraday:
            with st.spinner(f"Fetching intraday data for {ticker}..."):
                try:
                    from src.intraday import IntradayPredictor
                    ip     = IntradayPredictor(ticker=ticker, config=cfg)
                    result = ip.predict_all_timeframes()

                    # Confluence badge
                    conf       = result["confluence"]
                    conf_score = result["conf_score"]
                    conf_color = "#22C55E" if "BUY" in conf else "#EF4444" if "SELL" in conf else "#F59E0B"
                    st.markdown(
                        f"<div style='background:{conf_color}22;border:2px solid {conf_color};"
                        f"border-radius:12px;padding:16px;text-align:center;margin-bottom:16px'>"
                        f"<div style='font-size:0.8rem;color:#94A3B8'>MULTI-TIMEFRAME CONFLUENCE</div>"
                        f"<div style='font-size:1.8rem;font-weight:800;color:{conf_color}'>{conf}</div>"
                        f"<div style='font-size:0.85rem;color:#64748B'>"
                        f"Agreement: {conf_score:.0%} across timeframes</div></div>",
                        unsafe_allow_html=True,
                    )

                    # Timeframe cards
                    cols = st.columns(3)
                    for i, (tf, r) in enumerate(result["timeframes"].items()):
                        with cols[i]:
                            if r.get("status") != "ok":
                                st.error(f"{tf}\n{r.get('error','No data')[:50]}")
                                continue

                            d         = r["direction"]
                            arrow     = "▲" if d == "UP" else "▼"
                            color     = "#22C55E" if d == "UP" else "#EF4444"
                            strength  = r["strength"]
                            conf_pct  = f"{r['confidence']:.0%}"
                            t         = r["targets"]
                            ind       = r.get("indicators", {})
                            alert_badge = "STRONG SIGNAL" if r["is_alert"] else ""

                            st.markdown(
                                f"<div class='signal-card' style='background:#111827;border-color:{color}33;border-left-color:{color}'>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
                                f"<span style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748B'>{tf.upper()}</span>"
                                f"{'<span style=\"background:#F59E0B18;color:#F59E0B;font-size:0.62rem;font-weight:600;padding:2px 8px;border-radius:99px;border:1px solid #F59E0B30\">' + alert_badge + '</span>' if alert_badge else ''}"
                                f"</div>"
                                f"<div style='font-size:1.6rem;font-weight:800;color:{color};margin:4px 0;font-family:Syne,sans-serif'>"
                                f"{arrow} {d}</div>"
                                f"<div style='font-size:0.75rem;color:#475569;margin-bottom:12px'>"
                                f"Confidence: <span style='color:{color};font-weight:600'>{conf_pct}</span>"
                                f" &nbsp;·&nbsp; {strength}</div>"
                                f"<div style='border-top:1px solid #1E293B;padding-top:10px;display:flex;flex-direction:column;gap:5px'>"
                                f"<div class='info-row'><span class='info-key'>Entry</span>"
                                f"<span class='info-val'>₹{t['entry']:,.2f}</span></div>"
                                f"<div class='info-row'><span class='info-key'>Target</span>"
                                f"<span style='color:#22C55E;font-family:DM Mono,monospace;font-size:0.78rem;font-weight:600'>₹{t['target']:,.2f} (+{t['reward_pct']:.1f}%)</span></div>"
                                f"<div class='info-row'><span class='info-key'>Stop Loss</span>"
                                f"<span style='color:#EF4444;font-family:DM Mono,monospace;font-size:0.78rem;font-weight:600'>₹{t['stop_loss']:,.2f} (-{t['risk_pct']:.1f}%)</span></div>"
                                f"<div class='info-row'><span class='info-key'>R:R Ratio</span>"
                                f"<span class='info-val'>1:{t['rr_ratio']:.1f}</span></div>"
                                f"</div>"
                                f"<div style='margin-top:10px;padding-top:8px;border-top:1px solid #1E293B;"
                                f"font-size:0.68rem;color:#334155;font-family:DM Mono,monospace'>"
                                f"RSI: {ind.get('RSI','—')} &nbsp;·&nbsp; "
                                f"Vol: {ind.get('Volume_ratio','—')}x &nbsp;·&nbsp; "
                                f"VWAP dev: {ind.get('VWAP_dev','—')}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    # Alert summary
                    st.markdown("---")
                    strong_tfs = [tf for tf, r in result["timeframes"].items()
                                  if r.get("is_alert") and r.get("status") == "ok"]
                    if strong_tfs:
                        st.success(
                            f"Strong signal detected on: {', '.join(strong_tfs)}  |  "
                            f"Run `python scripts/run_intraday.py --alert` to send WhatsApp/Email alert"
                        )
                    else:
                        st.info("No strong signals right now. Signal strength < 65% on all timeframes.")

                    st.caption(
                        f"Not financial advice. For research only.  |  "
                        f"Scanned at {result['timestamp']}"
                    )

                except Exception as e:
                    st.error(f"Intraday scan failed: {e}")
        else:
            st.markdown(
                "<div class='empty-state'>"
                "<div class='empty-state-icon'>&#9651;</div>"
                "<div class='empty-state-title'>Ready to Scan</div>"
                "<div class='empty-state-sub'>Click <b>Scan Now</b> to run intraday analysis<br>"
                "Fetches live 5min · 15min · 1hr data from NSE</div>"
                "</div>",
                unsafe_allow_html=True,
            )


    # ── Tab 5: Prediction History ──────────────────────────────────────────────
    with tab5:
        import plotly.express as px

        store = get_store()

        # ── Header + controls ─────────────────────────────────────────────────
        h_col1, h_col2, h_col3, h_col4 = st.columns([2, 1, 1, 1])
        with h_col1:
            st.markdown("### Prediction History")
        with h_col2:
            hist_days = st.selectbox("Period", [7, 14, 30, 60, 90, 180, 365], index=2,
                                     key="hist_days")
        with h_col3:
            hist_tf = st.selectbox("Timeframe", ["all", "daily", "5min", "15min", "1hr"],
                                   key="hist_tf")
        with h_col4:
            hist_ticker = st.selectbox(
                "Ticker",
                ["All"] + sorted(cfg["data"]["tickers"]),
                key="hist_ticker",
            )

        tf_filter     = None if hist_tf == "all" else hist_tf
        ticker_filter = None if hist_ticker == "All" else hist_ticker

        # Update outcomes button
        oc1, oc2, oc3 = st.columns([1, 1, 4])
        with oc1:
            if st.button("Update Outcomes", use_container_width=True):
                with st.spinner("Fetching actual prices..."):
                    n = store.update_outcomes(days_back=hist_days)
                st.success(f"Updated {n} prediction outcomes")
                st.rerun()
        with oc2:
            if st.button("Export Excel", use_container_width=True):
                with st.spinner("Generating Excel..."):
                    try:
                        path = store.export_excel()
                        with open(path, "rb") as f:
                            st.download_button(
                                "Download",
                                data=f.read(),
                                file_name="nse_predictions.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    except Exception as e:
                        st.error(f"Export failed: {e}")

        st.divider()

        # ── Overall accuracy stats ─────────────────────────────────────────────
        stats = store.get_accuracy_stats(
            ticker=ticker_filter, timeframe=tf_filter, days_back=hist_days
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        def metric_card(col, label, value, color="#F1F5F9", sub=None):
            col.markdown(
                f"<div class='stat-card'>"
                f"<div class='stat-label'>{label}</div>"
                f"<div class='stat-value' style='color:{color}'>{value}</div>"
                f"{'<div style=\"font-size:0.68rem;color:#475569;margin-top:4px\">' + str(sub) + '</div>' if sub else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

        acc_color  = "#22C55E" if stats["accuracy"] >= 0.55 else "#EF4444" if stats["accuracy"] > 0 else "#64748B"
        metric_card(m1, "TOTAL PREDICTIONS", stats["total"])
        metric_card(m2, "EVALUATED",         stats["evaluated"])
        metric_card(m3, "CORRECT",           stats["correct"], "#22C55E")
        metric_card(m4, "ACCURACY",          f"{stats['accuracy']:.1%}", acc_color)
        metric_card(m5, "WIN STREAK",        f"{stats['win_streak']}+" if stats["win_streak"] > 2 else str(stats["win_streak"]))
        metric_card(m6, "PENDING",           stats["pending"], "#F59E0B")

        st.markdown(
            f"<div style='display:flex;gap:16px;margin:12px 0'>"
            f"<span style='font-size:0.8rem;color:#64748B'>▲ UP accuracy: "
            f"<span style='color:#22C55E'>{stats['up_accuracy']:.1%}</span></span>"
            f"<span style='font-size:0.8rem;color:#64748B'>▼ DOWN accuracy: "
            f"<span style='color:#EF4444'>{stats['down_accuracy']:.1%}</span></span>"
            f"<span style='font-size:0.8rem;color:#64748B'>Avg confidence: "
            f"<span style='color:#94A3B8'>{stats['avg_confidence']:.1%}</span></span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: Daily accuracy trend
        with chart_col1:
            st.markdown("#### Win Rate Over Time")
            df_daily = store.get_daily_accuracy(days_back=hist_days)
            if len(df_daily) > 0:
                df_daily["date"] = pd.to_datetime(df_daily["date"])
                # 7-day rolling accuracy
                df_daily["rolling_acc"] = df_daily["accuracy"].rolling(7, min_periods=1).mean()

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_daily["date"], y=df_daily["accuracy"],
                    name="Daily Acc", marker_color="#1E293B",
                    opacity=0.6,
                ))
                fig.add_trace(go.Scatter(
                    x=df_daily["date"], y=df_daily["rolling_acc"],
                    name="7d Rolling", line=dict(color="#6366F1", width=2),
                    mode="lines",
                ))
                fig.add_hline(y=0.5, line_dash="dash", line_color="#EF4444",
                              annotation_text="50% baseline")
                fig.add_hline(y=0.55, line_dash="dot", line_color="#22C55E",
                              annotation_text="55% target")
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="#0A0E1A",
                    plot_bgcolor="#0A0E1A", height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", y=-0.2),
                    yaxis=dict(tickformat=".0%", range=[0, 1]),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No evaluated predictions yet. Click **Update Outcomes** after market close.")

        # Chart 2: Accuracy by ticker heatmap
        with chart_col2:
            st.markdown("#### Accuracy by Ticker")
            df_by_tick = store.get_accuracy_by_ticker(days_back=hist_days)
            if len(df_by_tick) > 0:
                df_by_tick = df_by_tick[df_by_tick["predictions"] >= 3].head(20)
                df_by_tick["ticker_short"] = df_by_tick["ticker"].str.replace(".NS", "")
                df_by_tick["color"] = df_by_tick["accuracy"].apply(
                    lambda x: "#22C55E" if x >= 0.55 else "#EF4444" if x < 0.50 else "#F59E0B"
                )
                fig2 = go.Figure(go.Bar(
                    x=df_by_tick["accuracy"],
                    y=df_by_tick["ticker_short"],
                    orientation="h",
                    marker_color=df_by_tick["color"],
                    text=df_by_tick["accuracy"].apply(lambda x: f"{x:.0%}"),
                    textposition="outside",
                    customdata=df_by_tick["predictions"],
                    hovertemplate="%{y}: %{x:.1%} (%{customdata} predictions)<extra></extra>",
                ))
                fig2.add_vline(x=0.5, line_dash="dash", line_color="#EF4444")
                fig2.update_layout(
                    template="plotly_dark", paper_bgcolor="#0A0E1A",
                    plot_bgcolor="#0A0E1A", height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(tickformat=".0%", range=[0, 1]),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Need at least 3 evaluated predictions per ticker to show chart.")

        st.divider()

        # ── Prediction table ──────────────────────────────────────────────────
        st.markdown("#### All Predictions")
        df_hist = store.get_history(
            ticker=ticker_filter, timeframe=tf_filter, days_back=hist_days, limit=500
        )

        if len(df_hist) == 0:
            st.info(
                "No predictions stored yet.\n\n"
                "Predictions are automatically saved every time you view a ticker "
                "in this dashboard, or run `python scripts/run_alerts.py`."
            )
        else:
            # Format for display
            display = df_hist[[
                "date", "ticker", "timeframe", "direction", "confidence",
                "price_at_pred", "target_price", "stop_loss", "regime",
                "actual_price", "actual_direction", "correct", "model_version",
            ]].copy()

            display["date"]       = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
            display["confidence"] = display["confidence"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
            display["price_at_pred"] = display["price_at_pred"].apply(
                lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"
            )
            display["actual_price"] = display["actual_price"].apply(
                lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"
            )
            display["target_price"] = display["target_price"].apply(
                lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"
            )
            display["stop_loss"] = display["stop_loss"].apply(
                lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—"
            )
            display["correct"] = display["correct"].apply(
                lambda x: "PASS" if x == 1 else "FAIL" if x == 0 else "PEND"
            )
            display["ticker"] = display["ticker"].str.replace(".NS", "")
            display.columns = [
                "Date", "Ticker", "TF", "Direction", "Confidence",
                "Entry", "Target", "Stop", "Regime",
                "Actual", "Actual Dir", "Result", "Model",
            ]

            # Color rows
            def color_result(val):
                if val == "PASS": return "color: #22C55E; font-weight: 600"
                if val == "FAIL": return "color: #EF4444; font-weight: 600"
                return "color: #F59E0B"

            def color_dir(val):
                if "UP" in str(val):   return "color: #22C55E"
                if "DOWN" in str(val): return "color: #EF4444"
                return ""

            styled = (
                display.style
                .applymap(color_result, subset=["Result"])
                .applymap(color_dir,    subset=["Direction", "Actual Dir"])
                .set_properties(**{
                    "background-color": "#111827",
                    "color": "#E2E8F0",
                    "font-size": "0.78rem",
                })
            )
            st.dataframe(styled, use_container_width=True, height=420)
            st.caption(f"Showing {len(df_hist)} predictions. PASS = Correct  ·  FAIL = Wrong  ·  PEND = Awaiting outcome")

    # ── Tab 6: Watchlist ───────────────────────────────────────────────────────
    with tab6:
        wm = get_watchlist()
        st.markdown("### Watchlist")

        # ── Add / Remove controls ─────────────────────────────────────────────
        wc1, wc2, wc3 = st.columns([2, 1, 1])
        with wc1:
            new_ticker = st.text_input("Add ticker (e.g. WIPRO or WIPRO.NS)",
                                       placeholder="WIPRO.NS", key="wl_add_input")
        with wc2:
            st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
            if st.button("Add", use_container_width=True, key="wl_add_btn"):
                if new_ticker.strip():
                    ok = wm.add(new_ticker.strip())
                    if ok:
                        st.success(f"Added {new_ticker.strip().upper()}")
                        st.rerun()
                    else:
                        st.warning("Already in watchlist or list is full (max 20)")
            st.markdown("</div>", unsafe_allow_html=True)
        with wc3:
            tickers_list = wm.get_tickers()
            st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
            if tickers_list:
                rm_ticker = st.selectbox("Remove", ["—"] + tickers_list,
                                         key="wl_rm_select", label_visibility="collapsed")
                if rm_ticker != "—" and st.button("Remove", key="wl_rm_btn",
                                                    use_container_width=True):
                    wm.remove(rm_ticker)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption(f"{wm.count()}/20 stocks in watchlist")
        st.divider()

        if wm.count() == 0:
            st.markdown(
                "<div class='empty-state'>"
                "<div class='empty-state-icon'>&#9734;</div>"
                "<div class='empty-state-title'>Watchlist is empty</div>"
                "<div class='empty-state-sub'>Add up to 20 NSE stocks using the form above.</div>"
                "</div>", unsafe_allow_html=True,
            )
        else:
            with st.spinner("Fetching live data..."):
                wl_data = wm.fetch_all(config=cfg)

            # ── Refresh button ────────────────────────────────────────────────
            rc1, rc2 = st.columns([1, 5])
            with rc1:
                if st.button("Refresh", key="wl_refresh"):
                    st.rerun()

            # ── Summary bar ───────────────────────────────────────────────────
            valid   = [s for s in wl_data if s.get("price")]
            gainers = sum(1 for s in valid if s.get("change_pct", 0) > 0)
            losers  = len(valid) - gainers
            bulls   = sum(1 for s in valid if s.get("direction") == "UP")
            bears   = sum(1 for s in valid if s.get("direction") == "DOWN")

            sm1, sm2, sm3, sm4 = st.columns(4)
            for col, label, val, color in [
                (sm1, "GAINERS",       f"+ {gainers}", "#22C55E"),
                (sm2, "LOSERS",        f"- {losers}",  "#EF4444"),
                (sm3, "BULLISH CALLS", f"{bulls}",  "#6366F1"),
                (sm4, "BEARISH CALLS", f"{bears}",  "#F59E0B"),
            ]:
                col.markdown(
                    f"<div class='stat-card'>"
                    f"<div class='stat-label'>{label}</div>"
                    f"<div class='stat-value' style='color:{color}'>{val}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Stock cards (3 per row) ────────────────────────────────────────
            for row_start in range(0, len(wl_data), 3):
                row_stocks = wl_data[row_start:row_start+3]
                cols = st.columns(3)
                for col, s in zip(cols, row_stocks):
                    with col:
                        if s.get("error") and not s.get("price"):
                            st.error(f"{s['ticker']}\n{s['error'][:60]}")
                            continue

                        chg      = s.get("change_pct", 0)
                        chg_abs  = s.get("change_abs", 0)
                        price    = s.get("price", 0)
                        is_up    = chg >= 0
                        chg_col  = "#22C55E" if is_up else "#EF4444"
                        chg_sym  = "▲" if is_up else "▼"
                        regime   = s.get("regime", "UNKNOWN")
                        reg_col  = s.get("regime_color", "#64748B")
                        reg_em   = s.get("regime_emoji", "")
                        direction = s.get("direction")
                        conf      = s.get("confidence")
                        dir_col   = "#22C55E" if direction == "UP" else "#EF4444" if direction == "DOWN" else "#64748B"
                        dir_sym   = "▲" if direction == "UP" else "▼" if direction == "DOWN" else "—"
                        vol_r     = s.get("volume_ratio", 1)

                        # Sparkline using plotly
                        spark = s.get("sparkline", [])
                        if spark:
                            spark_col = "#22C55E" if spark[-1] >= spark[0] else "#EF4444"
                            fig_s = go.Figure(go.Scatter(
                                y=spark, mode="lines",
                                line=dict(color=spark_col, width=1.5),
                                fill="tozeroy",
                                fillcolor=spark_col.replace(")", ",0.1)").replace("rgb", "rgba"),
                            ))
                            fig_s.update_layout(
                                height=60, margin=dict(l=0,r=0,t=0,b=0),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                showlegend=False,
                            )

                        st.markdown(
                            f"<div style='background:var(--bg-card);border:1px solid #1E293B;"
                            f"border-radius:12px;padding:16px;margin-bottom:4px'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
                            f"<div>"
                            f"<div style='font-weight:700;color:#F8FAFC;font-size:0.9rem'>"
                            f"{s['ticker'].replace('.NS','')}</div>"
                            f"<div style='color:#475569;font-size:0.7rem'>{s.get('name','')[:20]}</div>"
                            f"</div>"
                            f"<div style='text-align:right'>"
                            f"<div style='font-size:1.1rem;font-weight:700;color:#F8FAFC'>₹{price:,.2f}</div>"
                            f"<div style='font-size:0.78rem;color:{chg_col}'>"
                            f"{chg_sym} {abs(chg):.2f}% (₹{abs(chg_abs):.2f})</div>"
                            f"</div></div>"
                            f"<div style='display:flex;gap:8px;margin-top:8px;flex-wrap:wrap'>"
                            f"<span style='background:{reg_col}22;color:{reg_col};"
                            f"font-size:0.65rem;padding:2px 7px;border-radius:99px'>"
                            f"{regime}</span>"
                            f"{'<span style="background:' + dir_col + '22;color:' + dir_col + ';font-size:0.65rem;padding:2px 7px;border-radius:99px">' + dir_sym + ' ' + (direction or '—') + (' ' + f'{conf:.0%}' if conf else '') + '</span>' if direction else ''}"
                            f"{'<span style="background:#F59E0B22;color:#F59E0B;font-size:0.65rem;padding:2px 7px;border-radius:99px">Vol ' + str(vol_r) + 'x</span>' if vol_r > 1.5 else ''}"
                            f"</div>"
                            f"<div style='display:flex;justify-content:space-between;"
                            f"margin-top:8px;font-size:0.68rem;color:#475569'>"
                            f"<span>52W H: ₹{s.get('high_52w',0):,.0f}</span>"
                            f"<span>52W L: ₹{s.get('low_52w',0):,.0f}</span>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )
                        if spark:
                            st.plotly_chart(fig_s, use_container_width=True,
                                            config={"displayModeBar": False})

    # ── Tab 7: Portfolio ───────────────────────────────────────────────────────
    with tab7:
        import plotly.express as px

        st.markdown("### Portfolio & Holdings")
        st.markdown("<div style='height:3px;background:linear-gradient(90deg,#1D4ED8,#6366F1,#8B5CF6);border-radius:99px;margin-bottom:20px;opacity:0.7'></div>", unsafe_allow_html=True)

        # ── Broker connection panel ───────────────────────────────────────────
        with st.expander("Broker Connection — Angel One", expanded=False):
            angel_connected = cfg.get("broker", {}).get("angel_one", {}).get("enabled", False)
            if angel_connected:
                st.success("Angel One configured in config.yaml")
                st.caption("Holdings will be fetched from your Angel One account.")
            else:
                st.info(
                    "**Angel One not configured.** Using manual holdings.\n\n"
                    "To connect Angel One, add to `config.yaml`:\n"
                    "```yaml\n"
                    "broker:\n"
                    "  angel_one:\n"
                    "    enabled:   true\n"
                    "    api_key:   your_api_key\n"
                    "    client_id: your_client_id\n"
                    "    password:  your_mpin\n"
                    "    totp_key:  your_totp_secret\n"
                    "```\n"
                    "Then run: `pip install smartapi-python pyotp`"
                )

        # ── Manual entry form ─────────────────────────────────────────────────
        manual_mgr = ManualHoldingsManager()
        with st.expander("Add / Remove Holdings Manually", expanded=False):
            hc1, hc2, hc3, hc4 = st.columns([2, 1, 1, 1])
            with hc1:
                h_ticker = st.text_input("Ticker", placeholder="RELIANCE.NS", key="h_ticker")
            with hc2:
                h_qty    = st.number_input("Qty", min_value=0.0, step=1.0, key="h_qty")
            with hc3:
                h_price  = st.number_input("Avg Price ₹", min_value=0.0, step=0.5, key="h_price")
            with hc4:
                st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
                if st.button("Add Holding", use_container_width=True, key="h_add"):
                    if h_ticker and h_qty > 0 and h_price > 0:
                        manual_mgr.add(h_ticker.strip(), h_qty, h_price)
                        st.success(f"Added {h_ticker.upper()}")
                        st.rerun()
                    else:
                        st.warning("Fill all fields")
                st.markdown("</div>", unsafe_allow_html=True)

            # Remove holding
            raw_holdings = manual_mgr.get_all_raw()
            if raw_holdings:
                rm_opts = ["—"] + [h["ticker"] for h in raw_holdings]
                rm_sel  = st.selectbox("Remove holding", rm_opts, key="h_rm_sel")
                if rm_sel != "—" and st.button("Remove", key="h_rm_btn"):
                    manual_mgr.remove(rm_sel)
                    st.rerun()

        # ── Fetch holdings ────────────────────────────────────────────────────
        with st.spinner("Loading holdings..."):
            broker, broker_type = get_broker(cfg)
            holdings = broker.get_holdings()

            # Merge with manual if using Angel One (show both)
            if broker_type == "angel":
                manual_h = manual_mgr.get_holdings()
                manual_tickers = {h["ticker"] for h in manual_h}
                angel_tickers  = {h["ticker"] for h in holdings}
                extra = [h for h in manual_h if h["ticker"] not in angel_tickers]
                holdings = holdings + extra

        if not holdings:
            st.markdown(
                "<div class='empty-state'>"
                "<div class='empty-state-icon'>&#9632;</div>"
                "<div class='empty-state-title'>No holdings found</div>"
                "<div class='empty-state-sub'>Add positions manually using the form above.</div>"
                "</div>", unsafe_allow_html=True,
            )
        else:
            df_h = pd.DataFrame(holdings)

            # ── Portfolio summary cards ───────────────────────────────────────
            total_invested = df_h["cost_value"].sum()
            total_current  = df_h["current_value"].sum()
            total_pnl      = total_current - total_invested
            total_pnl_pct  = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            pnl_color      = "#22C55E" if total_pnl >= 0 else "#EF4444"

            pc1, pc2, pc3, pc4 = st.columns(4)
            for col, label, val, color in [
                (pc1, "INVESTED",       f"₹{total_invested:,.0f}",   "#94A3B8"),
                (pc2, "CURRENT VALUE",  f"₹{total_current:,.0f}",    "#F1F5F9"),
                (pc3, "TOTAL P&L",      f"{'▲' if total_pnl>=0 else '▼'} ₹{abs(total_pnl):,.0f}", pnl_color),
                (pc4, "RETURN",         f"{total_pnl_pct:+.2f}%",    pnl_color),
            ]:
                col.markdown(
                    f"<div class='stat-card'>"
                    f"<div class='stat-label'>{label}</div>"
                    f"<div class='stat-value' style='color:{color};font-size:1.4rem'>{val}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts row ────────────────────────────────────────────────────
            ch1, ch2 = st.columns(2)

            with ch1:
                st.markdown("#### P&L by Stock")
                df_sorted = df_h.sort_values("pnl")
                colors    = ["#22C55E" if x >= 0 else "#EF4444" for x in df_sorted["pnl"]]
                fig_pnl   = go.Figure(go.Bar(
                    x=df_sorted["pnl"],
                    y=df_sorted["ticker"].str.replace(".NS",""),
                    orientation="h",
                    marker_color=colors,
                    text=df_sorted["pnl"].apply(lambda x: f"₹{x:+,.0f}"),
                    textposition="outside",
                ))
                fig_pnl.add_vline(x=0, line_color="#475569")
                fig_pnl.update_layout(
                    template="plotly_dark", paper_bgcolor="#0A0E1A",
                    plot_bgcolor="#0A0E1A", height=300,
                    margin=dict(l=0,r=0,t=10,b=0),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

            with ch2:
                st.markdown("#### Portfolio Allocation")
                fig_pie = go.Figure(go.Pie(
                    labels=df_h["ticker"].str.replace(".NS",""),
                    values=df_h["current_value"],
                    hole=0.45,
                    marker=dict(colors=px.colors.qualitative.Set3),
                    textinfo="percent+label",
                    textfont_size=10,
                ))
                fig_pie.update_layout(
                    template="plotly_dark", paper_bgcolor="#0A0E1A",
                    height=300, margin=dict(l=0,r=0,t=10,b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # ── Portfolio vs NIFTY chart ──────────────────────────────────────
            st.markdown("#### Portfolio vs NIFTY 50")
            try:
                import yfinance as yf
                nifty_df = yf.download("^NSEI", period="1y",
                                       auto_adjust=True, progress=False)
                if isinstance(nifty_df.columns, pd.MultiIndex):
                    nifty_df.columns = [c[0] for c in nifty_df.columns]
                if nifty_df.index.tz is not None:
                    nifty_df.index = nifty_df.index.tz_localize(None)

                nifty_ret = (nifty_df["Close"] / nifty_df["Close"].iloc[0] - 1) * 100

                # Weighted portfolio return
                tickers_weights = dict(
                    zip(df_h["ticker"], df_h["current_value"] / df_h["current_value"].sum())
                )
                port_ret = pd.Series(0.0, index=nifty_df.index)
                for t, w in tickers_weights.items():
                    try:
                        tdf = yf.download(t, period="1y",
                                          auto_adjust=True, progress=False)
                        if isinstance(tdf.columns, pd.MultiIndex):
                            tdf.columns = [c[0] for c in tdf.columns]
                        if tdf.index.tz is not None:
                            tdf.index = tdf.index.tz_localize(None)
                        tret = (tdf["Close"].reindex(nifty_df.index).ffill()
                                / tdf["Close"].iloc[0] - 1) * 100
                        port_ret += tret.fillna(0) * w
                    except Exception:
                        pass

                fig_vs = go.Figure()
                fig_vs.add_trace(go.Scatter(
                    x=nifty_df.index, y=port_ret,
                    name="My Portfolio", line=dict(color="#6366F1", width=2),
                ))
                fig_vs.add_trace(go.Scatter(
                    x=nifty_df.index, y=nifty_ret,
                    name="NIFTY 50", line=dict(color="#475569", width=1.5, dash="dot"),
                ))
                fig_vs.add_hline(y=0, line_color="#334155")
                fig_vs.update_layout(
                    template="plotly_dark", paper_bgcolor="#0A0E1A",
                    plot_bgcolor="#0A0E1A", height=280,
                    margin=dict(l=0,r=0,t=10,b=0),
                    yaxis_ticksuffix="%",
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig_vs, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load portfolio vs NIFTY chart: {e}")

            # ── Prediction P&L table ──────────────────────────────────────────
            st.markdown("#### Prediction P&L — Did ML Calls Make Money?")
            st.caption("Tracks whether ML predictions for YOUR holdings were correct and profitable")
            try:
                store   = get_store()
                held_tickers = df_h["ticker"].tolist()
                pred_rows = []
                for hticker in held_tickers:
                    hist = store.get_history(ticker=hticker, days_back=30)
                    evl  = hist[hist["correct"].notna()]
                    if len(evl) == 0:
                        continue
                    evl = evl.copy()
                    evl["correct"] = evl["correct"].astype(int)
                    h_row = df_h[df_h["ticker"] == hticker].iloc[0]
                    pred_rows.append({
                        "Ticker":     hticker.replace(".NS",""),
                        "Holding":    f"₹{h_row['current_value']:,.0f}",
                        "P&L":        f"{'▲' if h_row['pnl']>=0 else '▼'} ₹{abs(h_row['pnl']):,.0f} ({h_row['pnl_pct']:+.1f}%)",
                        "ML Preds":   len(evl),
                        "Correct":    int(evl["correct"].sum()),
                        "ML Acc":     f"{evl['correct'].mean():.0%}",
                        "Last Signal":evl.iloc[0]["direction"] if len(evl)>0 else "—",
                    })
                if pred_rows:
                    df_pred = pd.DataFrame(pred_rows)
                    def color_acc(val):
                        try:
                            v = float(val.strip("%"))/100
                            return "color:#22C55E" if v>=0.55 else "color:#EF4444" if v<0.50 else "color:#F59E0B"
                        except Exception:
                            return ""
                    styled_pred = (
                        df_pred.style
                        .applymap(color_acc, subset=["ML Acc"])
                        .set_properties(**{"background-color":"#111827","color":"#E2E8F0","font-size":"0.8rem"})
                    )
                    st.dataframe(styled_pred, use_container_width=True)
                else:
                    st.info("No evaluated predictions for your holdings yet. Run predictions and check back after market close.")
            except Exception as e:
                st.warning(f"Could not load prediction P&L: {e}")

            # ── Holdings table ────────────────────────────────────────────────
            st.markdown("#### Holdings Detail")
            display_h = df_h[[
                "ticker","qty","avg_price","ltp","current_value","cost_value","pnl","pnl_pct"
            ]].copy()
            display_h.columns = ["Ticker","Qty","Avg Price","LTP","Current","Invested","P&L","Return%"]
            display_h["Ticker"]   = display_h["Ticker"].str.replace(".NS","")
            display_h["Avg Price"]= display_h["Avg Price"].apply(lambda x: f"₹{x:,.2f}")
            display_h["LTP"]      = display_h["LTP"].apply(lambda x: f"₹{x:,.2f}")
            display_h["Current"]  = display_h["Current"].apply(lambda x: f"₹{x:,.0f}")
            display_h["Invested"] = display_h["Invested"].apply(lambda x: f"₹{x:,.0f}")
            display_h["P&L"]      = display_h["P&L"].apply(lambda x: f"₹{x:+,.0f}")
            display_h["Return%"]  = display_h["Return%"].apply(lambda x: f"{x:+.2f}%")

            def color_pnl(val):
                try:
                    return "color:#22C55E" if float(val.replace("₹","").replace(",","").replace("+","")) >= 0 else "color:#EF4444"
                except Exception:
                    return ""

            styled_h = (
                display_h.style
                .applymap(color_pnl, subset=["P&L","Return%"])
                .set_properties(**{"background-color":"#111827","color":"#E2E8F0","font-size":"0.82rem"})
            )
            st.dataframe(styled_h, use_container_width=True, height=300)

            # Excel export
            if st.button("Export Holdings to Excel", key="port_export"):
                try:
                    path = str(project_path('results', 'portfolio.xlsx'))
                    ensure_dirs(Path(path).parent)
                    with pd.ExcelWriter(path, engine="openpyxl") as writer:
                        df_h.to_excel(writer, sheet_name="Holdings", index=False)
                    with open(path, "rb") as fx:
                        st.download_button(
                            "Download Excel",
                            data=fx.read(),
                            file_name="portfolio.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")

if __name__ == "__main__":
    main()