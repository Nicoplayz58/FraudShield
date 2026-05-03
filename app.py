from __future__ import annotations

import os
import queue
import threading
import time
from datetime import date, timedelta
from datetime import timezone
from pathlib import Path
from textwrap import dedent
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from db.connection import create_database_engine, get_database_url
from db.schema import ensure_dashboard_indexes
from queries.analytics import (
    get_amount_bucket_summary,
    get_category_summary,
    get_date_bounds,
    get_fraud_over_time,
    get_hourly_summary,
    get_kpis,
    get_merchant_summary,
    get_prediction_date_bounds,
    get_risk_distribution,
    get_transaction_detail,
    get_unique_categories,
    get_unique_merchants,
    load_recent_transactions_with_predictions,
)
from simulation.realtime import SimulationConfig, run_realtime_simulation


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield · Transaction monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATION_RUN_LOCK = threading.Lock()
try:
    BOGOTA_TZ = ZoneInfo("America/Bogota")
except Exception:
    BOGOTA_TZ = timezone(timedelta(hours=-5))
SIMULATION_VOLUME_PRESETS = {
    "Small (~100 transactions)": 100,
    "Medium (~1,000 transactions)": 1000,
    "Large (~10,000 transactions)": 10000,
}
SIMULATION_SPEED_PRESETS = {
    "Slow (realistic pacing)": {"batch_size": 25, "pause_seconds": 1.0, "timeout_seconds": 30.0, "retries": 3},
    "Normal": {"batch_size": 100, "pause_seconds": 0.25, "timeout_seconds": 25.0, "retries": 3},
    "Fast (stress test)": {"batch_size": 500, "pause_seconds": 0.0, "timeout_seconds": 20.0, "retries": 2},
}
SIMULATION_STATE_DEFAULT = {
    "running": False,
    "status": "Idle",
    "message": "",
    "requested_total": 0,
    "actual_total": 0,
    "sent_transactions": 0,
    "batches_sent": 0,
    "failures": 0,
    "progress": 0.0,
    "started_at": None,
    "finished_at": None,
    "estimated_seconds": None,
    "summary": None,
    "error": None,
    "params": None,
}


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#070b12",
    "bg_soft": "#0b111c",
    "panel": "#0f1726",
    "panel_2": "#131c2e",
    "border": "#1f2a40",
    "border_soft": "#172033",
    "text": "#eaf1ff",
    "muted": "#8094b8",
    "subtle": "#5b6e91",
    "accent": "#4da3ff",
    "accent_2": "#7c5cff",
    "success": "#34d399",
    "warning": "#fbbf24",
    "danger": "#f87171",
    "danger_soft": "rgba(248, 113, 113, 0.14)",
    "success_soft": "rgba(52, 211, 153, 0.14)",
    "warning_soft": "rgba(251, 191, 36, 0.16)",
    "accent_soft": "rgba(77, 163, 255, 0.16)",
}


def _inject_css() -> None:
    css = dedent(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
    :root {{
        --bg: {PALETTE['bg']};
        --bg-soft: {PALETTE['bg_soft']};
        --panel: {PALETTE['panel']};
        --panel-2: {PALETTE['panel_2']};
        --border: {PALETTE['border']};
        --border-soft: {PALETTE['border_soft']};
        --text: {PALETTE['text']};
        --muted: {PALETTE['muted']};
        --subtle: {PALETTE['subtle']};
        --accent: {PALETTE['accent']};
        --accent-2: {PALETTE['accent_2']};
        --success: {PALETTE['success']};
        --warning: {PALETTE['warning']};
        --danger: {PALETTE['danger']};
        --accent-soft: {PALETTE['accent_soft']};
        --danger-soft: {PALETTE['danger_soft']};
        --success-soft: {PALETTE['success_soft']};
        --warning-soft: {PALETTE['warning_soft']};
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.35);
    }}
    html, body, .stApp {{
        background: var(--bg) !important;
        color: var(--text);
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-feature-settings: "cv02", "cv03", "cv11";
    }}
    .stApp {{
        background:
            radial-gradient(1200px 600px at 0% -10%, rgba(77,163,255,0.08), transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, rgba(124,92,255,0.06), transparent 60%),
            var(--bg) !important;
    }}
    #MainMenu, footer {{ visibility: hidden; height: 0; }}
    header[data-testid="stHeader"] {{
        background: transparent;
        height: 0;
        min-height: 0;
        overflow: visible;
    }}
    button[data-testid="collapsedControl"] {{
        position: fixed;
        top: 1.35rem;
        left: 1rem;
        z-index: 1000;
        visibility: visible !important;
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 999px;
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow-md);
        opacity: 0.95;
    }}
    button[data-testid="collapsedControl"]:hover {{
        background: var(--accent-soft) !important;
        border-color: var(--accent) !important;
    }}
    .block-container {{
        padding-top: 0rem !important;
        padding-bottom: 3rem !important;
        max-width: 1600px;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text);
        font-family: "Inter", sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    h1 {{ font-size: clamp(1.3rem, 1.6vw, 1.6rem) !important; }}
    h2 {{ font-size: clamp(1.05rem, 1.3vw, 1.25rem) !important; }}
    h3 {{ font-size: clamp(0.92rem, 1.1vw, 1.05rem) !important; }}
    p, label, span {{ color: var(--text); }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}
    code, pre, .mono {{ font-family: "JetBrains Mono", monospace; }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0a1019 0%, #070b12 100%);
        border-right: 1px solid var(--border-soft);
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem !important; }}
    section[data-testid="stSidebar"] h3 {{
        text-transform: uppercase;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 600;
        margin: 1.1rem 0 0.4rem 0;
    }}
    section[data-testid="stSidebar"] hr {{ border-color: var(--border-soft); }}
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stDateInput input,
    .stTextInput input,
    .stNumberInput input {{
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text) !important;
        font-size: 0.85rem;
    }}
    .stSlider [data-baseweb="slider"] > div > div {{ background: var(--accent) !important; }}
    .stSlider [role="slider"] {{ background: var(--accent) !important; box-shadow: 0 0 0 4px var(--accent-soft) !important; }}
    .stButton > button, .stDownloadButton > button {{
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500;
        font-size: 0.84rem;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: var(--accent-soft) !important;
        border-color: var(--accent) !important;
        color: #fff !important;
        transform: translateY(-1px);
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, var(--panel) 0%, var(--panel-2) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        transition: border-color 0.15s ease, transform 0.15s ease;
    }}
    div[data-testid="stMetric"]:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
    div[data-testid="stMetricLabel"] p {{
        color: var(--muted);
        font-size: 0.74rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--text);
        font-size: clamp(1.4rem, 2vw, 1.8rem) !important;
        font-weight: 700;
        font-feature-settings: "tnum";
    }}
    .fs-card {{
        background: linear-gradient(160deg, var(--panel) 0%, var(--panel-2) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
    }}
    .fs-card-title {{
        display: flex;
        align-items: center;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.85rem;
    }}
    .fs-card-title .fs-dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 12px var(--accent);
        margin-right: 8px;
        display: inline-block;
    }}
    .fs-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 14px 20px;
        background: linear-gradient(135deg, rgba(77,163,255,0.08), rgba(124,92,255,0.05));
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }}
    .fs-brand {{ display: flex; align-items: center; gap: 12px; }}
    .fs-logo {{
        width: 38px; height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        box-shadow: 0 4px 16px rgba(77,163,255,0.35);
    }}
    .fs-brand-text {{ display: flex; flex-direction: column; line-height: 1.15; }}
    .fs-brand-name {{ font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em; color: var(--text); }}
    .fs-brand-sub {{ font-size: 0.75rem; color: var(--muted); }}
    .fs-status {{
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 12px;
        background: var(--success-soft);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 999px;
        font-size: 0.75rem;
        color: var(--success);
        font-weight: 500;
    }}
    .fs-status .pulse {{
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 0 0 rgba(52,211,153,0.7);
        animation: fs-pulse 1.8s infinite;
    }}
    @keyframes fs-pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0.6); }}
        70% {{ box-shadow: 0 0 0 8px rgba(52,211,153,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0); }}
    }}
    .fs-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1rem; }}
    .fs-chip {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 999px;
        font-size: 0.72rem;
        color: var(--muted);
    }}
    .fs-chip strong {{ color: var(--text); font-weight: 500; }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: var(--panel);
        padding: 4px;
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: var(--radius-sm);
        padding: 8px 18px;
        color: var(--muted);
        font-weight: 500;
        font-size: 0.85rem;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{ background: var(--accent-soft) !important; color: var(--text) !important; }}
    [data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: var(--radius-md); overflow: hidden; background: var(--panel); }}
    [data-testid="stDataFrame"] [role="columnheader"] {{
        background: var(--panel-2) !important;
        color: var(--muted) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-baseweb="notification"] {{ background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }}
    @media (max-width: 768px) {{
        .block-container {{ padding-left: 0.8rem !important; padding-right: 0.8rem !important; }}
        .fs-header {{ flex-direction: column; align-items: flex-start; }}
        div[data-testid="stMetric"] {{ padding: 14px; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
    }}
    </style>
    """)
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Engine + cached queries
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_engine():
    load_dotenv()
    engine = create_database_engine(get_database_url(os.getenv("DATABASE_URL")))
    ensure_dashboard_indexes(engine)
    return engine


@st.cache_data(ttl=10, show_spinner=False)
def cached_date_bounds():
    return get_date_bounds(engine=get_engine())


@st.cache_data(ttl=10, show_spinner=False)
def cached_prediction_date_bounds():
    return get_prediction_date_bounds(engine=get_engine())


@st.cache_data(ttl=5, show_spinner=False)
def cached_recent_transactions(limit, start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant, category, only_fraud):
    return load_recent_transactions_with_predictions(
        engine=get_engine(), limit=limit, start_date=start_date, end_date=end_date,
        created_start_date=created_start_date, created_end_date=created_end_date,
        min_risk_score=min_risk_score, merchant=merchant, category=category, only_fraud=only_fraud,
    )


@st.cache_data(ttl=5, show_spinner=False)
def cached_kpis(start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant, category, only_fraud):
    return get_kpis(
        engine=get_engine(), start_date=start_date, end_date=end_date,
        created_start_date=created_start_date, created_end_date=created_end_date,
        min_risk_score=min_risk_score, merchant=merchant, category=category, only_fraud=only_fraud,
    )


@st.cache_data(ttl=10, show_spinner=False)
def cached_fraud_over_time(start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant, category, only_fraud):
    return get_fraud_over_time(
        engine=get_engine(), start_date=start_date, end_date=end_date,
        created_start_date=created_start_date, created_end_date=created_end_date,
        min_risk_score=min_risk_score, merchant=merchant, category=category, only_fraud=only_fraud,
    )


@st.cache_data(ttl=10, show_spinner=False)
def cached_risk_distribution(start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant, category, only_fraud):
    return get_risk_distribution(
        engine=get_engine(), start_date=start_date, end_date=end_date,
        created_start_date=created_start_date, created_end_date=created_end_date,
        min_risk_score=min_risk_score, merchant=merchant, category=category, only_fraud=only_fraud,
    )


@st.cache_data(ttl=10, show_spinner=False)
def cached_category_summary(top_n, start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant, category, only_fraud):
    return get_category_summary(
        engine=get_engine(), top_n=top_n, start_date=start_date, end_date=end_date,
        created_start_date=created_start_date, created_end_date=created_end_date,
        min_risk_score=min_risk_score, merchant=merchant, category=category, only_fraud=only_fraud,
    )


@st.cache_data(ttl=10, show_spinner=False)
def cached_merchant_summary(top_n, start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant, category, only_fraud):
    return get_merchant_summary(
        engine=get_engine(), top_n=top_n, start_date=start_date, end_date=end_date,
        created_start_date=created_start_date, created_end_date=created_end_date,
        min_risk_score=min_risk_score, merchant=merchant, category=category, only_fraud=only_fraud,
    )


@st.cache_data(ttl=10, show_spinner=False)
def cached_hourly_summary(start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant, category, only_fraud):
    return get_hourly_summary(
        engine=get_engine(), start_date=start_date, end_date=end_date,
        created_start_date=created_start_date, created_end_date=created_end_date,
        min_risk_score=min_risk_score, merchant=merchant, category=category, only_fraud=only_fraud,
    )


@st.cache_data(ttl=10, show_spinner=False)
def cached_amount_summary(start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant, category, only_fraud):
    return get_amount_bucket_summary(
        engine=get_engine(), start_date=start_date, end_date=end_date,
        created_start_date=created_start_date, created_end_date=created_end_date,
        min_risk_score=min_risk_score, merchant=merchant, category=category, only_fraud=only_fraud,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_merchants():
    return get_unique_merchants(engine=get_engine())


@st.cache_data(ttl=300, show_spinner=False)
def cached_categories():
    return get_unique_categories(engine=get_engine())


# ---------------------------------------------------------------------------
# Chart catalog
# ---------------------------------------------------------------------------
CHART_CATALOG = [
    {"key": "fraud_over_time", "label": "Fraud count by day"},
    {"key": "avg_risk_over_time", "label": "Average risk by day"},
    {"key": "risk_distribution", "label": "Risk score distribution"},
    {"key": "category_fraud_count", "label": "Fraud count by category"},
    {"key": "category_fraud_rate", "label": "Fraud rate by category"},
    {"key": "category_avg_risk", "label": "Average risk by category"},
    {"key": "category_max_risk", "label": "Max risk by category"},
    {"key": "merchant_fraud_count", "label": "Fraud count by merchant"},
    {"key": "merchant_fraud_rate", "label": "Fraud rate by merchant"},
    {"key": "merchant_avg_risk", "label": "Average risk by merchant"},
    {"key": "hourly_fraud_count", "label": "Fraud count by hour"},
    {"key": "hourly_avg_risk", "label": "Average risk by hour"},
    {"key": "amount_fraud_rate", "label": "Fraud rate by amount bucket"},
]
CHART_LABEL_TO_KEY = {item["label"]: item["key"] for item in CHART_CATALOG}
CHART_KEY_TO_LABEL = {item["key"]: item["label"] for item in CHART_CATALOG}
CHART_PRESETS = {
    "Operational": ["Fraud count by day", "Risk score distribution", "Fraud count by category", "Fraud count by merchant"],
    "Category focus": ["Fraud count by category", "Fraud rate by category", "Average risk by category", "Max risk by category"],
    "Merchant focus": ["Fraud count by merchant", "Fraud rate by merchant", "Average risk by merchant"],
    "Temporal": ["Fraud count by day", "Average risk by day", "Fraud count by hour", "Average risk by hour", "Fraud rate by amount bucket"],
    "All charts": [item["label"] for item in CHART_CATALOG],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_money(value):
    return f"${value:,.2f}"


def _format_pct(value):
    return f"{value:.3f}"


def _format_compact(value):
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{int(value)}"


def _prediction_status_text(value):
    normalized = str(value).lower()
    if normalized == "fraud":
        return "Fraud"
    if normalized == "legit":
        return "Legit"
    return "Pending"


def _simulation_state() -> dict:
    if "simulation_state" not in st.session_state:
        st.session_state["simulation_state"] = dict(SIMULATION_STATE_DEFAULT)
    return st.session_state["simulation_state"]


def _simulation_events() -> queue.Queue:
    if "simulation_events" not in st.session_state:
        st.session_state["simulation_events"] = queue.Queue()
    return st.session_state["simulation_events"]


def _compact_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _normalize_date_range(selected_range):
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        return selected_range[0], selected_range[1], True
    return None, None, False


def _format_bogota_datetime(value) -> str:
    if pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts.tz_convert(BOGOTA_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _estimate_eta_seconds(state: dict) -> float | None:
    started_at = state.get("started_at")
    sent = int(state.get("sent_transactions") or 0)
    total = int(state.get("actual_total") or 0)
    if not started_at or sent <= 0 or total <= sent:
        return None

    elapsed = max(time.time() - float(started_at), 0.001)
    rate = sent / elapsed
    if rate <= 0:
        return None

    return max((total - sent) / rate, 0.0)


def _simulation_worker(*, params: dict[str, object], event_queue: queue.Queue) -> None:
    acquired = False
    try:
        acquired = SIMULATION_RUN_LOCK.acquire(blocking=False)
        if not acquired:
            event_queue.put({"type": "error", "message": "Ya hay una simulación en ejecución."})
            return

        requested_total = int(params["requested_total"])
        batch_size = int(params["batch_size"])
        pause_seconds = float(params["pause_seconds"])
        timeout_seconds = float(params["timeout_seconds"])
        retries = int(params["retries"])
        api_url = str(params["api_url"])
        database_url = str(params.get("database_url") or os.getenv("DATABASE_URL") or "")
        if not database_url:
            raise RuntimeError("DATABASE_URL no esta configurada.")

        event_queue.put({"type": "status", "message": "Preparando simulación...", "running": True})

        iterations = max((requested_total + batch_size - 1) // batch_size, 1)
        event_queue.put({
            "type": "start",
            "requested_total": requested_total,
            "actual_total": requested_total,
            "pool_size": requested_total,
            "available_after_filters": requested_total,
            "message": "Simulación en curso",
        })

        summary = run_realtime_simulation(
            SimulationConfig(
                source="db",
                batch_size=batch_size,
                iterations=iterations,
                continuous=False,
                pause_seconds=pause_seconds,
                endpoint_url=api_url,
                timeout_seconds=timeout_seconds,
                retries=retries,
                database_url=database_url,
                table_name="transactions",
                order_by="trans_num",
            )
        )

        sent_transactions = int(summary.get("transactions_sent", 0))
        batches_sent = int(summary.get("batches_sent", 0))
        failures = int(summary.get("failures", 0))
        total_latency_seconds = float(summary.get("total_latency_seconds", 0.0))

        event_queue.put({
            "type": "done",
            "status": "Simulación completada",
            "sent_transactions": sent_transactions,
            "batches_sent": batches_sent,
            "failures": failures,
            "actual_total": sent_transactions,
            "summary": {
                "requested_total": requested_total,
                "actual_total": sent_transactions,
                "batches_sent": batches_sent,
                "failures": failures,
                "total_latency_seconds": total_latency_seconds,
                "available_after_filters": requested_total,
            },
        })
    except Exception as exc:
        event_queue.put({"type": "error", "message": f"Fallo la simulación: {type(exc).__name__}: {exc}"})
    finally:
        if acquired and SIMULATION_RUN_LOCK.locked():
            SIMULATION_RUN_LOCK.release()


def _sync_simulation_state() -> bool:
    state = _simulation_state()
    event_queue = _simulation_events()
    changed = False

    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break

        changed = True
        event_type = event.get("type")

        if event_type == "status":
            state["status"] = str(event.get("message", state["status"]))
            state["message"] = str(event.get("message", ""))
            state["running"] = bool(event.get("running", True))
            continue

        if event_type == "warning":
            state["message"] = str(event.get("message", ""))
            st.session_state["simulation_warning"] = str(event.get("message", ""))
            continue

        if event_type == "start":
            state["running"] = True
            state["status"] = str(event.get("message", "Simulación en curso"))
            state["message"] = str(event.get("message", ""))
            state["requested_total"] = int(event.get("requested_total", state["requested_total"]))
            state["actual_total"] = int(event.get("actual_total", state["actual_total"]))
            state["sent_transactions"] = 0
            state["batches_sent"] = 0
            state["failures"] = 0
            state["started_at"] = time.time()
            state["finished_at"] = None
            state["summary"] = None
            state["error"] = None
            continue

        if event_type == "progress":
            state["running"] = True
            state["status"] = str(event.get("status", state["status"]))
            state["sent_transactions"] = int(event.get("sent_transactions", state["sent_transactions"]))
            state["batches_sent"] = int(event.get("batches_sent", state["batches_sent"]))
            state["failures"] = int(event.get("failures", state["failures"]))
            state["actual_total"] = int(event.get("actual_total", state["actual_total"]))
            state["progress"] = (state["sent_transactions"] / state["actual_total"]) if state["actual_total"] else 0.0
            state["estimated_seconds"] = _estimate_eta_seconds(state)
            continue

        if event_type == "done":
            state["running"] = False
            state["status"] = str(event.get("status", "Simulación completada"))
            state["sent_transactions"] = int(event.get("sent_transactions", state["sent_transactions"]))
            state["batches_sent"] = int(event.get("batches_sent", state["batches_sent"]))
            state["failures"] = int(event.get("failures", state["failures"]))
            state["actual_total"] = int(event.get("actual_total", state["actual_total"]))
            state["progress"] = 1.0 if state["actual_total"] else 0.0
            state["estimated_seconds"] = 0.0
            state["finished_at"] = time.time()
            state["summary"] = event.get("summary")
            state["message"] = str(event.get("status", ""))
            continue

        if event_type == "error":
            state["running"] = False
            state["status"] = "Error"
            state["error"] = str(event.get("message", ""))
            state["message"] = str(event.get("message", ""))
            state["finished_at"] = time.time()
            continue

    if changed:
        st.cache_data.clear()

    return changed


def _start_simulation(*, volume_label: str, speed_label: str) -> None:
    state = _simulation_state()

    if state.get("running") or SIMULATION_RUN_LOCK.locked():
        st.warning("Ya hay una simulación en ejecución.")
        return

    requested_total = int(SIMULATION_VOLUME_PRESETS[volume_label])
    speed = SIMULATION_SPEED_PRESETS[speed_label]
    params = {
        "requested_total": requested_total,
        "batch_size": int(speed["batch_size"]),
        "pause_seconds": float(speed["pause_seconds"]),
        "timeout_seconds": float(speed["timeout_seconds"]),
        "retries": int(speed["retries"]),
        "api_url": os.getenv("API_URL", "http://127.0.0.1:8000"),
        "database_url": os.getenv("DATABASE_URL"),
    }

    state.update({
        "running": True,
        "status": "Preparando simulación...",
        "message": "",
        "requested_total": requested_total,
        "actual_total": requested_total,
        "sent_transactions": 0,
        "batches_sent": 0,
        "failures": 0,
        "progress": 0.0,
        "started_at": time.time(),
        "finished_at": None,
        "estimated_seconds": None,
        "summary": None,
        "error": None,
        "params": params,
    })
    st.session_state.pop("simulation_warning", None)

    event_queue = _simulation_events()
    worker = threading.Thread(
        target=_simulation_worker,
        kwargs={"params": params, "event_queue": event_queue},
        daemon=True,
    )
    st.session_state["simulation_thread"] = worker
    worker.start()
    st.rerun()


# ---------------------------------------------------------------------------
# Header (single line HTML — no blank lines!)
# ---------------------------------------------------------------------------
def _render_header(refresh_seconds: int):
    html = (
        '<div class="fs-header">'
        '<div class="fs-brand">'
        '<div class="fs-logo">🛡️</div>'
        '<div class="fs-brand-text">'
        '<span class="fs-brand-name">FraudShield</span>'
        '<span class="fs-brand-sub">Real-time transaction monitoring</span>'
        '</div></div>'
        f'<div class="fs-status"><span class="pulse"></span>Live · auto-refresh {refresh_seconds}s</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_filter_chips(start_date, end_date, created_start_date, created_end_date, merchant, category, min_risk, only_fraud):
    chips = [
        f'<span class="fs-chip">Range <strong>{start_date} → {end_date}</strong></span>',
        f'<span class="fs-chip">Created_at <strong>{created_start_date} → {created_end_date}</strong></span>',
        f'<span class="fs-chip">Merchant <strong>{merchant or "All"}</strong></span>',
        f'<span class="fs-chip">Category <strong>{category or "All"}</strong></span>',
        f'<span class="fs-chip">Min risk <strong>{min_risk:.2f}</strong></span>',
    ]
    if only_fraud:
        chips.append('<span class="fs-chip">Only <strong>fraud</strong></span>')
    st.markdown('<div class="fs-chips">' + "".join(chips) + '</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
def _render_kpis(kpis):
    total = int(kpis["total_transactions"])
    frauds = int(kpis["fraud_count"])
    rate = (frauds / total * 100) if total else 0.0
    cols = st.columns(4)
    cols[0].metric("Total transactions", _format_compact(total))
    cols[1].metric("Fraud count", _format_compact(frauds), delta=f"{rate:.2f}% rate", delta_color="inverse")
    cols[2].metric("Amount saved", _format_money(kpis["amount_saved"]))
    cols[3].metric("Avg fraud amount", _format_money(kpis["avg_fraud_amount"]))


# ---------------------------------------------------------------------------
# Sidebar (filters + builder)
# ---------------------------------------------------------------------------
def _render_filters(*, min_bound, max_bound, default_start, default_end, created_min_bound, created_max_bound, created_default_start, created_default_end):
    with st.sidebar:
        sidebar_brand = (
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">'
            '<div class="fs-logo" style="width:32px;height:32px;font-size:0.95rem;">🛡️</div>'
            '<span style="font-weight:700;font-size:1rem;color:var(--text);">FraudShield</span>'
            '</div>'
        )
        st.markdown(sidebar_brand, unsafe_allow_html=True)

        if st.button("↻ Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("### Time window")
        time_window_mode = st.selectbox(
            "Range mode",
            ["Custom range", "All historical transactions"],
            index=0,
            key="time_window_mode",
        )

        if time_window_mode == "All historical transactions":
            start_date = min_bound or default_start
            end_date = max_bound or default_end
            time_window_ready = True
            st.caption(f"Using full history: {start_date} to {end_date}")
        else:
            selected_range = st.date_input(
                "Date range", value=(default_start, default_end),
                min_value=min_bound, max_value=max_bound, label_visibility="collapsed",
            )
            start_date, end_date, time_window_ready = _normalize_date_range(selected_range)

        st.markdown("### Created-at window")
        created_window_mode = st.selectbox(
            "Created-at range mode",
            ["Custom range", "All historical created_at"],
            index=0,
            key="created_window_mode",
        )

        if created_window_mode == "All historical created_at":
            created_start_date = created_min_bound or created_default_start
            created_end_date = created_max_bound or created_default_end
            created_window_ready = True
            st.caption(f"Using full created_at history: {created_start_date} to {created_end_date}")
        else:
            created_selected_range = st.date_input(
                "Created-at date range", value=(created_default_start, created_default_end),
                min_value=created_min_bound, max_value=created_max_bound, label_visibility="collapsed",
            )
            created_start_date, created_end_date, created_window_ready = _normalize_date_range(created_selected_range)

        st.markdown("### Filters")
        merchants = cached_merchants()
        categories = cached_categories()
        merchant = st.selectbox("Merchant", ["All"] + merchants)
        category = st.selectbox("Category", ["All"] + categories)
        min_risk_score = st.slider("Min risk score", 0.0, 1.0, 0.0, 0.01)
        only_fraud = st.toggle("Only fraud predictions", value=False)

        st.markdown("### Display")
        table_limit = st.slider("Rows in table", 100, 200, 150, 10)
        top_merchants = st.slider("Top merchants/categories", 5, 20, 10, 1)
        high_risk_threshold = st.slider("High-risk threshold", 0.0, 1.0, 0.8, 0.01)

    return (
        start_date, end_date,
        created_start_date, created_end_date,
        None if merchant == "All" else merchant,
        None if category == "All" else category,
        float(min_risk_score), int(table_limit), bool(only_fraud),
        int(top_merchants), float(high_risk_threshold),
        bool(time_window_ready and created_window_ready),
    )


def _render_dashboard_builder():
    with st.sidebar:
        st.markdown("### Dashboard builder")
        preset_name = st.selectbox("Preset", list(CHART_PRESETS.keys()))

        if "chart_selection" not in st.session_state:
            st.session_state["chart_selection"] = list(CHART_PRESETS[preset_name])

        c1, c2 = st.columns(2)
        if c1.button("Apply", use_container_width=True):
            st.session_state["chart_selection"] = list(CHART_PRESETS[preset_name])
        if c2.button("Clear", use_container_width=True):
            st.session_state["chart_selection"] = []

        selected_labels = st.multiselect(
            "Charts to display",
            options=[item["label"] for item in CHART_CATALOG],
            key="chart_selection",
        )
    return [CHART_LABEL_TO_KEY[label] for label in selected_labels]


def _render_simulation_control():
    state = _simulation_state()

    with st.sidebar.expander("Simulation Control", expanded=bool(state.get("running"))):
        st.caption("Generate transactions and post them to the API in controlled batches.")

        volume_label = st.selectbox(
            "Simulation volume",
            options=list(SIMULATION_VOLUME_PRESETS.keys()),
            index=1,
            key="simulation_volume_label",
        )
        speed_label = st.selectbox(
            "Transaction speed",
            options=list(SIMULATION_SPEED_PRESETS.keys()),
            index=1,
            key="simulation_speed_label",
        )
        st.caption(
            "Slow sends smaller batches with longer pauses, Normal balances pacing and throughput, and Fast uses bigger batches with minimal pause."
        )

        expected_total = int(SIMULATION_VOLUME_PRESETS[volume_label])
        batch_size = int(SIMULATION_SPEED_PRESETS[speed_label]["batch_size"])
        pause_seconds = float(SIMULATION_SPEED_PRESETS[speed_label]["pause_seconds"])
        estimated_batches = max((expected_total + batch_size - 1) // batch_size, 1)
        estimated_duration = max(estimated_batches * pause_seconds, 0.0)

        st.caption(
            f"Estimated load: {_compact_count(expected_total)} tx · {estimated_batches} batches · ~{estimated_duration:.1f}s pacing"
        )

        start_disabled = bool(state.get("running"))
        if st.button("Start Simulation", use_container_width=True, disabled=start_disabled):
            _start_simulation(
                volume_label=volume_label,
                speed_label=speed_label,
            )

        status_box = st.empty()
        summary_box = st.empty()

        if state.get("message"):
            status_box.info(str(state.get("message")))
        else:
            status_box.caption(f"Status: {state.get('status', 'Idle')}")

        total = int(state.get("actual_total") or state.get("requested_total") or expected_total)
        progress_value = float(state.get("progress") or 0.0)
        metrics = st.columns(3)
        metrics[0].metric("Sent", _compact_count(int(state.get("sent_transactions") or 0)))
        metrics[1].metric("Target", _compact_count(total))
        metrics[2].metric("Failures", _compact_count(int(state.get("failures") or 0)))
        st.progress(int(round(max(0.0, min(progress_value, 1.0)) * 100)))

        eta_seconds = state.get("estimated_seconds")
        if state.get("running") and eta_seconds is not None:
            eta_text = str(timedelta(seconds=max(int(eta_seconds), 0)))
            st.caption(f"Estimated remaining time: {eta_text}")

        if st.session_state.get("simulation_warning"):
            st.warning(str(st.session_state["simulation_warning"]))

        if state.get("error"):
            summary_box.error(str(state.get("error")))
        elif state.get("summary"):
            summary = state["summary"] or {}
            summary_box.success(
                f"Completed {summary.get('actual_total', total)} transactions across {summary.get('batches_sent', 0)} batches."
            )
            st.caption(f"Total latency: {float(summary.get('total_latency_seconds', 0.0)):.2f}s")

    return state


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _apply_chart_theme(fig, *, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter, sans-serif", size=12),
        margin=dict(l=8, r=8, t=20, b=8),
        height=height,
        showlegend=False,
        hoverlabel=dict(bgcolor=PALETTE["panel_2"], bordercolor=PALETTE["border"], font_size=12, font_family="Inter"),
    )
    fig.update_xaxes(
        gridcolor=PALETTE["border_soft"], zerolinecolor=PALETTE["border_soft"],
        linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["muted"], size=11), title=None,
    )
    fig.update_yaxes(
        gridcolor=PALETTE["border_soft"], zerolinecolor=PALETTE["border_soft"],
        linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["muted"], size=11), title=None,
    )
    return fig


def _render_chart_card(title, figure, *, height=240):
    title_html = f'<div class="fs-card-title"><span class="fs-dot"></span>{title}</div>'
    st.markdown(f'<div class="fs-card">{title_html}</div>', unsafe_allow_html=True)
    st.plotly_chart(_apply_chart_theme(figure, height=height), use_container_width=True, config={"displayModeBar": False})


def _build_chart_figure(chart_key, *, fraud_time, category_summary, merchant_summary, hourly_summary, amount_summary, risk_dist):
    if chart_key == "fraud_over_time" and not fraud_time.empty:
        figure = px.area(fraud_time, x="date", y="fraud_count")
        figure.update_traces(line=dict(color=PALETTE["accent"], width=2.5), fillcolor="rgba(77,163,255,0.12)")
        return CHART_KEY_TO_LABEL[chart_key], figure
    if chart_key == "avg_risk_over_time" and not fraud_time.empty:
        figure = px.line(fraud_time, x="date", y="avg_risk_score", markers=True)
        figure.update_traces(line=dict(color=PALETTE["accent_2"], width=2.5), marker=dict(color=PALETTE["accent_2"], size=6))
        return CHART_KEY_TO_LABEL[chart_key], figure
    if chart_key == "risk_distribution" and risk_dist is not None and not risk_dist.empty:
        figure = px.bar(risk_dist, x="risk_score_bucket", y="count", color_discrete_sequence=[PALETTE["accent"]])
        figure.update_traces(marker_line_width=0)
        return CHART_KEY_TO_LABEL[chart_key], figure
    if chart_key.startswith("category_") and category_summary is not None and not category_summary.empty:
        metric_map = {
            "category_fraud_count": "fraud_count", "category_fraud_rate": "fraud_rate",
            "category_avg_risk": "avg_risk_score", "category_max_risk": "max_risk_score",
        }
        col = metric_map.get(chart_key)
        if col:
            figure = px.bar(
                category_summary.sort_values(col, ascending=True),
                x=col, y="category", orientation="h", color_discrete_sequence=[PALETTE["accent"]],
            )
            figure.update_traces(marker_line_width=0)
            if col == "fraud_rate":
                figure.update_xaxes(tickformat=".0%")
            return CHART_KEY_TO_LABEL[chart_key], figure
    if chart_key.startswith("merchant_") and merchant_summary is not None and not merchant_summary.empty:
        metric_map = {
            "merchant_fraud_count": "fraud_count", "merchant_fraud_rate": "fraud_rate",
            "merchant_avg_risk": "avg_risk_score",
        }
        col = metric_map.get(chart_key)
        if col:
            figure = px.bar(
                merchant_summary.sort_values(col, ascending=True),
                x=col, y="merchant", orientation="h", color_discrete_sequence=[PALETTE["accent_2"]],
            )
            figure.update_traces(marker_line_width=0)
            if col == "fraud_rate":
                figure.update_xaxes(tickformat=".0%")
            return CHART_KEY_TO_LABEL[chart_key], figure
    if chart_key.startswith("hourly_") and hourly_summary is not None and not hourly_summary.empty:
        metric_map = {"hourly_fraud_count": "fraud_count", "hourly_avg_risk": "avg_risk_score"}
        col = metric_map.get(chart_key)
        if col:
            figure = px.bar(hourly_summary, x="hour_of_day", y=col, color_discrete_sequence=[PALETTE["accent"]])
            figure.update_traces(marker_line_width=0)
            return CHART_KEY_TO_LABEL[chart_key], figure
    if chart_key == "amount_fraud_rate" and amount_summary is not None and not amount_summary.empty:
        amount_view = amount_summary.copy()
        amount_view["bucket_label"] = amount_view["bucket_start"].astype(int).map(lambda v: f"${v:,.0f}-${v+99:,.0f}")
        figure = px.bar(amount_view, x="bucket_label", y="fraud_rate", color_discrete_sequence=[PALETTE["warning"]])
        figure.update_traces(marker_line_width=0)
        figure.update_xaxes(tickangle=-35)
        figure.update_yaxes(tickformat=".0%")
        return CHART_KEY_TO_LABEL[chart_key], figure
    return None, None


def _render_chart_grid(selected_chart_keys, *, fraud_time, category_summary, merchant_summary, hourly_summary, amount_summary, risk_dist):
    if not selected_chart_keys:
        st.info("Select at least one chart in the dashboard builder.")
        return

    chart_items = []
    for chart_key in selected_chart_keys:
        title, figure = _build_chart_figure(
            chart_key, fraud_time=fraud_time, category_summary=category_summary,
            merchant_summary=merchant_summary, hourly_summary=hourly_summary,
            amount_summary=amount_summary, risk_dist=risk_dist,
        )
        if figure is not None and title is not None:
            chart_items.append((title, figure))

    if not chart_items:
        st.info("No chart data found for the selected filters.")
        return

    for index in range(0, len(chart_items), 2):
        cols = st.columns(2, gap="small")
        for col, (title, figure) in zip(cols, chart_items[index:index + 2]):
            with col:
                _render_chart_card(title, figure)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _style_table(frame, highlight_threshold):
    def highlight_row(row):
        risk_value = row.get("risk_score")
        prediction_value = str(row.get("prediction_status", "pending")).lower()
        if pd.notna(risk_value) and float(risk_value) >= highlight_threshold:
            return ["background-color: rgba(248,113,113,0.10); color: #ffe7e7"] * len(row)
        if prediction_value == "pending":
            return ["background-color: rgba(255,255,255,0.02); color: #b8c5dc"] * len(row)
        return [""] * len(row)

    return (
        frame.style.apply(highlight_row, axis=1)
        .format({
            "amt": _format_money,
            "risk_score": lambda v: _format_pct(float(v)) if pd.notna(v) else "pending",
            "transaction_created_at": _format_bogota_datetime,
            "prediction_created_at": _format_bogota_datetime,
        })
    )


def _render_top_risk_panel(latest_df):
    st.markdown('<div class="fs-card-title"><span class="fs-dot"></span>Top risky transactions</div>', unsafe_allow_html=True)
    top_risk = latest_df.copy()
    sort_created_at = top_risk["prediction_created_at"].fillna(top_risk["transaction_created_at"])
    top_risk = top_risk.assign(_sort_created_at=sort_created_at).sort_values(
        ["risk_score", "_sort_created_at"], ascending=[False, False], na_position="last"
    ).head(5)
    if top_risk.empty:
        st.info("No transactions found.")
        return
    if "prediction_status" not in top_risk.columns:
        top_risk["prediction_status"] = top_risk["prediction"].astype(str)
    top_risk["prediction_status"] = top_risk["prediction_status"].map(_prediction_status_text)
    st.dataframe(
        top_risk[["trans_num", "merchant", "risk_score", "prediction_status"]],
        use_container_width=True, hide_index=True, height=240,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _inject_css()

    _sync_simulation_state()
    refresh_interval = 1000 if _simulation_state().get("running") else 7000
    st_autorefresh(interval=refresh_interval, key="fraudshield_autorefresh")

    min_bound, max_bound = cached_date_bounds()
    created_min_bound, created_max_bound = cached_prediction_date_bounds()
    default_end = max_bound or date.today()
    default_start = min_bound or (default_end - timedelta(days=7))
    created_default_end = created_max_bound or date.today()
    created_default_start = created_min_bound or (created_default_end - timedelta(days=7))

    _render_header(max(int(refresh_interval / 1000), 1))

    (
        start_date, end_date, created_start_date, created_end_date, merchant_value, category_value, min_risk_score,
        latest_limit, only_fraud, top_merchants, high_risk_threshold, filters_ready,
    ) = _render_filters(
        min_bound=min_bound, max_bound=max_bound,
        default_start=default_start, default_end=default_end,
        created_min_bound=created_min_bound, created_max_bound=created_max_bound,
        created_default_start=created_default_start, created_default_end=created_default_end,
    )
    selected_chart_keys = _render_dashboard_builder()
    _render_simulation_control()

    if not filters_ready:
        st.info("Complete both date ranges to run the queries.")
        st.stop()

    _render_filter_chips(start_date, end_date, created_start_date, created_end_date, merchant_value, category_value, min_risk_score, only_fraud)

    need_fraud_time = any(k in {"fraud_over_time", "avg_risk_over_time"} for k in selected_chart_keys)
    need_category = any(k.startswith("category_") for k in selected_chart_keys)
    need_merchant = any(k.startswith("merchant_") for k in selected_chart_keys)
    need_hourly = any(k.startswith("hourly_") for k in selected_chart_keys)
    need_amount = "amount_fraud_rate" in selected_chart_keys
    need_risk_dist = "risk_distribution" in selected_chart_keys

    try:
        with st.spinner("Loading data…"):
            kpis = cached_kpis(start_date, end_date, created_start_date, created_end_date, min_risk_score, merchant_value, category_value, only_fraud)
            latest_df = cached_recent_transactions(
                int(latest_limit), start_date, end_date, created_start_date, created_end_date, float(min_risk_score),
                merchant_value, category_value, bool(only_fraud),
            )
            fraud_time = (
                cached_fraud_over_time(start_date, end_date, created_start_date, created_end_date, float(min_risk_score), merchant_value, category_value, bool(only_fraud))
                if need_fraud_time else pd.DataFrame()
            )
            category_summary = (
                cached_category_summary(int(top_merchants), start_date, end_date, created_start_date, created_end_date, float(min_risk_score), merchant_value, category_value, bool(only_fraud))
                if need_category else None
            )
            merchant_summary = (
                cached_merchant_summary(int(top_merchants), start_date, end_date, created_start_date, created_end_date, float(min_risk_score), merchant_value, category_value, bool(only_fraud))
                if need_merchant else None
            )
            hourly_summary = (
                cached_hourly_summary(start_date, end_date, created_start_date, created_end_date, float(min_risk_score), merchant_value, category_value, bool(only_fraud))
                if need_hourly else None
            )
            amount_summary = (
                cached_amount_summary(start_date, end_date, created_start_date, created_end_date, float(min_risk_score), merchant_value, category_value, bool(only_fraud))
                if need_amount else None
            )
            risk_dist = (
                cached_risk_distribution(start_date, end_date, created_start_date, created_end_date, float(min_risk_score), merchant_value, category_value, bool(only_fraud))
                if need_risk_dist else None
            )
    except Exception as exc:
        st.error(f"Database query failed: {type(exc).__name__}: {exc}")
        st.stop()

    _render_kpis(kpis)

    if latest_df.empty:
        st.info("No transactions found for the selected filters.")
        return

    tab_overview, tab_tx, tab_drill = st.tabs(["📊  Overview", "📋  Transactions", "🔍  Drill-down"])

    with tab_overview:
        _render_top_risk_panel(latest_df)
        _render_chart_grid(
            selected_chart_keys,
            fraud_time=fraud_time,
            category_summary=category_summary,
            merchant_summary=merchant_summary,
            hourly_summary=hourly_summary,
            amount_summary=amount_summary,
            risk_dist=risk_dist,
        )

    with tab_tx:
        export_csv = latest_df.to_csv(index=False).encode("utf-8")
        head_l, head_r = st.columns([0.75, 0.25])
        with head_l:
            st.markdown(f"**{len(latest_df)} transactions** · highlighting risk ≥ {high_risk_threshold:.2f}")
        with head_r:
            st.download_button(
                "⬇  Export CSV", data=export_csv,
                file_name="transactions_with_predictions.csv",
                mime="text/csv", use_container_width=True,
            )

        table_df = latest_df.copy()
        table_df["prediction_status"] = table_df["prediction"].map(_prediction_status_text)
        table_df = table_df[["trans_num", "amt", "merchant", "category", "risk_score", "prediction_status", "rank", "transaction_created_at", "prediction_created_at"]]
        st.dataframe(
            _style_table(table_df, float(high_risk_threshold)),
            use_container_width=True, hide_index=True, height=520,
        )

    with tab_drill:
        c1, c2 = st.columns([0.95, 1.05], gap="medium")
        with c1:
            selected_trans_num = st.selectbox(
                "Transaction",
                options=latest_df["trans_num"].astype(str).tolist(),
                key="drilldown_transaction",
            )
            st.caption("Select a row from the current result set.")
        with c2:
            detail_df = get_transaction_detail(selected_trans_num, engine=get_engine())
            if detail_df.empty:
                st.info("No transaction detail found.")
            else:
                detail = detail_df.iloc[0].to_dict()
                summary_df = pd.DataFrame([{
                    "trans_num": detail.get("trans_num"),
                    "merchant": detail.get("merchant"),
                    "category": detail.get("category"),
                    "amt": detail.get("amt"),
                    "risk_score": detail.get("risk_score"),
                    "prediction": detail.get("prediction"),
                    "transaction_created_at": _format_bogota_datetime(detail.get("trans_date_trans_time")),
                    "prediction_created_at": _format_bogota_datetime(detail.get("prediction_created_at")),
                }])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
