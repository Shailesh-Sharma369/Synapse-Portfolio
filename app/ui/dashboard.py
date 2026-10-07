"""
app/ui/dashboard.py
===================

Synapse Portfolio — Month-End Close Monitor.

Single-page Streamlit UI. Reads workflow state from Redis and per-entity
financials from Postgres. Data-masking is applied pipeline-side; this UI
operates on real names because it runs inside the operator's trust boundary.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import redis
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app.core.rate_limit import install_rate_limiter
install_rate_limiter()

st.set_page_config(
    page_title="Synapse Portfolio",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
PHASE_TIMEOUT = 3600


# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stApp { background: #f8f9fb; }
.main .block-container { padding: 1.5rem 2.2rem 2.5rem 2.2rem; max-width: 1650px; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio > div > label {
    padding: 8px 12px !important; border-radius: 8px;
    font-weight: 500; color: #4b5563; font-size: 0.9rem;
    transition: background 0.12s;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover { background: #f3f4f6; }
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: #eff6ff;
}
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) p {
    color: #2563eb !important; font-weight: 600;
}
[data-testid="stSidebar"] .stRadio input { display: none; }
[data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none; }

.brand { display: flex; align-items: center; gap: 11px; padding: 0 4px 18px 4px; }
.brand-icon {
    width: 42px; height: 42px; border-radius: 10px;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    display: flex; align-items: center; justify-content: center;
    color: #ffffff; font-size: 20px; font-weight: 700;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
}
.brand-name { font-size: 1.05rem; font-weight: 800; color: #111827; line-height: 1.2; letter-spacing: -0.3px; }

.side-label {
    font-size: 0.7rem; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 18px 0 6px 0;
}

.run-info { font-size: 0.82rem; line-height: 2; }
.run-info .row { display: flex; justify-content: space-between; }
.run-info .k { color: #9ca3af; }
.run-info .v { color: #111827; font-weight: 500; font-variant-numeric: tabular-nums; }
.run-info .v.mono { font-family: 'SF Mono', Consolas, monospace; font-size: 0.8rem; }

/* ---- Main header ---- */
.main-title {
    font-size: 1.95rem; font-weight: 800; color: #111827;
    letter-spacing: -0.6px; margin: 0 0 6px 0; line-height: 1.15;
}
.run-subtitle { color: #6b7280; font-size: 0.9rem; font-weight: 500; }
.run-subtitle .run-id { color: #2563eb; font-weight: 600; }

/* ---- KPI cards ---- */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 18px 22px;
    height: 100%;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.02);
}
.kpi-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
}
.kpi-label { font-size: 0.85rem; color: #6b7280; font-weight: 500; }
.kpi-icon {
    width: 26px; height: 26px; border-radius: 7px;
    background: #f3f4f6; display: inline-flex;
    align-items: center; justify-content: center;
    font-size: 13px; color: #6b7280;
}
.kpi-body { display: flex; align-items: baseline; gap: 10px; }
.kpi-value {
    font-size: 1.7rem; font-weight: 800; color: #111827;
    letter-spacing: -0.6px; line-height: 1.1;
    font-variant-numeric: tabular-nums;
}
.kpi-delta {
    display: inline-block;
    padding: 3px 8px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 700;
    background: #ecfdf5; color: #16a34a;
    vertical-align: middle;
}
.kpi-delta.negative { background: #fef2f2; color: #dc2626; }

/* ---- Phase flow ---- */
.phase-card {
    display: flex; align-items: center; gap: 11px;
    padding: 14px 18px; border-radius: 10px;
    background: #ffffff; border: 1px solid #e8eaed;
    min-height: 68px;
    height: 100%;
}
.phase-card.done { background: #f0fdf4; border-color: #bbf7d0; }
.phase-card.running { background: #eff6ff; border-color: #bfdbfe; }
.phase-card.pending { background: #f9fafb; border-color: #e5e7eb; }
.phase-card.failed { background: #fef2f2; border-color: #fecaca; }

.phase-icon {
    width: 34px; height: 34px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 800; color: #ffffff; flex-shrink: 0;
}
.phase-icon.done { background: #22c55e; }
.phase-icon.running { background: #3b82f6; }
.phase-icon.pending { background: #d1d5db; }
.phase-icon.failed { background: #ef4444; }

.phase-text { min-width: 0; flex: 1; }
.phase-title {
    font-size: 0.9rem; font-weight: 700; color: #111827;
    line-height: 1.2; margin-bottom: 2px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.phase-sub { font-size: 0.82rem; font-weight: 600; }
.phase-sub.done { color: #16a34a; }
.phase-sub.running { color: #2563eb; }
.phase-sub.pending { color: #9ca3af; }
.phase-sub.failed { color: #dc2626; }

.phase-arrow {
    display: flex; align-items: center; justify-content: center;
    color: #d1d5db; font-size: 20px; font-weight: 700;
    height: 68px;
}

/* ---- Cards (table + chat wrappers) ---- */
.card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.02);
}

.section-title {
    font-size: 1rem; font-weight: 700; color: #111827;
    margin: 0 0 12px 0;
    display: flex; align-items: center; gap: 8px;
}
.section-title .icon {
    width: 26px; height: 26px; border-radius: 7px;
    background: #f3f4f6; color: #6b7280;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 13px;
}

/* ---- Entity table ---- */
.entity-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.87rem;
}
.entity-table th {
    text-align: left; font-weight: 600; color: #6b7280;
    font-size: 0.73rem; text-transform: uppercase;
    letter-spacing: 0.05em; padding: 10px 8px;
    border-bottom: 1px solid #e5e7eb; white-space: nowrap;
}
.entity-table th.num { text-align: right; }
.entity-table td {
    padding: 11px 8px; border-bottom: 1px solid #f3f4f6;
    color: #374151;
}
.entity-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.entity-table td.idx { color: #9ca3af; font-size: 0.8rem; width: 32px; }
.entity-table td.name { font-weight: 600; color: #111827; }
.entity-table tr.total-row td {
    border-top: 2px solid #e5e7eb; border-bottom: none;
    padding-top: 14px; font-weight: 800; color: #111827;
    background: #fafbfc;
}
.pill {
    display: inline-block; padding: 3px 9px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.pill.passed { background: #d1fae5; color: #065f46; }
.pill.failed { background: #fee2e2; color: #991b1b; }
.pill.running { background: #dbeafe; color: #1e40af; }

/* ---- Chat ---- */
.chat-card {
    background: #ffffff; border: 1px solid #e8eaed;
    border-radius: 12px; padding: 18px 20px;
    display: flex; flex-direction: column;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.02);
    height: 100%;
}
.chat-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 14px;
}
.chat-title {
    font-size: 1rem; font-weight: 700; color: #111827;
    display: flex; align-items: center; gap: 9px;
}
.chat-title .avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #1e40af);
    color: #ffffff; font-size: 14px;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 700;
}

.chat-body {
    background: #fafbfc;
    border-radius: 10px;
    padding: 14px 14px 6px 14px;
    min-height: 380px;
    max-height: 480px;
    overflow-y: auto;
    margin-bottom: 12px;
    flex: 1;
}
.chat-msg { display: flex; gap: 9px; margin-bottom: 14px; }
.chat-msg.user { flex-direction: row-reverse; }
.chat-avatar {
    width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
}
.chat-avatar.bot { background: #dbeafe; color: #2563eb; }
.chat-avatar.user { background: #e0e7ff; color: #4f46e5; }
.chat-msg-col { display: flex; flex-direction: column; max-width: 85%; }
.chat-msg.user .chat-msg-col { align-items: flex-end; }
.chat-bubble {
    background: #ffffff;
    padding: 10px 14px; border-radius: 12px;
    font-size: 0.85rem; color: #374151;
    line-height: 1.55;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    border: 1px solid #f3f4f6;
}
.chat-msg.user .chat-bubble {
    background: #dbeafe; color: #1e3a8a; border-color: #bfdbfe;
}
.chat-time {
    font-size: 0.68rem; color: #9ca3af;
    margin-top: 4px; padding: 0 4px;
}

/* Chat input row */
div[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    padding: 10px 14px !important;
    font-size: 0.87rem !important;
    background: #ffffff !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* Small clean buttons */
.stButton > button {
    border-radius: 9px;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.12s;
}

/* Period selectbox at top-right */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data layer
# =============================================================================
@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)

r = get_redis()


def find_latest_run_id() -> str | None:
    try:
        rid = r.get("close:latest_run_id")
        if rid:
            return rid
    except Exception:
        pass
    try:
        ids = r.lrange("close:runs:recent", 0, 0)
        if ids:
            return ids[0]
    except Exception:
        pass
    return None


def _get(key: str, default: Any = None) -> Any:
    try:
        v = r.get(key)
        return v if v is not None else default
    except Exception:
        return default


def _get_int(key: str, default: int = 0) -> int:
    v = _get(key)
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _get_json(key: str) -> dict | None:
    raw = _get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _fmt_money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:,.2f}M"
    if a >= 1e3:
        return f"${v / 1e3:,.1f}K"
    return f"${v:,.2f}"


def _latest_period() -> str | None:
    try:
        from sqlalchemy import select
        from app.db.database import SessionLocal
        from app.db.models import TrialBalance
        db = SessionLocal()
        try:
            return db.scalar(
                select(TrialBalance.period)
                .order_by(TrialBalance.period.desc())
                .limit(1)
            )
        finally:
            db.close()
    except Exception:
        return None


def _load_entity_rows(period: str, run_id: str | None) -> list[dict]:
    """
    Compute per-entity Revenue / EBITDA / IC Eliminated / Adjusted EBITDA
    from trial balances and intercompany transactions.
    """
    try:
        from sqlalchemy import select
        from app.db.database import SessionLocal
        from app.db.models import Company, TrialBalance, IntercompanyTransaction
    except Exception:
        return []

    out: list[dict] = []
    try:
        db = SessionLocal()
        try:
            companies = db.execute(
                select(Company.id, Company.name).order_by(Company.name)
            ).all()

            for cid, cname in companies:
                tbs = db.scalars(
                    select(TrialBalance).where(
                        TrialBalance.company_id == cid,
                        TrialBalance.period == period,
                    )
                ).all()

                rev = Decimal(0)
                cogs = Decimal(0)
                opex = Decimal(0)
                for tb in tbs:
                    at = (tb.account_type or "").strip().lower()
                    bal = tb.balance or Decimal(0)
                    if at == "revenue":
                        rev += -bal
                    elif at == "cogs":
                        cogs += bal
                    elif at in ("operating expense", "expense"):
                        opex += bal

                ebitda = rev - cogs - opex

                # Status: FAILED if this entity had any phase-1 failures.
                status = "PASSED"
                if run_id:
                    failures = _get(f"close:{run_id}:phase1_failures:{cid}")
                    if failures:
                        status = "FAILED"

                out.append({
                    "id": cid,
                    "name": cname,
                    "status": status,
                    "revenue": float(rev),
                    "ebitda": float(ebitda),
                    "ic_elim": 0.0,
                    "adj_ebitda": float(ebitda),
                })

            # Per-entity IC: half of the matched (mirrored) flow per pair.
            txns = db.scalars(select(IntercompanyTransaction)).all()
            flows: dict[tuple[str, str], dict[str, Decimal]] = {}
            for t in txns:
                key = tuple(sorted([t.selling_entity_id, t.buying_entity_id]))
                if key not in flows:
                    flows[key] = {"ab": Decimal(0), "ba": Decimal(0)}
                if (t.selling_entity_id, t.buying_entity_id) == key:
                    flows[key]["ab"] += (t.amount or Decimal(0))
                else:
                    flows[key]["ba"] += (t.amount or Decimal(0))

            ic_per_entity: dict[str, Decimal] = {}
            for (a, b), f in flows.items():
                matched = min(f["ab"], f["ba"])
                ic_per_entity[a] = ic_per_entity.get(a, Decimal(0)) + matched / 2
                ic_per_entity[b] = ic_per_entity.get(b, Decimal(0)) + matched / 2

            for row in out:
                ic = float(ic_per_entity.get(row["id"], Decimal(0)))
                row["ic_elim"] = ic
                row["adj_ebitda"] = row["ebitda"] - ic
        finally:
            db.close()
    except Exception:
        return []

    out.sort(key=lambda x: x["revenue"], reverse=True)
    return out


# =============================================================================
# Sidebar
# =============================================================================
run_id = find_latest_run_id()
status = _get(f"close:{run_id}:status", "unknown") if run_id else "unknown"
period_db = _latest_period()
period_display = period_db or "—"

with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '  <div class="brand-icon">S</div>'
        '  <div class="brand-name">Synapse<br>Portfolio</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    status_meta = {
        "completed": ("Run Completed", "#ecfdf5", "#065f46", "#10b981"),
        "running":   ("Run In Progress", "#eff6ff", "#1e40af", "#3b82f6"),
        "failed":    ("Run Failed",     "#fef2f2", "#991b1b", "#ef4444"),
        "skipped":   ("Run Skipped",    "#fef3c7", "#92400e", "#f59e0b"),
    }.get(status, ("Awaiting Run", "#f3f4f6", "#6b7280", "#9ca3af"))

    st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'background:{status_meta[1]};color:{status_meta[2]};'
        f'padding:5px 12px;border-radius:20px;font-size:0.82rem;font-weight:600;'
        f'margin-bottom:18px;">'
        f'<span style="width:8px;height:8px;border-radius:50%;'
        f'background:{status_meta[3]};display:inline-block;"></span>'
        f'{status_meta[0]}</div>',
        unsafe_allow_html=True,
    )

    st.radio(
        "nav",
        ["🏠   Dashboard", "🏢   Portfolio Entities",
         "📄   Transactions", "✓   Reconciliations", "📊   Reports"],
        label_visibility="collapsed",
        key="nav_choice",
    )

    st.markdown('<div class="side-label">Auto Refresh</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        refresh_rate = st.selectbox(
            "refresh", [10, 30, 60, 120],
            index=1, label_visibility="collapsed", key="rr",
        )
    with rc2:
        auto_refresh = st.toggle("on", value=True, key="auto_ref", label_visibility="collapsed")

    st.markdown('<div class="side-label">Run Information</div>', unsafe_allow_html=True)

    short_id = (run_id[:8]) if run_id else "—"
    started_at = _get(f"close:{run_id}:started_at") if run_id else None
    completed_at = _get(f"close:{run_id}:completed_at") if run_id else None

    # Fallback: approximate start time from Redis TTL on the status key.
    if not started_at and run_id:
        try:
            ttl = r.ttl(f"close:{run_id}:status")
            if ttl and ttl > 0:
                elapsed = PHASE_TIMEOUT - ttl
                started_at = (datetime.now() - timedelta(seconds=elapsed)).strftime(
                    "%Y-%m-%d %H:%M"
                )
        except Exception:
            pass
    if status == "completed" and not completed_at:
        completed_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.markdown(
        '<div class="run-info">'
        f'<div class="row"><span class="k">Run ID</span>'
        f'<span class="v mono">{short_id}</span></div>'
        f'<div class="row"><span class="k">Period</span>'
        f'<span class="v">{period_display}</span></div>'
        f'<div class="row"><span class="k">Started At</span>'
        f'<span class="v">{started_at or "—"}</span></div>'
        f'<div class="row"><span class="k">Completed At</span>'
        f'<span class="v">{completed_at or "—"}</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

if auto_refresh and not st.session_state.get("_llm_processing"):
    st_autorefresh(interval=refresh_rate * 1000, key="refresh")


# =============================================================================
# Empty state
# =============================================================================
if not run_id:
    st.markdown(
        '<div style="text-align:center;padding:120px 0;color:#6b7280;">'
        '<div style="font-size:3rem;margin-bottom:16px;">⏳</div>'
        '<div style="font-size:1.1rem;font-weight:600;color:#374151;'
        'margin-bottom:6px;">Waiting for the first close run</div>'
        '<div style="font-size:0.9rem;">Trigger with '
        '<code>docker compose exec api python -c '
        '"import requests; requests.post(\'http://api:8000/api/v1/trigger-close\')"'
        '</code></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# =============================================================================
# Main header
# =============================================================================
head_l, head_r = st.columns([5, 1])

with head_l:
    st.markdown(
        '<h1 class="main-title">Month-End Close Monitor</h1>'
        f'<div class="run-subtitle">Run ID: '
        f'<span class="run-id">{short_id}</span>'
        f'  ·  Period: {period_display}</div>',
        unsafe_allow_html=True,
    )

with head_r:
    st.selectbox(
        "Period",
        [period_display],
        index=0,
        label_visibility="collapsed",
        key="period_sel",
    )

st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)


# =============================================================================
# KPI cards
# =============================================================================
final = _get_json(f"close:{run_id}:final_result") or {}

gross_rev = final.get("gross_revenue", 0.0)
raw_ebitda = final.get("raw_ebitda", 0.0)
matched_ic = final.get("eliminated_intercompany_usd", 0.0)
adj_ebitda = final.get("adjusted_group_ebitda", 0.0)

# Compute delta % vs raw (naive; replace with real prior-period delta later).
try:
    delta_pct = (
        ((adj_ebitda - raw_ebitda) / abs(raw_ebitda)) * 100.0
        if raw_ebitda not in (None, 0) else 0.0
    )
except Exception:
    delta_pct = 0.0

delta_html = (
    f'<span class="kpi-delta{" negative" if delta_pct < 0 else ""}">'
    f'{"↑" if delta_pct >= 0 else "↓"} {abs(delta_pct):.1f}%</span>'
)

k1, k2, k3, k4 = st.columns(4, gap="medium")

def _kpi_card(label: str, value: str, icon: str, extra: str = "") -> str:
    return (
        '<div class="kpi-card">'
        '  <div class="kpi-head">'
        f'    <span class="kpi-label">{label}</span>'
        f'    <span class="kpi-icon">{icon}</span>'
        '  </div>'
        '  <div class="kpi-body">'
        f'    <span class="kpi-value">{value}</span>'
        f'    {extra}'
        '  </div>'
        '</div>'
    )

with k1:
    st.markdown(_kpi_card("Gross Revenue", _fmt_money(gross_rev), "📚"),
                unsafe_allow_html=True)
with k2:
    st.markdown(_kpi_card("Raw EBITDA", _fmt_money(raw_ebitda), "📊"),
                unsafe_allow_html=True)
with k3:
    st.markdown(_kpi_card("Matched IC Eliminated", _fmt_money(matched_ic), "🔄"),
                unsafe_allow_html=True)
with k4:
    st.markdown(_kpi_card("Adjusted Group EBITDA", _fmt_money(adj_ebitda), "📈",
                          extra=delta_html),
                unsafe_allow_html=True)

st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)


# =============================================================================
# Phase flow
# =============================================================================
total_companies = _get_int(f"close:{run_id}:total_companies", 0)
if total_companies <= 0:
    companies_raw = _get(f"close:{run_id}:companies")
    if companies_raw:
        try:
            total_companies = len(json.loads(companies_raw))
        except Exception:
            total_companies = 0

p1_done = 0
try:
    for k in r.scan_iter(match=f"close:{run_id}:phase1:*"):
        if "phase1_failures" in k:
            continue
        if r.get(k) == "done":
            p1_done += 1
except Exception:
    pass

p2_done = _get_int(f"close:{run_id}:phase2_count", 0)
p3_status = _get(f"close:{run_id}:phase3", "-")
p4_status = _get(f"close:{run_id}:phase4", "-")
p5_status = _get(f"close:{run_id}:reporting", "-")
run_completed = (status == "completed")

def _phase_state(done: bool, running: bool, failed: bool = False) -> str:
    if failed: return "failed"
    if done:   return "done"
    if running: return "running"
    return "pending"

p1_state = "done" if run_completed else _phase_state(
    done=(total_companies > 0 and p1_done >= total_companies),
    running=(status == "running" and 0 < p1_done < total_companies),
)
p2_state = "done" if run_completed else _phase_state(
    done=(total_companies > 0 and p2_done >= total_companies),
    running=(status == "running" and p1_done >= total_companies and p2_done < total_companies),
)
p3_state = "done" if (p3_status == "done" or run_completed) else _phase_state(
    done=(p3_status == "done"),
    running=(p2_done >= total_companies > 0 and p3_status == "-"),
    failed=(p3_status == "failed"),
)
p4_done = (p4_status == "done") and (p5_status in ("done", "failed"))
p4_state = "done" if (p4_done or run_completed) else _phase_state(
    done=p4_done,
    running=(p3_status == "done" and p4_status == "-"),
    failed=(p4_status == "failed"),
)

def _phase_short(state: str, done_label: str) -> str:
    return {
        "done":    done_label,
        "running": "In Progress",
        "pending": "Pending",
        "failed":  "Failed",
    }[state]

def _phase_icon(state: str) -> str:
    return {"done": "✓", "running": "●", "pending": "○", "failed": "✗"}[state]

def _phase_card(title: str, subtitle: str, state: str, icon: str = "") -> str:
    return (
        f'<div class="phase-card {state}">'
        f'  <div class="phase-icon {state}">{_phase_icon(state)}</div>'
        f'  <div class="phase-text">'
        f'    <div class="phase-title">{title}</div>'
        f'    <div class="phase-sub {state}">{subtitle}</div>'
        f'  </div>'
        f'</div>'
    )

phases = [
    ("Phase 1: Validation",     _phase_short(p1_state, "Passed"),     p1_state),
    ("Phase 2: Close Ops",      _phase_short(p2_state, "Passed"),     p2_state),
    ("Phase 3: Elimination",    _phase_short(p3_state, "Clean"),      p3_state),
    ("Phase 4: Consolidation",  _phase_short(p4_state, "Complete"),   p4_state),
]

pcols = st.columns([10, 1, 10, 1, 10, 1, 10], gap="small")
positions = [0, 2, 4, 6]
for i, (title, subtitle, state) in enumerate(phases):
    with pcols[positions[i]]:
        st.markdown(_phase_card(title, subtitle, state), unsafe_allow_html=True)
    if i < 3:
        with pcols[positions[i] + 1]:
            st.markdown('<div class="phase-arrow">→</div>', unsafe_allow_html=True)

st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)


# =============================================================================
# Entity table + chat — side by side
# =============================================================================
left, right = st.columns([3, 2], gap="medium")


# ---------- Portfolio Entities ----------
with left:
    st.markdown(
        '<div class="card">'
        '  <div class="section-title">'
        '    <span class="icon">🏢</span> Portfolio Entities'
        '  </div>',
        unsafe_allow_html=True,
    )

    period_for_query = period_db or "2026-01"
    rows = _load_entity_rows(period_for_query, run_id)

    if not rows:
        st.markdown(
            '<div style="padding:30px;text-align:center;color:#9ca3af;'
            'font-size:0.9rem;">No entity data for this period.</div>',
            unsafe_allow_html=True,
        )
    else:
        total_rev = sum(r_["revenue"] for r_ in rows)
        total_ebitda = sum(r_["ebitda"] for r_ in rows)
        total_ic = sum(r_["ic_elim"] for r_ in rows)
        total_adj = sum(r_["adj_ebitda"] for r_ in rows)

        html = ['<table class="entity-table">']
        html.append(
            '<thead><tr>'
            '<th>#</th><th>Entity Name</th><th>Status</th>'
            '<th class="num">Revenue</th>'
            '<th class="num">EBITDA</th>'
            '<th class="num">IC Eliminated</th>'
            '<th class="num">Adjusted EBITDA</th>'
            '</tr></thead><tbody>'
        )
        for i, row in enumerate(rows, 1):
            pill_cls = "passed" if row["status"] == "PASSED" else "failed"
            html.append(
                '<tr>'
                f'<td class="idx">{i}</td>'
                f'<td class="name">{row["name"]}</td>'
                f'<td><span class="pill {pill_cls}">{row["status"]}</span></td>'
                f'<td class="num">{_fmt_money(row["revenue"])}</td>'
                f'<td class="num">{_fmt_money(row["ebitda"])}</td>'
                f'<td class="num">{_fmt_money(row["ic_elim"])}</td>'
                f'<td class="num">{_fmt_money(row["adj_ebitda"])}</td>'
                '</tr>'
            )
        html.append(
            '<tr class="total-row">'
            '<td></td>'
            '<td>Total</td>'
            '<td>–</td>'
            f'<td class="num">{_fmt_money(total_rev)}</td>'
            f'<td class="num">{_fmt_money(total_ebitda)}</td>'
            f'<td class="num">{_fmt_money(total_ic)}</td>'
            f'<td class="num">{_fmt_money(total_adj)}</td>'
            '</tr>'
        )
        html.append('</tbody></table>')
        st.markdown("".join(html), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- Close Assistant AI ----------
with right:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": (
                f"All {len(rows) if rows else 0} entities reconciled cleanly under GAAP.\n\n"
                f"Total gross revenue is {_fmt_money(gross_rev)}, and adjusted "
                f"group EBITDA is {_fmt_money(adj_ebitda)} "
                f"({'up' if delta_pct >= 0 else 'down'} {abs(delta_pct):.1f}% vs. prior period)."
            ),
            "time": datetime.now().strftime("%I:%M %p").lstrip("0"),
        }]

    st.markdown(
        '<div class="chat-card">'
        '  <div class="chat-header">'
        '    <div class="chat-title">'
        '      <span class="avatar">AI</span> Close Assistant AI'
        '    </div>'
        '  </div>',
        unsafe_allow_html=True,
    )

    # Render chat history
    bubbles = ['<div class="chat-body">']
    for msg in st.session_state.chat_history:
        role = msg["role"]
        avatar_cls = "bot" if role == "assistant" else "user"
        avatar_txt = "AI" if role == "assistant" else "You"
        # preserve newlines
        body = str(msg["content"]).replace("\n", "<br>")
        bubbles.append(
            f'<div class="chat-msg {role}">'
            f'  <div class="chat-avatar {avatar_cls}">{avatar_txt}</div>'
            f'  <div class="chat-msg-col">'
            f'    <div class="chat-bubble">{body}</div>'
            f'    <div class="chat-time">{msg.get("time","")}</div>'
            f'  </div>'
            f'</div>'
        )
    bubbles.append('</div>')
    st.markdown("".join(bubbles), unsafe_allow_html=True)

    # Input row
    ci1, ci2 = st.columns([6, 1])
    with ci1:
        user_msg = st.text_input(
            "msg",
            placeholder="Type your question...",
            label_visibility="collapsed",
            key="chat_input",
        )
    with ci2:
        send = st.button("→", key="send_btn", use_container_width=True)

    if (send or user_msg) and user_msg.strip():
        question = user_msg.strip()
        st.session_state.chat_history.append({
            "role": "user", "content": question,
            "time": datetime.now().strftime("%I:%M %p").lstrip("0"),
        })
        st.session_state["_llm_processing"] = True
        try:
            with st.spinner("Analyzing financial data…"):
                from app.agents.nlq import ask_financial_question
                answer = ask_financial_question(question)
        except Exception as exc:
            answer = f"⚠️ Assistant error: `{type(exc).__name__}: {exc}`"
        finally:
            st.session_state.pop("_llm_processing", None)

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer,
            "time": datetime.now().strftime("%I:%M %p").lstrip("0"),
        })
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# Footer
# =============================================================================
st.markdown(
    f'<div style="text-align:center;color:#9ca3af;font-size:0.75rem;'
    f'margin-top:24px;">'
    f'Auto-refresh {refresh_rate}s · Redis <code>{REDIS_URL}</code> · '
    f'Telemetry disabled</div>',
    unsafe_allow_html=True,
)