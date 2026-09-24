"""
Zero-Click Executive Monitor — Apex Capital Month-End Close.

Layout:
    ┌─── Sidebar ───┬────── Main ──────┬─── Assistant ───┐
    │ Settings      │  Header          │  Close          │
    │ Theme accent  │  4 Big Phases    │  Assistant      │
    │ Refresh rate  │  Agentic cards   │  (chat)         │
    │               │  Consolidated    │                 │
    │               │  Drill-Down      │                 │
    └───────────────┴──────────────────┴─────────────────┘
"""

from __future__ import annotations

import json
import os
from typing import Any

import redis
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from app.core.rate_limit import install_rate_limiter
install_rate_limiter()

st.set_page_config(
    page_title="Month-End Close Monitor",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


# ============================================================================
# SESSION STATE
# ============================================================================
if "theme_accent" not in st.session_state:
    st.session_state.theme_accent = "Light"
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

ACCENT = {
    "Light": {
        "primary": "#2563eb", "primary_soft": "rgba(37, 99, 235, 0.10)",
        "success": "#16a34a", "success_soft": "rgba(22, 163, 74, 0.10)",
        "danger": "#dc2626", "danger_soft": "rgba(220, 38, 38, 0.10)",
        "warning": "#d97706", "neutral": "#6b7280",
        "chip_bg": "#F1F3F5", "chip_border": "#E9ECEF",
    },
    "Dark": {
        "primary": "#58a6ff", "primary_soft": "rgba(88, 166, 255, 0.15)",
        "success": "#3fb950", "success_soft": "rgba(63, 185, 80, 0.15)",
        "danger": "#f85149", "danger_soft": "rgba(248, 81, 73, 0.15)",
        "warning": "#d29922", "neutral": "#8b949e",
        "chip_bg": "rgba(255, 255, 255, 0.05)", "chip_border": "rgba(255, 255, 255, 0.12)",
    },
}
A = ACCENT[st.session_state.theme_accent]


# ============================================================================
# CSS
# ============================================================================
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        font-feature-settings: "tnum" 1, "ss01" 1;
    }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 1.2rem; max-width: 1600px; }}
    #MainMenu, footer {{ visibility: hidden; }}

    .stProgress > div > div > div > div {{
        background-color: {A["primary"]} !important; border-radius: 8px;
    }}
    .stProgress > div > div > div {{
        background-color: {A["primary_soft"]} !important; border-radius: 8px;
    }}
    div[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
    .stButton > button:hover {{ border-color: {A["primary"]} !important; color: {A["primary"]} !important; }}
    div[data-testid="stChatMessage"] {{ border-radius: 10px; border-left: 3px solid {A["primary_soft"]}; }}

    .status-chip {{
        display: inline-block; background: {A["chip_bg"]}; border: 1px solid {A["chip_border"]};
        border-radius: 20px; padding: 3px 12px; font-size: 0.82rem; font-weight: 500;
    }}

    /* BIG PHASE CARDS */
    .phase-card {{
        border-radius: 14px; padding: 20px 22px; height: 100%;
        border: 1px solid {A["chip_border"]};
        background: {A["chip_bg"]};
        transition: all 0.2s ease;
    }}
    .phase-card.done {{
        background: {A["success_soft"]};
        border-color: {A["success"]};
    }}
    .phase-card.running {{
        background: {A["primary_soft"]};
        border-color: {A["primary"]};
    }}
    .phase-card.pending {{
        background: {A["chip_bg"]};
        border-color: {A["chip_border"]};
        opacity: 0.75;
    }}
    .phase-card.failed {{
        background: {A["danger_soft"]};
        border-color: {A["danger"]};
    }}
    .phase-icon {{
        font-size: 32px; line-height: 1; margin-bottom: 10px;
    }}
    .phase-title {{
        font-size: 17px; font-weight: 700; margin-bottom: 4px;
        letter-spacing: -0.2px;
    }}
    .phase-subtitle {{
        font-size: 12.5px; opacity: 0.7; margin-bottom: 14px;
        text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .phase-status {{
        display: inline-block; padding: 5px 12px; border-radius: 8px;
        font-size: 13px; font-weight: 600; letter-spacing: 0.02em;
    }}
    .phase-status.done {{
        background: {A["success"]}; color: white;
    }}
    .phase-status.running {{
        background: {A["primary"]}; color: white;
    }}
    .phase-status.pending {{
        background: {A["chip_bg"]}; color: {A["neutral"]};
        border: 1px solid {A["chip_border"]};
    }}
    .phase-status.failed {{
        background: {A["danger"]}; color: white;
    }}
    .phase-progress-label {{
        font-size: 13px; margin-top: 14px; opacity: 0.85;
    }}

    div[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 600; }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.75;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# REDIS
# ============================================================================
@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)

r = get_redis()


# ============================================================================
# HELPERS
# ============================================================================
def find_latest_run_id() -> str | None:
    try:
        explicit = r.get("close:latest_run_id")
        if explicit:
            return explicit
    except Exception:
        pass
    try:
        ids = r.lrange("close:runs:recent", 0, 0)
        if ids:
            return ids[0]
    except Exception:
        pass
    return None


def _get(key: str, default: str | None = None) -> str | None:
    return r.get(key) or default


def _get_int(key: str, default: int = 0) -> int:
    v = r.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _get_json(key: str) -> dict[str, Any] | None:
    raw = r.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _fmt_usd(v: float | int | None, compact: bool = False) -> str:
    if v is None:
        return "—"
    if compact:
        a = abs(v)
        if a >= 1_000_000_000:
            return f"${v / 1_000_000_000:,.2f}B"
        if a >= 1_000_000:
            return f"${v / 1_000_000:,.2f}M"
        if a >= 1_000:
            return f"${v / 1_000:,.1f}K"
    return f"${v:,.2f}"


def _latest_period_from_db() -> str | None:
    try:
        from sqlalchemy import select
        from app.db.database import SessionLocal
        from app.db.models import TrialBalance
        db = SessionLocal()
        try:
            return db.scalar(
                select(TrialBalance.period).order_by(TrialBalance.period.desc()).limit(1)
            )
        finally:
            db.close()
    except Exception:
        return None


def _discover_company_ids(run_id: str) -> list[str]:
    """
    Return the list of companies in THIS run.

    Priority:
      1. Explicit list written by the orchestrator (close:{run_id}:companies)
         — this is what makes the UI dynamic when CLOSE_COMPANIES filters
         the run to a subset.
      2. Discovered from phase1 keys (fallback).
      3. Full portfolio from DB (last resort, only if Redis is empty).
    """
    # 1. Explicit list (set by orchestrator at run start)
    explicit = r.get(f"close:{run_id}:companies")
    if explicit:
        try:
            parsed = json.loads(explicit)
            if isinstance(parsed, list) and parsed:
                return sorted(parsed)
        except Exception:
            pass

    # 2. Discover from phase1 keys
    ids: set[str] = set()
    try:
        for k in r.scan_iter(match=f"close:{run_id}:phase1:*"):
            if "phase1_failures" in k:
                continue
            suffix = k.rsplit(":", 1)[-1]
            if suffix and suffix != "count":
                ids.add(suffix)
    except Exception:
        pass
    if ids:
        return sorted(ids)

    # 3. Last resort: full portfolio from DB
    try:
        from sqlalchemy import select
        from app.db.database import SessionLocal
        from app.db.models import Company
        db = SessionLocal()
        try:
            ids = {cid for (cid,) in db.execute(select(Company.id)).all()}
        finally:
            db.close()
    except Exception:
        pass
    return sorted(ids)


# ---- Cached tools ----
@st.cache_data(ttl=300, show_spinner=False)
def _tool_tb(cid: str, p: str) -> dict:
    from app.agents.validator import analyze_trial_balance
    return analyze_trial_balance(cid, p)

@st.cache_data(ttl=300, show_spinner=False)
def _tool_variance(cid: str, p: str) -> dict:
    from app.agents.variance import analyze_variances
    return analyze_variances(cid, p)

@st.cache_data(ttl=300, show_spinner=False)
def _tool_cash(cid: str, p: str) -> dict:
    from app.agents.cash_flow import reconcile_cash_flow
    return reconcile_cash_flow(cid, p)

@st.cache_data(ttl=300, show_spinner=False)
def _tool_accruals(cid: str, p: str) -> dict:
    from app.agents.accrual_verification import verify_accruals
    return verify_accruals(cid, p)

@st.cache_data(ttl=300, show_spinner=False)
def _tool_revrec(cid: str, p: str) -> dict:
    from app.agents.revenue_recognition import verify_revenue_recognition
    return verify_revenue_recognition(cid, p)

@st.cache_data(ttl=300, show_spinner=False)
def _tool_expenses(cid: str, p: str) -> dict:
    from app.agents.expense_categorization import categorize_expenses
    return categorize_expenses(cid, p)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    col_toggle, col_theme_mini = st.columns([4, 1])
    with col_toggle:
        if st.session_state.sidebar_collapsed:
            if st.button("☰", key="sbx_exp", help="Expand"):
                st.session_state.sidebar_collapsed = False
                st.rerun()
        else:
            if st.button("◀  Collapse", key="sbx_col", help="Collapse"):
                st.session_state.sidebar_collapsed = True
                st.rerun()
    with col_theme_mini:
        if st.button("🌓", key="theme_btn", help="Toggle theme"):
            st.session_state.theme_accent = (
                "Dark" if st.session_state.theme_accent == "Light" else "Light"
            )
            st.rerun()
    st.divider()

    if st.session_state.sidebar_collapsed:
        refresh_rate = st.session_state.get("refresh_rate", 3)
    else:
        st.markdown("## Settings")
        st.markdown("**Accent theme**")
        tc = st.radio(
            "thm", options=["Light", "Dark"],
            index=0 if st.session_state.theme_accent == "Light" else 1,
            label_visibility="collapsed", key="thm_w",
        )
        if tc != st.session_state.theme_accent:
            st.session_state.theme_accent = tc
            st.rerun()

        st.markdown("**Auto-refresh (s)**")
        refresh_rate = st.selectbox(
            "rr", options=[3, 5, 10, 30], index=0,
            label_visibility="collapsed", key="refresh_rate",
        )
        st.divider()
        st.markdown("### Welcome, Controller 👨‍💼")
        st.caption("Auto-discovers the latest close run.")
        try:
            recent = r.lrange("close:runs:recent", 0, 4)
            if recent:
                st.markdown("**Recent runs**")
                for rid in recent:
                    st.caption(f"• `{rid[:8]}…`")
        except Exception:
            pass


# ---- Auto-refresh (paused during LLM chat) ----
if not st.session_state.get("_llm_processing"):
    st_autorefresh(interval=refresh_rate * 1000, key="monitor_refresh")


# ============================================================================
# MAIN LAYOUT
# ============================================================================
main_col, assistant_col = st.columns([7, 3], gap="large")


# ============================================================================
# LEFT — MAIN MONITOR
# ============================================================================
with main_col:
    st.markdown("<h2 style='margin-bottom:0;'>✅ Month-End Close Monitor</h2>", unsafe_allow_html=True)

    run_id = find_latest_run_id()
    if not run_id:
        st.info("⏳ Waiting for the first close run to start…")
        st.caption("Celery Beat fires a close daily at 9 AM. No manual trigger needed.")
        st.stop()

    status = _get(f"close:{run_id}:status", "unknown")
    total_companies = _get_int(f"close:{run_id}:total_companies", 0)
    if total_companies <= 0:
        _companies_raw = r.get(f"close:{run_id}:companies")
        if _companies_raw:
            try:
                total_companies = len(json.loads(_companies_raw))
            except Exception:
                total_companies = 0
    if total_companies <= 0:
        total_companies = len(_discover_company_ids(run_id))

    # ---- Header ----
    st.markdown(
        f"<div style='font-size:0.9rem; margin-top:6px; opacity:0.85;'>"
        f"Run ID: <code>{run_id}</code> &nbsp;·&nbsp; "
        f"Status: <span class='status-chip'>{status}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ========================================================================
    # 4 BIG PHASE CARDS
    # ========================================================================
    st.markdown("### Close Progress")

    # Phase 1: Parallel per-company (TB, Variance, Cash Flow)
    p1_done = 0
    for k in r.scan_iter(match=f"close:{run_id}:phase1:*"):
        if k.startswith(f"close:{run_id}:phase1_failures"):
            continue
        if r.get(k) == "done":
            p1_done += 1

    # Phase 2: Sequential per-company (Accrual, RevRec, Expense)
    p2_done = _get_int(f"close:{run_id}:phase2_count", 0)

    # Phase 3: Intercompany
    p3_status = _get(f"close:{run_id}:phase3", "-")
    p3_payload = _get_json(f"close:{run_id}:phase3:result")

    # Phase 4: Consolidation + Reporting
    p4_status = _get(f"close:{run_id}:phase4", "-")
    p5_status = _get(f"close:{run_id}:reporting", "-")

    def _phase_state(done_bool: bool, running_bool: bool, failed: bool = False) -> str:
        if failed:
            return "failed"
        if done_bool:
            return "done"
        if running_bool:
            return "running"
        return "pending"

    def _phase_label(state: str) -> str:
        return {
            "done": "✓ Done",
            "running": "● Running",
            "pending": "○ Pending",
            "failed": "✗ Failed",
        }[state]

        # If the run has reached 'completed', every phase 1–4 must render as
    # done regardless of whether individual Redis keys are still present.
    # Redis TTL expiry mid-run should never lie about completion state.
    run_completed = (status == "completed")

    # Phase 1 state
    p1_state = "done" if run_completed else _phase_state(
        done_bool=(total_companies > 0 and p1_done >= total_companies),
        running_bool=(status == "running" and total_companies > 0 and p1_done < total_companies),
    )
    # Phase 2 state
    p2_state = "done" if run_completed else _phase_state(
        done_bool=(total_companies > 0 and p2_done >= total_companies),
        running_bool=(
            status == "running"
            and total_companies > 0
            and p1_done >= total_companies
            and p2_done < total_companies
        ),
    )
    # Phase 3 state
    p3_state = "done" if (p3_status == "done" or run_completed) else _phase_state(
        done_bool=(p3_status == "done"),
        running_bool=(
            total_companies > 0
            and p2_done > 0
            and p2_done >= total_companies
            and p3_status == "-"
        ),
        failed=(p3_status == "failed"),
    )
    # Phase 4 state
    p4_done_bool = (p4_status == "done") and (p5_status in ("done", "failed"))
    p4_state = "done" if (p4_done_bool or run_completed) else _phase_state(
        done_bool=p4_done_bool,
        running_bool=(p3_status == "done" and p4_status == "-"),
        failed=(p4_status == "failed"),
    )

    phases_data = [
        {
            "icon": "📋",
            "title": "Phase 1 — Validation",
            "subtitle": "TB · Variance · Cash Flow",
            "state": p1_state,
            "progress": (p1_done, total_companies),
            "detail": f"{p1_done} of {total_companies} companies validated",
        },
        {
            "icon": "📊",
            "title": "Phase 2 — Close Ops",
            "subtitle": "Accrual · RevRec · Expense",
            "state": p2_state,
            "progress": (p2_done, total_companies),
            "detail": f"{p2_done} of {total_companies} companies processed",
        },
        {
            "icon": "🔁",
            "title": "Phase 3 — Elimination",
            "subtitle": "Intercompany · Asymmetry",
            "state": p3_state,
            "progress": None,
            "detail": (
                f"Matched IC ${p3_payload.get('matched_intercompany_usd', 0):,.0f} · "
                f"Asymmetry ${p3_payload.get('total_asymmetry_usd', 0):,.0f}"
                if p3_payload else "Awaiting Phase 2 completion"
            ),
        },
        {
            "icon": "🏛️",
            "title": "Phase 4 — Consolidation",
            "subtitle": "Group P&L · Reporting",
            "state": p4_state,
            "progress": None,
            "detail": (
                f"Adjusted EBITDA ${(_get_json(f'close:{run_id}:final_result') or {}).get('adjusted_group_ebitda', 0):,.0f} · "
                f"Emails {p5_status}"
                if _get_json(f"close:{run_id}:final_result") else
                "Awaiting Phase 3 completion"
            ),
        },
    ]

    pc1, pc2, pc3, pc4 = st.columns(4, gap="medium")
    for col, p in zip([pc1, pc2, pc3, pc4], phases_data):
        with col:
            st.markdown(
                f"<div class='phase-card {p['state']}'>"
                f"  <div class='phase-icon'>{p['icon']}</div>"
                f"  <div class='phase-title'>{p['title']}</div>"
                f"  <div class='phase-subtitle'>{p['subtitle']}</div>"
                f"  <div class='phase-status {p['state']}'>{_phase_label(p['state'])}</div>"
                f"  <div class='phase-progress-label'>{p['detail']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if p["progress"] is not None:
                cur, tot = p["progress"]
                st.progress(min(cur / tot, 1.0) if tot else 0.0)

    st.divider()

    # ========================================================================
    # AGENTIC DECISIONS
    # ========================================================================
    st.markdown("### Agentic Decisions")

    dec1, dec2 = st.columns(2)
    with dec1:
        st.markdown("**🧭 Pre-Flight (Orchestrator Agent)**")
        preflight = _get_json(f"close:{run_id}:preflight")
        if preflight:
            st.markdown(f"Proceed: **{'✅ Yes' if preflight.get('proceed') else '❌ No'}**")
            st.caption(f"Reason: {preflight.get('reason', '—')}")
        else:
            st.caption("Awaiting pre-flight decision…")

    with dec2:
        st.markdown("**🔍 Post-Flight (Orchestrator Agent)**")
        postflight = _get_json(f"close:{run_id}:postflight")
        if postflight:
            esc = postflight.get("escalate_to_human", False)
            st.markdown(f"Escalate: **{'🔴 Yes' if esc else '🟢 No'}**")
            st.caption(f"{postflight.get('status_summary', '—')}")
        else:
            st.caption("Awaiting post-flight review… (after Phase 4)")

    st.divider()
# ========================================================================
    # INTERCOMPANY ELIMINATION DETAILS
    # ========================================================================
    st.markdown("### Intercompany Elimination")

    p3_payload = _get_json(f"close:{run_id}:phase3:result")
    if p3_payload:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Transactions", p3_payload.get("total_transactions", 0))
        e2.metric("Unique Pairs", p3_payload.get("unique_pairs", 0))
        e3.metric(
            "Matched IC (net at group)",
            _fmt_usd(p3_payload.get("matched_intercompany_usd", 0), compact=True),
        )
        e4.metric(
            "Asymmetry (haircut)",
            _fmt_usd(p3_payload.get("total_asymmetry_usd", 0), compact=True),
        )

        mismatches = p3_payload.get("mismatches", [])
        if mismatches:
            st.markdown("**Mismatch Details**")
            rows = [
                {
                    "Rule": m.get("rule"),
                    "Entities": f"{m.get('seller_id')} → {m.get('buyer_id')}",
                    "Amount": _fmt_usd(m.get("amount", 0)),
                    "Detail": (m.get("detail", "") or "")[:90],
                }
                for m in mismatches[:15]
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.success("All intercompany pairs reconcile cleanly.")
    else:
        st.info("Awaiting Phase 3 elimination result…")
    # ========================================================================
    # CONSOLIDATED GROUP FINANCIALS
    # ========================================================================
    st.markdown("### Consolidated Group Financials")

    final = _get_json(f"close:{run_id}:final_result")
    if final:
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Gross Revenue", _fmt_usd(final.get("gross_revenue"), compact=True))
        with k2:
            st.metric("Raw EBITDA", _fmt_usd(final.get("raw_ebitda"), compact=True))
        with k3:
            adj = final.get("adjusted_group_ebitda", 0)
            st.metric(
                "Adjusted Group EBITDA",
                _fmt_usd(adj, compact=True),
                delta="positive" if adj >= 0 else "negative",
                delta_color="normal" if adj >= 0 else "inverse",
            )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**P&L Breakdown**")
            st.dataframe([
                {"Line": "Gross Revenue", "Value": _fmt_usd(final.get("gross_revenue"))},
                {"Line": "COGS", "Value": _fmt_usd(final.get("total_cogs"))},
                {"Line": "Gross Profit", "Value": _fmt_usd(final.get("gross_profit"))},
                {"Line": "OpEx", "Value": _fmt_usd(final.get("total_opex"))},
                {"Line": "Raw EBITDA", "Value": _fmt_usd(final.get("raw_ebitda"))},
            ], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**IC Elimination**")
            st.dataframe([
                {"Line": "Matched IC Eliminated", "Value": _fmt_usd(final.get("eliminated_intercompany_usd"))},
                {"Line": "Asymmetry Haircut", "Value": _fmt_usd(final.get("elimination_asymmetry"))},
                {"Line": "Adjusted EBITDA", "Value": _fmt_usd(final.get("adjusted_group_ebitda"))},
                {"Line": "Entities", "Value": str(final.get("entity_count", 0))},
            ], use_container_width=True, hide_index=True)

        if final.get("executive_summary"):
            st.markdown("**Executive Summary**")
            st.markdown(
                f"<div style='background:#eff6ff;border-left:3px solid #3b82f6;"
                f"padding:12px 16px;border-radius:6px;font-size:0.92rem;line-height:1.6;'>"
                f"{final['executive_summary']}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Awaiting Phase 4 consolidation…")

    st.divider()

    # ========================================================================
    # ENTITY DRILL-DOWN
    # ========================================================================
    st.markdown("### Entity Drill-Down")

    companies = _discover_company_ids(run_id)
    if not companies:
        st.caption("No entities yet.")
    else:
        selected = st.selectbox("Select a Company:", companies, key="drill_c")
        period = _latest_period_from_db() or "2026-01"
        st.caption(f"Period: {period}")

        tab1, tab2 = st.tabs(["Phase 1 — Validation", "Phase 2 — Close Ops"])
        with tab1:
            st.markdown("#### Trial Balance Validator")
            tb = _tool_tb(selected, period)
            if tb.get("found"):
                t1, t2, t3 = st.columns(3)
                t1.metric("Accounts", tb.get("account_count", 0))
                t2.metric("Debits", _fmt_usd(tb.get("total_debits", 0), compact=True))
                t3.metric(
                    "Difference", _fmt_usd(tb.get("difference", 0)),
                    delta="balanced" if tb.get("is_balanced") else "unbalanced",
                    delta_color="normal" if tb.get("is_balanced") else "inverse",
                )
                if tb.get("issues"):
                    st.dataframe(tb["issues"], use_container_width=True, hide_index=True)
                else:
                    st.success("No trial balance issues.")

            st.markdown("#### Variance Analysis")
            var = _tool_variance(selected, period)
            if var.get("found"):
                v1, v2 = st.columns(2)
                v1.metric("Flagged", var.get("flagged_count", 0))
                v2.metric("Unfavorable", _fmt_usd(var.get("total_unfavorable", 0), compact=True))
                if var.get("flagged"):
                    st.dataframe(var["flagged"], use_container_width=True, hide_index=True)
                else:
                    st.success("No material variances.")

            st.markdown("#### Cash Flow Reconciliation")
            cf = _tool_cash(selected, period)
            if cf.get("found"):
                c1, c2 = st.columns(2)
                c1.metric("Status", "RECONCILED" if cf.get("within_tolerance") else "UNRECONCILED")
                c2.metric("Gap", _fmt_usd(cf.get("gap", 0)))

        with tab2:
            st.markdown("#### Accrual Verification")
            acc = _tool_accruals(selected, period)
            if acc.get("found"):
                a1, a2 = st.columns(2)
                a1.metric("Issues", acc.get("flagged_count", 0))
                a2.metric("Flagged $", _fmt_usd(acc.get("total_flagged_amount", 0), compact=True))

            st.markdown("#### Revenue Recognition")
            rev = _tool_revrec(selected, period)
            if rev.get("found"):
                r1, r2, r3 = st.columns(3)
                r1.metric("Issues", rev.get("flagged_count", 0))
                r2.metric("Flagged Value", _fmt_usd(rev.get("flagged_value", 0), compact=True))
                r3.metric("Month Revenue", _fmt_usd(rev.get("total_month_revenue_recognized", 0), compact=True))

            st.markdown("#### Expense Categorization")
            exp = _tool_expenses(selected, period)
            if exp.get("found"):
                e1, e2 = st.columns(2)
                e1.metric("Issues", exp.get("flagged_count", 0))
                e2.metric("Total OpEx", _fmt_usd(exp.get("total_expenses", 0), compact=True))


# ============================================================================
# RIGHT — CLOSE ASSISTANT AI
# ============================================================================
with assistant_col:
    st.markdown("### Close Assistant AI")

    try:
        from app.db.database import settings as _s
        _api_key_ok = bool(_s.gemini_api_key)
    except Exception:
        _api_key_ok = False

    if _api_key_ok:
        st.markdown(
            "<div style='font-size:0.78rem; opacity:0.7; margin-bottom:8px;'>"
            "🟢 Assistant online · grounded in deterministic tools</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:0.78rem; color:#dc2626; margin-bottom:8px;'>"
            "🔴 Assistant offline · GEMINI_API_KEY not configured</div>",
            unsafe_allow_html=True,
        )

    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "Welcome, Controller. 👋\n\n"
                "I can help with close analysis — trial balance, variances, "
                "accruals, revenue recognition, and consolidation.\n\n"
                "Ask me anything, or pick a suggestion below."
            ),
        })

    chat_box = st.container(height=520, border=True)
    pending = st.session_state.pop("_pending_question", None)

    with chat_box:
        has_user_msg = any(m["role"] == "user" for m in st.session_state.chat_history)

        if not has_user_msg and pending is None:
            st.caption("Try one of these:")
            suggestions = [
                "What's the TB Validator status?",
                "Why is R&D over budget at TechForge?",
                "Which entity has the biggest variance?",
            ]
            for i, s in enumerate(suggestions):
                if st.button(s, key=f"sug_{i}", width="stretch"):
                    st.session_state["_pending_question"] = s
                    st.session_state["_llm_processing"] = True   # ← pause refresh IMMEDIATELY
                    st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if pending is not None:
            with st.chat_message("user"):
                st.markdown(pending)

            # flag already True (set above); ensure it stays True during the run
            st.session_state["_llm_processing"] = True
            answer: str
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing financial data…"):
                        try:
                            from app.agents.nlq import ask_financial_question
                            answer = ask_financial_question(pending)
                        except Exception as exc:
                            answer = f"⚠️ Assistant error: `{type(exc).__name__}: {exc}`"
                    st.markdown(answer)
                st.session_state.chat_history.append({"role": "user", "content": pending})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            finally:
                st.session_state.pop("_llm_processing", None)

    user_input = st.chat_input("Type your question…", key="assistant_input")
    if user_input:
        st.session_state["_pending_question"] = user_input
        st.session_state["_llm_processing"] = True   # ← pause refresh IMMEDIATELY
        st.rerun()

    if any(m["role"] == "user" for m in st.session_state.chat_history):
        if st.button("🗑️ Clear conversation", key="clr_chat", width="stretch"):
            st.session_state.chat_history = []
            st.session_state.pop("_pending_question", None)
            st.session_state.pop("_llm_processing", None)
            st.rerun()


st.caption(
    f"Auto-refresh {refresh_rate}s · Accent {st.session_state.theme_accent} · "
    f"Redis `{REDIS_URL}`"
)