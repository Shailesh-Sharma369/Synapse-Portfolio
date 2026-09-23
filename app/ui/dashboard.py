"""
Zero-Click Executive Monitor for the Month-End Close pipeline.

Layout:
    ┌─── Sidebar ───┬───────── Main ─────────┬─── Assistant ───┐
    │ Collapsible   │  Header + Run info    │  Close          │
    │ Settings      │  Phase cards grid     │  Assistant      │
    │ Theme accent  │  Expanders (elim,     │  (chat)         │
    │ Refresh rate  │   financials, drill)  │                 │
    └───────────────┴───────────────────────┴─────────────────┘

Styling philosophy:
    - DO NOT override Streamlit's base theme (leave .stApp, containers alone)
    - ONLY override accent colors (progress bars, borders, badge chips)
    - Soft corporate palette that works in both light and dark modes
    - All numeric typography uses tabular-nums for alignment

Chat / auto-refresh coordination:
    - The assistant takes 5-25s per question via Gemini.
    - `streamlit_autorefresh` at 3s would rerun the script mid-LLM-call,
      wiping the spinner and clearing the pending state.
    - Fix: a single `_llm_processing` session flag is set for the ENTIRE
      duration of the LLM call and cleared in a `finally:` block. The
      auto-refresh registration at the top of the script checks this flag
      and skips itself while it is True.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import redis
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================================================
# PAGE CONFIG
# ============================================================================
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
    st.session_state.theme_accent = "Light"     # Light | Dark (ACCENT only)
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

ACCENT = {
    "Light": {
        "primary": "#2563eb",
        "primary_soft": "rgba(37, 99, 235, 0.10)",
        "success": "#16a34a",
        "success_soft": "rgba(22, 163, 74, 0.10)",
        "danger": "#dc2626",
        "danger_soft": "rgba(220, 38, 38, 0.10)",
        "warning": "#d97706",
        "neutral": "#6b7280",
        "chip_bg": "#F1F3F5",
        "chip_border": "#E9ECEF",
    },
    "Dark": {
        "primary": "#58a6ff",
        "primary_soft": "rgba(88, 166, 255, 0.15)",
        "success": "#3fb950",
        "success_soft": "rgba(63, 185, 80, 0.15)",
        "danger": "#f85149",
        "danger_soft": "rgba(248, 81, 73, 0.15)",
        "warning": "#d29922",
        "neutral": "#8b949e",
        "chip_bg": "rgba(255, 255, 255, 0.05)",
        "chip_border": "rgba(255, 255, 255, 0.12)",
    },
}

A = ACCENT[st.session_state.theme_accent]


# ============================================================================
# SOFT CORPORATE CSS — accents only, no surface overrides
# ============================================================================
def _accent_css() -> str:
    return f"""
    <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                         Roboto, Helvetica, Arial, sans-serif;
            font-feature-settings: "tnum" 1, "ss01" 1;
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            max-width: 1600px;
        }}
        #MainMenu, footer {{ visibility: hidden; }}

        .stProgress > div > div > div > div {{
            background-color: {A["primary"]} !important;
            border-radius: 6px;
        }}
        .stProgress > div > div > div {{
            background-color: {A["primary_soft"]} !important;
            border-radius: 6px;
        }}

        details > summary {{ font-weight: 500; padding: 6px 4px; }}
        details[open] > summary {{
            border-bottom: 1px solid {A["chip_border"]};
            padding-bottom: 10px;
            margin-bottom: 8px;
        }}

        div[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

        .stButton > button:hover {{
            border-color: {A["primary"]} !important;
            color: {A["primary"]} !important;
        }}

        div[data-testid="stChatMessage"] {{
            border-radius: 10px;
            border-left: 3px solid {A["primary_soft"]};
        }}

        .status-chip {{
            display: inline-block;
            background: {A["chip_bg"]};
            border: 1px solid {A["chip_border"]};
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.82rem;
            font-weight: 500;
        }}

        div[data-testid="stMetric"] {{ border-radius: 10px; padding: 10px 12px; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 600; }}
        div[data-testid="stMetricLabel"] {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.75;
        }}

        button[kind="header"] {{ display: none; }}
    </style>
    """

st.markdown(_accent_css(), unsafe_allow_html=True)


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
    try:
        candidates: list[tuple[int, str]] = []
        for key in r.scan_iter(match="close:*:status", count=200):
            parts = key.split(":")
            if len(parts) != 3:
                continue
            try:
                ttl = r.ttl(key)
            except Exception:
                ttl = 0
            candidates.append((ttl, parts[1]))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
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


def _status_dot(status: str | None) -> str:
    s = (status or "").lower()
    if s in ("completed", "done", "passed", "clean", "reconciled", "complete"):
        return "🟢"
    if s in ("running", "in_progress"):
        return "🟡"
    if s in ("failed", "mismatches_found", "unreconciled", "incomplete"):
        return "🔴"
    return "⚪"


def _latest_period_from_db() -> str | None:
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


def _discover_company_ids(run_id: str) -> list[str]:
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
    if not ids:
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


# ---- Cached deterministic tool wrappers ----------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _tool_tb(company_id: str, period: str) -> dict:
    from app.agents.validator import analyze_trial_balance
    return analyze_trial_balance(company_id, period)


@st.cache_data(ttl=300, show_spinner=False)
def _tool_variance(company_id: str, period: str) -> dict:
    from app.agents.variance import analyze_variances
    return analyze_variances(company_id, period)


@st.cache_data(ttl=300, show_spinner=False)
def _tool_cash(company_id: str, period: str) -> dict:
    from app.agents.cash_flow import reconcile_cash_flow
    return reconcile_cash_flow(company_id, period)


@st.cache_data(ttl=300, show_spinner=False)
def _tool_accruals(company_id: str, period: str) -> dict:
    from app.agents.accrual_verification import verify_accruals
    return verify_accruals(company_id, period)


@st.cache_data(ttl=300, show_spinner=False)
def _tool_revrec(company_id: str, period: str) -> dict:
    from app.agents.revenue_recognition import verify_revenue_recognition
    return verify_revenue_recognition(company_id, period)


@st.cache_data(ttl=300, show_spinner=False)
def _tool_expenses(company_id: str, period: str) -> dict:
    from app.agents.expense_categorization import categorize_expenses
    return categorize_expenses(company_id, period)


# ============================================================================
# SIDEBAR — collapsible to icon-only
# ============================================================================
with st.sidebar:
    col_toggle, col_theme_mini = st.columns([4, 1])

    with col_toggle:
        if st.session_state.sidebar_collapsed:
            if st.button("☰", key="sidebar_expand", help="Expand sidebar"):
                st.session_state.sidebar_collapsed = False
                st.rerun()
        else:
            if st.button("◀  Collapse", key="sidebar_collapse", help="Collapse to icons"):
                st.session_state.sidebar_collapsed = True
                st.rerun()

    with col_theme_mini:
        if st.button("🌓", key="theme_toggle_btn", help="Toggle accent theme"):
            st.session_state.theme_accent = (
                "Dark" if st.session_state.theme_accent == "Light" else "Light"
            )
            st.rerun()

    st.divider()

    if st.session_state.sidebar_collapsed:
        if st.button("⚙️", key="collapsed_settings", help="Settings"):
            st.session_state.sidebar_collapsed = False
            st.rerun()
        if st.button("🔄", key="collapsed_refresh", help="Refresh rate"):
            st.session_state.sidebar_collapsed = False
            st.rerun()
        if st.button("👤", key="collapsed_welcome", help="Welcome, Controller"):
            st.session_state.sidebar_collapsed = False
            st.rerun()
        refresh_rate = st.session_state.get("refresh_rate", 3)
    else:
        st.markdown("## Settings")

        st.markdown("**Accent theme**")
        theme_choice = st.radio(
            "accent_theme_radio",
            options=["Light", "Dark"],
            index=0 if st.session_state.theme_accent == "Light" else 1,
            label_visibility="collapsed",
            key="accent_theme_widget",
        )
        if theme_choice != st.session_state.theme_accent:
            st.session_state.theme_accent = theme_choice
            st.rerun()

        st.markdown("**Auto-refresh rate (s)**")
        refresh_rate = st.selectbox(
            "refresh_rate",
            options=[3, 5, 10, 30],
            index=0,
            label_visibility="collapsed",
            key="refresh_rate",
        )

        st.divider()
        st.markdown("### Welcome, Controller 👨‍💼")
        st.caption("This monitor auto-discovers the latest close run.")

        try:
            recent = r.lrange("close:runs:recent", 0, 4)
            if recent:
                st.markdown("**Recent runs**")
                for rid in recent:
                    st.caption(f"• `{rid[:8]}…`")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AUTO-REFRESH — paused while the LLM is in-flight.
#
# `_llm_processing` is set by the chat handler and cleared in a `finally:`
# block. Registering the timer only when that flag is absent guarantees the
# script cannot be rerun mid-LLM-call (which would wipe the spinner and
# orphan the pending question).
# ---------------------------------------------------------------------------
if not st.session_state.get("_llm_processing"):
    st_autorefresh(interval=refresh_rate * 1000, key="monitor_refresh")


# ============================================================================
# MAIN LAYOUT
# ============================================================================
main_col, assistant_col = st.columns([7, 3], gap="large")


# ============================================================================
# LEFT — MAIN MONITORING CONTENT
# ============================================================================
with main_col:
    st.markdown(
        "<h2 style='margin-bottom:0;'>✅ Month-End Close Monitor</h2>",
        unsafe_allow_html=True,
    )

    run_id = find_latest_run_id()

    if not run_id:
        st.info("⏳ Waiting for the first close run to start…")
        st.caption("This monitor auto-discovers the latest run. No action needed.")
        st.stop()

    status = _get(f"close:{run_id}:status", "unknown")
    total_companies = _get_int(f"close:{run_id}:total_companies", 0) or 8

    st.markdown(
        f"<div style='font-size:0.9rem; margin-top:6px; opacity:0.85;'>"
        f"Run ID: <code>{run_id}</code><br/>"
        f"Status: <span class='status-chip'>{_status_dot(status)} {status}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ---- Phase progress cards ------------------------------------------
    phase2_count = _get_int(f"close:{run_id}:phase2_count", 0)
    p1_done = 0
    for k in r.scan_iter(match=f"close:{run_id}:phase1:*"):
        if k.startswith(f"close:{run_id}:phase1_failures"):
            continue
        if r.get(k) == "done":
            p1_done += 1

    phase3_done = _get(f"close:{run_id}:phase3", "-")
    phase3_status = _get(f"close:{run_id}:phase3_status", "Awaiting")
    phase4_done = _get(f"close:{run_id}:phase4", "-")
    phase4_status = _get(f"close:{run_id}:phase4_status", "Awaiting")

    pc1, pc2, pc3, pc4 = st.columns(4)

    with pc1:
        st.markdown("**TB Validator 📋**")
        st.caption(f"Parallel: {p1_done}/{total_companies}")
        st.progress(min(p1_done / total_companies, 1.0) if total_companies else 0.0)

    with pc2:
        st.markdown("**CF Recon 💰**")
        st.caption(f"Parallel: {phase2_count}/{total_companies}")
        st.progress(min(phase2_count / total_companies, 1.0) if total_companies else 0.0)

    with pc3:
        st.markdown("**Elimination 🔁**")
        st.caption(f"Sequential: {phase3_status}")
        st.progress(1.0 if phase3_done == "done" else 0.0)

    with pc4:
        st.markdown("**Consolidation 🏛️**")
        st.caption(f"Sequential: {phase4_status}")
        st.progress(1.0 if phase4_done == "done" else 0.0)

    st.divider()

    st.markdown("### Detailed Explanations")

    # ---- Intercompany Elimination --------------------------------------
    with st.expander("Intercompany Elimination Details", expanded=True):
        st.caption("A detailed table of elimination transactions.")

        phase3_payload = _get_json(f"close:{run_id}:phase3:result")
        if phase3_payload and phase3_payload.get("mismatches"):
            rows = [
                {
                    "Rule": m["rule"],
                    "Entities": f"{m['seller_id']} → {m['buyer_id']}",
                    "Description": m["detail"][:80] + ("…" if len(m["detail"]) > 80 else ""),
                    "Mismatch Amt": _fmt_usd(m["amount"]),
                }
                for m in phase3_payload["mismatches"][:15]
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.dataframe(
                [{
                    "Rule": "awaiting…",
                    "Entities": "awaiting…",
                    "Description": "awaiting…",
                    "Mismatch Amt": "awaiting…",
                }],
                use_container_width=True,
                hide_index=True,
            )

    # ---- Consolidated Group Financials ---------------------------------
    with st.expander("Consolidated Group Financials", expanded=True):
        st.caption("Tables for P&L and Balance Sheet.")

        final = _get_json(f"close:{run_id}:final_result")

        fin_left, fin_mid, fin_right = st.columns([2, 2, 1.5])

        with fin_left:
            st.markdown("**P&L Table**")
            if final:
                pl_rows = [
                    {"Line": "Gross Revenue", "Value": _fmt_usd(final.get("gross_revenue"))},
                    {"Line": "COGS",          "Value": _fmt_usd(final.get("total_cogs"))},
                    {"Line": "Gross Profit",  "Value": _fmt_usd(final.get("gross_profit"))},
                    {"Line": "OpEx",          "Value": _fmt_usd(final.get("total_opex"))},
                    {"Line": "Raw EBITDA",    "Value": _fmt_usd(final.get("raw_ebitda"))},
                ]
                st.dataframe(pl_rows, use_container_width=True, hide_index=True)
            else:
                st.dataframe(
                    [{"Line": "awaiting…", "Value": "awaiting…"}],
                    use_container_width=True, hide_index=True,
                )

        with fin_mid:
            st.markdown("**Adjusted View**")
            if final:
                adj_rows = [
                    {"Line": "Raw EBITDA",            "Value": _fmt_usd(final.get("raw_ebitda"))},
                    {"Line": "Elim. Asymmetry",       "Value": _fmt_usd(final.get("elimination_asymmetry"))},
                    {"Line": "Adjusted Group EBITDA", "Value": _fmt_usd(final.get("adjusted_group_ebitda"))},
                    {"Line": "Entities",              "Value": str(final.get("entity_count", 0))},
                ]
                st.dataframe(adj_rows, use_container_width=True, hide_index=True)
            else:
                st.dataframe(
                    [{"Line": "awaiting…", "Value": "awaiting…"}],
                    use_container_width=True, hide_index=True,
                )

        with fin_right:
            st.markdown("**Raw EBITDA**")
            st.markdown(
                f"<div style='font-size:1.6rem; font-weight:600;'>"
                f"{_fmt_usd(final.get('raw_ebitda'), compact=True) if final else '$ await'}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Adjusted EBITDA**")
            adj_val = final.get("adjusted_group_ebitda") if final else None
            adj_color = A["danger"] if (adj_val or 0) < 0 else A["success"]
            st.markdown(
                f"<div style='font-size:1.6rem; font-weight:600; color:{adj_color};'>"
                f"{_fmt_usd(adj_val, compact=True) if final else '$ await'}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ---- Entity Drill-Down ---------------------------------------------
    with st.expander("Entity Drill-Down", expanded=False):
        st.caption("Select a company to inspect.")

        if status in ("running", "unknown"):
            st.info(
                "⏳ Entity data is currently being processed. "
                "Drill-down unlocks when the close completes."
            )
        else:
            period = _latest_period_from_db() or "2026-01"
            company_ids = _discover_company_ids(run_id)

            if not company_ids:
                st.caption("No entities available for this run.")
            else:
                selected = st.selectbox(
                    "Select a Company to inspect:",
                    company_ids,
                    key="drill_company",
                )
                st.caption(f"Period: {period}")

                tab1, tab2 = st.tabs(["Phase 1 — Validation", "Phase 2 — Close Ops"])

                with tab1:
                    tb = _tool_tb(selected, period)
                    if tb.get("found"):
                        t1, t2, t3 = st.columns(3)
                        t1.metric("Accounts", tb.get("account_count", 0))
                        t2.metric("Debits", _fmt_usd(tb.get("total_debits", 0), compact=True))
                        t3.metric(
                            "Difference",
                            _fmt_usd(tb.get("difference", 0)),
                            delta="balanced" if tb.get("is_balanced") else "unbalanced",
                            delta_color="normal" if tb.get("is_balanced") else "inverse",
                        )
                        if tb.get("issues"):
                            st.dataframe(tb["issues"], use_container_width=True, hide_index=True)
                        else:
                            st.success("No trial balance issues.")

                    st.divider()

                    var = _tool_variance(selected, period)
                    if var.get("found"):
                        v1, v2 = st.columns(2)
                        v1.metric("Flagged Variances", var.get("flagged_count", 0))
                        v2.metric(
                            "Unfavorable",
                            _fmt_usd(var.get("total_unfavorable", 0), compact=True),
                        )
                        if var.get("flagged"):
                            st.dataframe(var["flagged"], use_container_width=True, hide_index=True)
                        else:
                            st.success("No material variances.")

                    st.divider()

                    cf = _tool_cash(selected, period)
                    if cf.get("found"):
                        c1, c2 = st.columns(2)
                        c1.metric(
                            "Cash Recon",
                            "RECONCILED" if cf.get("within_tolerance") else "UNRECONCILED",
                        )
                        c2.metric("Gap", _fmt_usd(cf.get("gap", 0)))

                with tab2:
                    acc = _tool_accruals(selected, period)
                    if acc.get("found"):
                        a1, a2 = st.columns(2)
                        a1.metric("Accrual Issues", acc.get("flagged_count", 0))
                        a2.metric(
                            "Flagged $",
                            _fmt_usd(acc.get("total_flagged_amount", 0), compact=True),
                        )

                    st.divider()

                    rev = _tool_revrec(selected, period)
                    if rev.get("found"):
                        r1, r2 = st.columns(2)
                        r1.metric("Contract Issues", rev.get("flagged_count", 0))
                        r2.metric(
                            "Flagged Value",
                            _fmt_usd(rev.get("flagged_value", 0), compact=True),
                        )

                    st.divider()

                    exp = _tool_expenses(selected, period)
                    if exp.get("found"):
                        e1, e2 = st.columns(2)
                        e1.metric("Expense Issues", exp.get("flagged_count", 0))
                        e2.metric(
                            "Total OpEx",
                            _fmt_usd(exp.get("total_expenses", 0), compact=True),
                        )


# ============================================================================
# RIGHT — CLOSE ASSISTANT AI
# ============================================================================
with assistant_col:
    st.markdown("### Close Assistant AI")

    # ---- Connection status chip ----------------------------------------
    try:
        from app.db.database import settings as _app_settings
        _api_key_ok = bool(_app_settings.gemini_api_key)
    except Exception:
        _api_key_ok = False

    if _api_key_ok:
        st.markdown(
            "<div style='font-size:0.78rem; opacity:0.7; margin-bottom:8px;'>"
            "🟢 Assistant online · grounded in deterministic tools"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:0.78rem; color:#dc2626; margin-bottom:8px;'>"
            "🔴 Assistant offline · GEMINI_API_KEY not configured"
            "</div>",
            unsafe_allow_html=True,
        )

    # ---- Seed welcome message ------------------------------------------
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

    # ---- Chat container -------------------------------------------------
    chat_box = st.container(height=520, border=True)

    # Pop the pending question IMMEDIATELY. From this point forward, the
    # only signal that an LLM call is in-flight is `_llm_processing`.
    pending = st.session_state.pop("_pending_question", None)

    with chat_box:
        has_user_msg = any(m["role"] == "user" for m in st.session_state.chat_history)

        # Suggested prompts (only on very first visit)
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
                    st.rerun()

        # Render conversation history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # If a question is pending, run the LLM synchronously NOW.
        if pending is not None:
            # 1. Show the user bubble immediately
            with st.chat_message("user"):
                st.markdown(pending)

            # 2. Mark LLM as in-flight — this pauses auto-refresh above
            st.session_state["_llm_processing"] = True
            answer: str
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing financial data… (this may take ~30s)"):
                        try:
                            from app.agents.nlq import ask_financial_question
                            answer = ask_financial_question(pending)
                        except Exception as exc:  # noqa: BLE001
                            answer = f"⚠️ Assistant error: `{type(exc).__name__}: {exc}`"
                    st.markdown(answer)

                # 3. Persist turns only AFTER the LLM returns
                st.session_state.chat_history.append(
                    {"role": "user", "content": pending}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            finally:
                # 4. Always clear the in-flight flag, even on exception
                st.session_state.pop("_llm_processing", None)

    # ---- Chat input -----------------------------------------------------
    user_input = st.chat_input("Type your question…", key="assistant_input")
    if user_input:
        st.session_state["_pending_question"] = user_input
        st.rerun()

    # ---- Clear button ---------------------------------------------------
    if any(m["role"] == "user" for m in st.session_state.chat_history):
        if st.button("🗑️ Clear conversation", key="clear_chat", width="stretch"):
            st.session_state.chat_history = []
            st.session_state.pop("_pending_question", None)
            st.session_state.pop("_llm_processing", None)
            st.rerun()


# ============================================================================
# FOOTER
# ============================================================================
st.caption(
    f"Auto-refresh {refresh_rate}s · Accent {st.session_state.theme_accent} · "
    f"Redis `{REDIS_URL}` · Zero-click monitor"
)