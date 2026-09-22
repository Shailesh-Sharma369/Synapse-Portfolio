"""
Zero-Click Executive Monitor for the Month-End Close pipeline.

- Auto-discovers the latest run_id (no inputs, no buttons)
- Live-refreshes every 3s
- Shows Phase 1/2 progress, Phase 3 elimination mismatches, Phase 4 group P&L
- Read-only Entity Drill-Down: CFO picks a company, sees that entity's
  deterministic agent outputs live (no LLM calls — pure Python tools)

Launch:
    streamlit run app/ui/dashboard.py --server.port=8501 --server.address=0.0.0.0
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
# CONFIG
# ============================================================================
st.set_page_config(
    page_title="Close Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REFRESH_SECONDS = 3

# --- Kill Streamlit's default chrome for a wall-monitor look ---------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px; }
        #MainMenu {visibility: hidden;}
        header[data-testid="stHeader"] {display: none;}
        footer {visibility: hidden;}
        div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 600; }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6b7280;
                                            text-transform: uppercase; letter-spacing: .05em; }
        h2, h3 { margin-top: 0.4rem !important; margin-bottom: 0.4rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# REDIS
# ============================================================================
@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


r = get_redis()


# ============================================================================
# LATEST-RUN DISCOVERY (zero-click)
# ============================================================================
def find_latest_run_id() -> str | None:
    """
    Return the most recent run_id.

    Discovery order:
      1. `close:latest_run_id`  — set explicitly by the trigger endpoint.
      2. `close:runs:recent`    — LPUSH list, newest first.
      3. Key scan for `close:*:status` — last resort; picks the key with the
         highest TTL (all keys use uniform base TTL → highest = newest).
    """
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


# ============================================================================
# FORMATTERS
# ============================================================================
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


def _status_emoji(status: str | None) -> str:
    s = (status or "").lower()
    if s in ("completed", "done", "passed", "clean", "reconciled", "complete"):
        return "🟢"
    if s in ("running", "in_progress"):
        return "🟡"
    if s in ("failed", "mismatches_found", "unreconciled", "incomplete"):
        return "🔴"
    return "⚪"


def _phase_badge(value: str | None) -> str:
    if value == "done":
        return "✅"
    if value == "failed":
        return "❌"
    return "…"


# ============================================================================
# DRILL-DOWN HELPERS
# ============================================================================
def _discover_company_ids(run_id: str) -> list[str]:
    """Extract portfolio company IDs for a run from Redis, falling back to DB."""
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


def _resolve_period(run_id: str) -> str:
    """Resolve the accounting period for a run (Redis hint → latest in DB → default)."""
    try:
        hint = r.get(f"close:{run_id}:period")
        if hint:
            return hint
    except Exception:
        pass

    try:
        from sqlalchemy import select
        from app.db.database import SessionLocal
        from app.db.models import TrialBalance
        db = SessionLocal()
        try:
            latest = db.scalar(
                select(TrialBalance.period)
                .order_by(TrialBalance.period.desc())
                .limit(1)
            )
            if latest:
                return latest
        finally:
            db.close()
    except Exception:
        pass

    return "2026-01"


# ---- Cached deterministic tool wrappers -----------------------------------
# TTL 5 min — data only changes when the DB is re-seeded.
# show_spinner=False so 3s auto-refresh doesn't flash spinners.

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
# HEADER — UTC clock + auto-refresh indicator
# ============================================================================
st_autorefresh(interval=REFRESH_SECONDS * 1000, key="monitor_refresh")

now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
col_title, col_clock = st.columns([3, 1])
with col_title:
    st.markdown("## 📊 Month-End Close Monitor")
with col_clock:
    st.markdown(
        f"<div style='text-align:right; color:#6b7280; font-size:0.85rem; padding-top:1rem;'>"
        f"🔄 auto-refresh {REFRESH_SECONDS}s<br/>{now_utc}</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# NO-RUN STATE
# ============================================================================
run_id = find_latest_run_id()

if not run_id:
    st.info("⏳ Waiting for the first close run to start…")
    st.caption("This monitor auto-discovers the latest run. No action needed.")
    st.stop()


# ============================================================================
# LIVE STATE
# ============================================================================
status = _get(f"close:{run_id}:status", "unknown")
total_companies = _get_int(f"close:{run_id}:total_companies", 0)
phase2_count = _get_int(f"close:{run_id}:phase2_count", 0)

p1_done = 0
for k in r.scan_iter(match=f"close:{run_id}:phase1:*"):
    if k.startswith(f"close:{run_id}:phase1_failures"):
        continue
    if r.get(k) == "done":
        p1_done += 1

phase3_done = _get(f"close:{run_id}:phase3", "-")
phase3_status = _get(f"close:{run_id}:phase3_status", "-")
phase4_done = _get(f"close:{run_id}:phase4", "-")
phase4_status = _get(f"close:{run_id}:phase4_status", "-")


# ============================================================================
# STATUS BAR
# ============================================================================
st.markdown(
    f"**Run** `{run_id}` &nbsp;·&nbsp; **Status** {_status_emoji(status)} `{status}`",
    unsafe_allow_html=True,
)


# ============================================================================
# PHASE PROGRESS — 4 columns
# ============================================================================
st.markdown("### Pipeline Progress")
c1, c2, c3, c4 = st.columns(4)

with c1:
    total_1 = total_companies or 8
    st.metric("Phase 1 — Validation", f"{p1_done} / {total_1}",
              help="Trial Balance · Variance · Cash Flow (parallel per company)")
    st.progress(min(p1_done / total_1, 1.0) if total_1 else 0.0)

with c2:
    total_2 = total_companies or 8
    st.metric("Phase 2 — Close Ops", f"{phase2_count} / {total_2}",
              help="Accruals → Revenue → Expenses (sequential per company)")
    st.progress(min(phase2_count / total_2, 1.0) if total_2 else 0.0)

with c3:
    st.metric("Phase 3 — Elimination", phase3_status,
              help="Cross-company intercompany reconciliation")
    st.markdown(f"{_phase_badge(phase3_done)} phase3")

with c4:
    st.metric("Phase 4 — Consolidation", phase4_status,
              help="Group P&L and adjusted EBITDA")
    st.markdown(f"{_phase_badge(phase4_done)} phase4")


# ============================================================================
# PHASE 3 — INTERCOMPANY MISMATCHES
# ============================================================================
st.divider()
st.markdown("### 🔁 Intercompany Elimination")

phase3 = _get_json(f"close:{run_id}:phase3:result")

if phase3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", phase3.get("status", "—"))
    m2.metric("Transactions", f"{phase3.get('total_transactions', 0):,}")
    m3.metric("Entity Pairs", f"{phase3.get('unique_pairs', 0)}")
    m4.metric("Total Asymmetry", _fmt_usd(phase3.get("total_asymmetry_usd", 0.0), compact=True))

    mismatches = phase3.get("mismatches", [])
    if mismatches:
        with st.expander(f"⚠️ {len(mismatches)} mismatches detected — click to expand", expanded=False):
            rows = [
                {
                    "Rule": m["rule"],
                    "Seller → Buyer": f"{m['seller_id']} → {m['buyer_id']}",
                    "Amount": _fmt_usd(m["amount"]),
                }
                for m in mismatches[:20]
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if len(mismatches) > 20:
                st.caption(f"Showing top 20 of {len(mismatches)}.")
    else:
        st.success("All intercompany flows reconcile cleanly.")
else:
    st.caption("Awaiting Phase 3 completion…")


# ============================================================================
# PHASE 4 — CONSOLIDATED FINANCIALS
# ============================================================================
st.divider()
st.markdown("### 💰 Consolidated Group Financials")

final = _get_json(f"close:{run_id}:final_result")

if final:
    h1, h2, h3 = st.columns(3)
    h1.metric("Gross Revenue", _fmt_usd(final.get("gross_revenue"), compact=True))
    h2.metric("Raw EBITDA", _fmt_usd(final.get("raw_ebitda"), compact=True))
    h3.metric(
        "Adjusted Group EBITDA",
        _fmt_usd(final.get("adjusted_group_ebitda"), compact=True),
        delta=f"-{_fmt_usd(final.get('elimination_asymmetry'), compact=True)} (elim.)",
        delta_color="inverse",
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("COGS", _fmt_usd(final.get("total_cogs"), compact=True))
    s2.metric("Gross Profit", _fmt_usd(final.get("gross_profit"), compact=True))
    s3.metric("OpEx", _fmt_usd(final.get("total_opex"), compact=True))
    s4.metric("Entities", f"{final.get('entity_count', 0)}")

    st.markdown("#### 📝 Executive Summary")
    st.info(final.get("executive_summary", "—"))

    with st.expander("Full payload"):
        st.json(final)
else:
    st.caption("Awaiting Phase 4 completion…")


# ============================================================================
# ENTITY DRILL-DOWN (read-only, uses deterministic tools directly)
# ============================================================================
st.divider()
st.markdown("### 🔍 Entity Drill-Down")
st.caption(
    "Read-only per-company inspection. Data comes from deterministic Python tools "
    "(no LLM). Not tied to the run — reflects the current state of the database."
)

period = _resolve_period(run_id)
company_ids = _discover_company_ids(run_id)

if not company_ids:
    st.caption("No entities available for this run.")
else:
    sel_col, info_col = st.columns([2, 3])
    with sel_col:
        selected = st.selectbox(
            "Portfolio company",
            company_ids,
            key="drill_company",
            label_visibility="collapsed",
        )
    with info_col:
        st.markdown(
            f"<div style='padding-top:0.5rem; color:#6b7280;'>"
            f"Selected <b>{selected}</b> · Period <code>{period}</code></div>",
            unsafe_allow_html=True,
        )

    tab1, tab2 = st.tabs(["Phase 1 — Validation", "Phase 2 — Close Ops"])

    # ---- Phase 1 tab -----------------------------------------------------
    with tab1:
        # Trial Balance
        tb = _tool_tb(selected, period)
        st.markdown("**Trial Balance Validation**")
        if tb.get("found"):
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Accounts", tb.get("account_count", 0))
            t2.metric("Debits", _fmt_usd(tb.get("total_debits", 0), compact=True))
            t3.metric("Credits", _fmt_usd(tb.get("total_credits", 0), compact=True))
            t4.metric(
                "Difference",
                _fmt_usd(tb.get("difference", 0)),
                delta="balanced" if tb.get("is_balanced") else "unbalanced",
                delta_color="normal" if tb.get("is_balanced") else "inverse",
            )
            if tb.get("issues"):
                st.dataframe(tb["issues"], use_container_width=True, hide_index=True)
            else:
                st.success("No trial balance issues.")
        else:
            st.warning(tb.get("error", "No trial balance data."))

        st.divider()

        # Variance
        var = _tool_variance(selected, period)
        st.markdown("**Variance Analysis**")
        if var.get("found"):
            v1, v2, v3 = st.columns(3)
            v1.metric("Accounts Compared", var.get("accounts_compared", 0))
            v2.metric("Flagged", var.get("flagged_count", 0))
            v3.metric("Unfavorable", _fmt_usd(var.get("total_unfavorable", 0), compact=True))
            if var.get("flagged"):
                st.dataframe(var["flagged"], use_container_width=True, hide_index=True)
            else:
                st.success("No material variances.")
        else:
            st.warning(var.get("error", "No variance data."))

        st.divider()

        # Cash Flow
        cf = _tool_cash(selected, period)
        st.markdown("**Cash Flow Reconciliation**")
        if cf.get("found"):
            cf1, cf2, cf3 = st.columns(3)
            cf1.metric(
                "Status",
                "RECONCILED" if cf.get("within_tolerance") else "UNRECONCILED",
            )
            cf2.metric("Gap", _fmt_usd(cf.get("gap", 0)))
            cf3.metric("Tolerance", _fmt_usd(cf.get("tolerance_usd", 0)))
            if cf.get("top_transactions"):
                st.caption("Top bank transactions")
                st.dataframe(cf["top_transactions"], use_container_width=True, hide_index=True)
        else:
            st.warning(cf.get("error", "No cash flow data."))

    # ---- Phase 2 tab -----------------------------------------------------
    with tab2:
        # Accruals
        acc = _tool_accruals(selected, period)
        st.markdown("**Accrual Verification**")
        if acc.get("found"):
            a1, a2, a3 = st.columns(3)
            a1.metric("Accruals", acc.get("total_accruals", 0))
            a2.metric("Flagged", acc.get("flagged_count", 0))
            a3.metric("Flagged $", _fmt_usd(acc.get("total_flagged_amount", 0), compact=True))
            if acc.get("issues"):
                with st.expander(f"{len(acc['issues'])} accrual issues — click to expand", expanded=False):
                    st.dataframe(acc["issues"][:25], use_container_width=True, hide_index=True)
                    if len(acc["issues"]) > 25:
                        st.caption(f"Showing top 25 of {len(acc['issues'])}.")
            else:
                st.success("All accruals current.")
        else:
            st.warning(acc.get("error", "No accrual data."))

        st.divider()

        # Revenue Recognition
        rev = _tool_revrec(selected, period)
        st.markdown("**Revenue Recognition (ASC 606)**")
        if rev.get("found"):
            r1, r2, r3 = st.columns(3)
            r1.metric("Contracts Examined", rev.get("contracts_examined", 0))
            r2.metric("Flagged", rev.get("flagged_count", 0))
            r3.metric("Flagged Value", _fmt_usd(rev.get("flagged_value", 0), compact=True))
            if rev.get("issues"):
                st.dataframe(rev["issues"], use_container_width=True, hide_index=True)
            else:
                st.success("All contracts compliant.")
        else:
            st.warning(rev.get("error", "No revenue contract data."))

        st.divider()

        # Expense Categorization
        exp = _tool_expenses(selected, period)
        st.markdown("**Expense Categorization**")
        if exp.get("found"):
            e1, e2, e3 = st.columns(3)
            e1.metric("Accounts", exp.get("expense_accounts_examined", 0))
            e2.metric("Flagged", exp.get("flagged_count", 0))
            e3.metric("Total OpEx", _fmt_usd(exp.get("total_expenses", 0), compact=True))
            if exp.get("issues"):
                st.dataframe(exp["issues"], use_container_width=True, hide_index=True)
            else:
                st.success("All expenses categorized cleanly.")
        else:
            st.warning(exp.get("error", "No expense data."))


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption(
    f"Zero-click monitor · discovers latest run automatically · "
    f"drill-down uses deterministic Python tools · "
    f"Redis `{REDIS_URL}` · refresh every {REFRESH_SECONDS}s"
)