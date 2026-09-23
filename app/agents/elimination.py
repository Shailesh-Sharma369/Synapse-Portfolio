"""
Intercompany Elimination Agent — Phase 3, Cross-Company Group.

================================================================================
BUSINESS PROBLEM (GAAP CONSOLIDATION)
================================================================================
When the PE fund consolidates its 8 portfolio companies into a single set of
financial statements, any transaction BETWEEN two portfolio companies must be
ELIMINATED. Otherwise, group revenue and group expenses are both inflated by
the intercompany markup — the classic "internal sales double-count" problem.
    
Example (why this matters):
    Company A sells $500K of services to Company B.
        - In Company A's books: Revenue += $500K
        - In Company B's books: Expense += $500K (or Inventory += $500K)
    At group level, this nets to zero economic activity (money never left the
    group). If A and B both book their side but the amounts don't MATCH (e.g.
    A booked $500K, B booked $480K), the group financials are mis-stated.

An intercompany reconciliation finds these asymmetries BEFORE consolidation.

TYPES OF ASYMMETRIES THIS AGENT DETECTS
---------------------------------------
    R1. DIRECTIONAL_ASYMMETRY
        Pair (A, B) has A→B flow of $500K but B→A flow of $0.
        In a mirrored-ledger world, B should have recorded its side of the
        SAME transaction (as a payable/purchase). Missing mirror = red flag.

    R2. AMOUNT_MISMATCH
        A→B = $500K, B→A = $480K on mirror pairs. The $20K difference is the
        unreconciled amount. Common cause: FX, cut-off timing, data entry.

    R3. DUPLICATE_TRANSACTION
        Same (seller, buyer, amount, date) appears more than once. Almost
        always a data-load bug that inflates both sides.

    R4. ORPHAN_ENTITY
        A transaction references a company_id that doesn't exist in the
        companies table. Would crash a naive consolidation.

    R5. HIGH_SINGLE_DIRECTION_VOLUME
        A single-direction flow (only A→B, no reverse) exceeds materiality.
        Escalated severity vs a small one-sided flow.

================================================================================
HYBRID DESIGN (Python Math + LLM Reasoning)
================================================================================
- Python does ALL of:
    - grouping, summing, pair matrix construction
    - asymmetry math, duplicate detection, orphan detection
    - materiality threshold checks
- Gemini does ONLY:
    - prioritizing which asymmetries matter
    - writing a controller-ready narrative
    - structuring output via Pydantic schema
- The LLM NEVER sees the raw transaction table and NEVER recalculates.
================================================================================
"""

from __future__ import annotations

import json
import logging
import time
import os
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import Company, IntercompanyTransaction

logger = logging.getLogger(__name__)
_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes") 


# ---------------------------------------------------------------------------
# Materiality constants — tune per PE firm's consolidation policy.
# ---------------------------------------------------------------------------
# Any asymmetry (|flow_AB - flow_BA|) above this is flagged.
ASYMMETRY_MATERIALITY_USD = Decimal("10000")

# Any single-direction flow above this with no mirror is escalated to HIGH.
SINGLE_DIRECTION_MATERIALITY_USD = Decimal("100000")


# =============================================================================
# 1. STRUCTURED OUTPUT SCHEMA
# =============================================================================

class IntercompanyPair(BaseModel):
    """One directional pair of entities (A→B and B→A flows summarised together)."""

    seller_id: str
    buyer_id: str
    flow_ab: float = Field(..., description="Total A→B amount for the period.")
    flow_ba: float = Field(..., description="Total B→A amount for the period.")
    asymmetry: float = Field(..., description="abs(flow_ab - flow_ba). Zero means perfect mirror.")
    direction: str = Field(
        ..., description="'MIRRORED' if both directions have flow, 'ONE_WAY' otherwise."
    )
    severity: str = Field(..., description="'HIGH', 'MEDIUM', or 'LOW'.")


class EliminationMismatch(BaseModel):
    """A specific rule violation for one entity pair."""

    seller_id: str
    buyer_id: str
    rule: str = Field(
        ...,
        description="One of: DIRECTIONAL_ASYMMETRY, AMOUNT_MISMATCH, DUPLICATE_TRANSACTION, "
                    "ORPHAN_ENTITY, HIGH_SINGLE_DIRECTION_VOLUME.",
    )
    amount: float = Field(..., description="Dollar amount implicated in this mismatch.")
    detail: str = Field(..., description="One-sentence explanation for the controller.")


class EliminationResult(BaseModel):
    """Final structured verdict returned by the agent for one period."""

    period: str
    status: str = Field(..., description="'CLEAN' if no mismatches, else 'MISMATCHES_FOUND'.")
    total_transactions: int
    unique_pairs: int
    mismatch_count: int
    total_asymmetry_usd: float = Field(
        ..., description="Sum of |flow_ab - flow_ba| across all mirrored pairs."
    )
    # NEW: matched IC flow — the portion that must be eliminated at group level.
    matched_intercompany_usd: float = Field(
        0.0,
        description="Sum of min(flow_ab, flow_ba) across all pairs — the amount "
                    "that nets to zero at group level.",
    )
    top_asymmetric_pairs: list[IntercompanyPair] = Field(default_factory=list)
    mismatches: list[EliminationMismatch] = Field(default_factory=list)
    summary: str

# =============================================================================
# 2. DETERMINISTIC ANALYSIS TOOL  (Python owns all math)
# =============================================================================

def _period_bounds(period: str) -> tuple[date, date]:
    """
    Return (first_day, last_day) for 'YYYY-MM'.

    We filter on the IC transactions' `date` column rather than a separate
    period column because our schema stores only the date.
    """
    y, m = (int(x) for x in period.split("-"))
    start = date(y, m, 1)
    if m == 12:
        end = date(y, 12, 31)
    else:
        end = date(y, m + 1, 1) - timedelta(days=1)
    return start, end


def verify_intercompany_eliminations(period: str) -> dict[str, Any]:
    """
    Deterministic intercompany reconciliation across ALL portfolio companies.

    This is the cross-company tool — unlike Phase 1/2 tools, it doesn't
    scope to a single company_id. It reads the full IC transaction ledger
    for the period and produces a pair-by-pair analysis.

    ALGORITHM
    ---------
    1. Load all IC transactions whose `date` falls within `period`.
    2. Detect orphan entities: transactions referencing company IDs not in
       the companies table. These are excluded from pair analysis (they can't
       be paired) but reported separately.
    3. Group transactions by directional pair (seller_id, buyer_id). For each
       pair, compute:
           flow_ab = sum of amounts where seller=A, buyer=B
           flow_ba = sum of amounts where seller=B, buyer=A
       Canonicalize (A,B) and (B,A) into a single record keyed by sorted IDs.
    4. For each canonical pair:
         - asymmetry = |flow_ab - flow_ba|
         - direction = MIRRORED if both > 0, else ONE_WAY
         - flag DIRECTIONAL_ASYMMETRY if asymmetry > materiality
         - flag HIGH_SINGLE_DIRECTION_VOLUME if ONE_WAY and max_flow > $100K
    5. Detect DUPLICATE_TRANSACTION: identical (seller, buyer, amount, date)
       appearing more than once.
    6. Sort mismatches by absolute amount descending.

    Returns:
        JSON-serialisable dict. Never raises.
    """
    db = SessionLocal()
    try:
        start, end = _period_bounds(period)

        # ---- 1. Load IC transactions for the period ----------------------
        txns = db.scalars(
            select(IntercompanyTransaction).where(
                IntercompanyTransaction.date >= start,
                IntercompanyTransaction.date <= end,
            )
        ).all()

        if not txns:
            return {
                "period": period, "found": False,
                "error": f"No intercompany transactions found for period {period}.",
                "total_transactions": 0, "unique_pairs": 0,
                "mismatches": [], "pairs": [],
                "total_asymmetry_usd": 0.0,
            }

        # ---- 2. Load valid company IDs (for orphan detection) ------------
        valid_company_ids = {
            cid for (cid,) in db.execute(select(Company.id)).all()
        }

        # ---- 3. Directional flow accumulator -----------------------------
        # Key: (seller, buyer) as ordered tuple → total amount
        flows: dict[tuple[str, str], Decimal] = defaultdict(lambda: Decimal("0"))
        # Key: (seller, buyer, amount_str, date_str) → occurrence count (duplicates)
        txn_signatures: dict[tuple[str, str, str, str], int] = defaultdict(int)

        orphan_txns: list[IntercompanyTransaction] = []

        for t in txns:
            seller = t.selling_entity_id
            buyer = t.buying_entity_id

            # Orphan check: either endpoint not in companies table
            if seller not in valid_company_ids or buyer not in valid_company_ids:
                orphan_txns.append(t)
                continue

            amount = t.amount or Decimal("0")
            flows[(seller, buyer)] += amount

            # Duplicate signature (excludes txn_id by design — identical
            # business facts with different IDs is the pattern we hunt)
            sig = (
                seller,
                buyer,
                str(amount),
                t.date.isoformat() if t.date else "",
            )
            txn_signatures[sig] += 1

        # ---- 4. Build canonical pairs (A,B) and (B,A) merged --------------
        # We canonicalize on sorted IDs so (A→B) and (B→A) collide cleanly.
        canonical: dict[tuple[str, str], dict[str, Decimal]] = {}
        for (a, b), amt in flows.items():
            key = tuple(sorted([a, b]))
            if key not in canonical:
                canonical[key] = {"ab": Decimal("0"), "ba": Decimal("0")}
            # Determine which original direction this amount belongs to
            if (a, b) == (key[0], key[1]):
                canonical[key]["ab"] += amt
            else:
                canonical[key]["ba"] += amt

        # ---- 5. Score each pair ------------------------------------------
        pairs: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []
        total_asymmetry = Decimal("0")

        for (lo, hi), agg in canonical.items():
            flow_ab = agg["ab"]
            flow_ba = agg["ba"]
            asymmetry = abs(flow_ab - flow_ba)
            total_asymmetry += asymmetry

            has_ab = flow_ab > 0
            has_ba = flow_ba > 0
            direction = "MIRRORED" if (has_ab and has_ba) else "ONE_WAY"

            # Severity grading:
            #   HIGH   = asymmetry > single_direction_materiality (100K)
            #   MEDIUM = asymmetry > asymmetry_materiality (10K)
            #   LOW    = below threshold
            if asymmetry > SINGLE_DIRECTION_MATERIALITY_USD:
                severity = "HIGH"
            elif asymmetry > ASYMMETRY_MATERIALITY_USD:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            pairs.append({
                "seller_id": lo,
                "buyer_id": hi,
                "flow_ab": float(flow_ab),
                "flow_ba": float(flow_ba),
                "asymmetry": float(asymmetry),
                "direction": direction,
                "severity": severity,
            })

            # R1 / R5: DIRECTIONAL_ASYMMETRY
            if direction == "MIRRORED" and asymmetry > ASYMMETRY_MATERIALITY_USD:
                mismatches.append({
                    "seller_id": lo, "buyer_id": hi,
                    "rule": "DIRECTIONAL_ASYMMETRY",
                    "amount": float(asymmetry),
                    "detail": (
                        f"Mirrored flows disagree by ${asymmetry}: "
                        f"{lo}→{hi}=${flow_ab}, {hi}→{lo}=${flow_ba}."
                    ),
                })
            elif direction == "ONE_WAY" and max(flow_ab, flow_ba) > SINGLE_DIRECTION_MATERIALITY_USD:
                mismatches.append({
                    "seller_id": lo, "buyer_id": hi,
                    "rule": "HIGH_SINGLE_DIRECTION_VOLUME",
                    "amount": float(max(flow_ab, flow_ba)),
                    "detail": (
                        f"One-way flow of ${max(flow_ab, flow_ba)} between {lo} and {hi} "
                        f"with no reverse entries — likely a missing mirror booking."
                    ),
                })
        matched_intercompany = Decimal("0")
        for (lo, hi), agg in canonical.items():
            matched_intercompany += min(agg["ab"], agg["ba"])

        # ---- 6. Duplicate detection --------------------------------------
        for (seller, buyer, amount_str, date_str), count in txn_signatures.items():
            if count > 1:
                mismatches.append({
                    "seller_id": seller, "buyer_id": buyer,
                    "rule": "DUPLICATE_TRANSACTION",
                    "amount": float(Decimal(amount_str)),
                    "detail": (
                        f"Identical transaction (seller={seller}, buyer={buyer}, "
                        f"amount={amount_str}, date={date_str}) appears {count} times."
                    ),
                })

        # ---- 7. Orphan entities ------------------------------------------
        for t in orphan_txns:
            mismatches.append({
                "seller_id": t.selling_entity_id or "UNKNOWN",
                "buyer_id": t.buying_entity_id or "UNKNOWN",
                "rule": "ORPHAN_ENTITY",
                "amount": float(t.amount or Decimal("0")),
                "detail": (
                    f"Transaction {t.transaction_id} references an entity not in the "
                    f"companies table — cannot be consolidated."
                ),
            })

        # Sort all mismatches by absolute amount, biggest first
        mismatches.sort(key=lambda x: abs(x["amount"]), reverse=True)
        # Sort pairs the same way — LLM narrative picks the top off this list
        pairs.sort(key=lambda x: abs(x["asymmetry"]), reverse=True)

        return {
            "period": period,
            "found": True,
            "total_transactions": len(txns),
            "unique_pairs": len(canonical),
            "mismatch_count": len(mismatches),
            "total_asymmetry_usd": float(total_asymmetry),
            "top_asymmetric_pairs": pairs[:10],  # top 10 for the LLM
            "matched_intercompany_usd": float(matched_intercompany),
            "mismatches": mismatches,
            "orphan_count": len(orphan_txns),
            "note": (
                f"{len(mismatches)} elimination mismatches detected across "
                f"{len(canonical)} pairs."
                if mismatches else "All intercompany flows reconcile cleanly."
            ),
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("verify_intercompany_eliminations failed for %s", period)
        return {
            "period": period, "found": False,
            "error": str(exc),
            "total_transactions": 0, "unique_pairs": 0,
            "mismatches": [], "pairs": [],
            "total_asymmetry_usd": 0.0,
        }
    finally:
        db.close()

# =============================================================================
# TOOL 2 — get_pair_detail
# =============================================================================

def get_pair_detail(entity_a: str, entity_b: str, period: str) -> dict[str, Any]:
    """
    Return the raw IC transactions between two specific entities in a period.

    Use this after `verify_intercompany_eliminations` flags a pair, to see
    the underlying transactions that caused the asymmetry.
    """
    from datetime import date, timedelta
    y, m = (int(x) for x in period.split("-"))
    start = date(y, m, 1)
    end = (date(y, 12, 31) if m == 12 else date(y, m + 1, 1) - timedelta(days=1))

    db = SessionLocal()
    try:
        rows = db.scalars(
            select(IntercompanyTransaction).where(
                IntercompanyTransaction.date >= start,
                IntercompanyTransaction.date <= end,
            )
        ).all()

        filtered = [
            {"transaction_id": t.transaction_id,
             "date": t.date.isoformat(),
             "seller_id": t.selling_entity_id,
             "buyer_id": t.buying_entity_id,
             "amount": float(t.amount or 0),
             "description": t.description or ""}
            for t in rows
            if {t.selling_entity_id, t.buying_entity_id} == {entity_a, entity_b}
        ]
        return {"entity_a": entity_a, "entity_b": entity_b, "period": period,
                "transactions": filtered, "count": len(filtered)}
    finally:
        db.close()


# =============================================================================
# TOOL 3 — get_elimination_candidates
# =============================================================================

def get_elimination_candidates(period: str, min_amount: float = 50000.0) -> list[dict[str, Any]]:
    """
    List pairs whose matched IC flow exceeds `min_amount`.

    These are the pairs the consolidation must net. Returns total matched
    and asymmetry for each so the LLM can reason about which matter.
    """
    facts = verify_intercompany_eliminations(period)
    if not facts.get("found"):
        return []
    out = []
    for p in facts.get("top_asymmetric_pairs", []):
        matched = min(p["flow_ab"], p["flow_ba"])
        if matched >= min_amount:
            out.append({
                "pair": f"{p['seller_id']} ↔ {p['buyer_id']}",
                "matched_amount": matched,
                "asymmetry": p["asymmetry"],
                "severity": p["severity"],
            })
    return out

def _latest_period_globally() -> str | None:
    """
    Find the latest month for which ANY intercompany transaction exists.

    Phase 3 is company-agnostic, so we can't use the per-company helper from
    Phase 1/2. This scans the IC ledger directly.
    """
    db = SessionLocal()
    try:
        latest_date = db.scalar(
            select(IntercompanyTransaction.date)
            .order_by(IntercompanyTransaction.date.desc())
            .limit(1)
        )
        if not latest_date:
            return None
        return latest_date.strftime("%Y-%m")
    finally:
        db.close()


# =============================================================================
# 3. AGNO AGENT DEFINITION
# =============================================================================

AGENT_INSTRUCTIONS = [
    "You are an Intercompany Elimination Agent for a PE month-end close system.",
    "You have THREE tools. Reason about which to call.",
    "",
    "TOOLS:",
    "  1. verify_intercompany_eliminations(period) — baseline. ALWAYS first.",
    "  2. get_pair_detail(entity_a, entity_b, period) — raw transactions for",
    "     one pair. Use on the top asymmetric pair to see what caused the gap.",
    "  3. get_elimination_candidates(period, min_amount) — pairs with material",
    "     matched IC flow (> $50K default). These MUST be netted at group level.",
    "",
    "WORKFLOW:",
    "  - Step 1: verify_intercompany_eliminations.",
    "  - Step 2: get_elimination_candidates to see which pairs need netting.",
    "  - Step 3: get_pair_detail on the largest asymmetry to see the source.",
    "  - Step 4: STOP after 3-4 tool calls.",
    "",
    "STRICT RULES:",
    "1. NEVER invent numbers.",
    "2. Include EVERY mismatch from the tool.",
    "3. status='CLEAN' iff mismatch_count == 0, else 'MISMATCHES_FOUND'.",
    "4. `summary` = 3-4 sentences leading with the largest mismatch.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return Agent(
        name="Intercompany Elimination Agent",
        model=Gemini(id="gemini-3.1-flash-lite", api_key=settings.gemini_api_key),
        tools=[
            verify_intercompany_eliminations,
            get_pair_detail,
            get_elimination_candidates,
        ],
        description="Reconciles intercompany flows and identifies elimination candidates.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=EliminationResult,
        markdown=False,
        use_json_mode=True,
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=4,
    )

# =============================================================================
# 4. PUBLIC ENTRYPOINT
# =============================================================================

def _safe_parse(content: Any) -> EliminationResult:
    """
    Parse the agent response into EliminationResult.

    Detects Gemini error payloads (which arrive as dicts/JSON strings with an
    'error' key on 429/503) and raises cleanly so the retry/fallback layer
    can do its job.
    """
    if isinstance(content, EliminationResult):
        return content
    if isinstance(content, dict):
        if "error" in content:
            raise RuntimeError(f"Gemini API error: {content['error']}")
        return EliminationResult(**content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"Gemini API error: {parsed['error']}")
            return EliminationResult(**parsed)
        raise ValueError(f"Non-JSON agent response: {stripped[:200]}")
    raise ValueError(f"Unexpected agent response type: {type(content)}")


def run_elimination(period: str | None = None) -> EliminationResult:
    """
    Run the Intercompany Elimination Agent for one period.

    Args:
        period: 'YYYY-MM'. If None, resolves to the latest period present in
                the intercompany ledger.

    Returns:
        EliminationResult. Falls back to a Python-only verdict if the LLM is
        unreachable — the orchestrator never blocks on a Gemini rate-limit.
    """
    if period is None:
        period = _latest_period_globally()
        if period is None:
            return EliminationResult(
                period="UNKNOWN", status="MISMATCHES_FOUND",
                total_transactions=0, unique_pairs=0, mismatch_count=0,
                total_asymmetry_usd=0.0,
                top_asymmetric_pairs=[], mismatches=[],
                summary="No intercompany transactions exist in the ledger.",
            )

    # ---- TIER 1: deterministic pre-check --------------------------------
    precheck = verify_intercompany_eliminations(period)
    if precheck.get("found") and precheck.get("mismatch_count", 0) == 0:
        logger.info("[Elimination] %s clean — skipping LLM.", period)
        pairs = [IntercompanyPair(**p) for p in precheck.get("top_asymmetric_pairs", [])]
        return EliminationResult(
            period=period, status="CLEAN",
            total_transactions=precheck.get("total_transactions", 0),
            unique_pairs=precheck.get("unique_pairs", 0),
            mismatch_count=0,
            total_asymmetry_usd=precheck.get("total_asymmetry_usd", 0.0),
            matched_intercompany_usd=precheck.get("matched_intercompany_usd", 0.0),
            top_asymmetric_pairs=pairs, mismatches=[],
            summary=(
                f"All {precheck.get('unique_pairs', 0)} intercompany pairs "
                f"reconcile cleanly across {precheck.get('total_transactions', 0)} "
                f"transactions. Matched IC flow "
                f"${precheck.get('matched_intercompany_usd', 0.0):,.2f}. "
                f"LLM reasoning skipped."
            ),
        )

    prompt = (
    f"Reconcile intercompany eliminations for period={period!r}. "
    f"Use tools to identify elimination candidates and investigate the "
    f"largest asymmetry. Then produce the structured result."
)

    # ---- Try LLM path up to 3 times on transient errors ------------------
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            agent = _build_agent()
            response = agent.run(prompt)
            return _safe_parse(response.content)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            if any(k in msg for k in ("503", "unavailable", "429", "quota", "timeout", "deadline")) and attempt < 2:
                import random as _rnd; time.sleep((5 + _rnd.random() * 3) * (attempt + 1))  # 2s, 4s
                logger.warning("Transient error on elimination attempt %d — retrying.", attempt + 1)
                continue
            break

    # ---- Deterministic fallback ------------------------------------------
    logger.warning(
        "LLM path failed for elimination %s (%s) — falling back to deterministic result.",
        period, type(last_exc).__name__,
    )
    facts = verify_intercompany_eliminations(period)

    pairs = [IntercompanyPair(**p) for p in facts.get("top_asymmetric_pairs", [])]
    mismatches = [
        EliminationMismatch(
            seller_id=m["seller_id"], buyer_id=m["buyer_id"],
            rule=m["rule"], amount=m["amount"], detail=m["detail"],
        )
        for m in facts.get("mismatches", [])
    ]

    return EliminationResult(
        period=period,
        status="CLEAN" if not mismatches else "MISMATCHES_FOUND",
        total_transactions=facts.get("total_transactions", 0),
        unique_pairs=facts.get("unique_pairs", 0),
        mismatch_count=len(mismatches),
        total_asymmetry_usd=facts.get("total_asymmetry_usd", 0.0),
        matched_intercompany_usd=facts.get("matched_intercompany_usd", 0.0),
        top_asymmetric_pairs=pairs,
        mismatches=mismatches,
        summary=(
            f"Deterministic result (LLM narrative skipped: {type(last_exc).__name__}). "
            f"Deterministic result: {len(mismatches)} intercompany mismatches "
            f"across {facts.get('unique_pairs', 0)} pairs, "
            f"aggregate asymmetry ${facts.get('total_asymmetry_usd', 0.0):,.2f}."
        ),
    )