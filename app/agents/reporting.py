"""
Reporting & Communication Agent — Phase 5, Final Notification.

================================================================================
PURPOSE
================================================================================
When the month-end close pipeline finishes (Phase 4 consolidation), the CFO
and stakeholders need a single, professional email summarising the group
financials. This agent:

    1. Reads the Phase 4 ConsolidationResult from Redis
    2. Renders it into a clean HTML email via Jinja2
    3. Sends it via Resend (or logs to stdout if no API key — mock mode)

GRACEFUL DEGRADATION
--------------------
The reporting step is downstream of the actual financial work. If it fails,
the close is still valid — the CFO can always re-query Redis. Therefore we:
    - Never raise on missing Redis data (return False, log warning)
    - Never raise on missing RESEND_API_KEY (mock mode — print HTML)
    - Never raise on Resend API errors (log the error, return False)
================================================================================
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis
from jinja2 import Template

from app.db.database import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML TEMPLATE — table-based layout for maximum email-client compatibility
# (Gmail, Outlook, Apple Mail all mangle modern CSS; tables just work).
# Inline styles only — no <style> blocks, no external CSS, no web fonts.
# ---------------------------------------------------------------------------
EMAIL_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Month-End Close Report</title></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#111827;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 0;">
  <tr><td align="center">
    <table width="640" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,0.06);">

      <!-- Header -->
      <tr><td style="background:#0f172a;padding:28px 40px;">
        <div style="color:#ffffff;font-size:20px;font-weight:600;letter-spacing:-0.3px;">Synapse Capital</div>
        <div style="color:#94a3b8;font-size:13px;margin-top:4px;">Autonomous Month-End Close</div>
      </td></tr>

      <!-- Status strip -->
      <tr><td style="background:#065f46;padding:12px 40px;">
        <div style="color:#d1fae5;font-size:13px;font-weight:500;">&check; CLOSE COMPLETED &nbsp;&middot;&nbsp; {{ period }}</div>
      </td></tr>

      <!-- Run ID -->
      <tr><td style="padding:32px 40px 8px;">
        <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Run ID</div>
        <div style="font-size:12px;color:#374151;font-family:'Courier New',monospace;margin-top:4px;">{{ run_id }}</div>
      </td></tr>

      <!-- KPI cards -->
      <tr><td style="padding:20px 40px 8px;">
        <table width="100%" cellpadding="0" cellspacing="0"><tr>
          <td width="32%" style="vertical-align:top;">
            <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px;">
              <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Gross Revenue</div>
              <div style="font-size:22px;font-weight:600;color:#111827;margin-top:6px;">{{ gross_revenue }}</div>
            </div>
          </td>
          <td width="2%"></td>
          <td width="32%" style="vertical-align:top;">
            <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px;">
              <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Raw EBITDA</div>
              <div style="font-size:22px;font-weight:600;color:#111827;margin-top:6px;">{{ raw_ebitda }}</div>
            </div>
          </td>
          <td width="2%"></td>
          <td width="32%" style="vertical-align:top;">
            <div style="background:{{ adj_bg }};border:1px solid {{ adj_border }};border-radius:8px;padding:16px;">
              <div style="font-size:11px;color:{{ adj_label }};text-transform:uppercase;letter-spacing:.06em;">Adjusted Group EBITDA</div>
              <div style="font-size:22px;font-weight:600;color:{{ adj_value }};margin-top:6px;">{{ adjusted_group_ebitda }}</div>
            </div>
          </td>
        </tr></table>
      </td></tr>

      <!-- Executive Summary -->
      <tr><td style="padding:28px 40px 8px;">
        <div style="font-size:15px;font-weight:600;color:#111827;margin-bottom:12px;">Executive Summary</div>
        <div style="background:#eff6ff;border-left:3px solid #3b82f6;padding:16px 18px;border-radius:6px;font-size:14px;line-height:1.65;color:#1e3a8a;">
          {{ executive_summary }}
        </div>
      </td></tr>

      <!-- Secondary metrics -->
      <tr><td style="padding:24px 40px 8px;">
        <table width="100%" cellpadding="0" cellspacing="0" style="border-top:1px solid #e5e7eb;">
          <tr>
            <td width="25%" style="padding:16px 0 4px;">
              <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">COGS</div>
              <div style="font-size:15px;color:#374151;margin-top:4px;">{{ total_cogs }}</div>
            </td>
            <td width="25%" style="padding:16px 0 4px;">
              <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Gross Profit</div>
              <div style="font-size:15px;color:#374151;margin-top:4px;">{{ gross_profit }}</div>
            </td>
            <td width="25%" style="padding:16px 0 4px;">
              <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">OpEx</div>
              <div style="font-size:15px;color:#374151;margin-top:4px;">{{ total_opex }}</div>
            </td>
            <td width="25%" style="padding:16px 0 4px;">
              <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Entities</div>
              <div style="font-size:15px;color:#374151;margin-top:4px;">{{ entity_count }}</div>
            </td>
          </tr>
        </table>
      </td></tr>

      <!-- Elimination note -->
      <tr><td style="padding:8px 40px 16px;">
        <div style="font-size:12px;color:#6b7280;line-height:1.6;">
          Adjusted Group EBITDA = Raw EBITDA &minus; {{ elimination_asymmetry }} in intercompany elimination asymmetries
          detected at Phase 3. This conservative haircut is the number reported to LPs.
        </div>
      </td></tr>

      <!-- Footer -->
      <tr><td style="background:#f9fafb;padding:20px 40px;border-top:1px solid #e5e7eb;">
        <div style="font-size:12px;color:#9ca3af;line-height:1.6;">
          Generated by the Synapse Capital Autonomous Close System.<br>
          Do not reply to this email. Contact the finance team for inquiries.
        </div>
      </td></tr>

    </table>
  </td></tr>
</table>
</body>
</html>
"""


# =============================================================================
# FORMATTERS
# =============================================================================
def _fmt_usd(v: float | int | None) -> str:
    """Format as $X,XXX,XXX.XX. Never raises."""
    try:
        return f"${float(v or 0):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _render_html(run_id: str, final: dict[str, Any]) -> str:
    """
    Render the Jinja2 template with all data, including dynamic styling for
    the adjusted EBITDA card (green if positive, red if negative).
    """
    adjusted = float(final.get("adjusted_group_ebitda", 0.0))

    if adjusted >= 0:
        adj_bg, adj_border = "#ecfdf5", "#a7f3d0"
        adj_label, adj_value = "#047857", "#065f46"
    else:
        adj_bg, adj_border = "#fef2f2", "#fecaca"
        adj_label, adj_value = "#b91c1c", "#991b1b"

    return Template(EMAIL_TEMPLATE).render(
        run_id=run_id,
        period=final.get("period", "—"),
        gross_revenue=_fmt_usd(final.get("gross_revenue")),
        raw_ebitda=_fmt_usd(final.get("raw_ebitda")),
        adjusted_group_ebitda=_fmt_usd(adjusted),
        total_cogs=_fmt_usd(final.get("total_cogs")),
        gross_profit=_fmt_usd(final.get("gross_profit")),
        total_opex=_fmt_usd(final.get("total_opex")),
        elimination_asymmetry=_fmt_usd(final.get("elimination_asymmetry")),
        entity_count=final.get("entity_count", 0),
        executive_summary=final.get("executive_summary", "No summary available."),
        adj_bg=adj_bg,
        adj_border=adj_border,
        adj_label=adj_label,
        adj_value=adj_value,
    )


# =============================================================================
# PUBLIC ENTRYPOINT
# =============================================================================
def send_executive_summary_email(run_id: str) -> bool:
    """
    Send the month-end close executive summary email.

    Returns:
        True  — email sent (real or mocked)
        False — could not send (missing data, API error). Never raises.
    """
    # ---- 1. Fetch the Phase 4 result from Redis --------------------------
    try:
        r = redis.from_url(settings.redis_url, decode_responses=True)
        raw = r.get(f"close:{run_id}:final_result")
    except Exception as exc:  # noqa: BLE001
        logger.error("Reporting: Redis read failed for run %s: %s", run_id, exc)
        return False

    if not raw:
        logger.warning(
            "Reporting: no final_result found for run %s — nothing to send.",
            run_id,
        )
        return False

    try:
        final = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error("Reporting: failed to parse final_result JSON for %s: %s", run_id, exc)
        return False

    # ---- 2. Render the HTML email ----------------------------------------
    try:
        html_content = _render_html(run_id, final)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reporting: template render failed for %s: %s", run_id, exc)
        return False

    subject = (
        f"[Month-End Close] {final.get('period', 'Report')} — "
        f"Adjusted EBITDA {_fmt_usd(final.get('adjusted_group_ebitda'))}"
    )

    # ---- 3. MOCK MODE — no API key configured ----------------------------
    if not settings.resend_api_key:
        logger.info(
            "MOCK EMAIL SENT (no RESEND_API_KEY configured)\n"
            "  To:      %s\n"
            "  From:    %s\n"
            "  Subject: %s\n"
            "  HTML length: %d chars",
            settings.to_email, settings.from_email, subject, len(html_content),
        )
        return True

    # ---- 4. Real send via Resend -----------------------------------------
    try:
        import resend  # lazy import — only needed in real-send mode

        resend.api_key = settings.resend_api_key
        params = {
            "from": settings.from_email,
            "to": [settings.to_email],
            "subject": subject,
            "html": html_content,
        }
        response = resend.Emails.send(params)
        logger.info(
            "Reporting: email sent for run %s — Resend response id=%s",
            run_id, response.get("id") if isinstance(response, dict) else response,
        )
        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Reporting: Resend send failed for %s: %s", run_id, exc)
        return False