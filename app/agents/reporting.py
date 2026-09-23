"""
Reporting & Communication Agent — Phase 5 executor.

This module is the SEND LAYER. It renders Jinja2 HTML and calls Resend
(or logs to stdout in mock mode).

The DECISION layer lives in `app/agents/reporting_agent.py` — that Agno
agent decides which emails to send based on run state.

GRACEFUL DEGRADATION
--------------------
Never raises. Mock mode if RESEND_API_KEY unset. Logs and returns False
on any Resend error.
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
# HTML TEMPLATE — table-based layout for maximum email-client compatibility.
# ---------------------------------------------------------------------------
EMAIL_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Month-End Close Report</title></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#111827;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 0;">
  <tr><td align="center">
    <table width="640" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,0.06);">

      <tr><td style="background:#0f172a;padding:28px 40px;">
        <div style="color:#ffffff;font-size:20px;font-weight:600;letter-spacing:-0.3px;">Apex Capital Partners</div>
        <div style="color:#94a3b8;font-size:13px;margin-top:4px;">Autonomous Month-End Close</div>
      </td></tr>

      <tr><td style="background:#065f46;padding:12px 40px;">
        <div style="color:#d1fae5;font-size:13px;font-weight:500;">&check; CLOSE COMPLETED &nbsp;&middot;&nbsp; {{ period }}</div>
      </td></tr>

      <tr><td style="padding:32px 40px 8px;">
        <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Run ID</div>
        <div style="font-size:12px;color:#374151;font-family:'Courier New',monospace;margin-top:4px;">{{ run_id }}</div>
      </td></tr>

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

      <tr><td style="padding:28px 40px 8px;">
        <div style="font-size:15px;font-weight:600;color:#111827;margin-bottom:12px;">Executive Summary</div>
        <div style="background:#eff6ff;border-left:3px solid #3b82f6;padding:16px 18px;border-radius:6px;font-size:14px;line-height:1.65;color:#1e3a8a;">
          {{ executive_summary }}
        </div>
      </td></tr>

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

      <tr><td style="padding:8px 40px 16px;">
        <div style="font-size:12px;color:#6b7280;line-height:1.6;">
          Adjusted Group EBITDA = Raw EBITDA &minus; {{ elimination_asymmetry }} in intercompany elimination asymmetries
          detected at Phase 3. This conservative haircut is the number reported to LPs.
        </div>
      </td></tr>

      <tr><td style="background:#f9fafb;padding:20px 40px;border-top:1px solid #e5e7eb;">
        <div style="font-size:12px;color:#9ca3af;line-height:1.6;">
          Generated by the Apex Capital Autonomous Close System.<br>
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
    """Render the Jinja2 template with dynamic EBITDA card styling."""
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
        adj_bg=adj_bg, adj_border=adj_border,
        adj_label=adj_label, adj_value=adj_value,
    )


# =============================================================================
# PUBLIC SENDERS
# =============================================================================

def send_executive_summary_email(run_id: str) -> bool:
    """Send the completion email with the group financial summary."""
    try:
        r = redis.from_url(settings.redis_url, decode_responses=True)
        raw = r.get(f"close:{run_id}:final_result")
    except Exception as exc:  # noqa: BLE001
        logger.error("Reporting: Redis read failed for run %s: %s", run_id, exc)
        return False

    if not raw:
        logger.warning("Reporting: no final_result for run %s — nothing to send.", run_id)
        return False

    try:
        final = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error("Reporting: JSON parse failed for %s: %s", run_id, exc)
        return False

    try:
        html_content = _render_html(run_id, final)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reporting: template render failed for %s: %s", run_id, exc)
        return False

    subject = (
        f"[Month-End Close] {final.get('period', 'Report')} — "
        f"Adjusted EBITDA {_fmt_usd(final.get('adjusted_group_ebitda'))}"
    )

    if not settings.resend_api_key:
        logger.info(
            "MOCK EMAIL SENT (no RESEND_API_KEY configured)\n"
            "  To:      %s\n  From:    %s\n  Subject: %s\n  HTML length: %d chars",
            settings.to_email, settings.from_email, subject, len(html_content),
        )
        return True

    try:
        import resend
        resend.api_key = settings.resend_api_key
        response = resend.Emails.send({
            "from": settings.from_email,
            "to": [settings.to_email],
            "subject": subject,
            "html": html_content,
        })
        logger.info(
            "Reporting: email sent for run %s — Resend id=%s",
            run_id, response.get("id") if isinstance(response, dict) else response,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reporting: Resend send failed for %s: %s", run_id, exc)
        return False


def _send_with_template(run_id: str, header_text: str, subject: str) -> bool:
    """Shared sender for daily / weekly / issue-alert emails."""
    try:
        r = redis.from_url(settings.redis_url, decode_responses=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Reporting[%s]: Redis connect failed: %s", header_text, exc)
        return False

    raw = None
    try:
        raw = r.get(f"close:{run_id}:final_result")
    except Exception as exc:  # noqa: BLE001
        logger.error("Reporting[%s]: Redis GET failed: %s", header_text, exc)
        return False

    if not raw:
        try:
            raw = r.get(f"close:{run_id}:phase3:result")
        except Exception:
            raw = None

    if not raw:
        logger.warning(
            "Reporting[%s]: no data available for run %s — skipping.",
            header_text, run_id,
        )
        return False

    try:
        final = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error("Reporting[%s]: JSON parse failed: %s", header_text, exc)
        return False

    try:
        adjusted = float(final.get("adjusted_group_ebitda", 0.0) or 0.0)
        if adjusted >= 0:
            adj_bg, adj_border = "#ecfdf5", "#a7f3d0"
            adj_label, adj_value = "#047857", "#065f46"
        else:
            adj_bg, adj_border = "#fef2f2", "#fecaca"
            adj_label, adj_value = "#b91c1c", "#991b1b"

        html = Template(EMAIL_TEMPLATE).render(
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
            executive_summary=final.get(
                "executive_summary",
                final.get("summary", "No summary available yet."),
            ),
            adj_bg=adj_bg, adj_border=adj_border,
            adj_label=adj_label, adj_value=adj_value,
        )
        html = html.replace("CLOSE COMPLETED", header_text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reporting[%s]: render failed: %s", header_text, exc)
        return False

    if not settings.resend_api_key:
        logger.info(
            "MOCK EMAIL [%s]\n  To:      %s\n  Subject: %s\n  HTML length: %d",
            header_text, settings.to_email, subject, len(html),
        )
        return True

    try:
        import resend
        resend.api_key = settings.resend_api_key
        resend.Emails.send({
            "from": settings.from_email,
            "to": [settings.to_email],
            "subject": subject,
            "html": html,
        })
        logger.info("Reporting[%s]: sent for run %s", header_text, run_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reporting[%s]: Resend failed: %s", header_text, exc)
        return False


def send_daily_summary_email(run_id: str) -> bool:
    """Daily progress summary (scheduled 8 AM)."""
    return _send_with_template(
        run_id,
        header_text="DAILY PROGRESS",
        subject="Daily Progress — Month-End Close",
    )


def send_weekly_stakeholder_report(run_id: str) -> bool:
    """Weekly stakeholder report (scheduled Monday 8 AM)."""
    return _send_with_template(
        run_id,
        header_text="WEEKLY REPORT",
        subject="Weekly Stakeholder Report — Apex Capital",
    )


def send_issue_alert_email(run_id: str) -> bool:
    """Condition-triggered issue alert. Fires only if an issue exists."""
    try:
        r = redis.from_url(settings.redis_url, decode_responses=True)
        phase3_status = r.get(f"close:{run_id}:phase3_status")
        escalation = r.get(f"close:{run_id}:escalation")
    except Exception:  # noqa: BLE001
        return False

    has_issue = (phase3_status == "MISMATCHES_FOUND") or (escalation is not None)
    if not has_issue:
        logger.info("Issue alert skipped for run %s — no active issues.", run_id)
        return False

    reason = escalation or phase3_status or "unknown"
    return _send_with_template(
        run_id,
        header_text=f"ISSUES DETECTED — {reason}",
        subject="⚠️ Close Issues Detected — Action Required",
    )


# =============================================================================
# DISPATCH HELPER — used by reporting_agent
# =============================================================================

def send_by_type(run_id: str, email_type: str) -> bool:
    """
    Send an email by its type string. Used by the Reporting Agent's executor.

    Valid types: 'completion', 'daily_summary', 'weekly_report', 'issue_alert'.
    Returns True on successful send (or mock), False otherwise.
    """
    dispatch = {
        "completion": send_executive_summary_email,
        "daily_summary": send_daily_summary_email,
        "weekly_report": send_weekly_stakeholder_report,
        "issue_alert": send_issue_alert_email,
    }
    fn = dispatch.get(email_type)
    if fn is None:
        logger.warning("Unknown email type: %s", email_type)
        return False
    return fn(run_id)