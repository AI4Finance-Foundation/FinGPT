"""
Compliance Gate — KYT/AML screening before payout release.

Responsibilities:
- Screen destination wallet (if wallet delivery) against KYT provider
- Screen beneficiary against sanctions lists
- Apply corridor-specific risk thresholds
- Hold or release each task
- Emit compliance events to audit_logs and task_events topic

This service is deterministic — no LLM calls. The Audit Agent reads results later.
"""
from __future__ import annotations

import structlog
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = structlog.get_logger()


class ComplianceOutcome(str, Enum):
    PASS = "pass"
    HOLD = "hold"
    FAIL = "fail"


@dataclass
class ComplianceResult:
    task_id: str
    outcome: ComplianceOutcome
    risk_score: Optional[int]       # 0-10 from KYT provider
    sanctions_hit: bool
    hold_reason: Optional[str]
    provider_ref: Optional[str]
    raw_response: dict


class ComplianceGate:
    """
    Orchestrates KYT + sanctions screening for a single payout task.
    Calls the configured KYT provider via the MCP client.
    """

    def __init__(self, kyt_client, compliance_profile):
        self.kyt = kyt_client
        self.cp = compliance_profile

    async def screen(
        self,
        task_id: str,
        corridor_code: str,
        amount_usd: float,
        resolved_mode: str,
        recipient_wallet: Optional[str] = None,
        recipient_name: Optional[str] = None,
        beneficiary_data: Optional[dict] = None,
    ) -> ComplianceResult:
        beneficiary_data = beneficiary_data or {}
        provider_ref = None
        risk_score = None
        sanctions_hit = False
        raw_response = {}

        # ── Sanctions screening (always) ──────────────────────────────────────
        if self.cp.sanctions_screen_enabled and recipient_name:
            sanctions_result = await self.kyt.screen_sanctions(
                name=recipient_name,
                country=corridor_code,
            )
            sanctions_hit = sanctions_result.get("hit", False)
            raw_response["sanctions"] = sanctions_result

        if sanctions_hit:
            log.warning("sanctions_hit", task_id=task_id, name=recipient_name)
            return ComplianceResult(
                task_id=task_id,
                outcome=ComplianceOutcome.FAIL,
                risk_score=10,
                sanctions_hit=True,
                hold_reason="Sanctions match — task permanently failed",
                provider_ref=None,
                raw_response=raw_response,
            )

        # ── KYT / blockchain screening (wallet delivery only) ─────────────────
        if self.cp.kyt_enabled and resolved_mode == "wallet" and recipient_wallet:
            kyt_result = await self.kyt.screen_address(
                address=recipient_wallet,
                asset=beneficiary_data.get("asset", "USDT"),
                amount_usd=amount_usd,
            )
            risk_score = kyt_result.get("risk_score", 0)
            provider_ref = kyt_result.get("ref")
            raw_response["kyt"] = kyt_result

        # ── Apply corridor risk thresholds ────────────────────────────────────
        outcome = ComplianceOutcome.PASS
        hold_reason = None

        if risk_score is not None:
            if risk_score >= self.cp.manual_review_threshold:
                outcome = ComplianceOutcome.HOLD
                hold_reason = f"KYT risk score {risk_score} >= manual review threshold {self.cp.manual_review_threshold}"
            elif risk_score >= self.cp.kyt_risk_threshold and self.cp.auto_hold_on_high_risk:
                outcome = ComplianceOutcome.HOLD
                hold_reason = f"KYT risk score {risk_score} >= corridor threshold {self.cp.kyt_risk_threshold}"

        log.info(
            "compliance_screened",
            task_id=task_id,
            outcome=outcome,
            risk_score=risk_score,
            sanctions_hit=sanctions_hit,
        )

        return ComplianceResult(
            task_id=task_id,
            outcome=outcome,
            risk_score=risk_score,
            sanctions_hit=sanctions_hit,
            hold_reason=hold_reason,
            provider_ref=provider_ref,
            raw_response=raw_response,
        )
