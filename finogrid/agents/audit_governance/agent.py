"""
Audit & Governance Agent — powered by FinGPT RAG.

Responsibilities (ALL off hot path):
- Build readable audit narratives from raw audit_logs
- Flag architecture or config drift that could create compliance problems
- Answer compliance team questions: "What happened with batch X?"
- Generate SAR/CTR-ready summaries for regulatory inquiries
- Detect anomalous patterns in payout flows

Does NOT modify any payout state or routing configuration.
"""
from __future__ import annotations

import json
import structlog
from datetime import datetime, timezone
from typing import Optional

log = structlog.get_logger()

AUDIT_NARRATIVE_PROMPT = """
You are a compliance officer at a fintech company. Using the following audit log entries,
write a clear, factual narrative describing what happened with this payout.
Be precise about timestamps, amounts, corridors, and any compliance holds.

Audit events:
{events}

Question: {question}

Provide a concise, professional response suitable for regulatory review.
"""


class AuditGovernanceAgent:
    """
    Agent that reads audit_logs and produces human-readable compliance narratives.
    Uses FinGPT RAG for context retrieval from compliance docs.
    """

    def __init__(self, db_session_factory, knowledge_base=None, llm_client=None, agentweb_client=None):
        self.SessionLocal = db_session_factory
        self.kb = knowledge_base        # FinoGridKnowledgeBase instance
        self.llm = llm_client           # OpenAI or local FinGPT
        self.agentweb = agentweb_client  # AgentWeb client for business verification

    async def audit_batch(self, batch_id: str, question: Optional[str] = None) -> dict:
        """
        Generate an audit narrative for a batch.
        """
        events = await self._fetch_batch_events(batch_id)
        if not events:
            return {"batch_id": batch_id, "error": "No events found"}

        narrative = await self._generate_narrative(
            events=events,
            question=question or f"Summarize all activity for batch {batch_id}",
        )

        return {
            "batch_id": batch_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "event_count": len(events),
            "narrative": narrative,
            "events": events,
        }

    async def check_config_drift(self) -> list[dict]:
        """
        Compare current routing/compliance profiles against expected baseline.
        Flags unexpected changes that could create compliance gaps.
        """
        issues = []
        async with self.SessionLocal() as db:
            from sqlalchemy import select
            from ...database.models import RoutingProfile, ComplianceProfile

            # Check KYT is enabled for all corridors
            cp_result = await db.execute(select(ComplianceProfile))
            profiles = cp_result.scalars().all()
            for cp in profiles:
                if not cp.kyt_enabled:
                    issues.append({
                        "type": "kyt_disabled",
                        "severity": "critical",
                        "corridor": cp.corridor_code,
                        "message": f"KYT screening disabled for {cp.corridor_code} — compliance risk",
                    })
                if not cp.sanctions_screen_enabled:
                    issues.append({
                        "type": "sanctions_screen_disabled",
                        "severity": "critical",
                        "corridor": cp.corridor_code,
                        "message": f"Sanctions screening disabled for {cp.corridor_code}",
                    })

        return issues

    async def answer_compliance_question(self, question: str) -> str:
        """
        RAG-powered Q&A for compliance team.
        Retrieves relevant docs + audit context, then generates an answer.
        """
        context = ""
        if self.kb:
            context = self.kb.build_context(question, n_results=5)

        if self.llm:
            prompt = (
                f"Context from Finogrid compliance docs:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer as a compliance expert:"
            )
            response = await self.llm.chat(prompt)
            return response
        else:
            return f"Context retrieved:\n{context}\n\n(LLM not configured — showing raw context)"

    async def _fetch_batch_events(self, batch_id: str) -> list[dict]:
        from sqlalchemy import select
        from ...database.models import AuditLog

        async with self.SessionLocal() as db:
            result = await db.execute(
                select(AuditLog)
                .where(AuditLog.batch_id == batch_id)
                .order_by(AuditLog.created_at)
            )
            logs = result.scalars().all()
            return [
                {
                    "timestamp": l.created_at.isoformat(),
                    "action": l.action,
                    "actor": f"{l.actor_type}:{l.actor_id}",
                    "detail": l.detail,
                    "corridor": l.corridor_code,
                }
                for l in logs
            ]

    async def _generate_narrative(self, events: list[dict], question: str) -> str:
        events_str = json.dumps(events, indent=2)
        if self.llm:
            prompt = AUDIT_NARRATIVE_PROMPT.format(events=events_str, question=question)
            return await self.llm.chat(prompt)
        # Fallback: structured text summary
        lines = [f"Audit narrative for question: {question}", ""]
        for e in events:
            lines.append(f"[{e['timestamp']}] {e['action']} | {e.get('detail', '')}")
        return "\n".join(lines)

    async def verify_business_entity(self, business_name: str, country: str = "US") -> dict:
        """
        Verify business entity using AgentWeb data.
        Useful for KYB verification and compliance checks.
        """
        if not self.agentweb or not self.agentweb.is_configured():
            return {
                "business_name": business_name,
                "verified": False,
                "message": "AgentWeb client not configured",
                "note": "Set AGENTWEB_API_KEY in environment to enable business verification"
            }

        try:
            verification_result = await self.agentweb.verify_business_entity(
                business_name=business_name,
                country=country
            )
            
            log.info(
                "business_verification_attempt",
                business_name=business_name,
                verified=verification_result.get("verified", False)
            )
            
            return verification_result
        except Exception as exc:
            log.error("business_verification_error", business_name=business_name, error=str(exc))
            return {
                "business_name": business_name,
                "verified": False,
                "error": str(exc)
            }

    async def enhance_compliance_context(self, entity_name: str, country: str = "US") -> str:
        """
        Enhance compliance context with business data from AgentWeb.
        Returns formatted context string for LLM prompts.
        """
        if not self.agentweb or not self.agentweb.is_configured():
            return f"Business verification not available for {entity_name}."

        try:
            verification = await self.verify_business_entity(entity_name, country)
            
            if verification.get("verified"):
                business_data = verification.get("business_data", {})
                context_parts = [
                    f"Business: {business_data.get('name', 'N/A')}",
                    f"Category: {business_data.get('category', 'N/A')}",
                    f"Address: {business_data.get('address', 'N/A')}",
                    f"Phone: {business_data.get('phone', 'N/A')}",
                    f"Rating: {business_data.get('rating', 'N/A')}",
                ]
                return " | ".join(context_parts)
            else:
                return f"Business verification failed for {entity_name}: {verification.get('message', 'Unknown error')}"
        except Exception as exc:
            log.error("context_enhancement_error", entity_name=entity_name, error=str(exc))
            return f"Error enhancing context for {entity_name}: {str(exc)}"
