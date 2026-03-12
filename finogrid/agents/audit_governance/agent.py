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

    def __init__(self, db_session_factory, knowledge_base=None, llm_client=None):
        self.SessionLocal = db_session_factory
        self.kb = knowledge_base        # FinoGridKnowledgeBase instance
        self.llm = llm_client           # OpenAI or local FinGPT

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
