"""
Internal Product Support Agent — powered by FinGPT RAG.

Answers team questions from:
- Runbooks (docs/dr-runbook.md, docs/corridors/)
- Architecture docs
- Partner API documentation
- Incident history (from audit_logs)

Designed to replace ad-hoc Slack searches with a grounded, doc-backed assistant.
NOT customer-facing. NOT in the transaction hot path.
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import Optional

log = structlog.get_logger()

SUPPORT_SYSTEM_PROMPT = """
You are an internal technical support assistant for Finogrid, a stablecoin payout
orchestration platform. You answer questions from the engineering, ops, and compliance teams.

You have access to:
- Finogrid architecture documentation
- Per-corridor runbooks and policy docs
- Partner API integration notes
- Incident history summaries

Always cite your sources. If you don't know something, say so clearly.
Never speculate about regulatory or legal requirements — direct those to counsel.
"""


class InternalSupportAgent:
    """
    RAG-backed support agent for the Finogrid team.
    Uses FinoGridKnowledgeBase for retrieval + an LLM for generation.
    """

    def __init__(self, knowledge_base=None, llm_client=None):
        self.kb = knowledge_base
        self.llm = llm_client

    async def answer(self, question: str, session_id: Optional[str] = None) -> dict:
        """
        Answer a team member's question using RAG.
        Returns answer + sources.
        """
        log.info("support_question_received", question=question[:100], session=session_id)

        # Retrieve relevant context
        sources = []
        context = ""
        if self.kb:
            chunks = self.kb.query(question, n_results=6)
            sources = [{"source": c["source"], "excerpt": c["text"][:200]} for c in chunks]
            context = self.kb.build_context(question, n_results=6)

        # Generate answer
        if self.llm:
            prompt = (
                f"{SUPPORT_SYSTEM_PROMPT}\n\n"
                f"Relevant documentation:\n{context}\n\n"
                f"Team member question: {question}\n\n"
                f"Answer:"
            )
            answer = await self.llm.chat(prompt)
        else:
            answer = f"RAG context retrieved (LLM not configured):\n\n{context}"

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "answered_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
        }

    async def index_incident(self, incident_id: str, summary: str, resolution: str):
        """Add an incident summary to the knowledge base for future RAG retrieval."""
        if self.kb:
            doc = f"Incident: {incident_id}\n\nSummary:\n{summary}\n\nResolution:\n{resolution}"
            self.kb.index_document(
                doc_id=f"incident_{incident_id}",
                text=doc,
                metadata={"type": "incident", "incident_id": incident_id},
            )
            log.info("incident_indexed", incident_id=incident_id)
