"""
FinGPT RAG Adapter for Finogrid.

Adapts FinGPT's RAG pipeline to serve:
1. Internal Product Support Agent — answers team questions from runbooks, docs, incident history
2. Audit & Governance Agent — retrieves relevant compliance rules and audit context

Uses ChromaDB for local vector store; can be swapped to Vertex AI Matching Engine in prod.

Based on: FinGPT/fingpt/FinGPT_RAG/
"""
from __future__ import annotations

import structlog
from pathlib import Path
from typing import Optional

log = structlog.get_logger()

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    log.warning("chromadb_not_installed", hint="pip install chromadb")


class FinoGridKnowledgeBase:
    """
    Vector knowledge base for internal agent RAG.

    Documents indexed:
    - Architecture docs (docs/architecture.md)
    - Per-corridor runbooks (docs/corridors/)
    - Compliance profiles
    - Incident history (from audit_logs)
    - Partner API docs (scraped or uploaded)
    - DR runbook (docs/dr-runbook.md)
    """

    def __init__(
        self,
        persist_directory: str = "/tmp/finogrid_rag",
        collection_name: str = "finogrid_kb",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._client = None
        self._collection = None

    def init(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMA_AVAILABLE:
            raise ImportError("Install chromadb: pip install chromadb")

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("knowledge_base_initialized", collection=self.collection_name)

    def index_document(self, doc_id: str, text: str, metadata: Optional[dict] = None):
        """Add or update a document in the knowledge base."""
        if not self._collection:
            raise RuntimeError("Call init() first")

        # Chunk long docs
        chunks = self._chunk_text(text, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            self._collection.upsert(
                ids=[f"{doc_id}__chunk_{i}"],
                documents=[chunk],
                metadatas=[{**(metadata or {}), "source_id": doc_id, "chunk": i}],
            )
        log.info("document_indexed", doc_id=doc_id, chunks=len(chunks))

    def index_directory(self, directory: str, glob_pattern: str = "**/*.md"):
        """Recursively index all matching files in a directory."""
        root = Path(directory)
        for path in root.glob(glob_pattern):
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.index_document(
                doc_id=str(path.relative_to(root)),
                text=text,
                metadata={"file": str(path), "type": "documentation"},
            )

    def query(self, question: str, n_results: int = 5) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a question.
        Returns list of {"text": str, "source": str, "score": float}
        """
        if not self._collection:
            raise RuntimeError("Call init() first")

        results = self._collection.query(
            query_texts=[question],
            n_results=n_results,
        )
        output = []
        for i, doc in enumerate(results["documents"][0]):
            output.append({
                "text": doc,
                "source": results["metadatas"][0][i].get("source_id", "unknown"),
                "distance": results["distances"][0][i],
            })
        return output

    def build_context(self, question: str, n_results: int = 5) -> str:
        """Build a context string for LLM prompting."""
        chunks = self.query(question, n_results=n_results)
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )
        return context

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks
