"""
langchain.py — LangChain integration for vstash.

Provides a ``VstashRetriever`` that plugs vstash's hybrid semantic search
into any LangChain chain or agent.

    from vstash.langchain import VstashRetriever
    from vstash import Memory

    retriever = VstashRetriever(memory=Memory(project="docs"))
    docs = retriever.invoke("deployment strategy")

Compatible with LangSmith tracing out of the box — LangChain instruments
all BaseRetriever subclasses automatically.
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install it with: pip install vstash[langchain]"
    ) from exc
from pydantic import ConfigDict

from .memory import Memory


class VstashRetriever(BaseRetriever):
    """LangChain retriever backed by vstash hybrid semantic search.

    Uses vstash's Reciprocal Rank Fusion (vector + FTS5 keyword) to retrieve
    the most relevant document chunks for a given query. Fully local — no API
    calls required for retrieval.

    Args:
        memory: A vstash ``Memory`` instance (handles DB, embeddings, config).
        top_k: Number of chunks to retrieve per query (default: 5).
        project: Override the Memory's default project filter.
        collection: Override the Memory's default collection filter.
        layer: Filter by layer tag.

    Example::

        from vstash import Memory
        from vstash.langchain import VstashRetriever
        from langchain_openai import ChatOpenAI
        from langchain.chains import RetrievalQA

        mem = Memory(project="my_docs")
        mem.add("report.pdf")

        retriever = VstashRetriever(memory=mem, top_k=5)
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=retriever,
        )
        answer = chain.invoke("What are the key findings?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory: Memory
    top_k: int = 5
    project: str | None = None
    collection: str | None = None
    layer: str | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve relevant documents using vstash hybrid search.

        Args:
            query: Natural language search query.
            run_manager: LangChain callback manager (used for tracing).

        Returns:
            List of LangChain Document objects with chunk text and metadata.
        """
        # Build kwargs for Memory.search(), only passing overrides explicitly set
        # by the caller. Using exclude_unset=True means passing project=None
        # will correctly override a Memory scoped to a project.
        search_kwargs: dict[str, Any] = {"top_k": self.top_k}
        retriever_filters = self.model_dump(
            include={"project", "collection", "layer"}, exclude_unset=True
        )
        search_kwargs.update(retriever_filters)

        results = self.memory.search(query, **search_kwargs)

        return [
            Document(
                page_content=r.text,
                metadata={
                    "source": r.path,
                    "title": r.title,
                    "score": r.score,
                    "chunk": r.chunk,
                },
            )
            for r in results
        ]
