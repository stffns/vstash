# LangChain Integration

*Added in v0.4.0*

Use vstash as a retriever in any LangChain chain or agent. The retriever uses vstash's hybrid search (vector + keyword RRF) and returns standard LangChain `Document` objects.

---

## Install

```bash
pip install vstash[langchain]
```

---

## Quick Start

```python
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
```

---

## Filtering

Pass metadata filters to scope retrieval:

```python
retriever = VstashRetriever(
    memory=mem,
    top_k=5,
    project="alpha",
    collection="research",
    layer="summaries",
)
```

---

## Returned Documents

Each result is a standard LangChain `Document` with metadata:

```python
docs = retriever.get_relevant_documents("query")
for doc in docs:
    print(doc.page_content)  # chunk text
    print(doc.metadata["source"])  # file path or URL
    print(doc.metadata["title"])  # document title
    print(doc.metadata["score"])  # hybrid search score
```

---

## LangSmith Tracing

VstashRetriever is compatible with LangSmith tracing automatically — no extra configuration needed. Retrieval steps appear in your trace alongside LLM calls.
