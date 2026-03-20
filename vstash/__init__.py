"""
vstash — Local document memory with instant semantic search.

Stack:
  FastEmbed (ONNX)  → ~700 chunks/s embeddings, fully in-process
  sqlite-vec        → single-file vector store, no server
  FTS5 + RRF        → hybrid search (semantic + keyword)
  Cerebras / Ollama → 2000 tok/s inference or fully local

Usage:
    vstash add myfile.pdf
    vstash ask "what does this file say about X?"
    vstash chat
"""

__version__ = "0.1.0"
__author__ = "Jay"
