"""Central configuration for the book recommender.

All values can be overridden with environment variables so the same
codebase runs in "quick demo" mode (no external services, no downloads)
and in "production" mode (ChromaDB + sentence-transformers + Claude).
"""
import os

# --- Data ---
DATA_PATH = os.environ.get(
    "BOOKS_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "books.json"),
)

# --- Vector store backend ---
# "tfidf"  -> pure sklearn, no downloads, runs anywhere instantly (default demo mode)
# "chroma" -> ChromaDB + sentence-transformers, real semantic embeddings (production mode)
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "tfidf")

CHROMA_PERSIST_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "..", "chroma_store"),
)
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "books")
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)

TFIDF_STORE_PATH = os.environ.get(
    "TFIDF_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "chroma_store", "tfidf_store.pkl"),
)

# --- LLM agent ---
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- Retrieval pipeline ---
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))
DEFAULT_CANDIDATE_K = int(os.environ.get("DEFAULT_CANDIDATE_K", "15"))
