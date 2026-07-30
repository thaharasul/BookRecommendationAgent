"""Vector store abstraction.

Two backends implement the same interface:

- TfidfVectorStore: pure scikit-learn, in-memory + pickled to disk.
  Zero external downloads, works immediately. Used as the default demo
  backend so the whole system is runnable out of the box.

- ChromaVectorStore: real semantic embeddings via sentence-transformers,
  persisted in a ChromaDB collection with cosine similarity search. This
  is the production backend the system is designed around.

Both return results in the same shape:
    [{"id": ..., "title": ..., "author": ..., "genre": ..., "year": ...,
      "tags": ..., "description": ..., "similarity": float}, ...]
"""
import os
import pickle
from typing import List, Dict, Optional

from .data_loader import build_document_text, build_metadata


class VectorStoreBase:
    def index_books(self, books: List[Dict]) -> None:
        raise NotImplementedError

    def query(self, query_text: str, n_results: int = 10,
              genre_filter: Optional[str] = None,
              year_min: Optional[int] = None,
              year_max: Optional[int] = None,
              min_similarity: float = 0.05) -> List[Dict]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class TfidfVectorStore(VectorStoreBase):
    """Lightweight semantic-ish search using TF-IDF + cosine similarity.

    Not a substitute for true dense embeddings, but has the same interface
    as the Chroma backend, requires no downloads, and is good enough to
    demonstrate and unit-test the full retrieval -> agent -> generation
    pipeline offline.
    """

    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self._vectorizer = None
        self._matrix = None
        self._records: List[Dict] = []

    def index_books(self, books: List[Dict]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        documents = [build_document_text(b) for b in books]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(documents)
        self._records = [
            {"id": b["id"], "document": doc, **build_metadata(b), "description": b["description"]}
            for b, doc in zip(books, documents)
        ]
        self._persist()

    def _persist(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(
                {"vectorizer": self._vectorizer, "matrix": self._matrix, "records": self._records},
                f,
            )

    def load(self) -> bool:
        if not os.path.exists(self.persist_path):
            return False
        with open(self.persist_path, "rb") as f:
            state = pickle.load(f)
        self._vectorizer = state["vectorizer"]
        self._matrix = state["matrix"]
        self._records = state["records"]
        return True

    def count(self) -> int:
        return len(self._records)

    def query(self, query_text: str, n_results: int = 10,
              genre_filter: Optional[str] = None,
              year_min: Optional[int] = None,
              year_max: Optional[int] = None,
              min_similarity: float = 0.05) -> List[Dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self._vectorizer is None and not self.load():
            raise RuntimeError("Vector store is empty. Run ingest first.")

        q_vec = self._vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        ranked_idx = sims.argsort()[::-1]
        results = []
        for idx in ranked_idx:
            record = self._records[idx]
            if genre_filter and record["genre"].lower() != genre_filter.lower():
                continue
            if year_min is not None and record.get("year", 0) < year_min:
                continue
            if year_max is not None and record.get("year", 0) > year_max:
                continue
            similarity = float(sims[idx])
            if similarity < min_similarity:
                continue
            results.append({**record, "similarity": similarity})
            if len(results) >= n_results:
                break
        return results


class ChromaVectorStore(VectorStoreBase):
    """Production backend: ChromaDB + sentence-transformers embeddings.

    Requires: pip install chromadb sentence-transformers
    On first run, sentence-transformers will download the embedding model
    (requires internet access once; cached locally afterward).
    """

    def __init__(self, persist_dir: str, collection_name: str, embedding_model_name: str):
        import chromadb
        from chromadb.utils import embedding_functions

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_books(self, books: List[Dict]) -> None:
        ids = [b["id"] for b in books]
        documents = [build_document_text(b) for b in books]
        metadatas = [build_metadata(b) for b in books]
        # Chroma upsert batches internally; for 800+ records this is a single call.
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def count(self) -> int:
        return self.collection.count()

    def query(self, query_text: str, n_results: int = 10,
              genre_filter: Optional[str] = None,
              year_min: Optional[int] = None,
              year_max: Optional[int] = None,
              min_similarity: float = 0.05) -> List[Dict]:
        where = {"genre": genre_filter} if genre_filter else None
        raw = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )
        results = []
        ids = raw["ids"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]
        for id_, meta, dist in zip(ids, metadatas, distances):
            # Chroma returns cosine *distance*; convert to a similarity score.
            similarity = 1.0 - dist
            if similarity < min_similarity:
                continue
            if year_min is not None and meta.get("year", 0) < year_min:
                continue
            if year_max is not None and meta.get("year", 0) > year_max:
                continue
            results.append({"id": id_, "similarity": similarity, **meta})
        return results


def build_vector_store(config) -> VectorStoreBase:
    """Factory that returns the configured backend."""
    if config.VECTOR_BACKEND == "chroma":
        return ChromaVectorStore(
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
        )
    elif config.VECTOR_BACKEND == "tfidf":
        return TfidfVectorStore(persist_path=config.TFIDF_STORE_PATH)
    else:
        raise ValueError(f"Unknown VECTOR_BACKEND: {config.VECTOR_BACKEND}")
