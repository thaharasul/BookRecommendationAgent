"""Orchestrates the end-to-end RAG recommendation pipeline:

  query -> QueryAgent -> vector store retrieval -> RecommendationAgent -> results

This is the module the CLI and evaluation scripts call into.
"""
import time
from typing import Dict, List, Optional

from . import config as default_config
from .agents import QueryAgent, RecommendationAgent
from .data_loader import load_books
from .vector_store import build_vector_store


class BookRecommenderPipeline:
    def __init__(self, config=default_config, vector_store=None, recommendation_agent=None):
        self.config = config
        self.vector_store = vector_store or build_vector_store(config)
        self.query_agent = QueryAgent()
        self.recommendation_agent = recommendation_agent or RecommendationAgent(
            model=config.ANTHROPIC_MODEL, api_key=config.ANTHROPIC_API_KEY
        )

    def ingest(self, data_path: Optional[str] = None) -> int:
        """Load the catalog and (re)build the vector index. Returns record count."""
        books = load_books(data_path or self.config.DATA_PATH)
        self.vector_store.index_books(books)
        return len(books)

    def recommend(self, query: str, top_k: Optional[int] = None,
                  candidate_k: Optional[int] = None,
                  genre_filter: Optional[str] = None,
                  year_min: Optional[int] = None,
                  year_max: Optional[int] = None) -> Dict:
        top_k = top_k or self.config.DEFAULT_TOP_K
        candidate_k = candidate_k or self.config.DEFAULT_CANDIDATE_K

        t0 = time.perf_counter()

        interpreted = self.query_agent.interpret(query)
        effective_genre = genre_filter or interpreted["genre_filter"]
        candidates = self.vector_store.query(
            interpreted["query"],
            n_results=candidate_k,
            genre_filter=effective_genre,
            year_min=year_min,
            year_max=year_max,
        )
        recommendations = self.recommendation_agent.recommend(query, candidates, top_k=top_k)

        latency = time.perf_counter() - t0
        has_strong_matches = bool(recommendations)
        status_message = "No strong matches" if not has_strong_matches else "Matches found"

        return {
            "query": query,
            "detected_genre_filter": effective_genre,
            "candidates_retrieved": len(candidates),
            "latency_seconds": round(latency, 4),
            "used_llm": self.recommendation_agent.using_llm,
            "has_strong_matches": has_strong_matches,
            "status_message": status_message,
            "recommendations": recommendations,
        }
