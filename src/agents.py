"""Modular agents that make up the recommendation pipeline.

- QueryAgent: lightly interprets the user's free-text request (pulls out
  an optional genre filter) before retrieval runs.
- RecommendationAgent: takes retrieved candidates and produces the final,
  ranked, explained recommendations. Uses Claude when an API key is
  configured; otherwise falls back to a deterministic, template-based
  explanation so the pipeline still runs end to end with zero setup.
"""
import json
import re
from typing import List, Dict, Optional

KNOWN_GENRES = [
    "Science Fiction", "Fantasy", "Thriller", "Mystery", "Romance",
    "Nonfiction", "Self-Help", "Biography", "Business", "History",
    "Classic Literature", "Horror", "Young Adult", "Philosophy", "Poetry",
]


class QueryAgent:
    """Extracts light structure (e.g. a genre filter) from a free-text query."""

    def interpret(self, query: str) -> Dict:
        genre_filter = None
        lowered = query.lower()
        for genre in KNOWN_GENRES:
            if genre.lower() in lowered:
                genre_filter = genre
                break
        return {"query": query, "genre_filter": genre_filter}


class RecommendationAgent:
    """Ranks and explains candidates. LLM-backed with a rule-based fallback."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        self.model = model
        self._client = None
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                self._client = None

    @property
    def using_llm(self) -> bool:
        return self._client is not None

    def recommend(self, user_query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not candidates:
            return []
        if self._client is not None:
            try:
                return self._recommend_with_llm(user_query, candidates, top_k)
            except Exception:
                # Never let an LLM/network hiccup break the pipeline.
                return self._recommend_fallback(user_query, candidates, top_k)
        return self._recommend_fallback(user_query, candidates, top_k)

    # ---------------- LLM path ----------------

    def _recommend_with_llm(self, user_query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        catalog_block = "\n".join(
            f"{i+1}. [{c['id']}] \"{c['title']}\" by {c['author']} ({c['genre']}, {c.get('year', 'n/a')}) "
            f"- similarity {c['similarity']:.2f} - {c.get('description', '')}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "You are a book recommendation assistant. A user asked:\n"
            f'"{user_query}"\n\n'
            "Here are retrieved candidate books, already ranked by semantic similarity:\n"
            f"{catalog_block}\n\n"
            f"Choose the best {top_k} books for this user from the list above (you may "
            "reorder them if a lower-similarity book is actually a better fit). "
            "Respond with ONLY a JSON array, no other text, in this exact format:\n"
            '[{"id": "book_0001", "reason": "one or two sentence, specific reason this fits the request"}]'
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
        picks = json.loads(text)

        by_id = {c["id"]: c for c in candidates}
        results = []
        for pick in picks[:top_k]:
            book = by_id.get(pick["id"])
            if not book:
                continue
            results.append({**book, "reason": pick.get("reason", "")})
        return results

    # ---------------- Fallback path (no API key required) ----------------

    def _recommend_fallback(self, user_query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        query_terms = set(re.findall(r"[a-z']+", user_query.lower()))
        results = []
        for c in candidates[:top_k]:
            tag_terms = set(re.findall(r"[a-z']+", c.get("tags", "").lower()))
            overlap = query_terms & tag_terms
            if overlap:
                reason = (
                    f"Matches your request on {', '.join(sorted(overlap))}; "
                    f"also ranked highly by semantic similarity ({c['similarity']:.2f})."
                )
            else:
                reason = (
                    f"Ranked highly by semantic similarity to your query ({c['similarity']:.2f}) "
                    f"based on its genre ({c['genre']}) and description."
                )
            results.append({**c, "reason": reason})
        return results
