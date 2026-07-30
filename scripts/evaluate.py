"""Evaluates retrieval relevance and latency.

Compares a naive keyword-overlap baseline against the configured semantic
vector store (TF-IDF by default; ChromaDB + sentence-transformers in
production mode) on a small hand-labeled query set.

This prints REAL, measured numbers from your environment and dataset —
not fixed figures — so the results are honest and reproducible. Run it
after `python -m src.cli ingest`.

Usage:
    python scripts/evaluate.py
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config
from src.data_loader import load_books, build_document_text
from src.vector_store import build_vector_store

# Hand-labeled eval set: query -> set of book ids considered relevant.
# Relevance was judged manually against data/books.json.
EVAL_SET = [
    {
        "query": "epic fantasy war between kingdoms with multiple points of view",
        "relevant_ids": {"book_0014", "book_0017", "book_0023", "book_0083"},
    },
    {
        "query": "survive alone using science and engineering in space",
        "relevant_ids": {"book_0006", "book_0080"},
    },
    {
        "query": "psychological thriller with an unreliable narrator and a twist",
        "relevant_ids": {"book_0024", "book_0026", "book_0086"},
    },
    {
        "query": "vampires and gothic horror in a haunted setting",
        "relevant_ids": {"book_0065", "book_0068"},
    },
    {
        "query": "practical guide to building better daily habits",
        "relevant_ids": {"book_0043"},
    },
    {
        "query": "coming of age story about racism and justice in the American South",
        "relevant_ids": {"book_0056", "book_0061"},
    },
    {
        "query": "cyberpunk hacker fighting corporations with AI",
        "relevant_ids": {"book_0003", "book_0005"},
    },
    {
        "query": "enemies to lovers workplace romantic comedy",
        "relevant_ids": {"book_0038"},
    },
]


def baseline_keyword_search(books, query, n_results):
    """Naive baseline: rank by count of overlapping words with the description/title."""
    query_terms = set(re.findall(r"[a-z']+", query.lower()))
    scored = []
    for b in books:
        text = build_document_text(b).lower()
        doc_terms = set(re.findall(r"[a-z']+", text))
        overlap = len(query_terms & doc_terms)
        scored.append((overlap, b["id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [book_id for _, book_id in scored[:n_results]]


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def main():
    books = load_books(config.DATA_PATH)
    store = build_vector_store(config)
    store.index_books(books)

    k = 5
    baseline_scores, semantic_scores = [], []
    baseline_latencies, semantic_latencies = [], []

    print(f"Evaluating {len(EVAL_SET)} queries against {len(books)} books "
          f"(backend='{config.VECTOR_BACKEND}', k={k})\n")

    for case in EVAL_SET:
        query, relevant = case["query"], case["relevant_ids"]

        t0 = time.perf_counter()
        baseline_ids = baseline_keyword_search(books, query, n_results=k)
        baseline_latencies.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        semantic_results = store.query(query, n_results=k)
        semantic_latencies.append(time.perf_counter() - t0)
        semantic_ids = [r["id"] for r in semantic_results]

        bp = precision_at_k(baseline_ids, relevant, k)
        sp = precision_at_k(semantic_ids, relevant, k)
        baseline_scores.append(bp)
        semantic_scores.append(sp)

        print(f"- \"{query}\"")
        print(f"    baseline precision@{k}: {bp:.2f}   semantic precision@{k}: {sp:.2f}")

    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    avg_semantic = sum(semantic_scores) / len(semantic_scores)
    avg_baseline_latency = sum(baseline_latencies) / len(baseline_latencies)
    avg_semantic_latency = sum(semantic_latencies) / len(semantic_latencies)

    print("\n" + "=" * 60)
    print(f"Average precision@{k}  — baseline keyword search: {avg_baseline*100:.1f}%")
    print(f"Average precision@{k}  — semantic retrieval:      {avg_semantic*100:.1f}%")
    print(f"Average latency        — baseline keyword search: {avg_baseline_latency*1000:.1f} ms")
    print(f"Average latency        — semantic retrieval:      {avg_semantic_latency*1000:.1f} ms")
    print("=" * 60)
    print(
        "\nNote: these numbers are computed live from this run — they will vary "
        "with your dataset, eval set, and (in 'chroma' mode) embedding model. "
        "Swap VECTOR_BACKEND=chroma for real dense-embedding numbers."
    )


if __name__ == "__main__":
    main()
