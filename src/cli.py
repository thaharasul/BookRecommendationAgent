"""Command-line interface for the book recommender.

Usage:
    python -m src.cli ingest
    python -m src.cli recommend "a slow-burn political fantasy with multiple povs"
    python -m src.cli chat        # interactive loop
"""
import argparse
import json
import sys

from . import config
from .pipeline import BookRecommenderPipeline


def print_recommendations(result: dict):
    print(f"\nQuery: {result['query']}")
    if result["detected_genre_filter"]:
        print(f"Detected genre filter: {result['detected_genre_filter']}")
    print(f"Candidates retrieved: {result['candidates_retrieved']}")
    print(f"Latency: {result['latency_seconds']}s  |  LLM used: {result['used_llm']}")
    print("-" * 60)
    for i, rec in enumerate(result["recommendations"], start=1):
        print(f"{i}. {rec['title']} by {rec['author']} ({rec.get('year', 'n/a')}) "
              f"— {rec['genre']}  [similarity {rec['similarity']:.2f}]")
        print(f"   why: {rec['reason']}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Intelligent Book Recommendation System")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Build/rebuild the vector index from data/books.json")

    rec_parser = sub.add_parser("recommend", help="Get recommendations for a query")
    rec_parser.add_argument("query", type=str, help="Free-text description of what you want to read")
    rec_parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K)
    rec_parser.add_argument("--json", action="store_true", help="Print raw JSON output")

    sub.add_parser("chat", help="Interactive recommendation loop")

    args = parser.parse_args(argv)
    pipeline = BookRecommenderPipeline()

    if args.command == "ingest":
        n = pipeline.ingest()
        print(f"Indexed {n} books using backend='{config.VECTOR_BACKEND}'.")
        return

    if args.command == "recommend":
        result = pipeline.recommend(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_recommendations(result)
        return

    if args.command == "chat":
        print("Book Recommender — type a request, or 'quit' to exit.")
        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in {"quit", "exit"}:
                break
            if not query:
                continue
            result = pipeline.recommend(query)
            print_recommendations(result)


if __name__ == "__main__":
    main(sys.argv[1:])
