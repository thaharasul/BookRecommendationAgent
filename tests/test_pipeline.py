import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config
from src.pipeline import BookRecommenderPipeline


def make_pipeline(tmp_path):
    config.VECTOR_BACKEND = "tfidf"
    config.TFIDF_STORE_PATH = str(tmp_path / "test_store.pkl")
    config.ANTHROPIC_API_KEY = None  # force fallback agent path
    pipeline = BookRecommenderPipeline(config=config)
    pipeline.ingest()
    return pipeline


def test_ingest_indexes_all_books(tmp_path):
    pipeline = make_pipeline(tmp_path)
    assert pipeline.vector_store.count() > 0


def test_recommend_returns_top_k(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = pipeline.recommend("space survival science fiction", top_k=3)
    assert len(result["recommendations"]) == 3
    assert result["used_llm"] is False
    assert all("reason" in r for r in result["recommendations"])


def test_recommend_detects_genre_filter(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = pipeline.recommend("a fantasy novel about dragons")
    assert result["detected_genre_filter"] == "Fantasy"
    assert all(r["genre"] == "Fantasy" for r in result["recommendations"])


def test_relevant_book_surfaces_for_specific_query(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = pipeline.recommend("an astronaut stranded alone on Mars using science to survive")
    titles = [r["title"] for r in result["recommendations"]]
    assert "The Martian" in titles


def test_low_similarity_results_are_dropped(tmp_path):
    pipeline = make_pipeline(tmp_path)
    results = pipeline.vector_store.query("zzzz impossible words only", n_results=5)
    assert results == []


def test_genre_and_year_filters_are_applied(tmp_path):
    pipeline = make_pipeline(tmp_path)
    results = pipeline.vector_store.query(
        "space adventure",
        n_results=10,
        genre_filter="Science Fiction",
        year_min=2000,
        year_max=2015,
    )
    assert results
    assert all(r["genre"] == "Science Fiction" for r in results)
    assert all(2000 <= r["year"] <= 2015 for r in results)


def test_pipeline_reports_no_strong_matches_when_threshold_is_not_met(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = pipeline.recommend("zzzz impossible words only", top_k=3)
    assert result["has_strong_matches"] is False
    assert result["status_message"] == "No strong matches"
