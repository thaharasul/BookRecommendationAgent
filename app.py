"""Browser-based UI for the book recommender.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.

This is a thin web layer over the same BookRecommenderPipeline used by
the CLI — no logic is duplicated. Uses the tfidf backend by default so it
runs with zero external downloads (just `pip install flask scikit-learn`).
"""
import os
from flask import Flask, render_template, request, jsonify

from src import config
from src.pipeline import BookRecommenderPipeline

app = Flask(__name__)

# Build the pipeline once at startup and reuse it across requests.
pipeline = BookRecommenderPipeline(config=config)

# Auto-ingest on first launch if the index doesn't exist yet, so the user
# doesn't have to run a separate CLI command before using the web UI.
def ensure_index_ready():
    try:
        if pipeline.vector_store.count() == 0:
            raise RuntimeError
    except Exception:
        pipeline.ingest()

ensure_index_ready()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/recommend")
def api_recommend():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Please enter a search query."}), 400

    top_k = int(request.args.get("top_k", config.DEFAULT_TOP_K))
    result = pipeline.recommend(query, top_k=top_k)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nBook Recommender running — open http://127.0.0.1:{port} in your browser\n")
    app.run(debug=True, port=port)
