# Intelligent Book Recommendation System

An autonomous, RAG-based book recommendation engine. Retrieves candidates
from a vector store using semantic embeddings, then hands them to an LLM
agent that re-ranks and explains the picks in natural language.

Runs in two modes:

| Mode | Backend | Setup | Use for |
|---|---|---|---|
| **Demo** (default) | scikit-learn TF-IDF, in-memory/pickle | none — works immediately | trying it out, testing, CI |
| **Production** | ChromaDB + sentence-transformers embeddings | `pip install chromadb sentence-transformers` | real semantic search at scale |

Both modes share the exact same pipeline code (`src/pipeline.py`), agents,
and CLI — only the vector store implementation changes.

## Architecture

```
 user query
     │
     ▼
 QueryAgent          — pulls out light structure (e.g. a genre filter)
     │
     ▼
 VectorStore.query() — semantic retrieval of top-N candidates
     │  (TfidfVectorStore  or  ChromaVectorStore)
     ▼
 RecommendationAgent — re-ranks candidates + writes a reason for each
     │  (Claude via Anthropic API, or a deterministic fallback agent)
     ▼
 ranked, explained recommendations
```

`src/pipeline.py` — `BookRecommenderPipeline` orchestrates the three
stages above and reports retrieval count, latency, and whether the LLM
path was used.

## Project layout

```
book_recommender/
├── data/books.json          # 94 seed book records (title/author/genre/description/tags)
├── src/
│   ├── config.py            # all settings, overridable via env vars
│   ├── data_loader.py       # loads books.json, builds embeddable document text
│   ├── vector_store.py      # TfidfVectorStore + ChromaVectorStore (same interface)
│   ├── agents.py            # QueryAgent, RecommendationAgent (LLM + fallback)
│   ├── pipeline.py          # orchestrates the end-to-end RAG flow
│   └── cli.py                # `ingest` / `recommend` / `chat` commands
├── scripts/
│   ├── generate_dataset.py  # (re)generates data/books.json
│   └── evaluate.py          # measures retrieval precision + latency, baseline vs semantic
└── tests/test_pipeline.py   # end-to-end tests on the tfidf backend
```

## Browser UI

There's also a browser-based search page (a small Flask app over the same
pipeline used by the CLI):

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Type a description
of what you want to read into the search box and get ranked, explained
recommendations rendered as catalog cards. The index is built
automatically on first launch — no separate `ingest` step needed.

## Quick start (CLI, zero setup)

```bash
cd book_recommender
pip install -r requirements.txt      # only needs scikit-learn/numpy for this mode

python -m src.cli ingest
python -m src.cli recommend "a slow-burn political fantasy with multiple points of view"
python -m src.cli chat               # interactive loop
```

Sample output:

```
Query: a slow-burn political fantasy with multiple points of view
Detected genre filter: Fantasy
Candidates retrieved: 14
Latency: 0.02s  |  LLM used: False
------------------------------------------------------------
1. A Game of Thrones by George R.R. Martin (1996) — Fantasy  [similarity 0.18]
   why: Matches your request on fantasy, multiple, political; also ranked
        highly by semantic similarity (0.18).
...
```

## Production mode (real embeddings + ChromaDB + Claude)

```bash
pip install chromadb sentence-transformers anthropic
cp .env.example .env
# edit .env: VECTOR_BACKEND=chroma, and add ANTHROPIC_API_KEY for the LLM agent
export $(cat .env | xargs)

python -m src.cli ingest        # builds a persistent Chroma collection
python -m src.cli recommend "..."
```

The first run downloads the `all-MiniLM-L6-v2` sentence-transformer model
(~80MB, cached afterward). Without `ANTHROPIC_API_KEY` set, recommendations
still work — `RecommendationAgent` transparently falls back to a
deterministic, tag-overlap-based explanation generator so the pipeline
never breaks.

## Scaling the catalog to 800+ books

`data/books.json` ships with 94 hand-curated records to keep the repo
small. To scale up:

1. Add more records to `data/books.json` (or generate them from a source
   like Open Library / Google Books APIs) following the schema in
   `scripts/generate_dataset.py`.
2. Re-run `python -m src.cli ingest`.
3. Switch to `VECTOR_BACKEND=chroma` — ChromaDB's HNSW index keeps query
   latency roughly flat as the catalog grows, whereas the TF-IDF demo
   backend's cosine similarity is O(n) per query.

## Evaluating retrieval quality

```bash
python scripts/evaluate.py
```

This runs a small hand-labeled query set through both a naive
keyword-overlap baseline and the configured vector store, and prints
**real, measured** precision@5 and latency for each — no numbers are
hard-coded. Example run against the demo dataset:

```
Average precision@5  — baseline keyword search: 37.5%
Average precision@5  — semantic retrieval:      42.5%
Average latency        — baseline keyword search: 1.4 ms
Average latency        — semantic retrieval:      3.5 ms
```

TF-IDF vs. naive keyword overlap is a modest improvement since both are
lexical methods. Switching `VECTOR_BACKEND=chroma` (true dense semantic
embeddings) is where relevance typically jumps substantially, because it
matches on meaning rather than shared words — e.g. "a story about
grief and letting go" retrieving relevant books that never use those
exact words. Run the same script in that mode to get your own numbers
for a resume/portfolio writeup rather than quoting someone else's.

## Design notes / what maps to what

- **"LLMs, RAG, ChromaDB"** → `ChromaVectorStore` (dense retrieval) +
  `RecommendationAgent` (Claude generates the final explained picks) =
  classic retrieve-then-generate RAG.
- **"Improving relevance via semantic embeddings"** → `scripts/evaluate.py`
  is the harness to measure and report this honestly on your own data.
- **"Reduced latency through optimized vector search"** → Chroma's
  persistent HNSW index (`hnsw:space: cosine`) avoids recomputing
  embeddings/index on every query; `pipeline.py` times each stage so you
  can profile where time goes.
- **"Modular AI agents"** → `QueryAgent` and `RecommendationAgent` are
  independent, swappable, unit-testable components composed by
  `BookRecommenderPipeline`.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Tests run entirely on the tfidf backend (no network, no API key) so they're
fast and CI-friendly, while still exercising the real pipeline logic.
