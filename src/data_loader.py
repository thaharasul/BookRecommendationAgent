"""Loads and prepares the book catalog for indexing."""
import json
from typing import List, Dict


def load_books(path: str) -> List[Dict]:
    """Load the book catalog from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        books = json.load(f)
    for b in books:
        for field in ("id", "title", "author", "genre", "description"):
            if field not in b:
                raise ValueError(f"Book record missing required field '{field}': {b}")
        b.setdefault("tags", [])
        b.setdefault("year", None)
    return books


def build_document_text(book: Dict) -> str:
    """Flatten a book record into a single text blob for embedding.

    This is the text that actually gets embedded and searched against, so
    it's constructed to carry the signal that matters for semantic
    retrieval: title, author, genre, description, and tags.
    """
    tags = ", ".join(book.get("tags", []))
    return (
        f"Title: {book['title']}\n"
        f"Author: {book['author']}\n"
        f"Genre: {book['genre']}\n"
        f"Description: {book['description']}\n"
        f"Tags: {tags}"
    )


def build_metadata(book: Dict) -> Dict:
    """Metadata stored alongside each vector (used for filtering + display)."""
    return {
        "title": book["title"],
        "author": book["author"],
        "genre": book["genre"],
        "year": book.get("year") if book.get("year") is not None else 0,
        "tags": ", ".join(book.get("tags", [])),
    }
