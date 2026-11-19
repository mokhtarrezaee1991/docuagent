"""Beschreibt ausschließlich die Daten einer einzelnen Dokumentfrage."""

from typing import NotRequired, TypedDict


class RagResult(TypedDict):
    """Behält das bestehende Antwortformat des RAG-Service bei."""

    answer: str
    sources: list[dict]


class RagState(TypedDict):
    """Enthält weder Schlüssel noch PDFs, Embeddings oder Datenbankverbindungen."""

    query: str
    top_k: int
    retrieved_chunks: NotRequired[list[dict]]
    result: NotRequired[RagResult | None]
    abstention_reason: NotRequired[str | None]
