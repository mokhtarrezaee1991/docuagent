"""Kombiniert unveränderte semantische Suche und BM25 mittels Rangfusion."""

import os
import re

from rank_bm25 import BM25Okapi

from app.services.semantic_search import search_embedded_chunks


# Vor der Evaluation festgelegt; keine Optimierung anhand der Testfragen.
RRF_K = 60
BM25_PARAMETERS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}


def retrieval_mode(mode: str | None = None) -> str:
    """Wählt den Modus explizit oder aus der Prozessumgebung, ohne .env-Zugriff."""
    selected = mode if mode is not None else os.environ.get("DOCUAGENT_RETRIEVAL_MODE", "semantic")
    if selected not in ("semantic", "hybrid"):
        raise ValueError("Der Retrieval-Modus muss semantic oder hybrid sein.")
    return selected


def tokenize_keywords(text: str) -> list[str]:
    """Verwendet identische Unicode-Wortzerlegung für Dokumente und Fragen.

    Keine Anpassung an Evaluationsfragen, keine Stoppwortliste und kein Stemming.
    Zahlen bleiben erhalten; Groß-/Kleinschreibung wird vereinheitlicht.
    """
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class BM25Index:
    """Hält einen wiederverwendbaren Keyword-Index für einen Chunk-Schnappschuss."""

    def __init__(self, chunks: list[dict]):
        self.chunks = tuple(chunks)
        self.tokens = [tokenize_keywords(chunk["text"]) for chunk in self.chunks]
        self.term_sets = [set(tokens) for tokens in self.tokens]
        self.model = BM25Okapi(self.tokens, **BM25_PARAMETERS) if any(self.tokens) else None

    def search(self, query: str) -> list[dict]:
        """Gibt BM25-Treffer mit mindestens einer Wortübereinstimmung zurück.

        Treffer mit gleichem Score behalten die ursprüngliche Chunk-Reihenfolge.
        Ein leerer oder unbekannter Suchbegriff erzeugt keine künstliche Rangliste.
        """
        tokens = tokenize_keywords(query)
        if not tokens or self.model is None:
            return []
        scores = self.model.get_scores(tokens)
        terms = set(tokens)
        hits = [
            {**{key: value for key, value in chunk.items() if key != "embedding"}, "bm25_score": float(score)}
            for chunk, term_set, score in zip(self.chunks, self.term_sets, scores)
            if terms.intersection(term_set)
        ]
        return sorted(hits, key=lambda hit: hit["bm25_score"], reverse=True)


def reciprocal_rank_fusion(semantic_hits: list[dict], keyword_hits: list[dict], k: int = RRF_K) -> list[dict]:
    """Berechnet sum(1 / (k + Rang)) bei gleicher Gewichtung beider Ranglisten.

    similarity_score bleibt die Kosinus-Ähnlichkeit. rrf_score bestimmt die neue
    Reihenfolge und ist keine Kosinus-Ähnlichkeit oder Wahrscheinlichkeit.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("Die RRF-Konstante muss eine positive ganze Zahl sein.")
    results = {}
    for label, hits in (("semantic", semantic_hits), ("bm25", keyword_hits)):
        seen = set()
        for rank, hit in enumerate(hits, start=1):
            key = (hit.get("document_id"), hit["chunk_id"], hit["page_number"])
            if key in seen:
                continue
            seen.add(key)
            if key not in results:
                results[key] = {**hit, "rrf_score": 0.0, "bm25_score": 0.0, "semantic_rank": None, "bm25_rank": None}
            result = results[key]
            result["rrf_score"] += 1 / (k + rank)
            result[f"{label}_rank"] = rank
            if label == "bm25":
                result["bm25_score"] = hit["bm25_score"]
    # Python sortiert stabil: Bei Gleichstand entscheidet die semantische Reihenfolge.
    return sorted(results.values(), key=lambda hit: hit["rrf_score"], reverse=True)


def search_prepared(query: str, chunks: list[dict], top_k: int = 3, *, mode: str | None = None, keyword_index: BM25Index | None = None) -> list[dict]:
    """Sucht ohne neue Dokument-Embeddings semantisch oder hybrid."""
    selected = retrieval_mode(mode)
    if selected == "semantic":
        return search_embedded_chunks(query, chunks, top_k=top_k)
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k muss eine positive ganze Zahl sein.")
    if not query.strip() or not chunks:
        return []
    # Beide Verfahren betrachten den gesamten lokalen Index vor der Top-k-Auswahl.
    semantic_hits = search_embedded_chunks(query, chunks, top_k=len(chunks))
    keyword_hits = (keyword_index or BM25Index(chunks)).search(query)
    return reciprocal_rank_fusion(semantic_hits, keyword_hits)[:top_k]
