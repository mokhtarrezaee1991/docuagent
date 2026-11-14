"""Erzeugt lokale Embeddings und sucht ähnliche Textabschnitte."""

from functools import lru_cache
from math import sqrt


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Lädt das CPU-Modell bei Bedarf und behält es für weitere Aufrufe."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Gibt nichtleere Chunks mit zusätzlichem embedding zurück.

    Die Eingabe bleibt unverändert. Alle Texte werden gemeinsam kodiert;
    Chunks ohne Text oder mit reinem Leerraum werden übersprungen.
    """
    valid_chunks = [chunk for chunk in chunks if chunk.get("text", "").strip()]
    if not valid_chunks:
        return []

    embeddings = _get_model().encode(
        [chunk["text"] for chunk in valid_chunks], convert_to_numpy=True
    )
    return [
        {**chunk, "embedding": [float(value) for value in embedding]}
        for chunk, embedding in zip(valid_chunks, embeddings)
    ]


def _cosine_similarity(first, second) -> float:
    """Vergleicht die Richtung zweier Vektoren; Nullvektoren ergeben 0."""
    first_norm = sqrt(sum(value * value for value in first))
    second_norm = sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0
    score = sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
    # NumPy-Skalare werden für die JSON-Ausgabe in Python-float umgewandelt.
    return float(max(-1.0, min(1.0, score)))


def search_chunks(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Sucht in rohen Chunks und sortiert nach absteigender Kosinus-Ähnlichkeit.

    Leere Fragen liefern keine Treffer. top_k muss eine positive ganze Zahl
    sein. Chunk-Embeddings werden bei jeder Suche neu berechnet; das Modell
    selbst bleibt geladen. Bei Gleichstand bleibt die Eingabereihenfolge.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k muss eine positive ganze Zahl sein.")
    if not query.strip():
        return []

    return search_embedded_chunks(query, embed_chunks(chunks), top_k=top_k)


def search_embedded_chunks(
    query: str, embedded_chunks: list[dict], top_k: int = 3
) -> list[dict]:
    """Sucht mit vorbereiteten Embeddings aus embed_chunks.

    Nur die Frage wird neu kodiert. Die vorbereiteten Chunks bleiben
    unverändert und müssen mit demselben Modell erzeugt worden sein.
    Leere Fragen oder Listen liefern keine Treffer; top_k muss positiv sein.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k muss eine positive ganze Zahl sein.")
    if not query.strip():
        return []
    if not embedded_chunks:
        return []

    query_embedding = _get_model().encode([query], convert_to_numpy=True)[0]
    results = [
        {
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
            "similarity_score": _cosine_similarity(chunk["embedding"], query_embedding),
        }
        for chunk in embedded_chunks
    ]
    return sorted(results, key=lambda result: result["similarity_score"], reverse=True)[:top_k]
