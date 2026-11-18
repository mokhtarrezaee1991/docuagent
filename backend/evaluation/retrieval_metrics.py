"""Berechnet Retrieval-Metriken anhand nachprüfbarer Textreferenzen."""

import re


def normalize_text(text: str) -> str:
    """Ignoriert Großschreibung und PDF-bedingte Leerraum-/Zeilenunterschiede."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def matches_reference(hit: dict, reference: dict) -> bool:
    """Verlangt Dokument, Seite und alle Belegstellen im selben Chunk.

    Eine thematisch passende Beispielfrage allein zählt dadurch nicht als Treffer.
    Es erfolgt keine Bewertung einer generierten LLM-Antwort.
    """
    return (
        hit["filename"] == reference["filename"]
        and hit["page_number"] == reference["page_number"]
        and bool(reference["evidence"])
        and all(normalize_text(text) in normalize_text(hit["text"]) for text in reference["evidence"])
    )


def first_relevant_hit(ranking: list[dict], references: list[dict]) -> tuple[int | None, dict | None]:
    """Liefert den ersten relevanten Rang (ab 1) ohne Umordnung der Treffer."""
    for rank, hit in enumerate(ranking, start=1):
        if any(matches_reference(hit, reference) for reference in references):
            return rank, hit
    return None, None


def calculate_metrics(ranks: list[int | None]) -> dict:
    """Fehlende Treffer zählen mit 0; MRR nutzt die vollständige Rangliste."""
    if not ranks:
        raise ValueError("Für die Evaluation wird mindestens eine Frage benötigt.")
    if any(rank is not None and (type(rank) is not int or rank <= 0) for rank in ranks):
        raise ValueError("Ränge müssen positive ganze Zahlen oder None sein.")
    count = len(ranks)
    return {
        "question_count": count,
        "hit_rate_at_1": sum(rank == 1 for rank in ranks) / count,
        "hit_rate_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / count,
    }
