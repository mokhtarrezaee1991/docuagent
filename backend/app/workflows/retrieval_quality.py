"""Konservative Strukturdiagnose, keine Entscheidung über Antworten oder Retrys."""

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.workflows.search_policy import select_search


class QualityDiagnosis(BaseModel):
    """Sufficient ist für ein später validiertes Verfahren reserviert."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["no_context", "unknown", "sufficient"]
    reason: Literal["no_usable_text", "semantic_sufficiency_unassessed"]
    hit_count: int
    usable_text_count: int
    empty_text_count: int
    duplicate_count: int
    issues: tuple[str, ...]


def diagnose_retrieval(
    chunks: list[dict], *, documents: list[dict], document_ids: list[str] | None = None,
) -> QualityDiagnosis:
    """Prüft einen Treffer-Schnappschuss gegen aktuelle Dokumentmetadaten.

    Nutzbar heißt hier ausschließlich nichtleerer Text, nicht semantisch belegt.
    Fehlende Zitiermetadaten werden als Probleme ausgewiesen, vernichten aber
    keinen vorhandenen Text. Unbekannte IDs und Bereichsverletzungen sind Fehler.
    Fehlende Dokument-IDs sind nur bei unbeschränkter Auswahl diagnostizierbar;
    bei expliziter Auswahl kann ihr Bereich nicht geprüft werden: ValueError.
    Keine Score-Grenzen, Textheuristiken, Filterung oder Suchaufrufe.
    """
    plan = select_search(documents, document_ids=document_ids)
    known = {document["document_id"]: document for document in documents}
    scope = None if plan.document_ids is None else set(plan.document_ids)
    if not isinstance(chunks, list) or any(not isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("Treffer müssen als Liste von Dictionaries vorliegen.")
    if scope == set() and chunks:
        raise ValueError("Treffer liegen außerhalb der explizit leeren Dokumentauswahl.")
    usable, duplicates = 0, 0
    seen = {}
    issues = set()
    for chunk in chunks:
        text = chunk.get("text")
        if isinstance(text, str) and text.strip():
            usable += 1
        else:
            issues.add("empty_or_invalid_text")
        identifier = chunk.get("document_id")
        valid_id = isinstance(identifier, str) and bool(identifier.strip())
        if not valid_id:
            issues.add("missing_or_invalid_document_id")
            if scope is not None:
                raise ValueError("Der Dokumentbereich eines Treffers ist nicht überprüfbar.")
        elif identifier not in known:
            raise ValueError("Ein Treffer verweist auf eine unbekannte oder gelöschte Dokument-ID.")
        elif scope is not None and identifier not in scope:
            raise ValueError("Ein Treffer liegt außerhalb der erlaubten Dokumentauswahl.")
        if not isinstance(chunk.get("filename"), str) or not chunk["filename"].strip():
            issues.add("missing_or_invalid_filename")
        elif valid_id and chunk["filename"] != known[identifier]["filename"]:
            issues.add("filename_mismatch")
        valid_reference = True
        for field in ("chunk_id", "page_number"):
            if type(chunk.get(field)) is not int or chunk[field] <= 0:
                issues.add(f"missing_or_invalid_{field}")
                valid_reference = False
        score = chunk.get("similarity_score")
        if type(score) not in (int, float) or not isfinite(score):
            issues.add("missing_or_invalid_similarity_score")
        if valid_id and valid_reference:
            key = (identifier, chunk["chunk_id"], chunk["page_number"])
            if key in seen:
                duplicates += 1
                issues.add("duplicate_reference")
                if seen[key] != text:
                    issues.add("conflicting_duplicate_text")
            seen[key] = text
    return QualityDiagnosis(
        status="unknown" if usable else "no_context",
        reason="semantic_sufficiency_unassessed" if usable else "no_usable_text",
        hit_count=len(chunks), usable_text_count=usable,
        empty_text_count=len(chunks) - usable, duplicate_count=duplicates,
        issues=tuple(sorted(issues)),
    )
