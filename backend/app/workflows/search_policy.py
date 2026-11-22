"""Wählt Suchbereich und Suchmodus ausschließlich anhand expliziter Metadaten."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


SearchMode = Literal["semantic", "hybrid", "hybrid_de"]


class SearchPlan(BaseModel):
    """Validierter, unveränderlicher Auftrag ohne Texte oder Modellzugriffe.

    None bedeutet alle Dokumente, ein leeres Tupel ausdrücklich keines.
    Die Existenz der IDs prüft select_search anhand aktueller Metadaten;
    DocumentCollection.search prüft sie bei der späteren Ausführung erneut.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: SearchMode = "semantic"
    document_ids: tuple[str, ...] | None = None
    routing_reason: str

    @field_validator("document_ids")
    @classmethod
    def validate_ids(cls, value):
        """Verhindert leere IDs und vereinheitlicht Reihenfolge und Duplikate."""
        if value is not None:
            if any(not identifier.strip() for identifier in value):
                raise ValueError("Dokument-IDs dürfen nicht leer sein.")
            return tuple(sorted(set(value)))
        return None

    @field_validator("routing_reason")
    @classmethod
    def validate_reason(cls, value):
        """Jeder Auftrag benötigt einen nachvollziehbaren Entscheidungsgrund."""
        if not value.strip():
            raise ValueError("routing_reason darf nicht leer sein.")
        return value


def select_search(
    documents: list[dict], *, mode: str | None = None,
    document_ids: list[str] | None = None, filename: str | None = None,
) -> SearchPlan:
    """Plant deterministisch mit dem Ergebnis von collection.list_documents().

    Ohne Angaben gilt semantic über alle Dokumente, unabhängig von der
    Prozessumgebung. Dateinamen werden exakt und case-sensitiv verglichen;
    Freitext, Dateipfade und Dokumentinhalte werden nicht analysiert.
    IDs und Dateiname sind alternative Auswahlen, keine kombinierten Filter.
    Unbekannte/gelöschte IDs oder mehrdeutige Namen führen zu ValueError.
    """
    selected_mode = "semantic" if mode is None else mode
    if not isinstance(selected_mode, str) or selected_mode not in ("semantic", "hybrid", "hybrid_de"):
        raise ValueError("Der Suchmodus muss semantic, hybrid oder hybrid_de sein.")
    if document_ids is not None and filename is not None:
        raise ValueError("Bitte entweder Dokument-IDs oder einen Dateinamen auswählen.")
    if not isinstance(documents, list):
        raise ValueError("Aktuelle Dokument-Metadaten müssen als Liste vorliegen.")
    known = {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Ungültige Dokument-Metadaten.")
        identifier, name = document.get("document_id"), document.get("filename")
        if not isinstance(identifier, str) or not identifier.strip() or not isinstance(name, str) or not name.strip():
            raise ValueError("Dokument-Metadaten benötigen eine gültige ID und einen Dateinamen.")
        if identifier in known:
            raise ValueError("Dokument-Metadaten enthalten eine doppelte Dokument-ID.")
        known[identifier] = name

    scope = None
    scope_reason = "all_documents"
    if document_ids is not None:
        if not isinstance(document_ids, list) or any(
            not isinstance(identifier, str) or not identifier.strip() for identifier in document_ids
        ):
            raise ValueError("document_ids muss eine Liste nichtleerer Dokument-IDs sein.")
        scope = tuple(sorted(set(document_ids)))
        if set(scope).difference(known):
            raise ValueError("Mindestens eine Dokument-ID ist unbekannt oder wurde gelöscht.")
        scope_reason = "explicit_document_ids" if scope else "explicit_empty_selection"
    elif filename is not None:
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Der ausgewählte Dateiname darf nicht leer sein.")
        matches = [identifier for identifier, name in known.items() if name == filename]
        if not matches:
            raise ValueError("Zum Dateinamen wurde kein aktuelles Dokument gefunden.")
        if len(matches) > 1:
            raise ValueError("Der Dateiname ist mehrdeutig. Bitte eine Dokument-ID auswählen.")
        scope = tuple(matches)
        scope_reason = "unique_explicit_filename"

    mode_reason = "default_semantic" if mode is None else "explicit_mode"
    return SearchPlan(mode=selected_mode, document_ids=scope, routing_reason=f"{scope_reason};{mode_reason}")
