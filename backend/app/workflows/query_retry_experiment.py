"""Explizit aktivierbares Offline-Experiment, keine produktive Retry-Steuerung."""

from copy import deepcopy
from typing import Callable

from app.workflows.retrieval_quality import diagnose_retrieval
from app.workflows.search_policy import select_search


def normalize_query(query: str) -> str:
    """Vergleicht Schreibweise und Leerraum ohne semantische Umschreibung."""
    return " ".join(query.casefold().split())


def select_context(first: list[dict], second: list[dict], top_k: int) -> list[dict]:
    """Vorab feste Regel: Original-Rang 1 schützen, Rest nach gleicher RRF(60).

    Dedupliziert nach Dokument/Chunk/Seite. Bei Gleichstand Vorrang der ersten
    Liste. Metadaten und Originalscores stammen beim Überlappen aus Suche 1;
    verschiedene Query-Scores werden nicht direkt verglichen oder verändert.
    Schutz von Rang 1 ist kein Beweis, dass dieser Treffer tatsächlich relevant
    ist. Andere Originaltreffer können verdrängt werden; das Experiment misst es.
    """
    candidates, scores = {}, {}
    def key(hit):
        return (hit["document_id"], hit["chunk_id"], hit["page_number"])
    for hits in (first, second):
        seen = set()
        for rank, hit in enumerate(hits, 1):
            identity = key(hit)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.setdefault(identity, hit)
            scores[identity] = scores.get(identity, 0) + 1 / (60 + rank)
    ranked = sorted(candidates, key=lambda identity: scores[identity], reverse=True)
    if first:
        anchor = key(first[0])
        ranked.remove(anchor)
        ranked.insert(0, anchor)
    return deepcopy([candidates[identity] for identity in ranked[:top_k]])


def run_query_retry(
    query: str, *, retrieve: Callable, get_documents: Callable,
    alternative_query: str | None = None, enabled: bool = False,
    mode: str = "semantic", document_ids: list[str] | None = None, top_k: int = 3,
) -> dict:
    """Sucht einmal, optional genau ein zweites Mal mit vorgegebener Anfrage.

    Unknown aus der Strukturdiagnose ist KEIN Retry-Auslöser. Nur enabled und
    eine gültige andere Alternative erlauben den zweiten Versuch. Fehler werden
    weitergegeben, nie durch globale Suche oder eine dritte Suche ersetzt.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Die Originalfrage darf nicht leer sein.")
    if type(enabled) is not bool or type(top_k) is not int or top_k <= 0:
        raise ValueError("enabled und top_k sind ungültig.")
    plan = select_search(get_documents(), mode=mode, document_ids=document_ids)
    scope = None if plan.document_ids is None else list(plan.document_ids)
    def search(text):
        hits = retrieve(text, top_k=top_k, mode=plan.mode, document_ids=deepcopy(scope))
        diagnose_retrieval(hits, documents=get_documents(), document_ids=scope)
        return deepcopy(hits)
    first = search(query)
    second = []
    reason = "automation_disabled"
    if enabled:
        if scope == []:
            reason = "empty_selection"
        elif alternative_query is None:
            reason = "no_alternative"
        elif not isinstance(alternative_query, str) or not alternative_query.strip():
            reason = "invalid_alternative"
        elif normalize_query(alternative_query) == normalize_query(query):
            reason = "identical_alternative"
        else:
            second = search(alternative_query)
            reason = "explicit_experiment_second_search"
    retried = reason == "explicit_experiment_second_search"
    return {
        "original_query": query, "alternative_query": alternative_query if retried else None,
        "mode": plan.mode, "document_ids": scope, "top_k": top_k,
        "first_hits": first, "second_hits": second,
        "final_hits": select_context(first, second, top_k),
        "search_calls": 2 if retried else 1, "retry_count": int(retried), "reason": reason,
    }
