"""Prüft die reine Metadaten-Policy ohne Netzwerk, PDFs oder Modellzugriffe."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline
from app.workflows.search_policy import SearchPlan, select_search


@pytest.fixture
def documents():
    return [
        {"document_id": "a", "filename": "gleich.pdf", "page_count": 1, "chunk_count": 2},
        {"document_id": "b", "filename": "gleich.pdf", "page_count": 2, "chunk_count": 3},
        {"document_id": "c", "filename": "eindeutig.pdf", "page_count": 1, "chunk_count": 1},
    ]


def test_default_semantic_all_documents_ignores_environment(documents, monkeypatch):
    monkeypatch.setenv("DOCUAGENT_RETRIEVAL_MODE", "hybrid_de")
    plan = select_search(documents)
    assert plan.mode == "semantic"
    assert plan.document_ids is None
    assert plan.routing_reason == "all_documents;default_semantic"


@pytest.mark.parametrize("mode", ["semantic", "hybrid", "hybrid_de"])
def test_explicit_modes_do_not_change_scope(documents, mode):
    plan = select_search(documents, mode=mode)
    assert plan.mode == mode and plan.document_ids is None
    assert plan.routing_reason == "all_documents;explicit_mode"


@pytest.mark.parametrize("mode", ["bm25", "auto", "", "SEMANTIC", 3, True, []])
def test_invalid_modes(documents, mode):
    with pytest.raises(ValueError, match="Suchmodus"):
        select_search(documents, mode=mode)


def test_unique_filename_requires_exact_explicit_metadata(documents):
    plan = select_search(documents, filename="eindeutig.pdf")
    assert plan.document_ids == ("c",)
    assert plan.routing_reason == "unique_explicit_filename;default_semantic"
    for filename in ["EINDEUTIG.pdf", "laut eindeutig.pdf", "eindeutig", "../eindeutig.pdf"]:
        with pytest.raises(ValueError):
            select_search(documents, filename=filename)


def test_duplicate_filename_is_not_guessed(documents):
    with pytest.raises(ValueError, match="mehrdeutig"):
        select_search(documents, filename="gleich.pdf")
    assert select_search(documents, document_ids=["b"]).document_ids == ("b",)


def test_multiple_ids_and_duplicates_are_canonical(documents):
    plan = select_search(documents, document_ids=["c", "a", "a"])
    assert plan.document_ids == ("a", "c")
    assert plan == select_search(list(reversed(documents)), document_ids=["a", "c"])
    assert plan.routing_reason == "explicit_document_ids;default_semantic"


def test_empty_selection_remains_empty(documents):
    plan = select_search(documents, document_ids=[])
    assert plan.document_ids == ()
    assert plan.routing_reason == "explicit_empty_selection;default_semantic"
    assert select_search([]).document_ids is None
    assert select_search([], document_ids=[]).document_ids == ()


@pytest.mark.parametrize("ids", [["unbekannt"], ["a", "unbekannt"]])
def test_unknown_ids_reject_whole_selection(documents, ids):
    with pytest.raises(ValueError, match="unbekannt"):
        select_search(documents, document_ids=ids)


@pytest.mark.parametrize("ids", ["a", ("a",), [None], [1], [""], ["  "]])
def test_invalid_id_input(documents, ids):
    with pytest.raises(ValueError, match="document_ids"):
        select_search(documents, document_ids=ids)


@pytest.mark.parametrize("filename", ["", "  ", 1, []])
def test_invalid_filename(documents, filename):
    with pytest.raises(ValueError, match="Dateiname"):
        select_search(documents, filename=filename)


@pytest.mark.parametrize("ids", [[], ["a"]])
def test_competing_scope_inputs_are_rejected(documents, ids):
    with pytest.raises(ValueError, match="entweder"):
        select_search(documents, document_ids=ids, filename="eindeutig.pdf")


@pytest.mark.parametrize("mode", [None, "semantic", "hybrid", "hybrid_de"])
def test_scope_and_mode_are_independent_and_inputs_unchanged(documents, mode):
    original = deepcopy(documents)
    ids = ["c", "a", "c"]
    plan = select_search(documents, document_ids=ids, mode=mode)
    assert plan.document_ids == ("a", "c")
    assert plan.mode == (mode or "semantic")
    assert documents == original and ids == ["c", "a", "c"]


def test_current_collection_and_deleted_ids():
    collection = DocumentCollection()
    pipeline = DocumentRetrievalPipeline.from_record({
        "metadata": {"document_id": "a", "filename": "test.pdf", "page_count": 1, "chunk_count": 0},
        "chunks": [],
    })
    collection.add_prepared(pipeline)
    assert select_search(collection.list_documents(), document_ids=["a"]).document_ids == ("a",)
    assert collection.remove_document("a")
    with pytest.raises(ValueError, match="gelöscht"):
        select_search(collection.list_documents(), document_ids=["a"])
    with pytest.raises(ValueError, match="kein aktuelles"):
        select_search(collection.list_documents(), filename="test.pdf")


@pytest.mark.parametrize("metadata", [None, [{}], [None], [{"document_id": "", "filename": "a.pdf"}],
    [{"document_id": "a", "filename": "a.pdf"}, {"document_id": "a", "filename": "b.pdf"}]])
def test_invalid_metadata_fails_closed(metadata):
    with pytest.raises(ValueError):
        select_search(metadata)


def test_plan_is_typed_validated_and_immutable():
    plan = SearchPlan(document_ids=("b", "a", "a"), routing_reason="explicit")
    assert plan.document_ids == ("a", "b")
    for kwargs in [
        {"mode": "unknown", "routing_reason": "test"},
        {"document_ids": (1,), "routing_reason": "test"},
        {"document_ids": ("",), "routing_reason": "test"},
        {"routing_reason": " "},
        {"routing_reason": "test", "tool_name": "dynamic"},
    ]:
        with pytest.raises(ValidationError):
            SearchPlan(**kwargs)
    with pytest.raises(ValidationError):
        plan.mode = "hybrid"
