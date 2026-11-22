"""Prüft Dokumentfilter mit künstlichen Chunks und lokalen Mock-Embeddings."""

from unittest.mock import Mock

import pytest

from app.services import semantic_search
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline


MODES = ["semantic", "hybrid", "hybrid_de"]


def add(collection, identifier, text, vector):
    """Übernimmt einen vorbereiteten Chunk ohne PDF, Netzwerk oder Modell-Download."""
    collection.add_prepared(DocumentRetrievalPipeline.from_record({
        "metadata": {"document_id": identifier, "filename": "gleich.pdf", "page_count": 2, "chunk_count": 1},
        "chunks": [{"chunk_id": 1, "page_number": 2, "text": text, "embedding": vector}],
    }))


@pytest.fixture
def data(monkeypatch):
    model = Mock()
    model.encode.return_value = [[1.0, 0.0]]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    collection = DocumentCollection()
    add(collection, "a", "Passwort Zugang", [1.0, 0.0])
    add(collection, "b", "Passwörter Anmeldung", [0.8, 0.6])
    add(collection, "c", "Rechnungen bezahlen", [0.0, 1.0])
    return collection, model


@pytest.mark.parametrize("mode", MODES)
def test_single_document_and_same_filenames(data, mode):
    collection, model = data
    hits = collection.search("Passwort", mode=mode, document_ids=["b"], top_k=10)
    assert len(hits) == 1
    assert hits[0]["document_id"] == "b"
    assert hits[0]["filename"] == "gleich.pdf"
    assert hits[0]["chunk_id"] == 1 and hits[0]["page_number"] == 2
    assert hits[0]["text"] == "Passwörter Anmeldung"
    assert hits[0]["similarity_score"] == pytest.approx(0.8)
    model.encode.assert_called_once_with(["Passwort"], convert_to_numpy=True)


@pytest.mark.parametrize("mode", MODES)
def test_multiple_ids_duplicates_and_order(data, mode):
    collection, _ = data
    expected = collection.search("Passwort", mode=mode, document_ids=["b", "c"], top_k=10)
    actual = collection.search("Passwort", mode=mode, document_ids=["c", "b", "b"], top_k=10)
    assert actual == expected
    assert {hit["document_id"] for hit in actual} == {"b", "c"}


@pytest.mark.parametrize("mode", MODES)
def test_filter_precedes_top_k(data, mode):
    collection, _ = data
    assert collection.search("Passwort", mode=mode, top_k=1)[0]["document_id"] == "a"
    # Der globale Erstplatzierte wird ausgeschlossen; trotzdem entsteht ein Treffer.
    hits = collection.search("Passwort", mode=mode, document_ids=["b", "c"], top_k=1)
    assert len(hits) == 1 and hits[0]["document_id"] == "b"


@pytest.mark.parametrize("mode", MODES)
def test_none_matches_unfiltered_search(data, mode):
    collection, _ = data
    assert collection.search("Passwort", mode=mode) == collection.search("Passwort", mode=mode, document_ids=None)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ids", [["unbekannt"], ["a", "unbekannt"]])
def test_unknown_ids_reject_whole_request(data, mode, ids):
    collection, model = data
    with pytest.raises(ValueError, match="unbekannt"):
        collection.search("Passwort", mode=mode, document_ids=ids)
    model.encode.assert_not_called()


@pytest.mark.parametrize("mode", MODES)
def test_deleted_document_is_rejected(data, mode):
    collection, _ = data
    collection.search("Passwort", mode=mode, document_ids=["b"])
    assert collection.remove_document("b")
    with pytest.raises(ValueError, match="gelöscht"):
        collection.search("Passwort", mode=mode, document_ids=["b"])
    assert all(hit["document_id"] != "b" for hit in collection.search("Passwort", mode=mode))


@pytest.mark.parametrize("mode", MODES)
def test_empty_selection_returns_nothing_without_embedding(data, mode):
    collection, model = data
    assert collection.search("Passwort", mode=mode, document_ids=[]) == []
    model.encode.assert_not_called()
    with pytest.raises(ValueError, match="top_k"):
        collection.search("Passwort", mode=mode, document_ids=[], top_k=0)


@pytest.mark.parametrize("ids", ["a", ("a",), [None], [1], [""], ["  "], {"a"}])
def test_invalid_filter_types(data, ids):
    collection, model = data
    with pytest.raises(ValueError, match="document_ids"):
        collection.search("Passwort", document_ids=ids)
    model.encode.assert_not_called()


@pytest.mark.parametrize("mode", ["hybrid", "hybrid_de"])
def test_global_cache_cannot_escape_scope_or_change_scores(data, mode):
    collection, _ = data
    global_hits = collection.search("Passwort", mode=mode, top_k=10)
    global_index = collection._keyword_index
    # Vergleich mit einer tatsächlich separaten Sammlung desselben Teilbestands.
    isolated = DocumentCollection()
    add(isolated, "b", "Passwörter Anmeldung", [0.8, 0.6])
    add(isolated, "c", "Rechnungen bezahlen", [0.0, 1.0])
    expected = isolated.search("Passwort", mode=mode, top_k=10)
    assert collection.search("Passwort", mode=mode, document_ids=["b", "c"], top_k=10) == expected
    assert collection._keyword_index is global_index
    assert collection.search("Passwort", mode=mode, top_k=10) == global_hits
    assert collection.search("Passwort", mode=mode, document_ids=["c"])[0]["document_id"] == "c"


def test_mode_changes_and_document_addition_do_not_leave_stale_scope(data):
    collection, _ = data
    for mode in ["hybrid", "hybrid_de", "semantic", "hybrid"]:
        assert collection.search("Passwort", mode=mode, document_ids=["b"])[0]["document_id"] == "b"
    add(collection, "d", "Passwort neu", [1.0, 0.0])
    assert collection.search("Passwort", mode="hybrid_de", document_ids=["d"])[0]["document_id"] == "d"
