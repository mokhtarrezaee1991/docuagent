"""Prüft BM25, Rangfusion, Moduswahl und Indexaktualisierung ohne Netzwerk."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from rank_bm25 import BM25Okapi

from app.services import semantic_search
from app.services.document_store import SQLiteDocumentStore
from app.services.hybrid_search import (
    BM25Index, BM25_PARAMETERS, RRF_K, reciprocal_rank_fusion,
    retrieval_mode, search_prepared, tokenize_keywords,
)
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline


def chunks():
    return [
        {"document_id": name, "filename": "gleich.pdf", "chunk_id": 1, "page_number": 1,
         "text": text, "embedding": vector}
        for name, text, vector in [
            ("a", "Urlaub Anspruch", [1.0, 0.0]),
            ("b", "Passwort Mindestlänge", [0.8, 0.6]),
            ("c", "Rechnung Frist", [0.0, 1.0]),
        ]
    ]


@pytest.fixture
def model(monkeypatch):
    model = Mock()
    model.encode.return_value = [[1.0, 0.0]]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    return model


def test_keyword_tokenization():
    assert tokenize_keywords("PASSWÖRTER: 14 Zeichen; Straße!") == ["passwörter", "14", "zeichen", "strasse"]


def test_bm25_uses_library_scores_without_adjustments():
    data = chunks()
    index = BM25Index(data)
    library = BM25Okapi([tokenize_keywords(chunk["text"]) for chunk in data], **BM25_PARAMETERS)
    hits = index.search("PASSWORT")
    assert [hit["document_id"] for hit in hits] == ["b"]
    assert hits[0]["bm25_score"] == float(library.get_scores(["passwort"])[1])
    assert type(hits[0]["bm25_score"]) is float
    assert "embedding" not in hits[0]
    assert index.search("unbekannt") == []
    assert index.search(" ") == []


def test_empty_bm25_corpus():
    assert BM25Index([]).search("Text") == []
    assert BM25Index([{"text": "!!!"}]).search("Text") == []


def test_rrf_formula_preserves_cosine_and_document_identity():
    semantic = [{**chunk, "similarity_score": score} for chunk, score in zip(chunks(), [0.9, 0.8, 0.7])]
    keyword = [{**semantic[1], "bm25_score": 4.0}, {**semantic[2], "bm25_score": 2.0}]
    original = deepcopy(semantic)
    hits = reciprocal_rank_fusion(semantic, keyword)
    assert [hit["document_id"] for hit in hits] == ["b", "c", "a"]
    assert hits[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert hits[0]["similarity_score"] == 0.8
    assert hits[0]["bm25_score"] == 4.0
    assert hits[0]["semantic_rank"] == 2
    assert hits[0]["bm25_rank"] == 1
    assert semantic == original
    assert RRF_K == 60


def test_rrf_stable_tie_and_duplicate_protection():
    data = chunks()
    semantic = [{**chunk, "similarity_score": 0.5} for chunk in data[:2]]
    keyword = [{**chunk, "bm25_score": 1.0} for chunk in reversed(data[:2])]
    hits = reciprocal_rank_fusion(semantic, keyword)
    assert hits[0]["rrf_score"] == hits[1]["rrf_score"]
    assert [hit["document_id"] for hit in hits] == ["a", "b"]
    duplicate = reciprocal_rank_fusion([semantic[0], semantic[0]], [])
    assert len(duplicate) == 1
    assert duplicate[0]["rrf_score"] == pytest.approx(1 / 61)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_rrf_constant(k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], k=k)


def test_semantic_and_hybrid_are_separately_available(model, monkeypatch):
    monkeypatch.setenv("DOCUAGENT_RETRIEVAL_MODE", "hybrid")
    data = chunks()
    original = deepcopy(data)
    semantic = search_prepared("Passwort", data, mode="semantic")
    hybrid = search_prepared("Passwort", data)
    assert [hit["document_id"] for hit in semantic] == ["a", "b", "c"]
    assert hybrid[0]["document_id"] == "b"
    assert hybrid[0]["similarity_score"] == semantic[1]["similarity_score"]
    assert "rrf_score" not in semantic[0]
    assert data == original
    assert model.encode.call_count == 2
    assert all(call.args[0] == ["Passwort"] for call in model.encode.call_args_list)


def test_hybrid_without_keyword_match_keeps_semantic_order(model):
    semantic = search_prepared("Unbekannt", chunks(), mode="semantic")
    hybrid = search_prepared("Unbekannt", chunks(), mode="hybrid")
    assert [hit["document_id"] for hit in hybrid] == [hit["document_id"] for hit in semantic]


@pytest.mark.parametrize("top_k", [0, -1, True, "3", None])
def test_invalid_hybrid_top_k(top_k, model):
    with pytest.raises(ValueError):
        search_prepared("Frage", chunks(), top_k=top_k, mode="hybrid")
    model.encode.assert_not_called()


def test_empty_hybrid_input_and_mode_validation(model):
    assert search_prepared("", chunks(), mode="hybrid") == []
    assert search_prepared("Frage", [], mode="hybrid") == []
    with pytest.raises(ValueError):
        retrieval_mode("unknown")
    model.encode.assert_not_called()


def prepared(chunk):
    """Stellt künstliche, bereits vorhandene Vektoren ohne Modellaufruf her."""
    return DocumentRetrievalPipeline.from_record({
        "metadata": {"document_id": chunk["document_id"], "filename": chunk["filename"], "page_count": 1, "chunk_count": 1},
        "chunks": [{key: chunk[key] for key in ("chunk_id", "page_number", "text", "embedding")}],
    })


def test_bm25_cache_invalidation_and_sqlite_restore(model, tmp_path):
    data = chunks()
    collection = DocumentCollection()
    collection.add_prepared(prepared(data[0]))
    collection.search("Urlaub", mode="hybrid")
    first_index = collection._keyword_index
    collection.search("Passwort", mode="hybrid")
    assert collection._keyword_index is first_index
    collection.add_prepared(prepared(data[1]))
    assert collection._keyword_index is None
    assert collection.search("Passwort", top_k=1, mode="hybrid")[0]["document_id"] == "b"
    collection.remove_document("b")
    assert collection._keyword_index is None
    assert {hit["document_id"] for hit in collection.search("Passwort", mode="hybrid")} == {"a"}

    store = SQLiteDocumentStore(tmp_path / "hybrid.sqlite3")
    store.save_document(prepared(data[0]).to_record())
    store.save_document(prepared(data[1]).to_record())
    restored = DocumentCollection()
    model.encode.reset_mock()
    records, _ = SQLiteDocumentStore(store.path).load_documents()
    for record in records:
        restored.add_prepared(DocumentRetrievalPipeline.from_record(record))
    model.encode.assert_not_called()
    hits = restored.search("Passwort", top_k=1, mode="hybrid")
    assert hits[0]["document_id"] == "b"
    model.encode.assert_called_once_with(["Passwort"], convert_to_numpy=True)


def test_api_mode_switch_works_after_sqlite_restart(model, monkeypatch):
    """Der Prozessmodus benötigt keine Änderung an API-Anfragen oder Datenbank."""
    from fastapi.testclient import TestClient
    from app import main

    store = SQLiteDocumentStore(main.database_path())
    for chunk in chunks():
        store.save_document(prepared(chunk).to_record())
    with TestClient(main.app) as client:
        semantic = client.post("/documents/search", json={"query": "Passwort"})
        assert semantic.status_code == 200
        assert "rrf_score" not in semantic.json()[0]
        monkeypatch.setenv("DOCUAGENT_RETRIEVAL_MODE", "hybrid")
        hybrid = client.post("/documents/search", json={"query": "Passwort"})
        assert hybrid.status_code == 200
        assert hybrid.json()[0]["document_id"] == "b"
        assert type(hybrid.json()[0]["rrf_score"]) is float
    with TestClient(main.app) as client:
        assert client.post("/documents/search", json={"query": "Passwort"}).json() == hybrid.json()
    assert model.encode.call_count == 3
