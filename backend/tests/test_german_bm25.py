"""Prüft deutsche Wortnormalisierung ohne Netzwerk, LLM oder Testfragen-Tuning."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from rank_bm25 import BM25Okapi

from app.services import semantic_search
from app.services.hybrid_search import BM25Index, BM25_PARAMETERS, RRF_K, search_prepared, tokenize_keywords
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline


@pytest.mark.parametrize("singular, plural", [
    ("Passwort", "Passwörter"), ("Urlaubsantrag", "Urlaubsanträge"),
    ("Haus", "Häuser"), ("Rechnung", "Rechnungen"),
])
def test_german_singular_plural(singular, plural):
    assert tokenize_keywords(singular, normalization="german") == tokenize_keywords(plural, normalization="german")
    assert tokenize_keywords(singular) != tokenize_keywords(plural)


def test_unicode_numbers_and_compound_limitations():
    assert tokenize_keywords("Passwörter 14", normalization="german") == tokenize_keywords("Passwo\u0308rter 14", normalization="german")
    assert tokenize_keywords("14 02 30", normalization="german") == ["14", "02", "30"]
    assert tokenize_keywords("Rechnung", normalization="german") != tokenize_keywords("Lieferantenrechnungen", normalization="german")
    assert tokenize_keywords("Urlaub", normalization="german") != tokenize_keywords("Urlaubstage", normalization="german")
    assert tokenize_keywords("einreichen", normalization="german") != tokenize_keywords("eingereicht", normalization="german")


def data():
    return [
        {"document_id": name, "filename": "gleich.pdf", "chunk_id": 1, "page_number": 1,
         "text": text, "embedding": vector}
        for name, text, vector in [
            ("a", "Passwörter", [0.8, 0.6]),
            ("b", "Rechnungen", [1.0, 0.0]),
            ("c", "Urlaubsanträge", [0.0, 1.0]),
        ]
    ]


def test_documents_and_queries_use_same_stemming_and_library_scores():
    chunks = data()
    original = deepcopy(chunks)
    raw = BM25Index(chunks)
    german = BM25Index(chunks, normalization="german")
    assert raw.search("Passwort") == []
    hits = german.search("PASSWORT")
    assert len(hits) == 1 and hits[0]["document_id"] == "a"
    library = BM25Okapi([tokenize_keywords(chunk["text"], normalization="german") for chunk in chunks], **BM25_PARAMETERS)
    assert hits[0]["bm25_score"] == float(library.get_scores(tokenize_keywords("PASSWORT", normalization="german"))[0])
    assert chunks == original


def test_invalid_normalization():
    with pytest.raises(ValueError):
        tokenize_keywords("Text", normalization="invalid")
    with pytest.raises(ValueError):
        BM25Index([], normalization="invalid")


@pytest.fixture
def model(monkeypatch):
    model = Mock()
    model.encode.return_value = [[1.0, 0.0]]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    return model


def test_stemming_changes_only_keyword_branch(model):
    chunks = data()
    original = deepcopy(chunks)
    semantic = search_prepared("Passwort", chunks, mode="semantic")
    previous = search_prepared("Passwort", chunks, mode="hybrid")
    normalized = search_prepared("Passwort", chunks, mode="hybrid_de")
    assert [hit["document_id"] for hit in previous] == [hit["document_id"] for hit in semantic]
    assert normalized[0]["document_id"] == "a"
    assert normalized[0]["bm25_rank"] == 1
    assert normalized[0]["similarity_score"] == next(hit["similarity_score"] for hit in semantic if hit["document_id"] == "a")
    assert normalized[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert chunks == original
    assert RRF_K == 60
    assert model.encode.call_count == 3
    assert all(call.args[0] == ["Passwort"] for call in model.encode.call_args_list)


def test_incompatible_cached_index_is_rejected(model):
    with pytest.raises(ValueError, match="unterschiedliche Normalisierung"):
        search_prepared("Passwort", data(), mode="hybrid_de", keyword_index=BM25Index(data()))


def test_mode_specific_cache_and_document_removal(model):
    collection = DocumentCollection()
    for chunk in data():
        collection.add_prepared(DocumentRetrievalPipeline.from_record({
            "metadata": {"document_id": chunk["document_id"], "filename": chunk["filename"], "page_count": 1, "chunk_count": 1},
            "chunks": [{key: chunk[key] for key in ("chunk_id", "page_number", "text", "embedding")}],
        }))
    collection.search("Passwort", mode="hybrid")
    assert collection._keyword_index.normalization == "none"
    collection.search("Passwort", mode="hybrid_de")
    index = collection._keyword_index
    assert index.normalization == "german"
    collection.search("Rechnung", mode="hybrid_de")
    assert collection._keyword_index is index
    collection.search("Passwort", mode="hybrid")
    assert collection._keyword_index.normalization == "none"
    collection.remove_document("a")
    assert all(hit["document_id"] != "a" for hit in collection.search("Passwort", mode="hybrid_de"))


def test_process_mode_and_default_stay_compatible(model, monkeypatch):
    monkeypatch.delenv("DOCUAGENT_RETRIEVAL_MODE", raising=False)
    assert "rrf_score" not in search_prepared("Passwort", data())[0]
    monkeypatch.setenv("DOCUAGENT_RETRIEVAL_MODE", "hybrid_de")
    assert search_prepared("Passwort", data())[0]["document_id"] == "a"
