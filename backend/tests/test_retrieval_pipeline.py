"""Prüft die Pipeline mit künstlichen PDFs und einem Modell ohne Netzwerk."""

from unittest.mock import Mock

import pymupdf
import pytest

from app.services import semantic_search
from app.services.retrieval_pipeline import DocumentRetrievalPipeline


def make_pdf(*texts):
    """Erzeugt Testseiten direkt im Arbeitsspeicher."""
    with pymupdf.open() as document:
        for text in texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()


@pytest.fixture
def model(monkeypatch):
    vectors = {"Urlaub": [1.0, 0.0], "Rechnung": [0.0, 1.0]}
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: [vectors[text.strip()] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    return model


def test_multiple_questions_reuse_document_embeddings(model):
    pipeline = DocumentRetrievalPipeline()
    # Eine leere erste Seite prüft, dass originale Seitenzahlen erhalten bleiben.
    pipeline.load_pdf(make_pdf("", "Urlaub", "Rechnung"))
    first = pipeline.search("Urlaub", top_k=1)
    second = pipeline.search("Rechnung", top_k=1)
    assert first == [{"chunk_id": 1, "page_number": 2, "text": "Urlaub\n", "similarity_score": 1.0}]
    assert second == [{"chunk_id": 2, "page_number": 3, "text": "Rechnung\n", "similarity_score": 1.0}]
    assert [call.args[0] for call in model.encode.call_args_list] == [
        ["Urlaub\n", "Rechnung\n"], ["Urlaub"], ["Rechnung"]
    ]


def test_search_before_loading():
    with pytest.raises(ValueError, match="zuerst"):
        DocumentRetrievalPipeline().search("Frage")


@pytest.mark.parametrize("content", [b"", b"keine PDF", b"%PDF-1.7\nkaputt"])
def test_invalid_pdf(content, model):
    pipeline = DocumentRetrievalPipeline()
    with pytest.raises(ValueError):
        pipeline.load_pdf(content)
    model.encode.assert_not_called()


def test_pdf_without_text(model):
    with pytest.raises(ValueError, match="keinen extrahierbaren Text"):
        DocumentRetrievalPipeline().load_pdf(make_pdf(""))
    model.encode.assert_not_called()


@pytest.mark.parametrize("query", ["", " \n", None, 123])
def test_invalid_query(query, model):
    pipeline = DocumentRetrievalPipeline()
    pipeline.load_pdf(make_pdf("Urlaub"))
    with pytest.raises(ValueError, match="Suchfrage"):
        pipeline.search(query)
    assert model.encode.call_count == 1


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, None, "3"])
def test_invalid_top_k(top_k, model):
    pipeline = DocumentRetrievalPipeline()
    pipeline.load_pdf(make_pdf("Urlaub"))
    with pytest.raises(ValueError, match="top_k"):
        pipeline.search("Urlaub", top_k=top_k)
    assert model.encode.call_count == 1


def test_reload_replaces_document_and_failed_reload_preserves_it(model):
    pipeline = DocumentRetrievalPipeline()
    pipeline.load_pdf(make_pdf("Urlaub"))
    pipeline.load_pdf(make_pdf("Rechnung"))
    with pytest.raises(ValueError):
        pipeline.load_pdf(b"kaputt")
    hits = pipeline.search("Rechnung", top_k=10)
    assert len(hits) == 1
    assert hits[0]["text"] == "Rechnung\n"


def test_instances_have_separate_documents(model):
    first = DocumentRetrievalPipeline()
    second = DocumentRetrievalPipeline()
    first.load_pdf(make_pdf("Urlaub"))
    second.load_pdf(make_pdf("Rechnung"))
    assert first.search("Urlaub")[0]["text"] == "Urlaub\n"
    assert second.search("Rechnung")[0]["text"] == "Rechnung\n"
