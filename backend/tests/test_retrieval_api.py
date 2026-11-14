"""Prüft die Retrieval-Endpunkte mit isoliertem Index und Mock-Embeddings."""

from unittest.mock import Mock

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import semantic_search


def make_pdf(*texts):
    """Erzeugt ausschließlich künstliche PDF-Seiten im Speicher."""
    with pymupdf.open() as document:
        for text in texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()


@pytest.fixture
def api(monkeypatch):
    """Stellt den vorherigen Zustand nach jedem Test automatisch wieder her."""
    previous_pipeline = getattr(app.state, "retrieval_pipeline", None)
    app.state.retrieval_pipeline = None
    vectors = {"Urlaub": [1.0, 0.0], "Rechnung": [0.0, 1.0]}
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: [vectors[text.strip()] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    try:
        with TestClient(app) as client:
            yield client, model
    finally:
        app.state.retrieval_pipeline = previous_pipeline


def index(client, content, content_type="application/pdf"):
    return client.post("/documents/index", files={"file": ("test.pdf", content, content_type)})


def test_index_and_search_reuse_embeddings(api):
    client, model = api
    response = index(client, make_pdf("Urlaub", "Rechnung"))
    assert response.status_code == 200
    assert response.json() == {"page_count": 2, "chunk_count": 2}
    for query, page in [("Urlaub", 1), ("Rechnung", 2)]:
        response = client.post("/documents/search", json={"query": query, "top_k": 1})
        assert response.status_code == 200
        assert response.json() == [{
            "chunk_id": page, "page_number": page, "text": query + "\n", "similarity_score": 1.0
        }]
    assert [call.args[0] for call in model.encode.call_args_list] == [
        ["Urlaub\n", "Rechnung\n"], ["Urlaub"], ["Rechnung"]
    ]


def test_search_without_index(api):
    client, _ = api
    response = client.post("/documents/search", json={"query": "Urlaub"})
    assert response.status_code == 409
    assert "zuerst" in response.json()["detail"]


def test_index_and_search_with_numpy_float32(api):
    """Echte NumPy-Vektoren müssen als normale JSON-Zahlen zurückkommen."""
    client, model = api
    vectors = {
        "Urlaub": [1.0, 0.0],
        "Rechnung": [0.0, 1.0],
        "Frage": [0.8, 0.6],
    }
    model.encode.side_effect = lambda texts, **kwargs: np.array(
        [vectors[text.strip()] for text in texts], dtype=np.float32
    )
    assert index(client, make_pdf("Urlaub", "Rechnung")).status_code == 200

    # Prüft den tatsächlichen Service-Rückgabewert vor der Serialisierung.
    hits = app.state.retrieval_pipeline.search("Frage")
    for hit in hits:
        assert type(hit["similarity_score"]) is float
        assert type(hit["chunk_id"]) is int
        assert type(hit["page_number"]) is int
        assert type(hit["text"]) is str

    response = client.post("/documents/search", json={"query": "Frage"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == hits
    assert [hit["chunk_id"] for hit in hits] == [1, 2]
    assert [hit["similarity_score"] for hit in hits] == pytest.approx([0.8, 0.6])


@pytest.mark.parametrize("body", [
    {}, {"query": ""}, {"query": " \n"}, {"query": None}, {"query": 123},
    *[{"query": "Urlaub", "top_k": value} for value in (0, -1, True, 1.5, "3", None)],
])
def test_invalid_search_parameters(api, body):
    client, model = api
    assert client.post("/documents/search", json=body).status_code == 422
    model.encode.assert_not_called()


@pytest.mark.parametrize("content, content_type, status", [
    (b"text", "text/plain", 415),
    (b"text", "application/pdf", 400),
    (b"", "application/pdf", 400),
    (b"%PDF-1.7\nkaputt", "application/pdf", 400),
    (make_pdf(""), "application/pdf", 400),
    (b"%PDF-" + b" " * (10 * 1024 * 1024), "application/pdf", 413),
])
def test_invalid_upload(api, content, content_type, status):
    client, model = api
    assert index(client, content, content_type).status_code == status
    model.encode.assert_not_called()
    assert client.post("/documents/search", json={"query": "Urlaub"}).status_code == 409


def test_replace_index_and_preserve_it_after_failed_upload(api):
    client, _ = api
    assert index(client, make_pdf("Urlaub")).status_code == 200
    assert index(client, make_pdf("Rechnung")).status_code == 200
    assert index(client, b"kaputt").status_code == 400
    result = client.post("/documents/search", json={"query": "Rechnung"})
    assert result.status_code == 200
    assert len(result.json()) == 1
    assert result.json()[0]["text"] == "Rechnung\n"
