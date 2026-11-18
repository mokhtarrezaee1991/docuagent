"""Prüft FastAPI-Neustarts mit temporärer SQLite-Datei und lokalen Mock-Vektoren."""

from unittest.mock import Mock
import sqlite3

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import semantic_search


@pytest.fixture
def model(monkeypatch, tokenizer):
    model = Mock(tokenizer=tokenizer, max_seq_length=128)
    vectors = {"Urlaub": [1.0, 0.0], "Rechnung": [0.0, 1.0]}
    model.encode.side_effect = lambda texts, **kwargs: [vectors[text.strip()] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    # Ein versehentlicher OpenAI-Aufruf würde den Test sofort abbrechen.
    monkeypatch.setattr(main, "generate_answer", Mock(side_effect=AssertionError("Keine OpenAI-Aufrufe erlaubt")))
    return model


def add(client, text, endpoint="/documents"):
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), text)
        content = document.tobytes()
    response = client.post(endpoint, files={"file": ("gleich.pdf", content, "application/pdf")})
    assert response.status_code == 200
    return response.json()


def test_restart_restores_search_and_deletion(model, monkeypatch):
    with TestClient(main.app) as client:
        first = add(client, "Urlaub")
        second = add(client, "Rechnung")
        before = client.post("/documents/search", json={"query": "Urlaub"}).json()
    calls_before_restart = model.encode.call_count
    # Beim Start dürfen weder Tokenizer noch Modell oder Extraktion benötigt werden.
    loader = Mock(side_effect=AssertionError("Kein Modell beim Start laden"))
    monkeypatch.setattr(semantic_search, "_get_model", loader)
    with TestClient(main.app) as client:
        assert client.get("/documents").json() == [first, second]
        loader.assert_not_called()
        monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
        assert client.post("/documents/search", json={"query": "Urlaub"}).json() == before
        assert model.encode.call_count == calls_before_restart + 1
        assert client.delete(f"/documents/{first['document_id']}").status_code == 200
    with TestClient(main.app) as client:
        assert client.get("/documents").json() == [second]
        hits = client.post("/documents/search", json={"query": "Rechnung"}).json()
        assert len(hits) == 1 and hits[0]["document_id"] == second["document_id"]
        assert client.delete(f"/documents/{second['document_id']}").status_code == 200
    with TestClient(main.app) as client:
        assert client.get("/documents").json() == []
        assert client.post("/documents/search", json={"query": "Urlaub"}).status_code == 409


def test_legacy_replace_survives_restart(model):
    with TestClient(main.app) as client:
        add(client, "Rechnung")
        add(client, "Urlaub", "/documents/index")
        before = client.post("/documents/search", json={"query": "Urlaub"}).json()
    with TestClient(main.app) as client:
        assert len(client.get("/documents").json()) == 1
        assert client.post("/documents/search", json={"query": "Urlaub"}).json() == before
        add(client, "Rechnung")
    with TestClient(main.app) as client:
        assert len(client.get("/documents").json()) == 2


def test_database_failure_keeps_memory_and_persistent_data(model, monkeypatch):
    with TestClient(main.app) as client:
        first = add(client, "Urlaub")
        broken = Mock(side_effect=sqlite3.OperationalError("Interne SQL-Diagnose"))
        monkeypatch.setattr(main.app.state.document_store, "save_document", broken)
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), "Rechnung")
            content = document.tobytes()
        for endpoint in ("/documents", "/documents/index"):
            response = client.post(endpoint, files={"file": ("test.pdf", content, "application/pdf")})
            assert response.status_code == 503
            assert "Interne SQL-Diagnose" not in response.text
        monkeypatch.setattr(main.app.state.document_store, "delete_document", broken)
        assert client.delete(f"/documents/{first['document_id']}").status_code == 503
        assert client.get("/documents").json() == [first]
    with TestClient(main.app) as client:
        assert client.get("/documents").json() == [first]
