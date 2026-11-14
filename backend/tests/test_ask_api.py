"""Prüft Indexieren und Fragen mit künstlichen PDFs und gemockter OpenAI API."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import rag_service, semantic_search


@pytest.fixture
def api(monkeypatch, tokenizer):
    """Isoliert den Index; verhindert echte API-Aufrufe und das Lesen der .env."""
    previous = getattr(main.app.state, "retrieval_pipeline", None)
    main.app.state.retrieval_pipeline = None
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    dotenv = Mock()
    monkeypatch.setattr(main, "load_dotenv", dotenv)
    model = Mock()
    model.tokenizer = tokenizer
    model.max_seq_length = 128
    model.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    client = Mock()
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=client)
    constructor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(rag_service, "OpenAI", constructor)
    client.responses.parse.return_value = SimpleNamespace(
        status="completed",
        output_parsed=rag_service.RagAnswer(
            answer="Mitarbeiter haben 30 Urlaubstage pro Jahr.",
            sources=[rag_service.Source(chunk_id=1, page_number=1)],
        ),
    )
    try:
        with TestClient(main.app) as http:
            yield http, client, constructor, dotenv, model
    finally:
        main.app.state.retrieval_pipeline = previous


def index_document(http):
    """Indexiert eine kleine künstliche PDF im Arbeitsspeicher."""
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "30 Urlaubstage pro Jahr.")
        content = document.tobytes()
    response = http.post("/documents/index", files={"file": ("test.pdf", content, "application/pdf")})
    assert response.status_code == 200


def test_index_then_ask(api):
    http, client, _, dotenv, model = api
    index_document(http)
    for body in ({"query": "Wie viel Urlaub?"}, {"query": "Wie viel Urlaub?", "top_k": 1}):
        response = http.post("/documents/ask", json=body)
        assert response.status_code == 200
        assert response.json() == {
            "answer": "Mitarbeiter haben 30 Urlaubstage pro Jahr.",
            "sources": [{"chunk_id": 1, "page_number": 1}],
        }
    context = json.loads(client.responses.parse.call_args.kwargs["input"])["context"]
    assert context == [{"chunk_id": 1, "page_number": 1, "text": "30 Urlaubstage pro Jahr.\n"}]
    path = dotenv.call_args.kwargs["dotenv_path"]
    assert path == Path(main.__file__).resolve().parents[1] / ".env"
    assert path.is_absolute()
    assert dotenv.call_args.kwargs["override"] is False
    assert model.encode.call_count == 3


def test_ask_without_document(api):
    http, _, constructor, dotenv, _ = api
    response = http.post("/documents/ask", json={"query": "Frage"})
    assert response.status_code == 409
    constructor.assert_not_called()
    dotenv.assert_not_called()


@pytest.mark.parametrize("body", [
    {}, {"query": ""}, {"query": " \n"}, {"query": None}, {"query": 123},
    *[{"query": "Frage", "top_k": value} for value in (0, -1, True, 1.5, "3", None)],
])
def test_invalid_ask_parameters(api, body):
    http, _, constructor, _, _ = api
    assert http.post("/documents/ask", json=body).status_code == 422
    constructor.assert_not_called()


def test_missing_key(api, monkeypatch):
    http, _, constructor, _, _ = api
    index_document(http)
    monkeypatch.delenv("OPENAI_API_KEY")
    response = http.post("/documents/ask", json={"query": "Frage"})
    assert response.status_code == 503
    assert "nicht konfiguriert" in response.json()["detail"]
    constructor.assert_not_called()


def test_api_failure_is_sanitized(api):
    http, client, _, _, _ = api
    index_document(http)
    client.responses.parse.side_effect = APIConnectionError(
        message="Interne vertrauliche Diagnose", request=httpx.Request("POST", "https://example.invalid")
    )
    response = http.post("/documents/ask", json={"query": "Frage"})
    assert response.status_code == 502
    assert "Interne vertrauliche Diagnose" not in response.text
    assert "placeholder" not in response.text
