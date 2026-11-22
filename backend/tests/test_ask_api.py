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
from app.services import rag_service, semantic_search, retrieval_pipeline


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
    client.responses.parse.assert_called_once()


@pytest.mark.parametrize("has_key,expected_status", [(True, 200), (False, 503)])
def test_workflow_empty_context_preserves_key_requirement(api, monkeypatch, has_key, expected_status):
    """Lokaler Verzicht umgeht die bisherige API-Schlüsselprüfung nicht."""
    http, remote, constructor, dotenv, _ = api
    index_document(http)
    pipeline = main.app.state.retrieval_pipeline
    search = Mock(return_value=[])
    monkeypatch.setattr(pipeline, "search", search)
    generator = Mock(side_effect=AssertionError("Kein Generator bei leerem Kontext"))
    monkeypatch.setattr(main, "generate_answer", generator)
    if not has_key:
        monkeypatch.delenv("OPENAI_API_KEY")

    response = http.post("/documents/ask", json={"query": "Unbekannte Information", "top_k": 2})

    assert response.status_code == expected_status
    if has_key:
        assert response.json() == {"answer": "Die Information wurde im Dokument nicht gefunden.", "sources": []}
    else:
        assert "nicht konfiguriert" in response.json()["detail"]
    search.assert_called_once_with("Unbekannte Information", top_k=2, mode="semantic")
    dotenv.assert_called_once()
    generator.assert_not_called()
    constructor.assert_not_called()
    remote.responses.parse.assert_not_called()


def test_workflow_generate_once_and_preserve_llm_abstention(api):
    """Vorhandener Kontext erzwingt keine Antwort, wenn das LLM verzichtet."""
    http, remote, constructor, _, _ = api
    index_document(http)
    remote.responses.parse.return_value = SimpleNamespace(
        status="completed",
        output_parsed=rag_service.RagAnswer(
            answer="Die Information wurde im Dokument nicht gefunden.", sources=[],
        ),
    )
    response = http.post("/documents/ask", json={"query": "Wie hoch ist der Umsatz?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Die Information wurde im Dokument nicht gefunden.", "sources": []}
    constructor.assert_called_once_with(api_key="placeholder", timeout=30.0, max_retries=0)
    remote.responses.parse.assert_called_once()
    assert remote.responses.parse.call_args.kwargs["store"] is False


def test_no_document_precedes_missing_key(api, monkeypatch):
    """Auch mit fehlendem Schlüssel hat ein fehlender Index weiterhin Vorrang."""
    http, _, constructor, dotenv, _ = api
    monkeypatch.delenv("OPENAI_API_KEY")
    assert http.post("/documents/ask", json={"query": "Frage"}).status_code == 409
    dotenv.assert_not_called()
    constructor.assert_not_called()


def test_endpoint_invokes_existing_workflow(api, monkeypatch):
    """Der Endpunkt gibt nur das Ergebnis aus, keine internen State-Felder."""
    http, _, constructor, _, _ = api
    result = {"answer": "Testantwort", "sources": []}
    invoke = Mock(return_value={"result": result, "abstention_reason": None, "retrieved_chunks": []})
    monkeypatch.setattr(main.document_qa_workflow, "invoke", invoke)
    response = http.post("/documents/ask", json={"query": "Frage"})
    assert response.status_code == 200
    assert response.json() == result
    invoke.assert_called_once_with({"query": "Frage", "top_k": 3, "mode": "semantic"})
    constructor.assert_not_called()


@pytest.mark.parametrize("mode", [None, "hybrid", "hybrid_de"])
def test_workflow_preserves_retrieval_mode(api, monkeypatch, mode):
    """Semantic bleibt Standard, explizite alternative Modi funktionieren weiter."""
    http, remote, _, _, _ = api
    index_document(http)
    if mode is None:
        monkeypatch.delenv("DOCUAGENT_RETRIEVAL_MODE", raising=False)
    else:
        monkeypatch.setenv("DOCUAGENT_RETRIEVAL_MODE", mode)
    search = Mock(wraps=retrieval_pipeline.search_prepared)
    monkeypatch.setattr(retrieval_pipeline, "search_prepared", search)
    response = http.post("/documents/ask", json={"query": "Urlaubstage"})
    assert response.status_code == 200
    search.assert_called_once()
    assert search.call_args.kwargs["mode"] == (mode or "semantic")
    remote.responses.parse.assert_called_once()


def test_public_request_schema_and_ignored_extra_fields_remain_unchanged(api):
    """Interne Suchoptionen sind weiterhin keine öffentlichen Request-Felder."""
    http, _, _, _, _ = api
    index_document(http)
    properties = http.get("/openapi.json").json()["components"]["schemas"]["SearchRequest"]["properties"]
    assert set(properties) == {"query", "top_k"}
    # Pydantic ignorierte zusätzliche Felder schon bisher; sie steuern keine Suche.
    response = http.post("/documents/ask", json={"query": "Frage", "mode": "invalid", "document_ids": ["fremd"]})
    assert response.status_code == 200
    assert set(response.json()) == {"answer", "sources"}
