"""Testet die Antwortgenerierung ausschließlich mit gemockter OpenAI API."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import APIConnectionError
import httpx

from app.services import rag_service


@pytest.fixture
def api(monkeypatch):
    # Nur ein bedeutungsloser Platzhalter, kein echter API-Schlüssel.
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    client = Mock()
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=client)
    constructor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(rag_service, "OpenAI", constructor)
    client.responses.parse.return_value = SimpleNamespace(
        status="completed",
        output_parsed=rag_service.RagAnswer(
            answer="Es gibt 30 Urlaubstage (Chunk 1, Seite 2).",
            sources=[rag_service.Source(chunk_id=1, page_number=2)],
        ),
    )
    return constructor, client


def chunks():
    return [{"chunk_id": 1, "page_number": 2, "text": "30 Urlaubstage.", "embedding": [1, 0]}]


def test_answer_and_minimal_context(api):
    constructor, client = api
    result = rag_service.generate_answer("Wie viel Urlaub?", chunks())
    assert result["sources"] == [{"chunk_id": 1, "page_number": 2}]
    assert "30" in result["answer"]
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-4.1-mini"
    assert kwargs["store"] is False
    assert json.loads(kwargs["input"]) == {
        "query": "Wie viel Urlaub?",
        "context": [{"chunk_id": 1, "page_number": 2, "text": "30 Urlaubstage."}],
    }
    assert "untrusted input" in kwargs["instructions"]
    assert constructor.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize("source", [(99, 2), (1, 99)])
def test_unknown_sources_are_rejected(api, source):
    _, client = api
    client.responses.parse.return_value.output_parsed.sources = [
        rag_service.Source(chunk_id=source[0], page_number=source[1])
    ]
    with pytest.raises(RuntimeError, match="nicht übergebene"):
        rag_service.generate_answer("Frage", chunks())


def test_no_sources_are_invented(api):
    _, client = api
    client.responses.parse.return_value.output_parsed = rag_service.RagAnswer(
        answer="Die Information wurde im Dokument nicht gefunden.", sources=[]
    )
    assert rag_service.generate_answer("Frage", chunks())["sources"] == []


@pytest.mark.parametrize("query", ["", " \n", None])
def test_empty_question(api, query):
    constructor, _ = api
    with pytest.raises(ValueError, match="Frage"):
        rag_service.generate_answer(query, chunks())
    constructor.assert_not_called()


def test_missing_key(api, monkeypatch):
    constructor, _ = api
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        rag_service.generate_answer("Frage", chunks())
    constructor.assert_not_called()


def test_empty_context(api):
    constructor, _ = api
    assert rag_service.generate_answer("Frage", [])["sources"] == []
    constructor.assert_not_called()


def test_api_error(api):
    _, client = api
    client.responses.parse.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    with pytest.raises(RuntimeError, match="fehlgeschlagen"):
        rag_service.generate_answer("Frage", chunks())


@pytest.mark.parametrize("status, parsed", [("incomplete", None), ("completed", None)])
def test_incomplete_or_refused_answer(api, status, parsed):
    _, client = api
    client.responses.parse.return_value = SimpleNamespace(status=status, output_parsed=parsed)
    with pytest.raises(RuntimeError, match="keine vollständige"):
        rag_service.generate_answer("Frage", chunks())
