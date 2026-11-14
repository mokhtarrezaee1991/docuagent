"""Prüft UI-Hilfsfunktionen ohne Server und ohne OpenAI-Aufrufe."""

from unittest.mock import Mock

import httpx
import pytest

import streamlit_app as ui


def mock_backend(monkeypatch, handler):
    """Lenkt sämtliche HTTP-Anfragen in einen lokalen Mock-Transport."""
    client_class = httpx.Client
    monkeypatch.setattr(ui.httpx, "Client", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handler), **kwargs
    ))


def test_index_pdf_and_api_url(monkeypatch):
    monkeypatch.setenv("DOCUAGENT_API_URL", "http://backend.local/")
    def handler(request):
        assert str(request.url) == "http://backend.local/documents/index"
        assert b"%PDF-test" in request.content
        assert request.extensions["timeout"]["read"] == 300
        return httpx.Response(200, json={"page_count": 2, "chunk_count": 4})
    mock_backend(monkeypatch, handler)
    assert ui.index_pdf("test.pdf", b"%PDF-test") == {"page_count": 2, "chunk_count": 4}


def test_ask_question(monkeypatch):
    def handler(request):
        assert request.url.path == "/documents/ask"
        assert request.extensions["timeout"]["read"] == 90
        return httpx.Response(200, json={"answer": "Testantwort", "sources": [{"chunk_id": 1, "page_number": 2}]})
    mock_backend(monkeypatch, handler)
    assert ui.ask_question("Frage")["sources"] == [{"chunk_id": 1, "page_number": 2}]


@pytest.mark.parametrize("content", [b"", b"kein PDF", b"%PDF-" + b" " * ui.MAX_PDF_SIZE])
def test_invalid_pdf_does_not_send_request(monkeypatch, content):
    post = Mock()
    monkeypatch.setattr(ui, "post_backend", post)
    with pytest.raises(ValueError):
        ui.index_pdf("test.pdf", content)
    post.assert_not_called()


def test_exact_size_limit():
    ui.validate_pdf(b"%PDF-" + b" " * (ui.MAX_PDF_SIZE - 5))


@pytest.mark.parametrize("status", [400, 409, 413, 415, 422, 502, 503, 500])
def test_backend_errors_do_not_expose_response_details(monkeypatch, status):
    mock_backend(monkeypatch, lambda request: httpx.Response(status, json={"detail": "Interne vertrauliche Diagnose"}))
    with pytest.raises(ui.BackendError) as error:
        ui.ask_question("Frage")
    assert error.value.status_code == status
    assert "vertrauliche" not in str(error.value)


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_connection_errors(monkeypatch, error_type):
    def handler(request):
        raise error_type("Interne Diagnose", request=request)
    mock_backend(monkeypatch, handler)
    with pytest.raises(ui.BackendError):
        ui.ask_question("Frage")


def test_invalid_backend_response(monkeypatch):
    mock_backend(monkeypatch, lambda request: httpx.Response(200, text="kein JSON"))
    with pytest.raises(ui.BackendError, match="JSON"):
        ui.ask_question("Frage")


def test_ui_reruns_do_not_send_requests(monkeypatch):
    """Auch wiederholtes Rendern startet keine Backend-Anfrage."""
    from streamlit.testing.v1 import AppTest
    post = Mock(side_effect=AssertionError("Keine Anfrage beim Rendern erwartet"))
    monkeypatch.setattr(httpx.Client, "post", post)
    app = AppTest.from_file(ui.__file__).run()
    assert not app.exception
    assert app.title[0].value == "DocuAgent"
    assert app.chat_input[0].disabled
    app.run()
    assert not app.exception
    post.assert_not_called()
