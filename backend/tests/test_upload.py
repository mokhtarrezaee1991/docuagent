import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def pdf_bytes():
    """Erzeugt eine PDF mit zwei künstlichen Testseiten im Arbeitsspeicher."""
    with pymupdf.open() as document:
        for text in ("DocuAgent Testseite Eins", "DocuAgent Testseite Zwei"):
            page = document.new_page()
            page.insert_text((72, 72), text)
        return document.tobytes()


def test_upload_success(pdf_bytes):
    # Auch ohne PDF-Endung wird der tatsächliche PDF-Inhalt akzeptiert.
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("test.bin", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "test.bin",
        "page_count": 2,
        "pages": [
            {"page_number": 1, "text": "DocuAgent Testseite Eins\n"},
            {"page_number": 2, "text": "DocuAgent Testseite Zwei\n"},
        ],
    }


def test_upload_wrong_content_type(pdf_bytes):
    # Ein falscher Content-Type wird trotz gültiger PDF abgelehnt.
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", pdf_bytes, "text/plain")},
        )
    assert response.status_code == 415


@pytest.mark.parametrize("content", [b"keine PDF", b"%PDF-1.7\nkaputte PDF", b""])
def test_upload_invalid_pdf(content):
    # Dateiendung und Content-Type ersetzen keine Prüfung der PDF-Struktur.
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", content, "application/pdf")},
        )
    assert response.status_code == 400


def test_upload_too_large(pdf_bytes):
    # Vergrössert die künstliche PDF auf genau ein Byte über dem Limit.
    oversized_pdf = pdf_bytes + b" " * (10 * 1024 * 1024 + 1 - len(pdf_bytes))
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", oversized_pdf, "application/pdf")},
        )
    assert response.status_code == 413
