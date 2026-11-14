import pymupdf
import pytest

from app.services.pdf_reader import extract_pdf_text


def test_extract_pdf_text_two_pages():
    # Erzeugt zwei Testseiten im Arbeitsspeicher, ohne eine Datei zu speichern.
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "DocuAgent Testseite Eins")
        page = document.new_page()
        page.insert_text((72, 72), "DocuAgent Testseite Zwei")
        file_bytes = document.tobytes()

    result = extract_pdf_text(file_bytes)

    # Prueft Anzahl, Reihenfolge und Text der extrahierten Seiten.
    assert result["page_count"] == 2
    assert len(result["pages"]) == 2
    assert result["pages"][0]["page_number"] == 1
    assert result["pages"][1]["page_number"] == 2
    assert "DocuAgent Testseite Eins" in result["pages"][0]["text"]
    assert "DocuAgent Testseite Zwei" in result["pages"][1]["text"]


def test_extract_pdf_text_empty_bytes():
    # Leere Eingaben sollen einen ValueError ausloesen.
    with pytest.raises(ValueError):
        extract_pdf_text(b"")


def test_extract_pdf_text_invalid_bytes():
    # Diese kuenstlichen Bytes enthalten keine gueltige PDF-Datei.
    with pytest.raises(ValueError):
        extract_pdf_text(b"Dies ist keine PDF-Datei.")
