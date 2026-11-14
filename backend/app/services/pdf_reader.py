"""Liest den vorhandenen Text aus PDF-Dateien."""

import pymupdf


def extract_pdf_text(file_bytes: bytes) -> dict:
    """Extrahiert Text je Seite und meldet unlesbare PDFs als ValueError.

    Gescannte Seiten ohne Textschicht liefern einen leeren Text; OCR findet
    hier nicht statt.
    """
    if not file_bytes:
        raise ValueError("Die PDF-Datei ist leer.")

    try:
        # Der Kontextmanager schliesst die PDF auch bei einem Fehler.
        with pymupdf.open(stream=file_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Die PDF-Datei ist passwortgeschützt.")

            pages = []
            # enumerate beginnt hier bei 1, damit die Seitenzahlen lesbar sind.
            for page_number, page in enumerate(document, start=1):
                pages.append({"page_number": page_number, "text": page.get_text()})

            return {"page_count": len(pages), "pages": pages}
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise ValueError(
            "Die PDF-Datei ist ungültig oder beschädigt und konnte nicht gelesen werden."
        ) from error
