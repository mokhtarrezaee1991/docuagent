"""Verbindet PDF-Extraktion, Chunking und lokale semantische Suche."""

from app.services.pdf_reader import extract_pdf_text
from app.services.text_chunker import chunk_pages
from app.services.semantic_search import embed_chunks, search_embedded_chunks


class DocumentRetrievalPipeline:
    """Hält die vorberechneten Chunks eines Dokuments im Arbeitsspeicher."""

    def __init__(self):
        self._embedded_chunks = None

    def load_pdf(self, file_bytes: bytes) -> None:
        """Lädt eine PDF und berechnet ihre Embeddings einmalig.

        Ein erfolgreiches Laden ersetzt das vorige Dokument. Bei Fehlern
        bleibt ein zuvor geladenes Dokument erhalten.
        """
        document = extract_pdf_text(file_bytes)
        chunks = chunk_pages(document["pages"])
        if not chunks:
            raise ValueError("Die PDF-Datei enthält keinen extrahierbaren Text.")

        # Erst nach erfolgreicher Vorbereitung wird der Zustand ersetzt.
        embedded_chunks = embed_chunks(chunks)
        self._embedded_chunks = embedded_chunks

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Kodiert nur die Frage und sucht in den gespeicherten Chunks."""
        if self._embedded_chunks is None:
            raise ValueError("Bitte laden Sie zuerst eine PDF-Datei mit load_pdf.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Die Suchfrage muss ein nichtleerer Text sein.")
        # Die bestehende Suche prüft top_k und berechnet die Kosinus-Ähnlichkeit.
        return search_embedded_chunks(query, self._embedded_chunks, top_k=top_k)
