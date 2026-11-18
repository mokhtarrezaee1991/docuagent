"""Verbindet PDF-Extraktion, Chunking und lokale semantische Suche."""

from threading import RLock
from uuid import uuid4

from app.services.pdf_reader import extract_pdf_text
from app.services.text_chunker import chunk_pages_by_tokens
from app.services import semantic_search
from app.services.semantic_search import embed_chunks, search_embedded_chunks


class DocumentRetrievalPipeline:
    """Hält die vorberechneten Chunks eines Dokuments im Arbeitsspeicher."""

    def __init__(self):
        self._embedded_chunks = None
        self.document_id = str(uuid4())
        self.filename = "document.pdf"
        self.metadata = None

    def load_pdf(self, file_bytes: bytes) -> dict:
        """Lädt eine PDF, berechnet Embeddings und liefert Seiten-/Chunkanzahl.

        Ein erfolgreiches Laden ersetzt das vorige Dokument. Bei Fehlern
        bleibt ein zuvor geladenes Dokument erhalten.
        """
        document = extract_pdf_text(file_bytes)
        if not any(page["text"].strip() for page in document["pages"]):
            raise ValueError("Die PDF-Datei enthält keinen extrahierbaren Text.")
        model = semantic_search._get_model()
        chunks = chunk_pages_by_tokens(
            document["pages"], model.tokenizer, model.max_seq_length
        )

        # Erst nach erfolgreicher Vorbereitung wird der Zustand ersetzt.
        embedded_chunks = embed_chunks(chunks)
        self._embedded_chunks = embedded_chunks
        self.metadata = {"page_count": document["page_count"], "chunk_count": len(embedded_chunks)}
        return dict(self.metadata)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Kodiert nur die Frage und sucht in den gespeicherten Chunks."""
        if self._embedded_chunks is None:
            raise ValueError("Bitte laden Sie zuerst eine PDF-Datei mit load_pdf.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Die Suchfrage muss ein nichtleerer Text sein.")
        # Die bestehende Suche prüft top_k und berechnet die Kosinus-Ähnlichkeit.
        return search_embedded_chunks(query, self._embedded_chunks, top_k=top_k)

    def to_record(self) -> dict:
        """Gibt die bereits berechneten Daten für die lokale Speicherung zurück."""
        if self.metadata is None:
            raise ValueError("Das Dokument ist noch nicht vorbereitet.")
        return {
            "metadata": {"document_id": self.document_id, "filename": self.filename, **self.metadata},
            "chunks": self._embedded_chunks,
        }

    @classmethod
    def from_record(cls, record: dict):
        """Stellt ein vorbereitetes Dokument ohne Extraktion oder Modellaufruf wieder her."""
        pipeline = cls()
        metadata = record["metadata"]
        pipeline.document_id = metadata["document_id"]
        pipeline.filename = metadata["filename"]
        pipeline.metadata = {key: metadata[key] for key in ("page_count", "chunk_count")}
        pipeline._embedded_chunks = record["chunks"]
        return pipeline


class DocumentCollection:
    """Verwaltet vorbereitete Dokumente ohne Datenbank oder erneute Kodierung."""

    def __init__(self):
        self._documents = {}
        self._lock = RLock()

    def add_prepared(self, pipeline: DocumentRetrievalPipeline) -> dict:
        """Übernimmt ein erfolgreich vorbereitetes Dokument einschließlich Metadaten."""
        if pipeline.metadata is None:
            raise ValueError("Das Dokument ist noch nicht vorbereitet.")
        metadata = {
            "document_id": pipeline.document_id, "filename": pipeline.filename,
            **pipeline.metadata,
        }
        chunks = [
            {**chunk, "document_id": pipeline.document_id, "filename": pipeline.filename}
            for chunk in pipeline._embedded_chunks
        ]
        with self._lock:
            self._documents[pipeline.document_id] = (metadata, chunks)
        return dict(metadata)

    def list_documents(self) -> list[dict]:
        """Liefert ausschließlich Metadaten, keine Texte oder Embeddings."""
        with self._lock:
            return [dict(metadata) for metadata, _ in self._documents.values()]

    def remove_document(self, document_id: str) -> bool:
        """Entfernt ein Dokument anhand seiner ID, niemals anhand des Dateinamens."""
        with self._lock:
            return self._documents.pop(document_id, None) is not None

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Sucht global mit genau einem neuen Frage-Embedding über alle Dokumente."""
        with self._lock:
            chunks = [chunk for _, document_chunks in self._documents.values() for chunk in document_chunks]
        return search_embedded_chunks(query, chunks, top_k=top_k)
