"""Verbindet PDF-Extraktion, Chunking und lokale semantische Suche."""

from threading import RLock
from uuid import uuid4

from app.services.pdf_reader import extract_pdf_text
from app.services.text_chunker import chunk_pages_by_tokens
from app.services import semantic_search
from app.services.semantic_search import embed_chunks
from app.services.hybrid_search import BM25Index, retrieval_mode, search_prepared


class DocumentRetrievalPipeline:
    """Hält die vorberechneten Chunks eines Dokuments im Arbeitsspeicher."""

    def __init__(self):
        self._embedded_chunks = None
        self.document_id = str(uuid4())
        self.filename = "document.pdf"
        self.metadata = None
        self._keyword_index = None

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
        self._keyword_index = None
        self.metadata = {"page_count": document["page_count"], "chunk_count": len(embedded_chunks)}
        return dict(self.metadata)

    def search(self, query: str, top_k: int = 3, *, mode: str | None = None) -> list[dict]:
        """Kodiert nur die Frage und sucht in den gespeicherten Chunks."""
        if self._embedded_chunks is None:
            raise ValueError("Bitte laden Sie zuerst eine PDF-Datei mit load_pdf.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Die Suchfrage muss ein nichtleerer Text sein.")
        # Die bestehende Suche prüft top_k und berechnet die Kosinus-Ähnlichkeit.
        selected = retrieval_mode(mode)
        normalization = "german" if selected == "hybrid_de" else "none"
        if selected != "semantic" and (self._keyword_index is None or self._keyword_index.normalization != normalization):
            self._keyword_index = BM25Index(self._embedded_chunks, normalization=normalization)
        return search_prepared(query, self._embedded_chunks, top_k=top_k, mode=selected, keyword_index=self._keyword_index)

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
        self._keyword_index = None

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
            self._keyword_index = None
        return dict(metadata)

    def list_documents(self) -> list[dict]:
        """Liefert ausschließlich Metadaten, keine Texte oder Embeddings."""
        with self._lock:
            return [dict(metadata) for metadata, _ in self._documents.values()]

    def remove_document(self, document_id: str) -> bool:
        """Entfernt ein Dokument anhand seiner ID, niemals anhand des Dateinamens."""
        with self._lock:
            removed = self._documents.pop(document_id, None) is not None
            if removed:
                self._keyword_index = None
            return removed

    def search(
        self, query: str, top_k: int = 3, *, mode: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        """Sucht mit vorbereiteten Embeddings optional in ausgewählten Dokumenten.

        None durchsucht wie bisher alle Dokumente, [] keines. Doppelte IDs
        zählen einmal; die Indexreihenfolge bleibt erhalten. Unbekannte oder
        gelöschte IDs sowie ungültige Filtertypen führen zu ValueError.
        Auswahl und Prüfung verwenden denselben gesperrten Index-Schnappschuss.
        """
        selected = retrieval_mode(mode)
        with self._lock:
            scope = None
            if document_ids is not None:
                if not isinstance(document_ids, list) or any(
                    not isinstance(identifier, str) or not identifier.strip()
                    for identifier in document_ids
                ):
                    raise ValueError("document_ids muss eine Liste nichtleerer Dokument-IDs sein.")
                scope = set(document_ids)
                if scope.difference(self._documents):
                    raise ValueError("Mindestens eine Dokument-ID ist unbekannt oder wurde gelöscht.")
            chunks = [
                chunk for identifier, (_, document_chunks) in self._documents.items()
                if scope is None or identifier in scope
                for chunk in document_chunks
            ]
            normalization = "german" if selected == "hybrid_de" else "none"
            if scope is None:
                # Der globale Cache bleibt ausschließlich für ungefilterte Suchen.
                if selected != "semantic" and (self._keyword_index is None or self._keyword_index.normalization != normalization):
                    self._keyword_index = BM25Index(chunks, normalization=normalization)
                index = self._keyword_index
            else:
                # Lokaler Index verhindert fremde Treffer und globale BM25-Statistiken.
                index = BM25Index(chunks, normalization=normalization) if selected != "semantic" else None
        return search_prepared(query, chunks, top_k=top_k, mode=selected, keyword_index=index)
