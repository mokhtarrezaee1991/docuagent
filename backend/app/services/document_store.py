"""Speichert vorbereitete Dokumente lokal und transaktional in SQLite."""

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3


class SQLiteDocumentStore:
    """Speichert Texte und Zahlenvektoren, keine Original-PDFs oder API-Schlüssel."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    PRIMARY KEY (document_id, chunk_id)
                );
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def _connection(self):
        """Öffnet pro Vorgang eine Verbindung mit Rollback und sicherem Schließen."""
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection

    def save_document(self, record: dict, *, replace_all: bool = False) -> None:
        """Speichert ein Dokument atomar; replace_all erhält die alte Index-Semantik."""
        metadata = record["metadata"]
        # Vor der Transaktion serialisieren, damit ungültige Zahlen früh scheitern.
        rows = [(
            metadata["document_id"], chunk["chunk_id"], chunk["page_number"],
            chunk["text"], json.dumps(chunk["embedding"], allow_nan=False),
        ) for chunk in record["chunks"]]
        with self._connection() as connection:
            if replace_all:
                connection.execute("DELETE FROM documents")
            connection.execute(
                "INSERT INTO documents(document_id, filename, page_count) VALUES (?, ?, ?)",
                (metadata["document_id"], metadata["filename"], metadata["page_count"]),
            )
            connection.executemany(
                "INSERT INTO chunks(document_id, chunk_id, page_number, text, embedding) VALUES (?, ?, ?, ?, ?)", rows,
            )
            connection.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES ('mode', ?)",
                ("single" if replace_all else "multi",),
            )

    def load_documents(self) -> tuple[list[dict], bool]:
        """Lädt vorhandene Vektoren ohne Modellaufruf und bewahrt die Reihenfolge."""
        records = []
        with self._connection() as connection:
            for document_id, filename, page_count in connection.execute(
                "SELECT document_id, filename, page_count FROM documents ORDER BY rowid"
            ).fetchall():
                chunks = [
                    {"chunk_id": chunk_id, "page_number": page, "text": text, "embedding": json.loads(embedding)}
                    for chunk_id, page, text, embedding in connection.execute(
                        "SELECT chunk_id, page_number, text, embedding FROM chunks WHERE document_id = ? ORDER BY chunk_id", (document_id,)
                    )
                ]
                records.append({
                    "metadata": {"document_id": document_id, "filename": filename, "page_count": page_count, "chunk_count": len(chunks)},
                    "chunks": chunks,
                })
            mode = connection.execute("SELECT value FROM settings WHERE key = 'mode'").fetchone()
        return records, bool(mode and mode[0] == "single")

    def delete_document(self, document_id: str) -> bool:
        """Löscht das Dokument einschließlich seiner Chunks in einer Transaktion."""
        with self._connection() as connection:
            return connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,)).rowcount > 0
