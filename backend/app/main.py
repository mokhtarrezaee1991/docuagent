import os
from pathlib import Path
from threading import RLock
from contextlib import asynccontextmanager
import sqlite3

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, StrictStr, field_validator
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.services.pdf_reader import extract_pdf_text
from app.services.retrieval_pipeline import DocumentRetrievalPipeline, DocumentCollection
from app.services.rag_service import generate_answer
from app.services.document_store import SQLiteDocumentStore


MAX_PDF_SIZE = 10 * 1024 * 1024


def database_path() -> Path:
    """Verwendet einen stabilen lokalen Pfad; Tests können ihn isoliert ersetzen."""
    configured = os.environ.get("DOCUAGENT_DB_PATH")
    project_root = Path(__file__).resolve().parents[2]
    path = Path(configured) if configured else Path("data/storage/docuagent.sqlite3")
    return path if path.is_absolute() else project_root / path


@asynccontextmanager
async def lifespan(application):
    """Lädt den persistenten Index beim Start ohne Neuberechnung der Embeddings."""
    store = SQLiteDocumentStore(database_path())
    records, single = store.load_documents()
    restored = None
    if single and len(records) == 1:
        restored = DocumentRetrievalPipeline.from_record(records[0])
    elif records:
        restored = DocumentCollection()
        for record in records:
            restored.add_prepared(DocumentRetrievalPipeline.from_record(record))
    application.state.document_store = store
    application.state.retrieval_pipeline = restored
    try:
        yield
    finally:
        application.state.document_store = None


# Erstellt die FastAPI-Anwendung mit dem Projekttitel.
app = FastAPI(title="DocuAgent API", lifespan=lifespan)
# Der Suchindex wird beim Start aus SQLite wiederhergestellt.
app.state.retrieval_pipeline = None
documents_lock = RLock()


@app.exception_handler(sqlite3.Error)
async def storage_error_handler(request, error):
    """Gibt keine SQL-Anweisungen, Dokumenttexte oder internen Pfade aus."""
    return JSONResponse(status_code=503, content={"detail": "Die lokale Dokumentdatenbank ist derzeit nicht verfügbar. Bitte Speicherplatz und Zugriffsrechte prüfen."})


class SearchRequest(BaseModel):
    """Validiert die Frage und eine strikt ganzzahlige Trefferanzahl."""

    query: StrictStr
    top_k: int = Field(default=3, strict=True, gt=0)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value):
        """Lehnt leere Fragen und reinen Leerraum ab."""
        if not value.strip():
            raise ValueError("Die Suchfrage darf nicht leer sein.")
        return value


# Prüft, ob der Dienst erreichbar ist.
@app.get("/health")
def health():
    return {"status": "ok", "service": "DocuAgent"}


async def read_pdf_upload(file: UploadFile) -> bytes:
    """Prüft Typ, Größe und Signatur und schließt die Upload-Datei immer.

    Die vollständige Strukturprüfung übernimmt anschließend der PDF-Reader.
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=415, detail="Nur PDF-Dateien sind erlaubt.")

        # Ein zusätzliches Byte macht eine Überschreitung des Limits erkennbar.
        file_bytes = await file.read(MAX_PDF_SIZE + 1)
        if len(file_bytes) > MAX_PDF_SIZE:
            raise HTTPException(status_code=413, detail="Die Datei darf maximal 10 MB groß sein.")

        # Die Signatur allein reicht nicht: Danach prüft PyMuPDF den Inhalt.
        if not file_bytes.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Die Datei ist keine gültige PDF-Datei.")

        return file_bytes
    finally:
        # Schliesst die Upload-Datei auch bei abgelehnten Anfragen.
        await file.close()


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Prüft eine hochgeladene PDF und gibt ihren Text je Seite zurück."""
    file_bytes = await read_pdf_upload(file)
    try:
        result = extract_pdf_text(file_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"filename": file.filename, **result}


@app.post("/documents/index")
async def index_document(file: UploadFile = File(...)):
    """Bereitet eine PDF vor und ersetzt erst bei Erfolg den aktiven Index."""
    file_bytes = await read_pdf_upload(file)
    candidate = DocumentRetrievalPipeline()
    candidate.filename = file.filename or "document.pdf"
    try:
        # Modellberechnungen blockieren nicht die Ereignisschleife der API.
        metadata = await run_in_threadpool(candidate.load_pdf, file_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with documents_lock:
        app.state.document_store.save_document(candidate.to_record(), replace_all=True)
        app.state.retrieval_pipeline = candidate
    return metadata


@app.post("/documents")
async def add_document(file: UploadFile = File(...)):
    """Fügt eine PDF hinzu, ohne erfolgreich indexierte Dokumente zu ersetzen."""
    file_bytes = await read_pdf_upload(file)
    candidate = DocumentRetrievalPipeline()
    candidate.filename = file.filename or "document.pdf"
    try:
        await run_in_threadpool(candidate.load_pdf, file_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with documents_lock:
        app.state.document_store.save_document(candidate.to_record())
        current = app.state.retrieval_pipeline
        if not isinstance(current, DocumentCollection):
            collection = DocumentCollection()
            if current is not None:
                collection.add_prepared(current)
            app.state.retrieval_pipeline = collection
        return app.state.retrieval_pipeline.add_prepared(candidate)


@app.get("/documents")
def list_documents():
    """Listet Metadaten aller im Prozess gespeicherten Dokumente auf."""
    with documents_lock:
        current = app.state.retrieval_pipeline
        if current is None:
            return []
        if isinstance(current, DocumentCollection):
            return current.list_documents()
        return [{"document_id": current.document_id, "filename": current.filename, **current.metadata}]


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):
    """Entfernt ein Dokument und dessen Embeddings aus dem Prozessspeicher."""
    with documents_lock:
        current = app.state.retrieval_pipeline
        if not any(document["document_id"] == document_id for document in list_documents()):
            raise HTTPException(status_code=404, detail="Das Dokument wurde nicht gefunden.")
        app.state.document_store.delete_document(document_id)
        if isinstance(current, DocumentCollection):
            removed = current.remove_document(document_id)
        else:
            removed = current is not None and current.document_id == document_id
            if removed:
                app.state.retrieval_pipeline = None
        if not removed:
            raise HTTPException(status_code=404, detail="Das Dokument wurde nicht gefunden.")
    return {"document_id": document_id, "deleted": True}


@app.post("/documents/search")
def search_document(request: SearchRequest):
    """Sucht ohne erneute Dokumentkodierung im aktuellen Prozessindex."""
    pipeline = app.state.retrieval_pipeline
    if pipeline is None or (isinstance(pipeline, DocumentCollection) and not pipeline.list_documents()):
        raise HTTPException(status_code=409, detail="Bitte indexieren Sie zuerst eine PDF-Datei.")
    return pipeline.search(request.query, top_k=request.top_k)


@app.post("/documents/ask")
def ask_document(request: SearchRequest):
    """Beantwortet eine Frage mit dem aktiven Dokument als RAG-Kontext."""
    # Nutzt dieselbe Prüfung und Suche wie der bestehende Suchendpunkt.
    chunks = search_document(request)
    # Der feste Pfad funktioniert unabhängig vom Arbeitsverzeichnis.
    # Bereits gesetzte Umgebungsvariablen behalten Vorrang.
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY ist für die Antwortgenerierung nicht konfiguriert.")
    try:
        return generate_answer(request.query, chunks)
    except RuntimeError:
        # Keine internen Fehlerdetails oder vertraulichen Daten weitergeben.
        raise HTTPException(status_code=502, detail="Die KI-Antwort konnte nicht erstellt werden. Bitte API-Zugang und Verbindung prüfen.") from None
