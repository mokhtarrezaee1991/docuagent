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
from app.services.hybrid_search import retrieval_mode
from app.workflows.document_qa import build_document_qa_workflow


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


class AskRequest(SearchRequest):
    """Erweitert nur Dokumentfragen um eine optionale explizite ID-Auswahl."""

    document_ids: list[StrictStr] | None = Field(default=None, strict=True)

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, value):
        """Leere Auswahl bleibt leer; leere einzelne IDs werden abgewiesen."""
        if value is not None and any(not identifier.strip() for identifier in value):
            raise ValueError("Dokument-IDs dürfen nicht leer sein.")
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


def retrieve_for_ask(
    query: str, *, top_k: int, mode: str, document_ids: list[str] | None,
) -> list[dict]:
    """Löst den aktuellen Index auf und erhält die bisherige Schlüsselprüfung."""
    pipeline = app.state.retrieval_pipeline
    if pipeline is None or (isinstance(pipeline, DocumentCollection) and not pipeline.list_documents()):
        raise HTTPException(status_code=409, detail="Bitte indexieren Sie zuerst eine PDF-Datei.")
    if isinstance(pipeline, DocumentCollection):
        chunks = pipeline.search(query, top_k=top_k, mode=mode, document_ids=document_ids)
    else:
        # Der Legacy-Index liefert weiterhin Quellen ohne zusätzliche Dokumentfelder.
        # Auch im Legacy-Modus eine explizite Auswahl niemals ignorieren.
        if document_ids is not None and any(identifier != pipeline.document_id for identifier in document_ids):
            raise ValueError("Mindestens eine Dokument-ID ist unbekannt oder wurde gelöscht.")
        chunks = [] if document_ids == [] else pipeline.search(query, top_k=top_k, mode=mode)
    # Der feste Pfad funktioniert unabhängig vom Arbeitsverzeichnis.
    # Bereits gesetzte Umgebungsvariablen behalten Vorrang.
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY ist für die Antwortgenerierung nicht konfiguriert.")
    # Auch ohne Treffer bleibt ein fehlender Schlüssel wie bisher HTTP 503.
    # Schlüssel und Konfiguration gelangen niemals in den Workflow-State.
    return chunks


def generate_for_ask(query: str, chunks: list[dict]) -> dict:
    """Übersetzt ausschließlich Fehler der Antwortgenerierung wie bisher."""
    try:
        return generate_answer(query, chunks)
    except RuntimeError:
        # Keine internen Fehlerdetails oder vertraulichen Daten weitergeben.
        raise HTTPException(status_code=502, detail="Die KI-Antwort konnte nicht erstellt werden. Bitte API-Zugang und Verbindung prüfen.") from None


def documents_for_ask() -> list[dict]:
    """Erhält HTTP 409 bei leerem Index vor der Prüfung ausgewählter IDs."""
    documents = list_documents()
    if not documents:
        raise HTTPException(status_code=409, detail="Bitte indexieren Sie zuerst eine PDF-Datei.")
    return documents


# Ein Graph, aber eigener State je Aufruf; der Retriever liest den Index erst dann.
document_qa_workflow = build_document_qa_workflow(
    retrieve_for_ask, generate_for_ask, get_documents=documents_for_ask,
)


@app.post("/documents/ask")
def ask_document(request: AskRequest):
    """Beantwortet Dokumentfragen über den bestehenden LangGraph-Workflow."""
    # Die serverseitige Konfiguration hat Vorrang vor dem Policy-Standard.
    # Nur document_ids ist zusätzlich öffentlich, der Modus bleibt serverseitig.
    state_input = {"query": request.query, "top_k": request.top_k, "mode": retrieval_mode()}
    if request.document_ids is not None:
        # Insbesondere [] nicht in None oder eine globale Suche umwandeln.
        state_input["document_ids"] = request.document_ids
    try:
        state = document_qa_workflow.invoke(state_input)
    except ValueError:
        # Keine internen Metadaten, Pfade oder Exception-Details veröffentlichen.
        raise HTTPException(status_code=422, detail="Die Dokumentauswahl ist ungültig. Bitte aktuelle Dokument-IDs verwenden.") from None
    return state["result"]
