from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, StrictStr, field_validator
from starlette.concurrency import run_in_threadpool

from app.services.pdf_reader import extract_pdf_text
from app.services.retrieval_pipeline import DocumentRetrievalPipeline


MAX_PDF_SIZE = 10 * 1024 * 1024

# Erstellt die FastAPI-Anwendung mit dem Projekttitel.
app = FastAPI(title="DocuAgent API")
# Ein aktives Dokument pro Prozess; ein Neustart leert diesen Speicher.
app.state.retrieval_pipeline = None


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
    try:
        # Modellberechnungen blockieren nicht die Ereignisschleife der API.
        metadata = await run_in_threadpool(candidate.load_pdf, file_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    app.state.retrieval_pipeline = candidate
    return metadata


@app.post("/documents/search")
def search_document(request: SearchRequest):
    """Sucht ohne erneute Dokumentkodierung im aktuellen Prozessindex."""
    pipeline = app.state.retrieval_pipeline
    if pipeline is None:
        raise HTTPException(status_code=409, detail="Bitte indexieren Sie zuerst eine PDF-Datei.")
    return pipeline.search(request.query, top_k=request.top_k)
