from fastapi import FastAPI, File, HTTPException, UploadFile

from app.services.pdf_reader import extract_pdf_text


MAX_PDF_SIZE = 10 * 1024 * 1024

# Erstellt die FastAPI-Anwendung mit dem Projekttitel.
app = FastAPI(title="DocuAgent API")


# Prüft, ob der Dienst erreichbar ist.
@app.get("/health")
def health():
    return {"status": "ok", "service": "DocuAgent"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Prüft eine hochgeladene PDF und gibt ihren Text je Seite zurück."""
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

        try:
            result = extract_pdf_text(file_bytes)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {"filename": file.filename, **result}
    finally:
        # Schliesst die Upload-Datei auch bei abgelehnten Anfragen.
        await file.close()
