from fastapi import FastAPI

# Erstellt die FastAPI-Anwendung mit dem Projekttitel.
app = FastAPI(title="DocuAgent API")


# Prüft, ob der Dienst erreichbar ist.
@app.get("/health")
def health():
    return {"status": "ok", "service": "DocuAgent"}
