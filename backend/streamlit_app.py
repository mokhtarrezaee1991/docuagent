"""Lokale Weboberfläche für PDF-Indexierung und dokumentbasierte Antworten."""

import os

import httpx
import streamlit as st


MAX_PDF_SIZE = 10 * 1024 * 1024
INDEX_TIMEOUT = httpx.Timeout(300.0, connect=5.0)
ASK_TIMEOUT = httpx.Timeout(90.0, connect=5.0)


class BackendError(Exception):
    """Enthält eine sichere Fehlermeldung und optional den HTTP-Status."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def api_url() -> str:
    """Liest nur die Backend-Adresse, niemals Schlüssel oder eine .env-Datei."""
    return os.environ.get("DOCUAGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def validate_pdf(file_bytes: bytes) -> None:
    """Prüft Größe und Signatur; das Backend prüft die vollständige Struktur."""
    if len(file_bytes) > MAX_PDF_SIZE:
        raise ValueError("Die PDF darf maximal 10 MB groß sein.")
    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("Bitte wählen Sie eine gültige PDF-Datei aus.")


def post_backend(path: str, timeout: httpx.Timeout, **kwargs):
    """Sendet genau eine Anfrage ohne Wiederholungen oder rohe Fehlerdetails."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{api_url()}{path}", **kwargs)
    except httpx.TimeoutException:
        raise BackendError("Die Anfrage hat zu lange gedauert. Der Server kann noch arbeiten; bitte erneut indexieren, falls der Status unklar ist.") from None
    except httpx.RequestError:
        raise BackendError("FastAPI ist nicht erreichbar. Bitte Backend starten und die konfigurierte Adresse prüfen.") from None

    if response.is_error:
        messages = {
            400: "Die PDF ist beschädigt, ungültig oder enthält keinen lesbaren Text.",
            409: "Kein Dokument indexiert. Bitte die PDF erneut indexieren, etwa nach einem Backend-Neustart.",
            413: "Die Datei überschreitet das 10-MB-Limit.",
            415: "Das Backend akzeptiert ausschließlich PDF-Dateien.",
            422: "Die Anfrage ist ungültig. Bitte Frage und Trefferanzahl prüfen.",
            502: "Die KI-Antwort konnte nicht erstellt werden. Bitte API-Zugang und Verbindung im Backend prüfen.",
            503: "Die Antwortgenerierung ist nicht konfiguriert. Bitte OPENAI_API_KEY ausschließlich im Backend setzen.",
        }
        raise BackendError(messages.get(response.status_code, "Das Backend meldet einen Fehler. Bitte die lokale Serverkonfiguration prüfen."), response.status_code)
    try:
        return response.json()
    except ValueError:
        raise BackendError("Das Backend hat keine gültige JSON-Antwort geliefert.") from None


def index_pdf(filename: str, file_bytes: bytes) -> dict:
    """Überträgt eine PDF nur nach ausdrücklicher Indexierungsaktion."""
    validate_pdf(file_bytes)
    result = post_backend(
        "/documents/index", INDEX_TIMEOUT,
        files={"file": (filename, file_bytes, "application/pdf")},
    )
    if not isinstance(result, dict) or any(
        type(result.get(key)) is not int or result[key] <= 0
        for key in ("page_count", "chunk_count")
    ):
        raise BackendError("Das Backend hat ungültige Index-Metadaten geliefert.")
    return {key: result[key] for key in ("page_count", "chunk_count")}


def ask_question(query: str, top_k: int = 3) -> dict:
    """Fordert eine Antwort an; der API-Schlüssel bleibt allein im Backend."""
    if not query.strip():
        raise ValueError("Bitte geben Sie eine Frage ein.")
    result = post_backend("/documents/ask", ASK_TIMEOUT, json={"query": query, "top_k": top_k})
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str) or not isinstance(result.get("sources"), list):
        raise BackendError("Das Backend hat ein ungültiges Antwortformat geliefert.")
    for source in result["sources"]:
        if not isinstance(source, dict) or any(type(source.get(key)) is not int for key in ("chunk_id", "page_number")):
            raise BackendError("Das Backend hat ungültige Quellenangaben geliefert.")
    return {"answer": result["answer"], "sources": result["sources"]}


def show_message(message: dict) -> None:
    """Zeigt Chattext und Quellen ohne HTML-Ausführung an."""
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                with st.expander("Quellen anzeigen"):
                    for source in sources:
                        st.write(f"Seite {source['page_number']} · Chunk {source['chunk_id']}")
            else:
                st.caption("Keine Dokumentquelle angegeben.")


def main() -> None:
    """Verwaltet den UI-Zustand und sendet Anfragen nur bei Nutzeraktionen."""
    st.set_page_config(page_title="DocuAgent", page_icon="📄", layout="wide")
    st.title("DocuAgent")
    st.write("Laden Sie ein PDF hoch und erhalten Sie Antworten mit Quellen aus Ihrem Dokument.")
    st.caption("Lokaler Einzelbenutzer-Prototyp · Ein aktives Dokument · Keine dauerhafte Speicherung")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("document", None)

    with st.sidebar:
        st.header("Ihr Dokument")
        uploaded = st.file_uploader("PDF auswählen", type=["pdf"], help="Maximal 10 MB. Eine neue Auswahl wird erst nach dem Indexieren aktiv.")
        too_large = uploaded is not None and uploaded.size > MAX_PDF_SIZE
        if too_large:
            st.error("Die PDF darf maximal 10 MB groß sein.")
        if st.button("Dokument indexieren", type="primary", disabled=uploaded is None or too_large):
            try:
                with st.spinner("PDF wird gelesen und lokal indexiert …"):
                    metadata = index_pdf(uploaded.name, uploaded.getvalue())
                st.session_state.document = {"filename": uploaded.name, **metadata}
                st.session_state.messages = []
            except (ValueError, BackendError) as error:
                # Bei unklarem Serverzustand keine Fragen an ein falsches Dokument senden.
                if isinstance(error, BackendError) and error.status_code not in (400, 413, 415, 422):
                    st.session_state.document = None
                st.error(str(error))

        document = st.session_state.document
        if document:
            st.success("Dokument indexiert")
            st.text(document["filename"])
            left, right = st.columns(2)
            left.metric("Seiten", document["page_count"])
            right.metric("Chunks", document["chunk_count"])
        st.divider()
        st.caption("Der Index liegt im FastAPI-Arbeitsspeicher und geht bei einem Neustart verloren. Andere Sitzungen teilen dasselbe Dokument.")

    st.subheader("Fragen an Ihr Dokument")
    st.caption("Das Absenden einer Frage nutzt die kostenpflichtige OpenAI API. Quellenreferenzen garantieren keine inhaltliche Richtigkeit.")
    if not st.session_state.document:
        st.info("Bitte zuerst eine PDF auswählen und indexieren.")
    for message in st.session_state.messages:
        show_message(message)

    query = st.chat_input("Was möchten Sie über das Dokument wissen?", disabled=not st.session_state.document)
    if query:
        user_message = {"role": "user", "content": query}
        st.session_state.messages.append(user_message)
        show_message(user_message)
        try:
            with st.spinner("Passende Abschnitte werden gesucht und die Antwort wird erstellt …"):
                result = ask_question(query)
            message = {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            st.session_state.messages.append(message)
            show_message(message)
        except (ValueError, BackendError) as error:
            if isinstance(error, BackendError) and error.status_code == 409:
                st.session_state.document = None
            st.error(str(error))


if __name__ == "__main__":
    main()
