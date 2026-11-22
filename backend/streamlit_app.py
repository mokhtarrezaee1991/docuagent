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


def request_backend(method: str, path: str, timeout: httpx.Timeout, **kwargs):
    """Sendet genau eine Anfrage ohne Wiederholungen oder rohe Fehlerdetails."""
    try:
        with httpx.Client(timeout=timeout) as client:
            if method == "POST":
                response = client.post(f"{api_url()}{path}", **kwargs)
            else:
                response = client.request(method, f"{api_url()}{path}", **kwargs)
    except httpx.TimeoutException:
        raise BackendError("Die Anfrage hat zu lange gedauert. Der Server kann noch arbeiten; bitte erneut indexieren, falls der Status unklar ist.") from None
    except httpx.RequestError:
        raise BackendError("FastAPI ist nicht erreichbar. Bitte Backend starten und die konfigurierte Adresse prüfen.") from None

    if response.is_error:
        messages = {
            400: "Die PDF ist beschädigt, ungültig oder enthält keinen lesbaren Text.",
            404: "Das Dokument ist nicht mehr vorhanden. Bitte die Dokumentliste aktualisieren.",
            409: "Kein Dokument indexiert. Bitte die Dokumentliste aktualisieren oder eine PDF hinzufügen.",
            413: "Die Datei überschreitet das 10-MB-Limit.",
            415: "Das Backend akzeptiert ausschließlich PDF-Dateien.",
            422: "Die Anfrage oder Dokumentauswahl ist ungültig. Bitte Frage und Trefferanzahl prüfen und die Dokumentliste aktualisieren.",
            502: "Die KI-Antwort konnte nicht erstellt werden. Bitte API-Zugang und Verbindung im Backend prüfen.",
            503: "Das Backend ist derzeit nicht bereit. Bitte lokale Datenbank und API-Konfiguration im Backend prüfen.",
        }
        raise BackendError(messages.get(response.status_code, "Das Backend meldet einen Fehler. Bitte die lokale Serverkonfiguration prüfen."), response.status_code)
    try:
        return response.json()
    except ValueError:
        raise BackendError("Das Backend hat keine gültige JSON-Antwort geliefert.") from None


def post_backend(path: str, timeout: httpx.Timeout, **kwargs):
    """Behält den bisherigen POST-Helfer kompatibel bei."""
    return request_backend("POST", path, timeout, **kwargs)


def list_documents() -> list[dict]:
    """Lädt die Dokumentliste nur bei Start oder ausdrücklicher Aktualisierung."""
    result = request_backend("GET", "/documents", httpx.Timeout(10.0, connect=5.0))
    if not isinstance(result, list) or any(
        not isinstance(document, dict)
        or not isinstance(document.get("document_id"), str)
        or not isinstance(document.get("filename"), str)
        or any(type(document.get(key)) is not int for key in ("page_count", "chunk_count"))
        for document in result
    ):
        raise BackendError("Das Backend hat eine ungültige Dokumentliste geliefert.")
    return result


def remove_document(document_id: str) -> None:
    """Entfernt genau das Dokument mit der ausgewählten ID."""
    request_backend("DELETE", f"/documents/{document_id}", httpx.Timeout(10.0, connect=5.0))


def index_pdf(filename: str, file_bytes: bytes, *, append: bool = False) -> dict:
    """Überträgt eine PDF nur nach ausdrücklicher Indexierungsaktion."""
    validate_pdf(file_bytes)
    result = post_backend(
        "/documents" if append else "/documents/index", INDEX_TIMEOUT,
        files={"file": (filename, file_bytes, "application/pdf")},
    )
    if not isinstance(result, dict) or any(
        type(result.get(key)) is not int or result[key] <= 0
        for key in ("page_count", "chunk_count")
    ):
        raise BackendError("Das Backend hat ungültige Index-Metadaten geliefert.")
    metadata = {key: result[key] for key in ("page_count", "chunk_count")}
    if append:
        if not isinstance(result.get("document_id"), str) or not isinstance(result.get("filename"), str):
            raise BackendError("Das Backend hat keine gültige Dokument-ID geliefert.")
        metadata.update(document_id=result["document_id"], filename=result["filename"])
    return metadata


def ask_question(query: str, top_k: int = 3, *, document_ids: list[str] | None = None) -> dict:
    """Fordert eine Antwort an; der API-Schlüssel bleibt allein im Backend."""
    if not query.strip():
        raise ValueError("Bitte geben Sie eine Frage ein.")
    payload = {"query": query, "top_k": top_k}
    if document_ids is not None:
        payload["document_ids"] = document_ids
    result = post_backend("/documents/ask", ASK_TIMEOUT, json=payload)
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str) or not isinstance(result.get("sources"), list):
        raise BackendError("Das Backend hat ein ungültiges Antwortformat geliefert.")
    for source in result["sources"]:
        if not isinstance(source, dict) or any(type(source.get(key)) is not int for key in ("chunk_id", "page_number")):
            raise BackendError("Das Backend hat ungültige Quellenangaben geliefert.")
    return {"answer": result["answer"], "sources": result["sources"]}


def reconcile_selection(documents: list[dict], selected_ids: list[str]) -> tuple[list[str], list[str]]:
    """Entfernt veraltete IDs, ohne eine leere Auswahl zur globalen zu machen."""
    available = {document["document_id"] for document in documents}
    return (
        list(dict.fromkeys(identifier for identifier in selected_ids if identifier in available)),
        list(dict.fromkeys(identifier for identifier in selected_ids if identifier not in available)),
    )


def document_label(document: dict) -> str:
    """Die vollständige ID unterscheidet auch gleiche Namen und ID-Präfixe."""
    return f"{document['filename']} · ID: {document['document_id']}"


def show_message(message: dict) -> None:
    """Zeigt Chattext und Quellen ohne HTML-Ausführung an."""
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                with st.expander("Quellen anzeigen"):
                    for source in sources:
                        st.text(f"{source.get('filename', 'Dokument')} · Seite {source['page_number']} · Chunk {source['chunk_id']}")
                        if source.get("document_id"):
                            st.caption(f"Dokument-ID: {source['document_id']}")
            else:
                st.caption("Keine Dokumentquelle angegeben.")


def main() -> None:
    """Verwaltet den UI-Zustand und sendet Anfragen nur bei Nutzeraktionen."""
    st.set_page_config(page_title="DocuAgent", page_icon="📄", layout="wide")
    st.title("DocuAgent")
    st.write("Durchsuchen Sie mehrere PDFs gemeinsam und erhalten Sie Antworten mit Dokumentquellen.")
    st.caption("Lokaler Einzelbenutzer-Prototyp · Mehrere Dokumente · Lokale SQLite-Speicherung")
    st.session_state.setdefault("messages", [])
    if "documents" not in st.session_state:
        st.session_state.documents = []
        try:
            st.session_state.documents = list_documents()
        except BackendError as error:
            st.session_state.document_error = str(error)

    with st.sidebar:
        st.header("Ihre Dokumente")
        uploaded = st.file_uploader("PDFs auswählen", type=["pdf"], accept_multiple_files=True, help="Maximal 10 MB pro Datei. Jede Indexierung fügt ein Dokument mit eigener ID hinzu.")
        if st.button("Dokumente indexieren", type="primary", disabled=not uploaded):
            for file in uploaded:
                if file.size > MAX_PDF_SIZE:
                    st.error(f"{file.name}: Die PDF darf maximal 10 MB groß sein.")
                    continue
                try:
                    with st.spinner("PDF wird gelesen und lokal indexiert …"):
                        metadata = index_pdf(file.name, file.getvalue(), append=True)
                    st.session_state.documents.append(metadata)
                    st.session_state.messages = []
                    st.session_state.pop("document_error", None)
                    st.success(f"Indexiert: {file.name}")
                    st.session_state.documents = list_documents()
                    st.session_state.pop("selection_invalid", None)
                except (ValueError, BackendError) as error:
                    st.error(f"{file.name}: {error}")
        if st.button("Dokumentliste aktualisieren"):
            try:
                refreshed = list_documents()
                if refreshed != st.session_state.documents:
                    st.session_state.messages = []
                st.session_state.documents = refreshed
                st.session_state.pop("document_error", None)
                st.session_state.pop("selection_invalid", None)
            except BackendError as error:
                st.session_state.document_error = str(error)
        if st.session_state.get("document_error"):
            st.error(st.session_state.document_error)
        st.caption(f"{len(st.session_state.documents)} Dokumente indexiert")
        for document in st.session_state.documents:
            with st.container(border=True):
                st.text(document["filename"])
                st.caption(f"ID: {document['document_id']}")
                st.caption(f"{document['page_count']} Seiten · {document['chunk_count']} Chunks")
                if st.button("Entfernen", key=f"remove_{document['document_id']}"):
                    try:
                        remove_document(document["document_id"])
                        st.session_state.documents = [item for item in st.session_state.documents if item["document_id"] != document["document_id"]]
                        st.session_state.messages = []
                        try:
                            st.session_state.documents = list_documents()
                            st.session_state.pop("selection_invalid", None)
                        except BackendError as error:
                            st.session_state.document_error = str(error)
                        st.rerun()
                    except BackendError as error:
                        st.error(str(error))
        st.divider()
        st.caption("Dokumenttexte und Embeddings bleiben lokal in SQLite gespeichert und werden beim Backend-Start geladen. Andere Sitzungen teilen denselben Index.")

    st.subheader("Fragen an Ihre Dokumente")
    # Vor dem Widget-Aufbau abgleichen, damit gelöschte IDs nie unsichtbar bleiben.
    selected, removed = reconcile_selection(
        st.session_state.documents, st.session_state.get("selected_document_ids", []),
    )
    st.session_state.selected_document_ids = selected
    if removed:
        st.warning("Nicht mehr verfügbare Dokumente wurden aus Ihrer Auswahl entfernt. Bitte prüfen Sie die verbleibende Auswahl.")
    scope = st.radio("Suchbereich", ["Alle Dokumente", "Einzelne Dokumente"], key="document_scope")
    labels = {document["document_id"]: document_label(document) for document in st.session_state.documents}
    selected = st.multiselect(
        "Dokumente für die Frage", options=list(labels), key="selected_document_ids",
        format_func=labels.__getitem__, disabled=scope == "Alle Dokumente",
    )
    document_ids = None if scope == "Alle Dokumente" else list(selected)
    if document_ids == []:
        st.info("Keine Dokumente ausgewählt. Eine Frage wird mit leerer Auswahl gesendet; es erfolgt keine Suche in allen Dokumenten.")
    if st.session_state.get("selection_invalid"):
        st.warning("Die Dokumentauswahl konnte nicht verwendet werden. Bitte zuerst die Dokumentliste aktualisieren.")
    st.caption("Das Absenden einer Frage nutzt die kostenpflichtige OpenAI API. Quellenreferenzen garantieren keine inhaltliche Richtigkeit.")
    if not st.session_state.documents:
        st.info("Bitte zuerst mindestens eine PDF auswählen und indexieren.")
    for message in st.session_state.messages:
        show_message(message)

    query = st.chat_input("Was möchten Sie über Ihre Dokumente wissen?", disabled=not st.session_state.documents or st.session_state.get("selection_invalid", False))
    if query:
        user_message = {"role": "user", "content": query}
        st.session_state.messages.append(user_message)
        show_message(user_message)
        try:
            with st.spinner("Passende Abschnitte werden gesucht und die Antwort wird erstellt …"):
                result = ask_question(query, document_ids=document_ids)
            message = {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            st.session_state.messages.append(message)
            show_message(message)
        except (ValueError, BackendError) as error:
            if isinstance(error, BackendError) and error.status_code == 409:
                st.session_state.documents = []
            if isinstance(error, BackendError) and error.status_code == 422:
                # Bis zur ausdrücklichen Aktualisierung keine möglicherweise alten IDs senden.
                st.session_state.selection_invalid = True
            st.error(str(error))


if __name__ == "__main__":
    main()
