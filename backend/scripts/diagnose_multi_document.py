"""Diagnostiziert die beiden künstlichen Test-PDFs ohne kostenpflichtigen Aufruf."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main as api
from app.services import rag_service, semantic_search
from app.services.pdf_reader import extract_pdf_text
from app.services.text_chunker import chunk_pages_by_tokens
from fastapi.testclient import TestClient
from scripts.diagnose_rag import check_text_preserved


QUERY = "Wie lang muss ein Passwort sein?"
SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "sample_documents"
FILES = [SAMPLE_DIRECTORY / name for name in (
    "DocuAgent_Testdokument.pdf", "DocuAgent_IT_Sicherheitsrichtlinie.pdf"
)]


def diagnose(paths: list[Path], query: str = QUERY) -> dict:
    """Vergleicht extrahierte Seiten, Chunks, globale Rangfolge und LLM-Kontext."""
    model = semantic_search._get_model()
    documents = []
    contents = []
    for path in paths:
        content = path.read_bytes()
        contents.append(content)
        extracted = extract_pdf_text(content)
        chunks = chunk_pages_by_tokens(extracted["pages"], model.tokenizer, model.max_seq_length)
        documents.append({
            "filename": path.name, "pages": extracted["pages"],
            "text_preserved": check_text_preserved(extracted["pages"], chunks),
            "chunks": [{**chunk, "token_count": len(model.tokenizer(chunk["text"], truncation=False)["input_ids"])} for chunk in chunks],
        })

    captured = {}
    client = Mock()
    def capture_request(**kwargs):
        captured["llm_payload"] = json.loads(kwargs["input"])
        return SimpleNamespace(status="completed", output_parsed=rag_service.MultiDocumentAnswer(
            answer="Diagnose ohne LLM-Antwort.", sources=[]
        ))
    client.responses.parse.side_effect = capture_request
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=client)
    constructor.return_value.__exit__ = Mock(return_value=False)
    real_generate = api.generate_answer
    def capture_generate(question, chunks):
        captured["generate_answer_chunks"] = chunks
        return real_generate(question, chunks)

    previous = getattr(api.app.state, "retrieval_pipeline", None)
    api.app.state.retrieval_pipeline = None
    try:
        with TemporaryDirectory(prefix="docuagent-diagnose-") as directory, \
                patch.object(rag_service, "OpenAI", constructor), patch.object(api, "load_dotenv"), \
                patch.object(api, "generate_answer", side_effect=capture_generate), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "diagnostic-placeholder", "DOCUAGENT_DB_PATH": str(Path(directory) / "diagnose.sqlite3")}):
            with TestClient(api.app) as http:
                for path, content in zip(paths, contents):
                    response = http.post("/documents", files={"file": (path.name, content, "application/pdf")})
                    response.raise_for_status()
                count = sum(len(document["chunks"]) for document in documents)
                ranked = http.post("/documents/search", json={"query": query, "top_k": count})
                ranked.raise_for_status()
                asked = http.post("/documents/ask", json={"query": query, "top_k": 3})
                asked.raise_for_status()
                ranking = [{"rank": rank, **chunk} for rank, chunk in enumerate(ranked.json(), start=1)]
    finally:
        api.app.state.retrieval_pipeline = previous
    return {"query": query, "documents": documents, "ranking": ranking, **captured, "real_openai_calls": 0}


def main() -> int:
    for path in FILES:
        if not path.is_file():
            print(f"Künstliche Testdatei fehlt: {path}")
            return 1
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(json.dumps(diagnose(FILES), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
