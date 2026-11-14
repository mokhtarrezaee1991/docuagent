"""Untersucht ausschließlich die künstliche Test-PDF ohne OpenAI-Aufruf."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main as api
from app.services import rag_service, semantic_search
from app.services.pdf_reader import extract_pdf_text
from app.services.text_chunker import chunk_pages_by_tokens
from fastapi.testclient import TestClient


QUERY = "Wie viele Urlaubstage haben Mitarbeiter pro Jahr?"
DEFAULT_PDF = Path(__file__).resolve().parents[2] / "data" / "sample_documents" / "DocuAgent_Testdokument.pdf"


def check_text_preserved(pages: list[dict], chunks: list[dict]) -> bool:
    """Prüft Textabdeckung unabhängig von der Länge der Überlappung."""
    for page in pages:
        if not page["text"].strip():
            continue
        parts = [chunk["text"] for chunk in chunks if chunk["page_number"] == page["page_number"]]
        covered = 0
        previous_start = -1
        for part in parts:
            position = page["text"].find(part, previous_start + 1)
            if position < 0 or position > covered:
                return False
            previous_start = position
            covered = max(covered, position + len(part))
        if covered != len(page["text"]):
            return False
    return True


def diagnose(file_bytes: bytes) -> None:
    """Prüft Extraktion, Rangfolge und den echten Kontextaufbau mit API-Sperre."""
    extracted = extract_pdf_text(file_bytes)
    model = semantic_search._get_model()
    chunks = chunk_pages_by_tokens(extracted["pages"], model.tokenizer, model.max_seq_length)
    print(f"Seiten: {extracted['page_count']}; Chunks: {len(chunks)}")
    print(f"Chunking erhält extrahierten Text vollständig: {check_text_preserved(extracted['pages'], chunks)}")
    for page in extracted["pages"]:
        print(f"\nExtraktion Seite {page['page_number']} ({len(page['text'])} Zeichen):\n{page['text']}")

    print(f"\nModell: {semantic_search.MODEL_NAME}; Tokenlimit: {model.max_seq_length}")
    for chunk in chunks:
        tokens = model.tokenizer(chunk["text"], truncation=False)["input_ids"]
        print(f"\nChunk {chunk['chunk_id']}, Seite {chunk['page_number']}, {len(chunk['text'])} Zeichen, {len(tokens)} Tokens")
        print(f"Embedding wird gekürzt: {len(tokens) > model.max_seq_length}")
        print(chunk["text"])

    # Der echte Service baut den Kontext auf; nur der externe Client ist ersetzt.
    captured = {}
    client = Mock()
    def capture_request(**kwargs):
        captured["payload"] = json.loads(kwargs["input"])
        return SimpleNamespace(
            status="completed", output_parsed=rag_service.RagAnswer(
                answer="Diagnose: keine LLM-Antwort erzeugt.", sources=[]
            ),
        )
    client.responses.parse.side_effect = capture_request
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=client)
    constructor.return_value.__exit__ = Mock(return_value=False)
    real_generate_answer = api.generate_answer
    def capture_chunks(query, retrieved_chunks):
        captured["chunks"] = retrieved_chunks
        return real_generate_answer(query, retrieved_chunks)

    previous = getattr(api.app.state, "retrieval_pipeline", None)
    try:
        with patch.object(rag_service, "OpenAI", constructor), patch.object(api, "load_dotenv"), \
                patch.object(api, "generate_answer", side_effect=capture_chunks), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "diagnostic-placeholder"}):
            with TestClient(api.app) as http:
                indexed = http.post("/documents/index", files={"file": ("test.pdf", file_bytes, "application/pdf")})
                indexed.raise_for_status()
                print(f"\nLokale Indexierung: {indexed.json()}")
                # Alle Chunks zeigen, ob Urlaubspassagen unterhalb von Top-3 liegen.
                ranking = http.post("/documents/search", json={"query": QUERY, "top_k": max(1, len(chunks))})
                ranking.raise_for_status()
                print(f"\nFrage: {QUERY}\nGesamtrangfolge:")
                for rank, hit in enumerate(ranking.json(), start=1):
                    print(f"\nRang {rank}, Chunk {hit['chunk_id']}, Seite {hit['page_number']}, Score {hit['similarity_score']}")
                    print(hit["text"])
                asked = http.post("/documents/ask", json={"query": QUERY, "top_k": 3})
                asked.raise_for_status()
                print("\nTatsächlich an generate_answer übergebene Chunks:")
                print(json.dumps(captured.get("chunks", []), ensure_ascii=False, indent=2))
                print("\nVom echten rag_service aufgebauter OpenAI-Nutzkontext (Anfrage abgefangen):")
                print(json.dumps(captured.get("payload"), ensure_ascii=False, indent=2))
                print("\nKeine OpenAI API aufgerufen; keine inhaltliche LLM-Bewertung erfolgt.")
    finally:
        api.app.state.retrieval_pipeline = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Pfad ausschließlich zur künstlichen DocuAgent_Testdokument.pdf")
    args = parser.parse_args()
    if args.pdf.name != "DocuAgent_Testdokument.pdf":
        parser.error("Bitte ausschließlich die künstliche DocuAgent_Testdokument.pdf verwenden.")
    if not args.pdf.is_file():
        print(f"Test-PDF nicht gefunden. Bitte hier ablegen: {args.pdf}")
        print("Ohne Original-Testdatei ist keine konkrete Fehlerursache nachweisbar.")
        return 1
    # Nur das vorhandene lokale Modell verwenden, keine Downloads anstoßen.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    diagnose(args.pdf.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
