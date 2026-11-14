"""Zeigt lokales Retrieval; nur --live erlaubt einen kostenpflichtigen API-Aufruf."""

import argparse
import os
from pathlib import Path
import sys

import pymupdf


# Ermöglicht den direkten Start unabhängig vom Arbeitsverzeichnis.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.retrieval_pipeline import DocumentRetrievalPipeline
from app.services.rag_service import generate_answer


def main() -> int:
    """Erzeugt künstliche Daten und führt standardmäßig nur Retrieval aus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Erlaubt genau einen OpenAI-API-Aufruf ohne automatische Wiederholungen.",
    )
    args = parser.parse_args()

    # Ein Live-Lauf ohne Schlüssel wird vor jeder weiteren Verarbeitung beendet.
    if args.live and not os.environ.get("OPENAI_API_KEY", "").strip():
        print("Live-Lauf abgebrochen: OPENAI_API_KEY ist nicht in der Umgebung gesetzt.")
        return 1

    with pymupdf.open() as document:
        for text in (
            "Mitarbeiter haben Anspruch auf 30 Urlaubstage pro Jahr.",
            "Rechnungen müssen innerhalb von 14 Tagen bezahlt werden.",
            "Für die IT-Sicherheit müssen Mitarbeiter starke Passwörter verwenden.",
        ):
            page = document.new_page()
            page.insert_text((72, 72), text)
        file_bytes = document.tobytes()

    query = "Wie viele Urlaubstage haben Mitarbeiter pro Jahr?"
    print("Modus: LIVE" if args.live else "Modus: DRY-RUN – kein OpenAI-API-Aufruf", flush=True)
    print(f"Frage: {query}", flush=True)
    pipeline = DocumentRetrievalPipeline()
    metadata = pipeline.load_pdf(file_bytes)
    print(f"Künstliche PDF: {metadata['page_count']} Seiten, {metadata['chunk_count']} Chunks")
    hits = pipeline.search(query, top_k=2)
    print("\nGefundene Chunks und vorgesehener Dokumentkontext:")
    for rank, hit in enumerate(hits, start=1):
        print(f"\nTreffer {rank}: chunk_id={hit['chunk_id']}, page_number={hit['page_number']}")
        print(f"similarity_score: {hit['similarity_score']}")
        print(f"text: {hit['text'].strip()}")

    if not args.live:
        print("\nDry-Run beendet. Es wurde keine Antwort generiert und keine OpenAI API aufgerufen.")
        print("Ein kostenpflichtiger Test erfordert einen erneuten Start mit --live.")
        return 0

    # Genau ein Aufruf; generate_answer deaktiviert automatische API-Wiederholungen.
    result = generate_answer(query, hits)
    print(f"\nGenerierte Antwort: {result['answer']}")
    print(f"Quellen: {result['sources']}")
    print("Quellenreferenzen wurden geprüft; die Antwort ist nicht inhaltlich verifiziert.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        # Die Services liefern verständliche Meldungen ohne API-Schlüssel.
        print(f"Demo abgebrochen: {error}", file=sys.stderr)
        raise SystemExit(1)
