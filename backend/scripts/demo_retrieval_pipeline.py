"""Testet die lokale Retrieval-Pipeline mit einer künstlichen PDF."""

import sys
from pathlib import Path

import pymupdf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.retrieval_pipeline import DocumentRetrievalPipeline


def main():
    """Erzeugt drei Seiten und zeigt die unveränderten Suchergebnisse."""
    with pymupdf.open() as document:
        for text in (
            "Mitarbeiter haben Anspruch auf 30 Urlaubstage pro Jahr.",
            "Rechnungen müssen innerhalb von 14 Tagen bezahlt werden.",
            "Das Unternehmen verwendet Python zur Automatisierung von Geschäftsprozessen.",
        ):
            page = document.new_page()
            page.insert_text((72, 72), text)
        file_bytes = document.tobytes()

    pipeline = DocumentRetrievalPipeline()
    print("Lade künstliche PDF und berechne Dokument-Embeddings einmalig ...", flush=True)
    pipeline.load_pdf(file_bytes)
    for query, expected_page in (
        ("Wie viele Urlaubstage stehen Mitarbeitern jährlich zu?", 1),
        ("Wie lange habe ich Zeit, eine Rechnung zu bezahlen?", 2),
    ):
        print(f"\nFrage: {query}", flush=True)
        results = pipeline.search(query)
        for rank, hit in enumerate(results, start=1):
            print(f"Treffer {rank}: chunk_id={hit['chunk_id']}, page_number={hit['page_number']}")
            print(f"  text: {hit['text'].strip()}")
            print(f"  similarity_score: {hit['similarity_score']}")
        print(f"Erwartete Seite: {expected_page}; tatsächliche erste Seite: {results[0]['page_number']}")
        print("Passender erster Treffer." if results[0]["page_number"] == expected_page else "Erster Treffer passt nicht zur erwarteten Seite.")


if __name__ == "__main__":
    main()
