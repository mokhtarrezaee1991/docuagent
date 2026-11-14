"""Demonstriert die echte lokale Suche mit ausschließlich künstlichen Texten."""

import sys
from pathlib import Path


# Ermöglicht den direkten Start des Skripts aus beliebigen Arbeitsverzeichnissen.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.semantic_search import embed_chunks, search_embedded_chunks


def main():
    """Vergleicht fünf echte Suchläufe mit den erwarteten ersten Treffern."""
    chunks = [
        {
            "chunk_id": 1,
            "page_number": 1,
            "text": "Mitarbeiter haben Anspruch auf 30 Urlaubstage pro Jahr.",
        },
        {
            "chunk_id": 2,
            "page_number": 2,
            "text": "Rechnungen müssen innerhalb von 14 Tagen bezahlt werden.",
        },
        {
            "chunk_id": 3,
            "page_number": 3,
            "text": "Das Unternehmen verwendet Python zur Automatisierung von Geschäftsprozessen.",
        },
    ]
    questions = [
        ("Wie viele Tage kann ich im Jahr frei nehmen?", 1),
        ("Wie viele Urlaubstage stehen Mitarbeitern jährlich zu?", 1),
        ("Wie lange habe ich Zeit, eine Rechnung zu bezahlen?", 2),
        ("Welche Programmiersprache wird für die Automatisierung eingesetzt?", 3),
        ("Wie viele freie Tage stehen Beschäftigten pro Jahr zu?", 1),
    ]
    # Die Dokumenttexte werden einmalig für alle fünf Fragen kodiert.
    print(
        "Berechne Chunk-Embeddings einmalig und verwende sie für alle Fragen.",
        flush=True,
    )
    embedded_chunks = embed_chunks(chunks)
    correct_count = 0
    for number, (query, expected_chunk_id) in enumerate(questions, start=1):
        print(f"\nFrage {number}: {query}", flush=True)
        results = search_embedded_chunks(query, embedded_chunks, top_k=3)
        for rank, result in enumerate(results, start=1):
            print(f"\nTreffer {rank}:")
            for key in ("chunk_id", "page_number", "text", "similarity_score"):
                print(f"  {key}: {result[key]}")

        top_chunk_id = results[0]["chunk_id"] if results else None
        is_correct = top_chunk_id == expected_chunk_id
        correct_count += int(is_correct)
        print(f"\nPlatz 1: Chunk {top_chunk_id}; erwartet: Chunk {expected_chunk_id}")
        print("Bewertung: korrekt" if is_correct else "Bewertung: nicht korrekt", flush=True)

    accuracy = correct_count / len(questions)
    print(f"\nTop-1-Accuracy: {correct_count}/{len(questions)} = {accuracy:.2%}")


if __name__ == "__main__":
    main()
