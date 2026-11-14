"""Teilt Seitentexte in überlappende Abschnitte."""


def chunk_pages(
    pages: list[dict], chunk_size: int = 1000, overlap: int = 200
) -> list[dict]:
    """Erzeugt seitenweise Chunks mit dokumentweit fortlaufenden IDs ab 1.

    Größen und Überschneidungen werden in Zeichen gemessen. Leere Seiten
    und Seiten mit ausschließlich Leerraum werden übersprungen.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size muss größer als 0 sein.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap muss mindestens 0 und kleiner als chunk_size sein.")

    chunks = []
    for page in pages:
        text = page["text"]
        if not text or not text.strip():
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "page_number": page["page_number"],
                    "text": text[start:end],
                }
            )
            # Der letzte Chunk beendet die Seite ohne weitere Rest-Chunks.
            if end >= len(text):
                break
            start = end - overlap

    return chunks
