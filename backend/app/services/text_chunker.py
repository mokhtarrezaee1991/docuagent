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


def chunk_pages_by_tokens(
    pages: list[dict], tokenizer, max_tokens: int, overlap: int = 20
) -> list[dict]:
    """Teilt Originaltext anhand von Tokenpositionen ohne verlustreiches Dekodieren.

    max_tokens umfasst spezielle Tokens. overlap zählt Inhaltstokens; bei
    ungünstigen Teilwortgrenzen kann die Überlappung kleiner ausfallen.
    Jeder fertige Text wird erneut tokenisiert und gegen das Limit geprüft.
    Benötigt einen schnellen Tokenizer mit Zeichen-Offsets.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens muss eine ganze Zahl sein.")
    capacity = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError("Das Tokenlimit muss Platz für Inhalt und spezielle Tokens bieten.")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < capacity:
        raise ValueError("overlap muss mindestens 0 und kleiner als das Inhaltstokenlimit sein.")

    chunks = []
    for page in pages:
        text = page["text"]
        if not text or not text.strip():
            continue
        start = 0
        while start < len(text):
            remaining = text[start:]
            encoded = tokenizer(
                remaining, add_special_tokens=False, truncation=False,
                return_offsets_mapping=True,
            )
            offsets = encoded["offset_mapping"]
            # Bis zum Beginn des nächsten Tokens bleiben auch Leerzeichen erhalten.
            end = len(text) if len(offsets) <= capacity else start + offsets[capacity][0]
            # Erneutes Tokenisieren eines Ausschnitts kann andere Teilwörter erzeugen.
            while end > start and len(tokenizer(text[start:end], truncation=False)["input_ids"]) > max_tokens:
                end -= 1
            if end <= start:
                raise ValueError("Das Tokenlimit reicht nicht für den nächsten Textabschnitt.")

            content = text[start:end]
            chunks.append({"chunk_id": len(chunks) + 1, "page_number": page["page_number"], "text": content})
            if end == len(text):
                break

            chunk_offsets = tokenizer(
                content, add_special_tokens=False, truncation=False,
                return_offsets_mapping=True,
            )["offset_mapping"]
            next_start = end
            if overlap and chunk_offsets:
                next_start = start + chunk_offsets[max(0, len(chunk_offsets) - overlap)][0]
            # Garantierter Fortschritt, auch bei überlappenden Unicode-Offsets.
            start = next_start if next_start > start else end
    return chunks
