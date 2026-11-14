"""Prüft lokale Diagnosehilfen ohne echte Dokumente oder externe APIs."""

from scripts.diagnose_rag import check_text_preserved
from app.services.text_chunker import chunk_pages


def test_chunking_preserves_late_passage():
    """Auch Text hinter der ersten Chunk-Grenze bleibt erhalten."""
    text = "Künstlicher Fülltext. " * 70 + "Künstliche Urlaubspassage."
    pages = [{"page_number": 1, "text": text}]
    chunks = chunk_pages(pages)
    assert check_text_preserved(pages, chunks)
    assert any("Künstliche Urlaubspassage." in chunk["text"] for chunk in chunks)


def test_detects_lost_text():
    pages = [{"page_number": 1, "text": "Künstlicher Testtext"}]
    assert not check_text_preserved(pages, [{"page_number": 1, "text": "Künstlicher"}])
