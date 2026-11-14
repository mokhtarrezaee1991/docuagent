import pytest

from app.services.text_chunker import chunk_pages


def test_short_text():
    """Ein kurzer Text bleibt vollständig in einem Chunk erhalten."""
    assert chunk_pages([{"page_number": 3, "text": "Hallo DocuAgent"}]) == [
        {"chunk_id": 1, "page_number": 3, "text": "Hallo DocuAgent"}
    ]


def test_long_text_without_overlap():
    """Lange Texte werden ohne Textverlust und innerhalb des Limits geteilt."""
    text = "ABCDEFGHIJK"
    chunks = chunk_pages([{"page_number": 1, "text": text}], chunk_size=4, overlap=0)
    assert [chunk["text"] for chunk in chunks] == ["ABCD", "EFGH", "IJK"]
    assert all(len(chunk["text"]) <= 4 for chunk in chunks)
    assert "".join(chunk["text"] for chunk in chunks) == text


def test_overlap():
    """Nachbar-Chunks teilen genau die vorgegebenen Zeichen."""
    chunks = chunk_pages(
        [{"page_number": 1, "text": "ABCDEFGHIJKL"}], chunk_size=5, overlap=2
    )
    assert [chunk["text"] for chunk in chunks] == ["ABCDE", "DEFGH", "GHIJK", "JKL"]
    for previous, current in zip(chunks, chunks[1:]):
        assert previous["text"][-2:] == current["text"][:2]


def test_pages_stay_separate_and_ids_are_reproducible():
    """Seitenzahlen bleiben erhalten; IDs laufen über Seitengrenzen weiter."""
    pages = [{"page_number": 2, "text": "ABCDEF"}, {"page_number": 7, "text": "XYZ"}]
    expected = [
        {"chunk_id": 1, "page_number": 2, "text": "ABCD"},
        {"chunk_id": 2, "page_number": 2, "text": "DEF"},
        {"chunk_id": 3, "page_number": 7, "text": "XYZ"},
    ]
    assert chunk_pages(pages, chunk_size=4, overlap=1) == expected
    assert chunk_pages(pages, chunk_size=4, overlap=1) == expected


def test_empty_pages():
    """Leere Seiten und reine Leerzeichen erzeugen keine Chunks."""
    pages = [
        {"page_number": 1, "text": ""},
        {"page_number": 2, "text": " \n\t"},
        {"page_number": 3, "text": "Text"},
    ]
    assert chunk_pages(pages) == [{"chunk_id": 1, "page_number": 3, "text": "Text"}]
    assert chunk_pages([]) == []


@pytest.mark.parametrize("text", ["ABCD", "ABCDEFG"])
def test_last_chunk_ends_exactly_at_text_end(text):
    """Auch bei hoher Überschneidung entstehen keine überflüssigen Rest-Chunks."""
    chunks = chunk_pages([{"page_number": 1, "text": text}], chunk_size=4, overlap=3)
    assert len(chunks) == len(text) - 3
    assert chunks[-1]["text"] == text[-4:]


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)])
def test_invalid_parameters(chunk_size, overlap):
    """Ungültige Parameter werden auch bei leerer Eingabe abgelehnt."""
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=chunk_size, overlap=overlap)
