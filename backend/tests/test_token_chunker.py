"""Prüft Tokenlimit, Überlappung und verlustfreie Originaltexte."""

import pytest

from app.services.text_chunker import chunk_pages_by_tokens


def test_limit_includes_special_tokens_and_overlap(tokenizer):
    chunks = chunk_pages_by_tokens(
        [{"page_number": 1, "text": "ABCDEFGHIJKL"}], tokenizer, max_tokens=7, overlap=2
    )
    assert [chunk["text"] for chunk in chunks] == ["ABCDE", "DEFGH", "GHIJK", "JKL"]
    assert all(len(tokenizer(chunk["text"])["input_ids"]) <= 7 for chunk in chunks)
    assert all(a["text"][-2:] == b["text"][:2] for a, b in zip(chunks, chunks[1:]))


def test_complete_text_and_page_metadata(tokenizer):
    text = "  Grüße\n中文🙂 und Urlaub!  "
    pages = [{"page_number": 2, "text": text}, {"page_number": 3, "text": " \n"}, {"page_number": 5, "text": "Ende"}]
    chunks = chunk_pages_by_tokens(pages, tokenizer, max_tokens=8, overlap=0)
    assert "".join(chunk["text"] for chunk in chunks if chunk["page_number"] == 2) == text
    assert [chunk["chunk_id"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert {chunk["page_number"] for chunk in chunks} == {2, 5}
    assert chunk_pages_by_tokens(pages, tokenizer, 8, 0) == chunks


def test_high_overlap_terminates(tokenizer):
    chunks = chunk_pages_by_tokens([{"page_number": 1, "text": "ABCDEFG"}], tokenizer, 6, 3)
    assert [chunk["text"] for chunk in chunks] == ["ABCD", "BCDE", "CDEF", "DEFG"]


@pytest.mark.parametrize("limit, overlap", [(2, 0), (0, 0), (6, -1), (6, 4), (6, True), (True, 0)])
def test_invalid_parameters(tokenizer, limit, overlap):
    with pytest.raises(ValueError):
        chunk_pages_by_tokens([], tokenizer, limit, overlap)


def test_empty_pages(tokenizer):
    assert chunk_pages_by_tokens([{"page_number": 1, "text": " \n"}], tokenizer, 128) == []


def test_numbered_sections_separate_boilerplate_and_password_rule(tokenizer):
    """Die Regel bleibt ohne themenfremden Dokumentvorspann auffindbar."""
    prefix = "Fiktives Testdokument mit allgemeinen Hinweisen.\n"
    rule = "1. Passwörter und Anmeldung\nPasswörter müssen mindestens 14 Zeichen haben.\n"
    following = "2. Datensicherung\nTäglich sichern.\n"
    text = prefix + rule + following
    chunks = chunk_pages_by_tokens([{"page_number": 4, "text": text}], tokenizer, 128)
    assert [chunk["text"] for chunk in chunks] == [prefix, rule, following]
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert [chunk["chunk_id"] for chunk in chunks] == [1, 2, 3]
    assert all(chunk["page_number"] == 4 for chunk in chunks)
    assert all(len(tokenizer(chunk["text"])["input_ids"]) <= 128 for chunk in chunks)


def test_section_boundaries_preserve_initial_whitespace(tokenizer):
    text = " \n1. Regeln\nText\n2. Weiter\nMehr Text\n"
    chunks = chunk_pages_by_tokens([{"page_number": 1, "text": text}], tokenizer, 128)
    assert "".join(chunk["text"] for chunk in chunks) == text


def test_long_section_still_respects_token_limit(tokenizer):
    text = "1. Regeln\n" + "Künstlicher Inhalt. " * 25 + "\n2. Ende\nFertig."
    chunks = chunk_pages_by_tokens([{"page_number": 1, "text": text}], tokenizer, 128)
    assert len(chunks) > 2
    assert all(len(tokenizer(chunk["text"])["input_ids"]) <= 128 for chunk in chunks)
    assert chunks[-1]["text"] == "2. Ende\nFertig."
