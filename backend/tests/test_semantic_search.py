"""Prüft die Suche ohne echte Modelle, Downloads oder Netzwerkzugriffe."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.services import semantic_search


@pytest.fixture
def model(monkeypatch):
    """Ersetzt das Modell durch feste künstliche Vektoren."""
    vectors = {
        "Frage": [1.0, 0.0],
        "Andere Frage": [0.0, 1.0],
        "Passend": [2.0, 0.0],
        "Teilweise": [1.0, 1.0],
        "Anders": [0.0, 1.0],
        "Gegenteil": [-1.0, 0.0],
        "Null": [0.0, 0.0],
    }
    mock = Mock()
    mock.encode.side_effect = lambda texts, **kwargs: [vectors[text] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: mock)
    return mock


def make_chunks(*texts):
    """Erzeugt kleine Test-Chunks mit eindeutigen Metadaten."""
    return [
        {"chunk_id": index, "page_number": index + 10, "text": text}
        for index, text in enumerate(texts, start=1)
    ]


def test_embed_chunks(model):
    chunks = make_chunks("Passend", "", " \n", "Anders")
    original = deepcopy(chunks)
    result = semantic_search.embed_chunks(chunks)
    assert result == [
        {**chunks[0], "embedding": [2.0, 0.0]},
        {**chunks[3], "embedding": [0.0, 1.0]},
    ]
    assert chunks == original
    # Alle nichtleeren Texte werden mit einem einzigen Modellaufruf kodiert.
    model.encode.assert_called_once_with(["Passend", "Anders"], convert_to_numpy=True)


def test_search_sorted_with_metadata_and_default_limit(model):
    chunks = make_chunks("Anders", "Gegenteil", "Teilweise", "Passend")
    result = semantic_search.search_chunks("Frage", chunks)
    assert [hit["chunk_id"] for hit in result] == [4, 3, 1]
    assert [hit["similarity_score"] for hit in result] == pytest.approx([1, 2 ** -0.5, 0])
    assert result[0] == {**chunks[3], "similarity_score": 1.0}


def test_top_k_and_zero_vector(model):
    chunks = make_chunks("Null", "Gegenteil", "Passend")
    assert len(semantic_search.search_chunks("Frage", chunks, top_k=1)) == 1
    result = semantic_search.search_chunks("Frage", chunks, top_k=10)
    assert [hit["similarity_score"] for hit in result] == [1.0, 0.0, -1.0]


@pytest.mark.parametrize("query, chunks", [
    ("", make_chunks("Passend")),
    (" \n", make_chunks("Passend")),
    ("Frage", []),
    ("Frage", make_chunks("", " \n")),
])
def test_empty_input_does_not_load_model(monkeypatch, query, chunks):
    loader = Mock(side_effect=AssertionError("Das Modell darf nicht geladen werden."))
    monkeypatch.setattr(semantic_search, "_get_model", loader)
    assert semantic_search.search_chunks(query, chunks) == []
    assert semantic_search.embed_chunks([]) == []
    loader.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, 1.5, "3", None, True])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        semantic_search.search_chunks("Frage", [], top_k=top_k)


def test_model_is_loaded_only_once(monkeypatch):
    """Ein künstliches Importmodul prüft den Cache ohne Bibliotheksimport."""
    import sys
    from types import SimpleNamespace

    constructor = Mock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    semantic_search._get_model.cache_clear()
    try:
        assert semantic_search._get_model() is semantic_search._get_model()
        constructor.assert_called_once_with(semantic_search.MODEL_NAME, device="cpu")
    finally:
        semantic_search._get_model.cache_clear()


def test_reuse_embeddings_for_multiple_questions(model):
    """Nur ein Aufruf kodiert die Dokumente; weitere Aufrufe kodieren Fragen."""
    embedded = semantic_search.embed_chunks(make_chunks("Passend", "Anders"))
    original = deepcopy(embedded)
    first = semantic_search.search_embedded_chunks("Frage", embedded)
    second = semantic_search.search_embedded_chunks("Andere Frage", embedded)

    assert first[0]["chunk_id"] == 1
    assert second[0]["chunk_id"] == 2
    assert embedded == original
    assert [call.args[0] for call in model.encode.call_args_list] == [
        ["Passend", "Anders"], ["Frage"], ["Andere Frage"]
    ]


def test_prepared_search_matches_compatible_search(model):
    """Die Optimierung lässt Scores, Metadaten und Rangfolge unverändert."""
    chunks = make_chunks("Anders", "Gegenteil", "Passend", "Teilweise")
    embedded = semantic_search.embed_chunks(chunks)
    assert semantic_search.search_embedded_chunks("Frage", embedded, top_k=2) == (
        semantic_search.search_chunks("Frage", chunks, top_k=2)
    )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, "3", None, True])
def test_prepared_search_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        semantic_search.search_embedded_chunks("Frage", [], top_k=top_k)


def test_prepared_search_empty_input(monkeypatch):
    """Leere Eingaben benötigen keine Modellberechnung."""
    loader = Mock(side_effect=AssertionError("Kein Modellzugriff erwartet."))
    monkeypatch.setattr(semantic_search, "_get_model", loader)
    assert semantic_search.search_embedded_chunks("Frage", []) == []
    assert semantic_search.search_embedded_chunks(" \n", [{"embedding": [1, 0]}]) == []
    loader.assert_not_called()
