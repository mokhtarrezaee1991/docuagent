"""Prüft gemeinsame Suche und eindeutige Quellen ohne externe API-Aufrufe."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import rag_service, semantic_search


def pdf(text):
    """Erzeugt künstliche Daten ausschließlich im Speicher."""
    with pymupdf.open() as document:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
        return document.tobytes()


@pytest.fixture
def api(monkeypatch, tokenizer):
    previous = getattr(main.app.state, "retrieval_pipeline", None)
    main.app.state.retrieval_pipeline = None
    model = Mock(tokenizer=tokenizer, max_seq_length=128)
    vectors = {"Urlaub": [1.0, 0.0], "Rechnung": [0.0, 1.0]}
    model.encode.side_effect = lambda texts, **kwargs: [vectors[text.strip()] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    monkeypatch.setattr(main, "load_dotenv", Mock())
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    remote = Mock()
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=remote)
    constructor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(rag_service, "OpenAI", constructor)
    def answer(**kwargs):
        context = json.loads(kwargs["input"])["context"]
        return SimpleNamespace(status="completed", output_parsed=rag_service.MultiDocumentAnswer(
            answer="Künstliche Testantwort.",
            sources=[rag_service.DocumentSource(
                document_id=context[0]["document_id"], chunk_id=context[0]["chunk_id"], page_number=context[0]["page_number"]
            )],
        ))
    remote.responses.parse.side_effect = answer
    try:
        with TestClient(main.app) as client:
            yield client, model, remote
    finally:
        main.app.state.retrieval_pipeline = previous


def add(client, text, filename="gleich.pdf"):
    response = client.post("/documents", files={"file": (filename, pdf(text), "application/pdf")})
    assert response.status_code == 200
    return response.json()


def test_duplicate_names_global_search_and_embedding_reuse(api):
    client, model, _ = api
    first = add(client, "Urlaub")
    second = add(client, "Rechnung")
    assert first["document_id"] != second["document_id"]
    assert first["filename"] == second["filename"] == "gleich.pdf"
    assert client.get("/documents").json() == [first, second]
    for query, expected in [("Urlaub", first), ("Rechnung", second)]:
        response = client.post("/documents/search", json={"query": query})
        assert response.status_code == 200
        hits = response.json()
        assert len(hits) == 2
        assert hits[0] == {
            "document_id": expected["document_id"], "filename": "gleich.pdf",
            "chunk_id": 1, "page_number": 1, "text": query + "\n", "similarity_score": 1.0,
        }
    assert [call.args[0] for call in model.encode.call_args_list] == [
        ["Urlaub\n"], ["Rechnung\n"], ["Urlaub"], ["Rechnung"]
    ]


def test_ask_uses_unambiguous_sources(api):
    client, _, remote = api
    first = add(client, "Urlaub")
    second = add(client, "Rechnung")
    for query, expected in [("Urlaub", first), ("Rechnung", second)]:
        result = client.post("/documents/ask", json={"query": query, "top_k": 2})
        assert result.status_code == 200
        assert result.json()["sources"] == [{
            "document_id": expected["document_id"], "filename": "gleich.pdf", "chunk_id": 1, "page_number": 1
        }]
    context = json.loads(remote.responses.parse.call_args.kwargs["input"])["context"]
    assert {chunk["document_id"] for chunk in context} == {first["document_id"], second["document_id"]}
    assert all("embedding" not in chunk for chunk in context)


def test_sources_from_other_documents_are_rejected(api):
    client, _, remote = api
    add(client, "Urlaub")
    remote.responses.parse.side_effect = None
    remote.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=rag_service.MultiDocumentAnswer(
        answer="Nicht belegt", sources=[rag_service.DocumentSource(document_id="fremd", chunk_id=1, page_number=1)]
    ))
    assert client.post("/documents/ask", json={"query": "Urlaub"}).status_code == 502


def test_same_chunk_number_from_excluded_document_is_rejected(api):
    """Eine existierende ID reicht nicht: Die Quelle muss im Suchkontext stehen."""
    client, _, remote = api
    add(client, "Urlaub")
    excluded = add(client, "Rechnung")
    remote.responses.parse.side_effect = None
    remote.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=rag_service.MultiDocumentAnswer(
        answer="Nicht im Kontext belegt", sources=[rag_service.DocumentSource(
            document_id=excluded["document_id"], chunk_id=1, page_number=1
        )],
    ))
    assert client.post("/documents/ask", json={"query": "Urlaub", "top_k": 1}).status_code == 502


def test_multi_document_index_keeps_token_limit(api):
    client, model, _ = api
    model.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for text in texts]
    # Mehrere kurze Zeilen passen auf die künstliche Seite und erzeugen mehrere Chunks.
    document = add(client, "Urlaub und Arbeitszeit.\n" * 20)
    assert document["chunk_count"] > 1
    encoded_texts = model.encode.call_args.args[0]
    assert all(len(model.tokenizer(text)["input_ids"]) <= 128 for text in encoded_texts)


def test_remove_by_id_keeps_same_named_document(api):
    client, _, _ = api
    first = add(client, "Urlaub")
    second = add(client, "Rechnung")
    assert client.delete(f"/documents/{first['document_id']}").status_code == 200
    assert client.get("/documents").json() == [second]
    hits = client.post("/documents/search", json={"query": "Urlaub"}).json()
    assert len(hits) == 1 and hits[0]["document_id"] == second["document_id"]
    assert client.delete(f"/documents/{first['document_id']}").status_code == 404
    assert client.delete(f"/documents/{second['document_id']}").status_code == 200
    assert client.get("/documents").json() == []
    assert client.post("/documents/search", json={"query": "Urlaub"}).status_code == 409
    assert client.post("/documents/ask", json={"query": "Urlaub"}).status_code == 409


@pytest.mark.parametrize("content, mime, status", [
    (b"", "application/pdf", 400), (b"%PDF-kaputt", "application/pdf", 400),
    (pdf(""), "application/pdf", 400), (pdf("Urlaub"), "text/plain", 415),
    (b"%PDF-" + b" " * (10 * 1024 * 1024), "application/pdf", 413),
])
def test_invalid_document_preserves_existing(api, content, mime, status):
    client, model, _ = api
    existing = add(client, "Urlaub")
    response = client.post("/documents", files={"file": ("test.pdf", content, mime)})
    assert response.status_code == status
    assert client.get("/documents").json() == [existing]
    assert model.encode.call_count == 1


def test_legacy_index_can_be_extended_and_still_replaces(api):
    client, _, _ = api
    old = client.post("/documents/index", files={"file": ("alt.pdf", pdf("Urlaub"), "application/pdf")})
    assert old.json() == {"page_count": 1, "chunk_count": 1}
    add(client, "Rechnung")
    assert len(client.get("/documents").json()) == 2
    client.post("/documents/index", files={"file": ("neu.pdf", pdf("Urlaub"), "application/pdf")})
    assert [document["filename"] for document in client.get("/documents").json()] == ["neu.pdf"]


def test_ask_workflow_sees_additions_deletions_and_replacement(api):
    """Ein einmal kompilierter Graph verwendet stets den aktuellen Dokumentindex."""
    client, model, remote = api
    first = add(client, "Urlaub")

    def ask(expected_ids, query="Urlaub"):
        remote.responses.parse.reset_mock()
        response = client.post("/documents/ask", json={"query": query, "top_k": 3})
        assert response.status_code == 200
        remote.responses.parse.assert_called_once()
        context = json.loads(remote.responses.parse.call_args.kwargs["input"])["context"]
        assert {chunk["document_id"] for chunk in context} == expected_ids
        assert response.json()["sources"][0]["document_id"] in expected_ids

    ask({first["document_id"]})
    second = add(client, "Rechnung")
    ask({first["document_id"], second["document_id"]})
    assert client.delete(f"/documents/{first['document_id']}").status_code == 200
    ask({second["document_id"]}, "Rechnung")

    # Der Legacy-Endpunkt ersetzt sogar das bisherige Collection-Objekt.
    replacement = client.post("/documents/index", files={"file": ("neu.pdf", pdf("Urlaub"), "application/pdf")})
    assert replacement.status_code == 200
    current = client.get("/documents").json()[0]
    remote.responses.parse.side_effect = None
    remote.responses.parse.return_value = SimpleNamespace(
        status="completed", output_parsed=rag_service.RagAnswer(
            answer="Testantwort", sources=[rag_service.Source(chunk_id=1, page_number=1)],
        ),
    )
    response = client.post("/documents/ask", json={"query": "Urlaub"})
    assert response.status_code == 200
    assert response.json()["sources"] == [{"chunk_id": 1, "page_number": 1}]
    context = json.loads(remote.responses.parse.call_args.kwargs["input"])["context"]
    assert context == [{"chunk_id": 1, "page_number": 1, "text": "Urlaub\n"}]
    assert client.delete(f"/documents/{current['document_id']}").status_code == 200
    before = remote.responses.parse.call_count
    assert client.post("/documents/ask", json={"query": "Urlaub"}).status_code == 409
    assert remote.responses.parse.call_count == before
    # Drei Indexierungen und vier Fragen: kein Dokument wurde erneut eingebettet.
    assert model.encode.call_count == 7


@pytest.mark.parametrize("mode", [None, "semantic", "hybrid", "hybrid_de"])
def test_ask_policy_uses_collection_metadata_and_configured_mode(api, monkeypatch, mode):
    """Die Policy sieht den aktuellen Index und erhält den serverseitigen Modus."""
    from app.workflows import document_qa
    client, _, remote = api
    first = add(client, "Urlaub")
    second = add(client, "Rechnung")
    if mode is None:
        monkeypatch.delenv("DOCUAGENT_RETRIEVAL_MODE", raising=False)
    else:
        monkeypatch.setenv("DOCUAGENT_RETRIEVAL_MODE", mode)
    collection = main.app.state.retrieval_pipeline
    metadata = Mock(wraps=collection.list_documents)
    search = Mock(wraps=collection.search)
    policy = Mock(wraps=document_qa.select_search_policy)
    monkeypatch.setattr(collection, "list_documents", metadata)
    monkeypatch.setattr(collection, "search", search)
    monkeypatch.setattr(document_qa, "select_search_policy", policy)

    result = client.post("/documents/ask", json={"query": "Urlaub", "top_k": 2})

    assert result.status_code == 200
    assert set(result.json()) == {"answer", "sources"}
    expected = mode or "semantic"
    policy.assert_called_once_with([first, second], mode=expected, document_ids=None)
    assert metadata.call_count >= 1
    search.assert_called_once_with("Urlaub", top_k=2, mode=expected, document_ids=None)
    remote.responses.parse.assert_called_once()
    assert {chunk["document_id"] for chunk in json.loads(remote.responses.parse.call_args.kwargs["input"])["context"]} == {first["document_id"], second["document_id"]}
