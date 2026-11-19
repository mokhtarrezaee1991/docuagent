"""Prüft den isolierten Graph mit künstlichem Kontext ohne externe Aufrufe."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from app.services import rag_service, semantic_search
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline
from app.workflows.document_qa import build_document_qa_workflow


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    """Ein versehentlicher echter Modell- oder OpenAI-Aufruf lässt Tests scheitern."""
    monkeypatch.setattr(rag_service, "OpenAI", Mock(side_effect=AssertionError("Keine OpenAI-Aufrufe")))
    monkeypatch.setattr(semantic_search, "_get_model", Mock(side_effect=AssertionError("Kein Modell-Download")))


def chunk(document_id="dokument-1"):
    """Liefert ausschließlich künstliche, eindeutig referenzierte Testdaten."""
    return {
        "document_id": document_id, "filename": "test.pdf", "chunk_id": 1,
        "page_number": 2, "text": "Mitarbeiter haben 30 Urlaubstage.",
        "similarity_score": 0.1,
    }


def test_generate_preserves_context_and_sources():
    chunks = [chunk()]
    source = {key: chunks[0][key] for key in ("document_id", "filename", "chunk_id", "page_number")}
    result = {"answer": "30 Urlaubstage.", "sources": [source]}
    retrieve = Mock(return_value=chunks)
    generator = Mock(return_value=result)
    workflow = build_document_qa_workflow(retrieve, generator)

    state = workflow.invoke({"query": "Wie viele Urlaubstage?", "top_k": 3})

    retrieve.assert_called_once_with("Wie viele Urlaubstage?", top_k=3)
    generator.assert_called_once_with("Wie viele Urlaubstage?", chunks)
    assert state["result"] == result
    assert state["retrieved_chunks"] == chunks
    assert state["abstention_reason"] is None
    # Auch geringe Scores werden weder gefiltert noch künstlich verändert.
    assert state["retrieved_chunks"][0]["similarity_score"] == 0.1


def test_empty_context_abstains_without_generator(monkeypatch):
    existing_generator = Mock(side_effect=AssertionError("RAG-Service darf nicht laufen"))
    monkeypatch.setattr(rag_service, "generate_answer", existing_generator)
    generator = Mock(side_effect=AssertionError("Generator darf nicht laufen"))
    workflow = build_document_qa_workflow(Mock(return_value=[]), generator)

    state = workflow.invoke({"query": "Wie viele Urlaubstage?", "top_k": 3})

    assert state["result"] == {"answer": "Die Information wurde im Dokument nicht gefunden.", "sources": []}
    assert state["abstention_reason"] == "no_context"
    generator.assert_not_called()
    existing_generator.assert_not_called()
    rag_service.OpenAI.assert_not_called()


def test_llm_abstention_is_preserved():
    refusal = {"answer": "Die Information wurde im Dokument nicht gefunden.", "sources": []}
    generator = Mock(return_value=refusal)
    workflow = build_document_qa_workflow(Mock(return_value=[chunk()]), generator)
    state = workflow.invoke({"query": "Wie hoch ist der Umsatz?", "top_k": 3})
    assert state["result"] == refusal
    generator.assert_called_once()


@pytest.mark.parametrize("failing_node", ["retrieve", "generate"])
def test_errors_propagate_without_retry(failing_node):
    error = RuntimeError("Künstlicher Servicefehler")
    retrieve = Mock(return_value=[chunk()])
    generator = Mock(return_value={"answer": "Test", "sources": []})
    (retrieve if failing_node == "retrieve" else generator).side_effect = error
    workflow = build_document_qa_workflow(retrieve, generator)
    with pytest.raises(RuntimeError) as caught:
        workflow.invoke({"query": "Frage", "top_k": 3})
    assert caught.value is error
    assert retrieve.call_count == 1
    assert generator.call_count == (1 if failing_node == "generate" else 0)


def test_default_generator_uses_existing_service(monkeypatch):
    generator = Mock(return_value={"answer": "Test", "sources": []})
    monkeypatch.setattr(rag_service, "generate_answer", generator)
    workflow = build_document_qa_workflow(Mock(return_value=[chunk()]))
    workflow.invoke({"query": "Frage", "top_k": 1})
    generator.assert_called_once_with("Frage", [chunk()])


def test_sequential_requests_do_not_share_state():
    chunks = [chunk()]
    retrieve = Mock(side_effect=[chunks, []])
    generator = Mock(return_value={"answer": "30", "sources": []})
    workflow = build_document_qa_workflow(retrieve, generator)
    first = workflow.invoke({"query": "Erste Frage", "top_k": 1})
    first["retrieved_chunks"][0]["text"] = "Geändert"
    second = workflow.invoke({"query": "Zweite Frage", "top_k": 2})
    assert chunks[0]["text"] == "Mitarbeiter haben 30 Urlaubstage."
    assert second["retrieved_chunks"] == []
    assert second["result"]["sources"] == []
    assert second["abstention_reason"] == "no_context"
    generator.assert_called_once()


def test_concurrent_requests_do_not_mix_context():
    def retrieve(query, *, top_k):
        return [chunk(query)]

    def generate(query, chunks):
        assert chunks[0]["document_id"] == query
        return {"answer": query, "sources": [{"document_id": query}]}

    workflow = build_document_qa_workflow(retrieve, generate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        states = list(executor.map(lambda query: workflow.invoke({"query": query, "top_k": 1}), ["A", "B"]))
    assert [state["result"]["answer"] for state in states] == ["A", "B"]


def test_current_collection_and_precomputed_embeddings_are_used(monkeypatch):
    # Vorbereitete Vektoren ersetzen eine PDF-Indexierung im isolierten Test.
    def collection(document_id):
        pipeline = DocumentRetrievalPipeline.from_record({
            "metadata": {"document_id": document_id, "filename": "test.pdf", "page_count": 2, "chunk_count": 1},
            "chunks": [{**chunk(document_id), "embedding": [1.0, 0.0]}],
        })
        documents = DocumentCollection()
        documents.add_prepared(pipeline)
        return documents

    model = Mock()
    model.encode.return_value = [[1.0, 0.0]]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    current = {"index": collection("A")}

    def retrieve(query, *, top_k):
        return current["index"].search(query, top_k=top_k)

    generator = Mock(return_value={"answer": "Test", "sources": []})
    workflow = build_document_qa_workflow(retrieve, generator)
    first = workflow.invoke({"query": "Frage A", "top_k": 1})
    current["index"] = collection("B")
    second = workflow.invoke({"query": "Frage B", "top_k": 1})
    assert first["retrieved_chunks"][0]["document_id"] == "A"
    assert second["retrieved_chunks"][0]["document_id"] == "B"
    assert [call.args[0] for call in model.encode.call_args_list] == [["Frage A"], ["Frage B"]]


@pytest.mark.parametrize("query,top_k", [("", 3), ("  ", 3), ("Frage", 0), ("Frage", True)])
def test_invalid_input_stops_before_retrieval(query, top_k):
    retrieve, generator = Mock(), Mock()
    workflow = build_document_qa_workflow(retrieve, generator)
    with pytest.raises(ValueError):
        workflow.invoke({"query": query, "top_k": top_k})
    retrieve.assert_not_called()
    generator.assert_not_called()
