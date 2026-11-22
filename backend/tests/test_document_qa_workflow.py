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


@pytest.fixture
def planned_collection(monkeypatch):
    """Gleiche Dateinamen und Chunk-IDs, aber eindeutig unterscheidbare Dokumente."""
    model = Mock()
    model.encode.return_value = [[1.0, 0.0]]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    collection = DocumentCollection()
    for identifier, vector in [("a", [1.0, 0.0]), ("b", [0.8, 0.6])]:
        collection.add_prepared(DocumentRetrievalPipeline.from_record({
            "metadata": {"document_id": identifier, "filename": "gleich.pdf", "page_count": 1, "chunk_count": 1},
            "chunks": [{"chunk_id": 1, "page_number": 1, "text": "Passwörter schützen", "embedding": vector}],
        }))
    return collection, model


def test_select_search_node_reuses_policy_and_defaults(planned_collection, monkeypatch):
    from app.workflows import document_qa
    collection, _ = planned_collection
    policy = Mock(wraps=document_qa.select_search_policy)
    monkeypatch.setattr(document_qa, "select_search_policy", policy)
    # Die neue Policy verwendet ihren eigenen Standard, keine automatische Hybridwahl.
    monkeypatch.setenv("DOCUAGENT_RETRIEVAL_MODE", "hybrid_de")
    graph = build_document_qa_workflow(
        collection.search, Mock(return_value={"answer": "Test", "sources": []}),
        get_documents=collection.list_documents,
    )
    state = graph.invoke({"query": "Passwort 14", "top_k": 3})
    policy.assert_called_once_with(collection.list_documents(), mode=None, document_ids=None)
    assert state["mode"] == "semantic" and state["document_ids"] is None
    assert state["routing_reason"] == "all_documents;default_semantic"
    assert {hit["document_id"] for hit in state["retrieved_chunks"]} == {"a", "b"}


@pytest.mark.parametrize("mode", ["semantic", "hybrid", "hybrid_de"])
def test_plan_filters_before_top_k_and_limits_generator_context(planned_collection, mode):
    collection, model = planned_collection
    generator = Mock(return_value={"answer": "Test", "sources": []})
    graph = build_document_qa_workflow(collection.search, generator, get_documents=collection.list_documents)
    state = graph.invoke({"query": "Passwort", "top_k": 1, "mode": mode, "document_ids": ["b", "b"]})
    assert state["mode"] == mode and state["document_ids"] == ["b"]
    assert state["routing_reason"] == "explicit_document_ids;explicit_mode"
    assert [hit["document_id"] for hit in state["retrieved_chunks"]] == ["b"]
    generator.assert_called_once_with("Passwort", state["retrieved_chunks"])
    model.encode.assert_called_once_with(["Passwort"], convert_to_numpy=True)


def test_empty_plan_abstains_without_generator_or_embedding(planned_collection):
    collection, model = planned_collection
    generator = Mock()
    graph = build_document_qa_workflow(collection.search, generator, get_documents=collection.list_documents)
    state = graph.invoke({"query": "Passwort", "top_k": 3, "document_ids": []})
    assert state["document_ids"] == []
    assert state["abstention_reason"] == "no_context"
    assert state["result"]["sources"] == []
    generator.assert_not_called()
    model.encode.assert_not_called()


@pytest.mark.parametrize("options", [{"document_ids": ["fremd"]}, {"mode": "bm25"}, {"document_ids": "a"}])
def test_invalid_plan_stops_before_retrieval(planned_collection, options):
    collection, model = planned_collection
    retrieve, generator = Mock(wraps=collection.search), Mock()
    graph = build_document_qa_workflow(retrieve, generator, get_documents=collection.list_documents)
    with pytest.raises(ValueError):
        graph.invoke({"query": "Frage", "top_k": 1, **options})
    retrieve.assert_not_called()
    generator.assert_not_called()
    model.encode.assert_not_called()


def test_metadata_is_current_and_deleted_ids_fail(planned_collection):
    collection, _ = planned_collection
    graph = build_document_qa_workflow(collection.search, Mock(return_value={"answer": "Test", "sources": []}), get_documents=collection.list_documents)
    graph.invoke({"query": "Frage", "top_k": 1, "document_ids": ["b"]})
    collection.remove_document("b")
    with pytest.raises(ValueError, match="gelöscht"):
        graph.invoke({"query": "Frage", "top_k": 1, "document_ids": ["b"]})


def test_deletion_between_planning_and_search_is_rechecked(planned_collection):
    collection, _ = planned_collection
    generator = Mock()
    def retrieve(query, *, top_k, mode, document_ids):
        collection.remove_document("b")
        return collection.search(query, top_k=top_k, mode=mode, document_ids=document_ids)

    graph = build_document_qa_workflow(retrieve, generator, get_documents=collection.list_documents)
    with pytest.raises(ValueError, match="gelöscht"):
        graph.invoke({"query": "Frage", "top_k": 1, "document_ids": ["b"]})
    generator.assert_not_called()


def test_plan_isolation_between_sequential_and_parallel_requests(planned_collection):
    collection, _ = planned_collection
    graph = build_document_qa_workflow(collection.search, Mock(return_value={"answer": "Test", "sources": []}), get_documents=collection.list_documents)
    graph.invoke({"query": "Frage", "top_k": 1, "mode": "hybrid_de", "document_ids": ["b"]})
    default = graph.invoke({"query": "Frage", "top_k": 3})
    assert default["mode"] == "semantic" and default["document_ids"] is None
    assert default["routing_reason"] == "all_documents;default_semantic"
    with ThreadPoolExecutor(max_workers=2) as executor:
        states = list(executor.map(lambda identifier: graph.invoke({
            "query": "Frage", "top_k": 1, "document_ids": [identifier],
        }), ["a", "b"]))
    assert [state["retrieved_chunks"][0]["document_id"] for state in states] == ["a", "b"]


@pytest.mark.parametrize("options", [{"mode": "semantic"}, {"document_ids": []}, {"document_ids": ["a"]}])
def test_legacy_callable_never_silently_ignores_options(options):
    retrieve, generator = Mock(), Mock()
    graph = build_document_qa_workflow(retrieve, generator)
    with pytest.raises(ValueError, match="Metadaten-Provider"):
        graph.invoke({"query": "Frage", "top_k": 1, **options})
    retrieve.assert_not_called()
    generator.assert_not_called()


def test_scoped_sources_are_validated_by_existing_rag_service(planned_collection, monkeypatch):
    from types import SimpleNamespace
    collection, _ = planned_collection
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(
        status="completed", output_parsed=rag_service.MultiDocumentAnswer(
            answer="Nicht belegt", sources=[rag_service.DocumentSource(document_id="a", chunk_id=1, page_number=1)],
        ),
    )
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=client)
    constructor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(rag_service, "OpenAI", constructor)
    graph = build_document_qa_workflow(collection.search, get_documents=collection.list_documents)
    with pytest.raises(RuntimeError, match="nicht übergebene"):
        graph.invoke({"query": "Frage", "top_k": 1, "document_ids": ["b"]})
    client.responses.parse.assert_called_once()
