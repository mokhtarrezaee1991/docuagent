"""Steuert bestehendes Retrieval und Antwortgenerierung ohne eigene Modelllogik."""

from copy import deepcopy
from typing import Callable, Literal, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.services import rag_service
from app.workflows.rag_state import RagResult, RagState


class Retriever(Protocol):
    """Passt zu search() der vorhandenen Dokument-Pipelines."""

    def __call__(self, query: str, *, top_k: int) -> list[dict]:
        ...


def build_document_qa_workflow(
    retrieve_chunks: Retriever,
    generator: Callable[[str, list[dict]], RagResult] | None = None,
) -> CompiledStateGraph:
    """Erstellt einen Graph ohne Checkpointer, Wiederholungen oder API-Aufruf.

    Aufruf: workflow.invoke({"query": frage, "top_k": 3}). Jeder Aufruf hat
    eigenen State. Der Retriever soll den aktuellen Index zur Aufrufzeit
    auflösen, statt einen später ersetzten Index dauerhaft einzuschließen.
    Die bestehende Pipeline behält ihre Modusauswahl mit Standard semantic.
    Ohne Test-Generator wird der vorhandene RAG-Service verwendet.
    """

    def retrieve(state: RagState) -> dict:
        """Validiert die Anfrage und sucht ohne neue Dokument-Embeddings."""
        query, top_k = state["query"], state["top_k"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Die Frage darf nicht leer sein.")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k muss eine positive ganze Zahl sein.")
        chunks = retrieve_chunks(query, top_k=top_k)
        # Keine veränderlichen Treffer zwischen Aufrufen teilen; alte Ausgaben löschen.
        return {"retrieved_chunks": deepcopy(chunks), "result": None, "abstention_reason": None}

    def route(state: RagState) -> Literal["abstain", "generate"]:
        """Treffer erlauben Generierung, beweisen aber keine Beantwortbarkeit."""
        return "generate" if state["retrieved_chunks"] else "abstain"

    def abstain(state: RagState) -> dict:
        """Liefert einen festen Hinweis ohne Aufruf eines Antwortgenerators."""
        return {
            "result": {"answer": "Die Information wurde im Dokument nicht gefunden.", "sources": []},
            "abstention_reason": "no_context",
        }

    def generate(state: RagState) -> dict:
        """Erhält LLM-Verzicht und Quellenvalidierung des bestehenden Service."""
        answer_generator = generator if generator is not None else rag_service.generate_answer
        result = answer_generator(state["query"], state["retrieved_chunks"])
        return {"result": deepcopy(result)}

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("abstain", abstain)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route, {"abstain": "abstain", "generate": "generate"})
    graph.add_edge("abstain", END)
    graph.add_edge("generate", END)
    return graph.compile()
