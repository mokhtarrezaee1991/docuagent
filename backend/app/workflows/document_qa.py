"""Steuert bestehendes Retrieval und Antwortgenerierung ohne eigene Modelllogik."""

from copy import deepcopy
from typing import Callable, Literal, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.services import rag_service
from app.services.hybrid_search import retrieval_mode
from app.workflows.rag_state import RagResult, RagState
from app.workflows.search_policy import SearchPlan, select_search as select_search_policy


class Retriever(Protocol):
    """Passt zu search() der vorhandenen Dokument-Pipelines."""

    def __call__(self, query: str, *, top_k: int) -> list[dict]:
        ...


class PlannedRetriever(Protocol):
    """Erweiterte Signatur von DocumentCollection.search für validierte Pläne."""

    def __call__(self, query: str, *, top_k: int, mode: str, document_ids: list[str] | None) -> list[dict]:
        ...


def build_document_qa_workflow(
    retrieve_chunks: Retriever | PlannedRetriever,
    generator: Callable[[str, list[dict]], RagResult] | None = None,
    *, get_documents: Callable[[], list[dict]] | None = None,
) -> CompiledStateGraph:
    """Erstellt einen Graph ohne Checkpointer, Wiederholungen oder API-Aufruf.

    Aufruf: workflow.invoke({"query": frage, "top_k": 3}). Jeder Aufruf hat
    eigenen State. Der Retriever soll den aktuellen Index zur Aufrufzeit
    auflösen, statt einen später ersetzten Index dauerhaft einzuschließen.
    Die bestehende Pipeline behält ihre Modusauswahl mit Standard semantic.
    Ohne Test-Generator wird der vorhandene RAG-Service verwendet.

    get_documents aktiviert die erweiterte Retrieval-Signatur mit mode und
    document_ids, z.B. collection.search und collection.list_documents. Der
    Provider wird je Anfrage aufgerufen; die Suche prüft IDs nochmals aktuell.
    Ohne Provider bleiben alte Callables und deren konfigurierte Moduswahl
    erhalten. Explizite Suchoptionen werden dann abgewiesen, niemals ignoriert.
    """

    def select_search(state: RagState) -> dict:
        """Validiert Eingaben und plant ausschließlich mit aktuellen Metadaten."""
        query, top_k = state["query"], state["top_k"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Die Frage darf nicht leer sein.")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k muss eine positive ganze Zahl sein.")
        if get_documents is None:
            if state.get("mode") is not None or state.get("document_ids") is not None:
                raise ValueError("Explizite Suchoptionen benötigen einen Metadaten-Provider und die erweiterte Retrieval-Schnittstelle.")
            # Alte API-Adapter behalten ihre bestehende Umgebungs-Konfiguration.
            configured = retrieval_mode()
            plan = select_search_policy([], mode=None if configured == "semantic" else configured)
        else:
            plan = select_search_policy(
                get_documents(), mode=state.get("mode"), document_ids=state.get("document_ids"),
            )
        return {
            "mode": plan.mode,
            "document_ids": None if plan.document_ids is None else list(plan.document_ids),
            "routing_reason": plan.routing_reason,
            "retrieved_chunks": [], "result": None, "abstention_reason": None,
        }

    def retrieve(state: RagState) -> dict:
        """Wendet den validierten Plan vor Ranking und Top-k in der Suche an."""
        plan = SearchPlan(
            mode=state["mode"],
            document_ids=None if state["document_ids"] is None else tuple(state["document_ids"]),
            routing_reason=state["routing_reason"],
        )
        if get_documents is None:
            chunks = retrieve_chunks(state["query"], top_k=state["top_k"])
        else:
            chunks = retrieve_chunks(
                state["query"], top_k=state["top_k"], mode=plan.mode,
                document_ids=None if plan.document_ids is None else list(plan.document_ids),
            )
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
    graph.add_node("select_search", select_search)
    graph.add_node("retrieve", retrieve)
    graph.add_node("abstain", abstain)
    graph.add_node("generate", generate)
    graph.add_edge(START, "select_search")
    graph.add_edge("select_search", "retrieve")
    graph.add_conditional_edges("retrieve", route, {"abstain": "abstain", "generate": "generate"})
    graph.add_edge("abstain", END)
    graph.add_edge("generate", END)
    return graph.compile()
