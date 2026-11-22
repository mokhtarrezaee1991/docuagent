"""Prüft harte Grenzen des isolierten Experiments ohne Modell oder LLM."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.workflows.query_retry_experiment import run_query_retry, select_context

DOCUMENTS = [{"document_id": "a", "filename": "test.pdf"}, {"document_id": "b", "filename": "test.pdf"}]


def hit(index=1, document_id="a"):
    return {"document_id": document_id, "filename": "test.pdf", "chunk_id": index,
            "page_number": 1, "text": f"Künstlicher Abschnitt {index}", "similarity_score": 0.5}


def run(retrieve, **kwargs):
    return run_query_retry("Original", retrieve=retrieve, get_documents=lambda: DOCUMENTS, **kwargs)


def test_disabled_is_single_pass_even_with_alternative():
    retrieve = Mock(return_value=[hit()])
    result = run(retrieve, alternative_query="Alternative")
    assert result["search_calls"] == 1 and result["retry_count"] == 0
    retrieve.assert_called_once_with("Original", top_k=3, mode="semantic", document_ids=None)


@pytest.mark.parametrize("alternative,reason", [(None, "no_alternative"), ("", "invalid_alternative"),
    ("  ", "invalid_alternative"), (123, "invalid_alternative"), ("  ORIGINAL \n", "identical_alternative")])
def test_invalid_or_identical_alternative_never_retries(alternative, reason):
    retrieve = Mock(return_value=[hit()])
    result = run(retrieve, enabled=True, alternative_query=alternative)
    assert result["reason"] == reason and result["retry_count"] == 0
    retrieve.assert_called_once()


@pytest.mark.parametrize("mode", ["semantic", "hybrid", "hybrid_de"])
def test_retry_maximum_one_and_scope_unchanged(mode):
    first, second = [hit(1), hit(2), hit(3)], [hit(4), hit(3), hit(5)]
    before = deepcopy(first)
    retrieve = Mock(side_effect=[first, second, AssertionError("Keine dritte Suche")])
    result = run(retrieve, enabled=True, alternative_query="Alternative", mode=mode, document_ids=["a"])
    assert result["retry_count"] == 1 and result["search_calls"] == 2
    assert [call.args[0] for call in retrieve.call_args_list] == ["Original", "Alternative"]
    assert all(call.kwargs == {"top_k": 3, "mode": mode, "document_ids": ["a"]} for call in retrieve.call_args_list)
    assert result["first_hits"] == before and result["second_hits"] == second
    assert len(result["final_hits"]) == 3 and result["final_hits"][0] == first[0]
    result["first_hits"][0]["text"] = "Geändert"
    assert first == before


def test_empty_selection_does_not_become_global_or_retry():
    retrieve = Mock(return_value=[])
    result = run(retrieve, document_ids=[], enabled=True, alternative_query="Alternative")
    assert result["reason"] == "empty_selection" and result["final_hits"] == []
    retrieve.assert_called_once_with("Original", top_k=3, mode="semantic", document_ids=[])


@pytest.mark.parametrize("failure_at", [1, 2])
def test_errors_propagate_without_third_search(failure_at):
    error = RuntimeError("Künstlicher Suchfehler")
    retrieve = Mock(side_effect=[error] if failure_at == 1 else [[hit()], error])
    with pytest.raises(RuntimeError) as caught:
        run(retrieve, enabled=True, alternative_query="Alternative")
    assert caught.value is error and retrieve.call_count == failure_at


def test_unknown_selection_is_error_before_search():
    retrieve = Mock()
    with pytest.raises(ValueError, match="unbekannt"):
        run(retrieve, document_ids=["gelöscht"], enabled=True, alternative_query="Alternative")
    retrieve.assert_not_called()


def test_outside_scope_is_rejected():
    retrieve = Mock(return_value=[hit(document_id="b")])
    with pytest.raises(ValueError, match="außerhalb"):
        run(retrieve, document_ids=["a"], enabled=True, alternative_query="Alternative")
    retrieve.assert_called_once()


def test_deleted_id_between_searches_is_error():
    documents = deepcopy(DOCUMENTS)
    def retrieve(query, **kwargs):
        if query == "Alternative":
            documents.pop(0)
        return [hit()]
    with pytest.raises(ValueError, match="gelöscht"):
        run_query_retry("Original", retrieve=retrieve, get_documents=lambda: documents,
                        enabled=True, alternative_query="Alternative", document_ids=["a"])


def test_context_deduplicates_and_keeps_source_metadata():
    first = [hit(1), hit(2), hit(3)]
    second = [{**hit(2), "similarity_score": 0.99}, hit(4)]
    final = select_context(first, second, 3)
    assert final[0] == first[0]
    assert next(item for item in final if item["chunk_id"] == 2)["similarity_score"] == 0.5
    assert len({(item["document_id"], item["chunk_id"], item["page_number"]) for item in final}) == len(final)


def test_lower_rank_original_hit_can_be_displaced_and_is_not_hidden():
    first = [hit(1), hit(2), hit(3)]
    final = select_context(first, [hit(4), hit(5), hit(6)], 3)
    assert final[0] == first[0] and len(final) == 3
    assert first[2] not in final


@pytest.mark.parametrize("options", [{"top_k": 0}, {"top_k": True}, {"enabled": 1}, {"mode": "auto"}])
def test_invalid_options(options):
    retrieve = Mock()
    with pytest.raises(ValueError):
        run(retrieve, **options)
    retrieve.assert_not_called()
