"""Prüft lokalen Evaluationspfad mit Modell-Dummy statt Downloads oder Netzwerk."""

from unittest.mock import Mock

import pytest

from app.services import rag_service, semantic_search
from scripts import evaluate_query_retry as evaluation


@pytest.fixture
def model(monkeypatch, tokenizer):
    fake = Mock(device="cpu", tokenizer=tokenizer, max_seq_length=128)
    fake.encode.side_effect = lambda texts, **kwargs: [[1.0, float(len(text) % 4) / 4] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: fake)
    monkeypatch.setattr(rag_service, "OpenAI", Mock(side_effect=AssertionError("Keine OpenAI-Aufrufe")))
    return fake


def test_mmr_actually_uses_candidates_beyond_original_top_three():
    candidates = [{"document_id": "a", "chunk_id": index, "page_number": 1, "similarity_score": 1 / index} for index in range(1, 5)]
    vectors = {("a", index, 1): vector for index, vector in enumerate(([0.95, 0.3], [0.9, 0.4], [0.8, 0.5], [0.8, -0.6]), 1)}
    selected = evaluation.mmr_context(candidates, vectors, [1, 0], anchor=candidates[0])
    assert selected[0] == candidates[0]
    assert candidates[3] in selected
    assert len(selected) == 3
    assert selected != candidates[:3]
    assert all(hit in candidates for hit in selected)


def test_mmr_handles_empty_duplicates_and_stable_ties():
    hits = [{"document_id": "a", "chunk_id": index, "page_number": 1} for index in range(1, 4)]
    vectors = {("a", index, 1): [1, 0] for index in range(1, 4)}
    assert evaluation.mmr_context([], vectors, [1, 0]) == []
    assert evaluation.mmr_context([hits[0], *hits], vectors, [1, 0], anchor=hits[0]) == hits


def test_local_mode_isolates_documents_reuses_embeddings_and_limits_retry(model, monkeypatch):
    embed = Mock(wraps=semantic_search.embed_chunks)
    monkeypatch.setattr(semantic_search, "embed_chunks", embed)
    original = evaluation.DocumentCollection.search
    calls = []
    def search(self, query, **kwargs):
        calls.append(kwargs)
        return original(self, query, **kwargs)
    monkeypatch.setattr(evaluation.DocumentCollection, "search", search)
    report = evaluation.evaluate(embedding_mode="local", repeats=2)
    assert report["input_kind"] == "actual_local_embeddings"
    assert report["embedding_model"] == semantic_search.MODEL_NAME
    assert embed.call_count == 4
    assert all("embedding" not in chunk for call in embed.call_args_list for chunk in call.args[0])
    # Je Split 8+8+13+13 Suchläufe, zweimal gemessen plus ein Warm-up.
    assert len(calls) == 2 * 42 * 3
    assert {call["top_k"] for call in calls} == {3, 6}
    for split in report["splits"].values():
        rows = split["questions"]
        for row in rows:
            for method in ("single_pass", "single_pass_budget", "retry", "retry_mmr_control"):
                value = row[method]
                assert value["retry_count"] <= 1 and value["search_calls"] <= 2
                assert len(value["final_hits"]) <= 3 and value["top_k"] == 3
                assert value["timing"]["sample_count"] == 2
                assert value["outside_scope_hits"] == 0
                assert all(hit["document_id"] in row["document_ids"] for hit in value["final_hits"])
                assert all(hit["document_id"] in row["document_ids"] for hit in value["first_hits"] + value["second_hits"])
            assert row["single_pass_budget"]["search_calls"] == 1
        for method in ("single_pass", "single_pass_budget", "retry"):
            ranks = [row[method]["rank"] for row in rows if row["references"]]
            assert all(split["summary"][method][key] == value for key, value in evaluation.metrics(ranks).items())
            assert split["summary"][method]["timing"]["sample_count"] == 16
    assert report["llm_calls"] == report["openai_api_calls"] == 0
    rag_service.OpenAI.assert_not_called()


@pytest.mark.parametrize("repeats", [0, 1, True, 51])
def test_invalid_local_repeats(repeats):
    with pytest.raises(ValueError, match="repeats"):
        evaluation.evaluate(embedding_mode="local", repeats=repeats)


def test_local_missing_model_is_not_replaced_by_fixtures(monkeypatch):
    monkeypatch.setattr(semantic_search, "_get_model", Mock(side_effect=RuntimeError("Modell fehlt")))
    with pytest.raises(RuntimeError, match="Modell fehlt"):
        evaluation.evaluate(embedding_mode="local")


def test_local_checks_token_limit(model):
    model.max_seq_length = 1
    with pytest.raises(ValueError, match="Tokenlimit"):
        evaluation.evaluate(embedding_mode="local")
    model.encode.assert_not_called()


def test_cli_local_options_and_exclusive_output(tmp_path, monkeypatch):
    run = Mock(return_value={"splits": {}})
    monkeypatch.setattr(evaluation, "evaluate", run)
    output = tmp_path / "local.json"
    assert evaluation.main(["--embeddings", "local", "--repeats", "2", "--output", str(output)]) == 0
    run.assert_called_once_with(embedding_mode="local", repeats=2)
    with pytest.raises(SystemExit):
        evaluation.main(["--embeddings", "local", "--output", str(output)])
    assert run.call_count == 1
