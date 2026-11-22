"""Prüft Split-Grenzen, Metriken und echte Suche ohne externe Modelle."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from app.services import rag_service, semantic_search
from scripts import evaluate_retrieval_modes as evaluation


@pytest.fixture(autouse=True)
def forbid_models(monkeypatch):
    monkeypatch.setattr(rag_service, "OpenAI", Mock(side_effect=AssertionError("Keine OpenAI-Aufrufe")))
    monkeypatch.setattr(semantic_search, "_get_model", Mock(side_effect=AssertionError("Kein Modell-Download")))


def test_metrics_at_three():
    result = evaluation.metrics([1, 2, 3, 4, None])
    assert result["hit_rate_at_1"] == pytest.approx(0.2)
    assert result["hit_rate_at_3"] == pytest.approx(0.6)
    assert result["mrr_at_3"] == pytest.approx((1 + 1/2 + 1/3) / 5)


@pytest.mark.parametrize("ranks", [[], [0], [-1], [True], ["1"]])
def test_invalid_ranks(ranks):
    with pytest.raises(ValueError):
        evaluation.metrics(ranks)


def test_ties_are_not_forced_into_one_winner():
    result = evaluation.compare_ranks({"semantic": 2, "hybrid": 1, "hybrid_de": 1})
    assert result["best_modes"] == ["hybrid", "hybrid_de"]
    assert result["versus_semantic"]["hybrid"] == "better"
    assert len(evaluation.compare_ranks(dict.fromkeys(evaluation.MODES, None))["best_modes"]) == 3


def test_splits_have_disjoint_documents_and_families():
    data = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    evaluation.validate_dataset(data)
    assert len(data["documents"]) == 6 and len(data["questions"]) == 12
    for key in ("family", "query"):
        groups = [{q[key] for q in data["questions"] if q["split"] == split} for split in evaluation.SPLITS]
        assert not groups[0] & groups[1]
    old = json.loads((evaluation.ROOT / "data/evaluation/retrieval_questions.json").read_text(encoding="utf-8"))
    assert not {q["query"] for q in old["questions"]} & {q["query"] for q in data["questions"]}


@pytest.mark.parametrize("error", ["family_leak", "document_leak", "bad_evidence", "bad_vector"])
def test_invalid_dataset_is_rejected(error):
    data = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    if error == "family_leak":
        data["questions"][-1]["family"] = data["questions"][0]["family"]
    elif error == "document_leak":
        data["questions"][-1]["document_ids"] = ["dev-sternwarte"]
    elif error == "bad_evidence":
        data["questions"][0]["references"][0]["evidence"] = "Erfunden"
    else:
        data["documents"][0]["chunks"][0]["embedding"] = [float("nan"), 0, 1]
    with pytest.raises(ValueError):
        evaluation.validate_dataset(data)


def test_real_retrieval_policy_equivalence_and_scope(monkeypatch):
    original = evaluation.DocumentCollection.search
    calls = []
    def search(self, query, **kwargs):
        calls.append(kwargs)
        return original(self, query, **kwargs)
    monkeypatch.setattr(evaluation.DocumentCollection, "search", search)
    report = evaluation.evaluate()
    assert len(calls) == 12 * 9
    assert report["llm_calls"] == report["openai_api_calls"] == 0
    for split in report["splits"].values():
        assert split["question_count"] == 6
        allowed = set(split["document_ids"])
        for row in split["questions"]:
            for value in row["methods"].values():
                assert value["outside_scope_hits"] == 0
                assert all(hit["document_id"] in allowed for hit in value["hits"])
            for fallback in evaluation.MODES:
                for state in ("disabled", "enabled"):
                    value = row["methods"][f"policy_{state}_{fallback}"]
                    assert value["hits"] == row["methods"][fallback]["hits"]
                    assert value["routing_correct"] and value["decision"]["used_fallback"]
    rag_service.OpenAI.assert_not_called()


def test_reference_requires_chunk_page_and_evidence():
    hit = {"document_id": "x", "chunk_id": 1, "page_number": 1, "text": "Beleg"}
    ref = {**hit, "evidence": "Beleg"}
    assert evaluation.rank_relevant([hit], [ref]) == 1
    for change in ({"page_number": 2}, {"chunk_id": 2}, {"document_id": "fremd"}, {"evidence": "fehlt"}):
        assert evaluation.rank_relevant([hit], [{**ref, **change}]) is None


def test_reproducible_except_runtime():
    def strip_times(report):
        result = deepcopy(report)
        for split in result["splits"].values():
            for summary in split["summary"].values():
                summary.pop("search_ms")
                summary.pop("routing_ms", None)
            for row in split["questions"]:
                for value in row["methods"].values():
                    value.pop("elapsed_ms")
                    value.pop("routing_ms", None)
        return result
    assert strip_times(evaluation.evaluate()) == strip_times(evaluation.evaluate())


def test_reports_are_exclusive(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    assert evaluation.main(["--output", str(output)]) == 0
    previous = output.read_bytes()
    run = Mock(side_effect=AssertionError("Nicht erneut starten"))
    monkeypatch.setattr(evaluation, "evaluate", run)
    with pytest.raises(SystemExit):
        evaluation.main(["--output", str(output)])
    assert output.read_bytes() == previous
    run.assert_not_called()


@pytest.fixture
def local_model(monkeypatch, tokenizer):
    """Lokaler Modell-Dummy prüft den neuen Pfad ohne Download oder Netzwerk."""
    model = Mock(device="cpu", tokenizer=tokenizer, max_seq_length=128)
    model.encode.side_effect = lambda texts, **kwargs: [[1.0, float(len(text) % 3), 0.5] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    return model


def test_local_embeddings_are_computed_once_and_reused(local_model, monkeypatch):
    embed = Mock(wraps=semantic_search.embed_chunks)
    monkeypatch.setattr(semantic_search, "embed_chunks", embed)
    report = evaluation.evaluate(embedding_mode="local", repeats=2)
    assert report["embedding_model"] == semantic_search.MODEL_NAME
    assert report["embedding_mode"] == "local"
    assert report["model_load_ms"] >= 0
    assert embed.call_count == 6
    assert all("embedding" not in chunk for call in embed.call_args_list for chunk in call.args[0])
    # Sechs Dokument-Batches; zwölf Fragen, sechs Methoden, Warm-up plus zwei Messungen.
    assert local_model.encode.call_count == 6 + 12 * 6 * 3
    for split in report["splits"].values():
        assert split["question_count"] == 6
        for row in split["questions"]:
            for mode in evaluation.MODES:
                fixed = row["methods"][mode]
                policy = row["methods"][f"policy_disabled_{mode}"]
                assert fixed["hits"] == policy["hits"]
                assert fixed["rank"] == policy["rank"]
                assert fixed["timing"]["sample_count"] == 2
                assert fixed["outside_scope_hits"] == 0
                assert all(hit["document_id"] in split["document_ids"] for hit in fixed["hits"])
        for mode in evaluation.MODES:
            expected = evaluation.metrics([row["methods"][mode]["rank"] for row in split["questions"]])
            summary = split["summary"][mode]
            assert all(summary[key] == value for key, value in expected.items())
            assert summary["timing"]["sample_count"] == 12
    assert report["llm_calls"] == report["openai_api_calls"] == 0


def test_timing_summary_has_empirical_percentile():
    assert evaluation.timing_summary([5, 1, 4, 2, 3]) == {"sample_count": 5, "median_ms": 3, "p95_ms": 5}


@pytest.mark.parametrize("repeats", [0, 1, True, 51, 2.5])
def test_local_invalid_repeats(repeats):
    with pytest.raises(ValueError, match="repeats"):
        evaluation.evaluate(embedding_mode="local", repeats=repeats)


def test_local_rejects_truncation(local_model):
    local_model.max_seq_length = 2
    with pytest.raises(ValueError, match="Tokenlimit"):
        evaluation.evaluate(embedding_mode="local", repeats=2)
    local_model.encode.assert_not_called()


def test_local_model_failure_does_not_fall_back_to_fixtures(monkeypatch):
    monkeypatch.setattr(semantic_search, "_get_model", Mock(side_effect=RuntimeError("Modell fehlt")))
    with pytest.raises(RuntimeError, match="Modell fehlt"):
        evaluation.evaluate(embedding_mode="local", repeats=2)


def test_cli_selects_local_mode_and_keeps_overwrite_protection(tmp_path, monkeypatch):
    run = Mock(return_value={"splits": {}})
    monkeypatch.setattr(evaluation, "evaluate", run)
    output = tmp_path / "local.json"
    assert evaluation.main(["--embeddings", "local", "--repeats", "3", "--output", str(output)]) == 0
    run.assert_called_once_with(embedding_mode="local", repeats=3)
    with pytest.raises(SystemExit):
        evaluation.main(["--embeddings", "local", "--output", str(output)])
    assert run.call_count == 1


def test_unknown_embedding_mode_is_rejected():
    with pytest.raises(ValueError, match="embedding_mode"):
        evaluation.evaluate(embedding_mode="unknown")
