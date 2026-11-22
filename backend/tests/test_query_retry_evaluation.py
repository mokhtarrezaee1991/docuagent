"""Prüft fair begrenzte Kontextbudgets und transparente Experimentmetriken."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from app.services import rag_service, semantic_search
from scripts import evaluate_query_retry as evaluation


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    monkeypatch.setattr(rag_service, "OpenAI", Mock(side_effect=AssertionError("Keine OpenAI-Aufrufe")))
    monkeypatch.setattr(semantic_search, "_get_model", Mock(side_effect=AssertionError("Kein Modell-Download")))


def test_actual_retrieval_budget_and_results(monkeypatch):
    original = evaluation.DocumentCollection.search
    calls = []
    def search(self, query, **kwargs):
        calls.append(kwargs)
        return original(self, query, **kwargs)
    monkeypatch.setattr(evaluation.DocumentCollection, "search", search)
    report = evaluation.evaluate()
    assert report["llm_calls"] == report["openai_api_calls"] == 0
    assert len(calls) == 58
    for split in report["splits"].values():
        summary = split["summary"]
        assert summary["case_count"] == 8 and summary["relevance_cases"] == 5
        assert summary["rescued_failures"] == 1 and summary["degraded_successes"] == 0
        assert summary["unnecessary_retries"] == 3
        assert summary["retry"]["search_calls"] == 13
        for method in ("single_pass", "single_pass_budget", "retry"):
            assert summary[method]["outside_scope_hits"] == 0
            for row in split["questions"]:
                assert len(row[method]["final_hits"]) <= 3
                assert row[method]["top_k"] == 3
        assert summary["single_pass"]["hit_rate_at_3"] == 0.4
        assert summary["retry"]["hit_rate_at_3"] == 0.6
        assert summary["single_pass_budget"]["hit_rate_at_3"] == 0.4
    rag_service.OpenAI.assert_not_called()


def test_full_evidence_is_separate_from_one_relevant_chunk():
    hits = [{"document_id": "a", "chunk_id": 1, "page_number": 1}]
    score = evaluation.score(hits, [("a", 1, 1), ("a", 2, 1)], ["a"])
    assert score["rank"] == 1 and not score["complete_evidence"]


def test_degraded_success_and_unnecessary_retry_metrics():
    report = evaluation.evaluate()
    row = deepcopy(report["splits"]["development"]["questions"][0])
    row["retry"]["complete_evidence"] = False
    summary = evaluation.summarize([row])
    assert summary["degraded_successes"] == 1
    assert summary["unnecessary_retries"] == 1


def test_disjoint_documents_and_families():
    data = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    assert len({split["family"] for split in data["splits"]}) == 2
    ids = [value for split in data["splits"] for value in (split["document_id"], split["excluded_id"])]
    assert len(set(ids)) == 4


def test_invalid_split_dataset(tmp_path):
    data = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    data["splits"][1]["family"] = data["splits"][0]["family"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="getrennt"):
        evaluation.evaluate(path)


def test_report_never_overwrites_existing_file(tmp_path, monkeypatch):
    output = tmp_path / "retry.json"
    assert evaluation.main(["--output", str(output)]) == 0
    original = output.read_bytes()
    run = Mock(side_effect=AssertionError("Nicht neu starten"))
    monkeypatch.setattr(evaluation, "evaluate", run)
    with pytest.raises(SystemExit):
        evaluation.main(["--output", str(output)])
    assert output.read_bytes() == original
    run.assert_not_called()
