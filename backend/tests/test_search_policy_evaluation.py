"""Prüft Offline-Auswertung, echte Suchaufrufe und unveränderte Berichte."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from app.services import rag_service, semantic_search
from scripts import evaluate_search_policy as evaluation


@pytest.fixture(autouse=True)
def forbid_external_models(monkeypatch):
    monkeypatch.setattr(rag_service, "OpenAI", Mock(side_effect=AssertionError("Keine OpenAI-Aufrufe")))
    monkeypatch.setattr(semantic_search, "_get_model", Mock(side_effect=AssertionError("Kein Modell-Download")))


def test_real_policy_to_retrieval_chain_and_metrics(monkeypatch):
    calls = []
    original = evaluation.DocumentCollection.search
    def search(self, query, **kwargs):
        calls.append(kwargs)
        return original(self, query, **kwargs)
    monkeypatch.setattr(evaluation.DocumentCollection, "search", search)
    report = evaluation.evaluate()
    summary = report["summary"]
    assert summary["case_count"] == summary["routing_correct"] == 18
    assert summary["expected_errors"] == summary["correct_error_handling"] == 6
    assert summary["baseline"]["hit_rate_at_1"] == pytest.approx(4 / 9)
    assert summary["baseline"]["hit_rate_at_3"] == 1
    assert summary["policy"]["hit_rate_at_1"] == 1
    assert summary["policy"]["outside_scope_hits"] == 0
    assert summary["baseline"]["outside_scope_hits"] > 0
    assert summary["policy"]["relevance_cases"] == 9
    assert len(calls) == 18 + 12
    assert {call["mode"] for call in calls} == {"semantic", "hybrid", "hybrid_de"}
    assert any(call.get("document_ids") == [] for call in calls)
    assert report["llm_calls"] == report["openai_api_calls"] == report["external_api_calls"] == 0
    assert report["elapsed_ms"] >= 0
    rag_service.OpenAI.assert_not_called()


def test_ranks_use_document_chunk_and_page():
    case = {"relevant_ids": ["a"], "expected_ids": ["a"]}
    hits = [
        {"document_id": "a", "chunk_id": 2, "page_number": 1},
        {"document_id": "b", "chunk_id": 1, "page_number": 1},
        {"document_id": "a", "chunk_id": 1, "page_number": 1},
    ]
    assert evaluation.score_hits(hits, case) == {"relevant_rank": 3, "outside_scope_hits": 1, "hit_count": 3}


def test_reproducible_except_timing():
    def without_timing(report):
        result = deepcopy(report)
        result.pop("elapsed_ms")
        for method in ("baseline", "policy"):
            result["summary"][method].pop("total_ms")
        for row in result["cases"]:
            for method in ("baseline", "policy"):
                row[method].pop("elapsed_ms")
        return result
    assert without_timing(evaluation.evaluate()) == without_timing(evaluation.evaluate())


def test_summary_does_not_turn_missing_relevance_into_success():
    report = evaluation.evaluate()
    row = deepcopy(report["cases"][2])
    row["policy"].update(relevant_rank=None, hit_count=0)
    summary = evaluation.summarize([row])
    assert summary["policy"]["hit_rate_at_1"] == 0
    assert summary["policy"]["hit_rate_at_3"] == 0
    assert summary["policy"]["mrr_at_3"] == 0


def test_cli_writes_new_report(tmp_path):
    output = tmp_path / "report.json"
    assert evaluation.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["routing_accuracy"] == 1


def test_existing_report_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("Vorhandener Bericht", encoding="utf-8")
    run = Mock(side_effect=AssertionError("Nicht starten"))
    monkeypatch.setattr(evaluation, "evaluate", run)
    with pytest.raises(SystemExit):
        evaluation.main(["--output", str(output)])
    run.assert_not_called()
    assert output.read_text(encoding="utf-8") == "Vorhandener Bericht"


def test_invalid_references_are_rejected(tmp_path):
    dataset = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    dataset["cases"][0]["relevant_ids"] = ["erfunden"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError, match="Referenz"):
        evaluation.evaluate(path)
