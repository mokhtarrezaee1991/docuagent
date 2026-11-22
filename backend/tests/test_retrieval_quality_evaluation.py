"""Prüft Diagnostikmetriken und klar getrennte semantische Referenzlabels."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from scripts import evaluate_retrieval_quality as evaluation


def test_fixture_evaluation_and_coverage():
    report = evaluation.evaluate()
    summary = report["summary"]
    assert summary["case_count"] == summary["correct_cases"] == 32
    assert summary["diagnosis_counts"] == {"unknown": 20, "no_context": 6}
    assert summary["false_no_context"] == summary["sufficient_count"] == 0
    assert summary["false_no_context_rate"] == 0
    assert summary["expected_errors"] == summary["correctly_rejected_errors"] == 6
    assert summary["scope_violations_injected"] == summary["scope_violations_rejected"] == 2
    assert len(summary["category_coverage"]) == 16
    assert all(count == 2 for count in summary["category_coverage"].values())
    assert report["llm_calls"] == report["openai_api_calls"] == report["retrieval_calls"] == 0
    assert all(split["case_count"] == 16 for split in report["splits"].values())


def test_unknown_is_not_a_semantic_answerability_decision():
    report = evaluation.evaluate()
    labels = report["summary"]["unknown_by_semantic_label"]
    assert labels["sufficient"] == 8
    assert labels["insufficient"] > 0 and labels["absent_from_documents"] > 0
    beyond = [row for row in report["cases"] if row["case"]["category"] == "relevant_beyond_top3"]
    assert all(row["case"]["full_relevant_rank"] == 4 and len(row["case"]["hits"]) == 3 for row in beyond)


def test_gold_labels_never_enter_diagnosis(monkeypatch):
    diagnose = Mock(wraps=evaluation.diagnose_retrieval)
    monkeypatch.setattr(evaluation, "diagnose_retrieval", diagnose)
    evaluation.evaluate()
    assert diagnose.call_count == 32
    assert all(set(call.kwargs) == {"documents", "document_ids"} and len(call.args) == 1 for call in diagnose.call_args_list)


def test_false_no_context_metric_detects_wrong_decision():
    row = deepcopy(evaluation.evaluate()["cases"][0])
    row["diagnosis"]["status"] = "no_context"
    summary = evaluation.summarize([row])
    assert summary["false_no_context"] == 1 and summary["false_no_context_rate"] == 1


def test_splits_have_separate_documents_and_families():
    report = evaluation.evaluate()
    groups = {split: [row["case"] for row in report["cases"] if row["case"]["split"] == split] for split in ("development", "test")}
    assert not {case["family"] for case in groups["development"]} & {case["family"] for case in groups["test"]}
    assert not {doc["document_id"] for case in groups["development"] for doc in case["documents"]} & {doc["document_id"] for case in groups["test"] for doc in case["documents"]}


def test_invalid_split_or_rank_references():
    data = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    broken = deepcopy(data)
    broken["splits"][1]["family"] = broken["splits"][0]["family"]
    with pytest.raises(ValueError, match="Fragefamilien"):
        evaluation.build_cases(broken)
    data["case_templates"][0]["expected_rank"] = 3
    with pytest.raises(ValueError, match="Rang"):
        evaluation.build_cases(data)


def test_cli_new_report_and_no_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "quality.json"
    assert evaluation.main(["--output", str(output)]) == 0
    before = output.read_bytes()
    run = Mock(side_effect=AssertionError("Nicht erneut ausführen"))
    monkeypatch.setattr(evaluation, "evaluate", run)
    with pytest.raises(SystemExit):
        evaluation.main(["--output", str(output)])
    run.assert_not_called()
    assert output.read_bytes() == before
