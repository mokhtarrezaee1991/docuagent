"""Prüft ausschließlich lokale manuelle Bewertungen mit künstlichen Antworten."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from evaluation.manual_reviews import aggregate_reviews, generated_cases, load_inputs, markdown_report, template, write_new
from scripts.review_answers import main


RUBRIC = Path(__file__).resolve().parents[2] / "data/evaluation/answer_review_rubric.json"


def report():
    return {"questions": [
        {"id": "A01", "query": "Frage eins?", "status": "completed", "answerable": True, "result": {"answer": "Künstliche Antwort eins.", "sources": []}},
        {"id": "A02", "query": "Frage zwei?", "status": "completed", "answerable": True, "result": {"answer": "Künstliche Antwort zwei.", "sources": []}},
        {"id": "A03", "query": "Nicht enthalten?", "status": "completed", "answerable": False, "result": {"answer": "Die Information wurde im Dokument nicht gefunden.", "sources": []}},
        {"id": "A04", "query": "Nicht generiert?", "status": "dry_run", "answerable": True, "result": None},
    ]}


def reviews():
    return template(report(), "source", "rubric", "Testreviewer")


def score(data, index, criterion, value):
    data["reviews"][index]["assessments"][criterion] = {"status": "scored", "score": value, "reason": "Künstliche Testbegründung mit Beleg."}


def test_template_only_contains_generated_answers_and_no_invented_scores():
    data = reviews()
    assert [row["question_id"] for row in data["reviews"]] == ["A01", "A02", "A03"]
    assert all(item["score"] is None for row in data["reviews"] for item in row["assessments"].values())
    assert data["reviews"][0]["assessments"]["abstention"]["status"] == "not_applicable"
    assert data["reviews"][2]["assessments"]["answer_correctness"]["status"] == "not_applicable"


def test_means_exclude_unreviewed_and_na_but_include_real_zero():
    data = reviews()
    score(data, 0, "answer_correctness", 2)
    score(data, 1, "answer_correctness", 0)
    score(data, 0, "faithfulness", 1)
    score(data, 2, "abstention", 2)
    data["reviews"][2]["assessments"]["citation_accuracy"] = {
        "status": "not_applicable", "score": None, "reason": "Verweigerung ohne Tatsachenbehauptungen."
    }
    result = aggregate_reviews(report(), data, "source", "rubric")
    assert result["criteria"]["answer_correctness"] == {"mean": 1.0, "reviewed_count": 2, "unreviewed_count": 0, "not_applicable_count": 1}
    assert result["criteria"]["faithfulness"]["mean"] == 1
    assert result["criteria"]["faithfulness"]["reviewed_count"] == 1
    assert result["criteria"]["faithfulness"]["unreviewed_count"] == 2
    assert result["criteria"]["citation_accuracy"]["mean"] is None
    assert result["criteria"]["citation_accuracy"]["not_applicable_count"] == 1
    assert result["criteria"]["abstention"]["mean"] == 2
    assert result["generated_cases"] == result["cases_with_scores"] == 3


def test_missing_reviews_or_criteria_are_unreviewed_not_zero():
    data = reviews()
    data["reviews"] = [{"question_id": "A01", "assessments": {}}]
    result = aggregate_reviews(report(), data, "source", "rubric")
    assert all(values["mean"] is None and values["reviewed_count"] == 0 and values["unreviewed_count"] == 3 for values in result["criteria"].values())
    assert result["cases_with_scores"] == 0


@pytest.mark.parametrize("value", [-1, 3, True, 1.0, "2", None])
def test_invalid_scores(value):
    data = reviews()
    score(data, 0, "faithfulness", value)
    with pytest.raises(ValueError, match="ganze Zahl"):
        aggregate_reviews(report(), data, "source", "rubric")


@pytest.mark.parametrize("status", ["unreviewed", "not_applicable"])
def test_score_on_unrated_status_is_rejected(status):
    data = reviews()
    score(data, 0, "faithfulness", 0)
    data["reviews"][0]["assessments"]["faithfulness"]["status"] = status
    with pytest.raises(ValueError, match="score: null"):
        aggregate_reviews(report(), data, "source", "rubric")


def test_missing_reason_and_reviewer_are_rejected():
    data = reviews()
    score(data, 0, "faithfulness", 2)
    data["reviews"][0]["assessments"]["faithfulness"]["reason"] = " "
    with pytest.raises(ValueError, match="Begründung"):
        aggregate_reviews(report(), data, "source", "rubric")
    score(data, 0, "faithfulness", 2)
    data["reviewer"] = ""
    with pytest.raises(ValueError, match="Reviewer"):
        aggregate_reviews(report(), data, "source", "rubric")


@pytest.mark.parametrize("field", ["source_report_sha256", "rubric_sha256"])
def test_stale_report_or_rubric_is_rejected(field):
    data = reviews()
    data[field] = "wrong"
    with pytest.raises(ValueError, match="SHA-256"):
        aggregate_reviews(report(), data, "source", "rubric")


def test_duplicate_unknown_and_not_generated_ids_are_rejected():
    for identifier in ("unknown", "A04", "A02"):
        data = reviews()
        data["reviews"][0]["question_id"] = identifier
        with pytest.raises(ValueError, match="Frage-ID"):
            aggregate_reviews(report(), data, "source", "rubric")


def test_wrong_criterion_and_inapplicable_score_are_rejected():
    data = reviews()
    data["reviews"][0]["assessments"]["typo"] = {"status": "unreviewed"}
    with pytest.raises(ValueError, match="unbekannte Kriterien"):
        aggregate_reviews(report(), data, "source", "rubric")
    for index, criterion in ((0, "abstention"), (2, "answer_correctness")):
        data = reviews()
        score(data, index, criterion, 0)
        with pytest.raises(ValueError, match="nicht anwendbar"):
            aggregate_reviews(report(), data, "source", "rubric")


def test_markdown_handles_zero_missing_values_and_escapes_cells():
    data = reviews()
    score(data, 0, "faithfulness", 0)
    data["reviews"][0]["assessments"]["faithfulness"]["reason"] = "Beleg | Zusatz\nneue Zeile"
    result = aggregate_reviews(report(), data, "source", "rubric")
    text = markdown_report(result, Path("report.json"))
    assert "| Faithfulness | 0.000 | 1 | 2 | 0 |" in text
    assert "nicht bewertet" in text
    assert "Beleg \\| Zusatz neue Zeile" in text


def test_dry_run_report_has_no_reviewable_answers():
    source = {"questions": [{"id": "A01", "status": "dry_run", "result": None}]}
    data = template(source, "source", "rubric")
    assert data["reviews"] == []
    result = aggregate_reviews(source, data, "source", "rubric")
    assert result["generated_cases"] == 0
    assert all(value["mean"] is None for value in result["criteria"].values())


def test_report_duplicate_id_and_missing_answer_rejected():
    source = report()
    source["questions"].append(deepcopy(source["questions"][0]))
    with pytest.raises(ValueError, match="eindeutige"):
        generated_cases(source)
    source = report()
    source["questions"][0]["result"] = None
    with pytest.raises(ValueError, match="keine gültige Antwort"):
        generated_cases(source)


def test_cli_never_overwrites_source_reviews_or_markdown(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(report()), encoding="utf-8")
    original = source.read_bytes()
    path = tmp_path / "reviews.json"
    output = tmp_path / "summary.md"
    base = ["--report", str(source), "--reviews", str(path), "--rubric", str(RUBRIC)]
    assert main(["init", *base, "--reviewer", "Testreviewer"]) == 0
    initial_reviews = path.read_bytes()
    assert main(["init", *base]) == 1
    assert path.read_bytes() == initial_reviews
    assert main(["summarize", *base, "--output", str(output)]) == 0
    original_markdown = output.read_bytes()
    assert main(["summarize", *base, "--output", str(output)]) == 1
    assert main(["summarize", *base, "--output", str(source)]) == 1
    assert output.read_bytes() == original_markdown
    assert source.read_bytes() == original
    assert path.read_bytes() == initial_reviews


def test_load_checks_embedded_rubric_and_fingerprints(tmp_path):
    source = tmp_path / "source.json"
    data = report()
    data["review_rubric"] = {"incorrect": True}
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stimmen nicht"):
        load_inputs(source, RUBRIC)
    del data["review_rubric"]
    source.write_text(json.dumps(data), encoding="utf-8")
    _, _, source_hash, rubric_hash = load_inputs(source, RUBRIC)
    assert source_hash == sha256(source.read_bytes()).hexdigest()
    assert rubric_hash == sha256(RUBRIC.read_bytes()).hexdigest()
