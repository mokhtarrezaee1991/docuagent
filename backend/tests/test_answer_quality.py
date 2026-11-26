"""Prüft reine Offline-Auswertung; Testnoten sind keine menschlichen Reviews."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from evaluation import answer_quality as quality
from scripts import evaluate_answer_quality as cli


@pytest.fixture
def inputs():
    return quality.load_inputs(cli.DATASET, cli.FIXTURES)


def row(report, identifier):
    return next(item for item in report["snapshots"] if item["snapshot_id"] == identifier)


def test_dataset_and_fixture_coverage(inputs):
    data, fixtures, *_ = inputs
    assert len(data.documents) == 4 and len(data.cases) == 16 and len(fixtures.snapshots) == 32
    families = [{case.family for case in data.cases if case.split == split} for split in ("development", "test")]
    assert not families[0] & families[1]
    assert all(sum(case.split == split for case in data.cases) == 8 for split in ("development", "test"))
    assert all(snapshot.provenance == "synthetic_fixture" for snapshot in fixtures.snapshots)


def test_technical_results_are_not_semantic_ratings(inputs):
    report = quality.evaluate(*inputs)
    assert report["technical"] == {"valid_source_count": 23, "source_count": 28, "reference_validity": 23 / 28, "snapshots_without_sources": 6}
    assert report["snapshot_count"] == report["snapshots_pending_review"] == 32
    assert report["pending_review_units"] == 178
    assert report["facet_outcomes"] == {"pending": 48}
    for values in report["semantic"].values():
        assert values["status"] == "pending" and values["mean_score_0_to_2"] is None
        assert values["scored_count"] == values["not_applicable_count"] == 0
        assert values["pending_count"] == values["unit_count"]
    assert report["llm_calls"] == report["openai_api_calls"] == 0


@pytest.mark.parametrize("identifier", ["D01-faulty", "D05-faulty", "D03-faulty", "D06-faulty", "D07-faulty", "T03-faulty"])
def test_false_numbers_negations_omissions_and_irrelevant_sources_remain_pending(inputs, identifier):
    result = row(quality.evaluate(*inputs), identifier)
    # Auch technisch gültige Quellen können falsche oder unvollständige Antworten begleiten.
    assert result["technical"]["reference_validity"] == 1
    assert all(claim["correctness"]["status"] == "pending" for claim in result["semantic_review"]["claims"])


@pytest.mark.parametrize("identifier", ["D04-faulty", "D08-faulty", "T02-faulty", "T06-faulty", "T08-faulty"])
def test_invalid_sources_are_checked_against_actual_context(inputs, identifier):
    result = row(quality.evaluate(*inputs), identifier)
    assert result["technical"]["reference_validity"] == 0


def test_missing_sources_are_not_automatically_semantic_failure_or_success(inputs):
    result = row(quality.evaluate(*inputs), "T01-faulty")
    assert result["technical"]["source_count"] == 0
    assert result["technical"]["reference_validity"] is None
    assert result["technical"]["has_factual_claims_without_sources"]
    assert result["semantic_review"]["claims"][0]["citation_coverage"]["status"] == "pending"


def test_partial_answer_corpus_and_context_are_distinct(inputs):
    data = inputs[0]
    partial = next(case for case in data.cases if case.case_id == "D04")
    assert partial.facets[0].in_allowed_corpus and partial.facets[0].in_context
    assert partial.facets[1].in_allowed_corpus and not partial.facets[1].in_context
    assert partial.facets[1].expected_behavior == "abstain"
    empty = next(case for case in data.cases if case.case_id == "T08")
    assert empty.allowed_document_ids == [] and empty.context == []
    assert not empty.facets[0].in_allowed_corpus


def test_manual_annotations_have_explicit_units_and_true_zero_counts(inputs):
    reviews = quality.review_template(*inputs)
    reviews.reviewer = "UNIT-TEST-SYNTHETIC-ANNOTATOR"
    first = reviews.snapshots[0]
    first.claim_inventory_complete = True
    first.claims[0].correctness = quality.Assessment(status="scored", score=2, reason="Testannotation, kein menschliches Urteil.")
    first.claims[1].correctness = quality.Assessment(status="scored", score=0, reason="Test einer echten Nullnote.")
    first.sources[0].claim_ids = [first.claims[0].claim_id]
    first.sources[0].source_precision = quality.Assessment(status="scored", score=2, reason="Synthetischer Verweis für Test.")
    first.facets[0].completeness = quality.Assessment(status="scored", score=1, reason="Teilweise, Testnote.")
    first.facets[1].completeness = quality.Assessment(status="not_applicable", reason="Explizite N/A-Testannotation.")
    report = quality.evaluate(*inputs, reviews=reviews)
    metric = report["semantic"]["correctness"]
    assert metric["mean_score_0_to_2"] == 1 and metric["scored_count"] == 2
    assert metric["pending_count"] == 32
    assert report["semantic"]["completeness"]["scored_count"] == 1
    assert report["semantic"]["completeness"]["not_applicable_count"] == 1
    assert report["semantic"]["completeness"]["mean_score_0_to_2"] == 1


def test_partial_and_abstention_outcomes_require_explicit_annotation(inputs):
    reviews = quality.review_template(*inputs)
    reviews.reviewer = "UNIT-TEST-ONLY"
    mappings = {"D04-good": ("correct_partial_answer", "justified_abstention"), "T04-faulty": ("false_abstention", "justified_abstention")}
    for item in reviews.snapshots:
        if item.snapshot_id in mappings:
            for facet, outcome in zip(item.facets, mappings[item.snapshot_id]):
                facet.outcome = outcome
                facet.reason = "Explizites synthetisches Testurteil."
    report = quality.evaluate(*inputs, reviews=reviews)
    assert report["facet_outcomes"]["correct_partial_answer"] == 1
    assert report["facet_outcomes"]["false_abstention"] == 1
    assert report["facet_outcomes"]["justified_abstention"] == 2
    assert report["facet_outcomes"]["pending"] == 44


@pytest.mark.parametrize("field", ["dataset_sha256", "fixtures_sha256", "snapshot_sha256"])
def test_wrong_review_binding_is_rejected(inputs, field):
    reviews = quality.review_template(*inputs)
    if field == "snapshot_sha256":
        reviews.snapshots[0].snapshot_sha256 = "0" * 64
    else:
        setattr(reviews, field, "0" * 64)
    with pytest.raises(ValueError, match="gehört nicht"):
        quality.evaluate(*inputs, reviews=reviews)


@pytest.mark.parametrize("fault", ["unknown_claim", "unknown_source", "unknown_facet", "duplicate_review", "no_reviewer", "no_reason", "bad_claim_link"])
def test_invalid_manual_review_references(inputs, fault):
    reviews = quality.review_template(*inputs)
    item = reviews.snapshots[0]
    if fault == "unknown_claim":
        item.claims[0].claim_id = "unknown"
    elif fault == "unknown_source":
        item.sources[0].source_index = 99
    elif fault == "unknown_facet":
        item.facets[0].facet_id = "unknown"
    elif fault == "duplicate_review":
        reviews.snapshots.append(item.model_copy(deep=True))
    elif fault == "no_reviewer":
        item.claim_inventory_complete = True
    elif fault == "no_reason":
        reviews.reviewer = "UNIT-TEST"
        item.facets[0].outcome = "false_abstention"
    else:
        item.sources[0].claim_ids = ["unknown"]
    with pytest.raises(ValueError):
        quality.evaluate(*inputs, reviews=reviews)


@pytest.mark.parametrize("assessment", [{"status":"scored","score":True,"reason":"Test"},
    {"status":"pending","score":0}, {"status":"scored","score":2,"reason":""},
    {"status":"not_applicable","reason":""}, {"status":"scored","score":3,"reason":"Test"}])
def test_strict_assessment_validation(assessment):
    with pytest.raises(ValidationError):
        quality.Assessment.model_validate(assessment)


@pytest.mark.parametrize("fault", ["version_bool", "scope", "context_text", "family", "reference_quote", "availability", "unknown_field"])
def test_dataset_strict_validation(inputs, fault):
    data = inputs[0].model_dump()
    if fault == "version_bool":
        data["version"] = True
    elif fault == "scope":
        data["cases"][0]["allowed_document_ids"] = ["test-sport"]
    elif fault == "context_text":
        data["cases"][0]["context"][0]["text"] = "Manipuliert"
    elif fault == "family":
        data["cases"][-1]["family"] = data["cases"][0]["family"]
    elif fault == "reference_quote":
        data["cases"][0]["facets"][0]["corpus_evidence"][0]["quote"] = "Nicht belegt"
    elif fault == "availability":
        data["cases"][0]["facets"][0]["in_context"] = False
    else:
        data["cases"][0]["unexpected"] = "Test"
    with pytest.raises(ValidationError):
        quality.Dataset.model_validate(data)


def test_fixture_claims_must_belong_to_answer(inputs):
    snapshot = inputs[1].snapshots[0].model_dump()
    snapshot["claims"][0]["text"] = "Steht nicht in der Antwort."
    with pytest.raises(ValidationError):
        quality.Snapshot.model_validate(snapshot)


def test_cli_offline_report_and_pending_review_template(tmp_path):
    output, review = tmp_path / "report.json", tmp_path / "reviews.json"
    assert cli.main(["--output", str(output), "--init-review", str(review)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    template = quality.Reviews.model_validate_json(review.read_bytes())
    assert report["snapshots_pending_review"] == 32 and template.reviewer == ""
    second = tmp_path / "second.json"
    assert cli.main(["--output", str(second), "--reviews", str(review)]) == 0
    assert json.loads(second.read_text(encoding="utf-8"))["semantic"] == report["semantic"]
    for args in (["--output", str(output)], ["--output", str(tmp_path / "new.json"), "--init-review", str(review)]):
        with pytest.raises(SystemExit):
            cli.main(args)
