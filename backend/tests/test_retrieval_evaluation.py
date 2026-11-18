"""Prüft Metriken und Referenzen ohne Modell-Downloads oder LLM-Aufrufe."""

import json
from unittest.mock import Mock

import pytest

from app.services.pdf_reader import extract_pdf_text
from evaluation.retrieval_metrics import calculate_metrics, first_relevant_hit, matches_reference
from scripts.evaluate_retrieval import DATASET, SAMPLES, evaluate, validate_references


def test_metrics_known_ranks_and_missing_hits():
    result = calculate_metrics([1, 2, 3, 4, None])
    assert result["question_count"] == 5
    assert result["hit_rate_at_1"] == pytest.approx(0.2)
    assert result["hit_rate_at_3"] == pytest.approx(0.6)
    assert result["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 5)


def test_mrr_is_not_cut_at_three():
    assert calculate_metrics([5])["mrr"] == pytest.approx(0.2)
    assert calculate_metrics([5])["hit_rate_at_3"] == 0


def test_perfect_and_missing_results():
    assert calculate_metrics([1, 1])["mrr"] == 1
    assert calculate_metrics([None, None])["mrr"] == 0
    assert calculate_metrics([None])["hit_rate_at_1"] == 0


@pytest.mark.parametrize("ranks", [[], [0], [-1], [True], [1.5], ["1"]])
def test_invalid_ranks(ranks):
    with pytest.raises(ValueError):
        calculate_metrics(ranks)


def test_matching_requires_document_page_and_evidence():
    reference = {"filename": "test.pdf", "page_number": 2, "evidence": ["mindestens 14 Zeichen"]}
    hit = {"filename": "test.pdf", "page_number": 2, "text": "Passwörter: MINDESTENS 14\nZeichen verwenden."}
    assert matches_reference(hit, reference)
    assert not matches_reference({**hit, "filename": "anders.pdf"}, reference)
    assert not matches_reference({**hit, "page_number": 1}, reference)
    assert not matches_reference({**hit, "text": "Wie lang muss ein Passwort sein?"}, reference)


def test_first_rank_accepts_alternative_sources_and_does_not_reorder():
    references = [{"filename": name, "page_number": 1, "evidence": ["Beleg"]} for name in ("a.pdf", "b.pdf")]
    ranking = [
        {"filename": "a.pdf", "page_number": 1, "text": "Nur eine Frage", "similarity_score": 0.9},
        {"filename": "b.pdf", "page_number": 1, "text": "Beleg", "similarity_score": 0.5},
        {"filename": "a.pdf", "page_number": 1, "text": "Beleg", "similarity_score": 0.4},
    ]
    assert first_relevant_hit(ranking, references) == (2, ranking[1])
    assert first_relevant_hit([], references) == (None, None)
    assert ranking[0]["similarity_score"] == 0.9


def test_dataset_has_verified_answers_in_real_artificial_pdfs():
    questions = json.loads(DATASET.read_text(encoding="utf-8"))["questions"]
    assert len(questions) >= 15
    names = {ref["filename"] for question in questions for ref in question["references"]}
    assert len(names) == 2
    documents = {name: extract_pdf_text((SAMPLES / name).read_bytes()) for name in names}
    validate_references(questions, documents)
    invalid = [{**questions[0], "references": [{"filename": "fehlt.pdf", "page_number": 1, "evidence": ["erfunden"]}]}]
    with pytest.raises(ValueError, match="nicht im PDF-Text"):
        validate_references(invalid, documents)


def test_evaluator_indexes_once_and_searches_full_ranking(monkeypatch):
    """Die Ablaufprüfung benötigt weder Server, SQLite noch ein echtes Modell."""
    from scripts import evaluate_retrieval as evaluation

    questions = json.loads(DATASET.read_text(encoding="utf-8"))["questions"]
    collection = Mock()
    counter = iter(range(2))
    collection.add_prepared.side_effect = lambda pipeline: {"document_id": str(next(counter)), "filename": pipeline.filename, "chunk_count": 10}
    def search(query, top_k):
        assert top_k == 20
        question = next(question for question in questions if question["query"] == query)
        ref = question["references"][0]
        return [{"document_id": "test", "filename": ref["filename"], "page_number": ref["page_number"], "chunk_id": 1,
                 "text": " ".join(ref["evidence"]), "similarity_score": 0.123}]
    collection.search.side_effect = search
    pipeline_class = Mock()
    monkeypatch.setattr(evaluation, "DocumentCollection", lambda: collection)
    monkeypatch.setattr(evaluation, "DocumentRetrievalPipeline", pipeline_class)
    report = evaluate()
    assert pipeline_class.call_count == 2
    assert pipeline_class.return_value.load_pdf.call_count == 2
    assert collection.search.call_count == len(questions)
    assert report["metrics"]["hit_rate_at_1"] == 1
    assert all(row["similarity_score"] == 0.123 for row in report["questions"])
