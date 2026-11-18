"""Testet Antwort-Evaluation ohne echte API-Aufrufe oder Modell-Downloads."""

import json
from unittest.mock import Mock

import pytest

from app.services import semantic_search
from evaluation.answer_metrics import assess_deterministic, pending_review, summarize
from scripts import evaluate_answers as evaluation


def case(answerable=True):
    return {
        "id": "test", "query": "Wie viel Urlaub?", "answerable": answerable,
        "expected_answer": "30 Urlaubstage.", "required_patterns": [r"\b30\b", "urlaub"] if answerable else [],
        "references": [{"filename": "test.pdf", "page_number": 1, "evidence": ["30 Urlaubstage"]}] if answerable else [],
        "context": [{"document_id": "doc", "filename": "test.pdf", "chunk_id": 1, "page_number": 1, "text": "30 Urlaubstage."}],
        "context_contains_expected_evidence": True if answerable else None,
    }


def answer(text="30 Urlaubstage."):
    return {"answer": text, "sources": [{"document_id": "doc", "filename": "test.pdf", "chunk_id": 1, "page_number": 1}]}


def test_fact_patterns_and_reference_validity():
    row = case()
    result = assess_deterministic(row, answer(), row["context"])
    assert result["correctness_fact_match_proxy"]
    assert result["citation_reference_validity"] == 1
    assert result["citation_expected_source_hit_proxy"]
    assert not assess_deterministic(row, answer("130 Urlaubstage."), row["context"])["correctness_fact_match_proxy"]
    wrong = answer()
    wrong["sources"][0]["document_id"] = "other"
    assert assess_deterministic(row, wrong, row["context"])["citation_reference_validity"] == 0


def test_valid_reference_is_not_semantic_citation_accuracy():
    row = case()
    row["context"][0]["text"] = "Ein thematisch unpassender Abschnitt."
    result = assess_deterministic(row, answer(), row["context"])
    assert result["citation_reference_validity"] == 1
    assert not result["citation_expected_source_hit_proxy"]
    assert pending_review()["citation_accuracy"]["score"] is None


def test_pattern_match_does_not_claim_full_correctness_or_faithfulness():
    row = case()
    result = assess_deterministic(row, answer("30 Urlaubstage sind falsch; tatsächlich gibt es 90."), row["context"])
    assert result["correctness_fact_match_proxy"]
    assert pending_review()["answer_correctness"]["score"] is None
    assert pending_review()["faithfulness"]["score"] is None


def test_abstention_and_missing_sources():
    refusal = {"answer": "Die Information wurde im Dokument nicht gefunden.", "sources": []}
    unknown = case(False)
    result = assess_deterministic(unknown, refusal, unknown["context"])
    assert result["correct_abstention_proxy"]
    assert result["citation_reference_validity"] is None
    assert result["correctness_fact_match_proxy"] is None
    known = case()
    assert assess_deterministic(known, refusal, known["context"])["unexpected_abstention"]
    assert not assess_deterministic(unknown, answer(refusal["answer"]), unknown["context"])["correct_abstention_proxy"]


def test_dry_run_never_calls_generator_or_reports_quality():
    forbidden = Mock(side_effect=AssertionError("Keine API-Aufrufe erlaubt"))
    report = evaluation.run_evaluation([case(), case(False)], generator=forbidden)
    forbidden.assert_not_called()
    assert report["api_calls_attempted"] == 0
    assert report["summary"]["completed_answers"] == 0
    assert report["summary"]["correctness_fact_match_proxy"] == {"value": None, "sample_count": 0}
    assert all(row["result"] is None for row in report["questions"])


@pytest.mark.parametrize("maximum", [0, 4, True, 1.5])
def test_live_call_limit_cannot_be_exceeded(maximum):
    with pytest.raises(ValueError):
        evaluation.run_evaluation([case()], mode="live", confirm_live=True, max_calls=maximum)


def test_live_needs_explicit_confirmation():
    with pytest.raises(ValueError, match="confirm-live"):
        evaluation.run_evaluation([case()], mode="live")


def test_missing_key_does_not_call_generator(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator = Mock()
    with pytest.raises(ValueError, match="Prozessumgebung"):
        evaluation.run_evaluation([case()], mode="live", confirm_live=True, generator=generator)
    generator.assert_not_called()


def test_mocked_live_is_bounded_and_cost_printed_first(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    def generate(query, context):
        if generator.call_count == 1:
            assert "Geschätzte Kosten" in capsys.readouterr().out
        return answer()
    generator = Mock(side_effect=generate)
    report = evaluation.run_evaluation([case() for _ in range(5)], mode="live", confirm_live=True, max_calls=3, generator=generator)
    assert generator.call_count == 3
    assert report["api_calls_attempted"] == 3
    assert report["summary"]["completed_answers"] == 3
    assert report["summary"]["correctness_fact_match_proxy"] == {"value": 1.0, "sample_count": 3}
    assert sum(row["status"].startswith("skipped") for row in report["questions"]) == 2


def test_error_stops_live_without_retry_or_leaking_details(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    generator = Mock(side_effect=RuntimeError("Interne vertrauliche Diagnose"))
    report = evaluation.run_evaluation([case(), case()], mode="live", confirm_live=True, generator=generator)
    generator.assert_called_once()
    assert report["summary"]["failed_answers"] == 1
    assert report["summary"]["completed_answers"] == 0
    assert "Interne vertrauliche Diagnose" not in json.dumps(report)


def test_pricing_estimate_arithmetic():
    cost = evaluation.estimate_cost([case(), case()])
    assert cost["planned_calls"] == 2
    assert cost["reserved_output_tokens"] == 2000
    assert cost["estimated_usd"] == pytest.approx((cost["estimated_input_tokens"] * 0.4 + 2000 * 1.6) / 1_000_000)
    assert cost["actual_cost_usd"] is None


def test_dataset_has_answerable_and_explicit_unknown_cases():
    cases = evaluation.load_cases()
    assert len(cases) >= 15
    assert sum(row["answerable"] for row in cases) == 20
    assert sum(not row["answerable"] for row in cases) == 4
    assert [row["id"] for row in evaluation.load_cases(["A01", "A11", "A21"])] == ["A01", "A11", "A21"]
    with pytest.raises(ValueError):
        evaluation.load_cases(["not-existing"])


def test_preparation_reuses_document_embeddings_without_network(monkeypatch, tokenizer):
    model = Mock(tokenizer=tokenizer, max_seq_length=128)
    model.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    forbidden = Mock(side_effect=AssertionError("Keine Antwortgenerierung im Dry-Run"))
    monkeypatch.setattr(evaluation, "generate_answer", forbidden)
    cases, documents = evaluation.prepare_cases(evaluation.load_cases())
    report = evaluation.run_evaluation(cases)
    assert len(cases) == 24 and len(documents) == 2
    assert model.encode.call_count == 2 + len(cases)
    assert report["summary"]["semantic_reviews_completed"] == 0
    forbidden.assert_not_called()


def test_cli_rejects_unconfirmed_live_before_model_loading(monkeypatch):
    prepare = Mock(side_effect=AssertionError("Nicht vorbereiten"))
    monkeypatch.setattr(evaluation, "prepare_cases", prepare)
    with pytest.raises(SystemExit):
        evaluation.main(["--mode", "live"])
    prepare.assert_not_called()


def test_cli_dry_run_writes_json_without_answer_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "prepare_cases", lambda cases: ([case()], []))
    output = tmp_path / "report.json"
    assert evaluation.main(["--mode", "dry-run", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["api_calls_attempted"] == 0
    assert report["summary"]["citation_reference_validity"]["value"] is None


def test_live_contract_with_mocked_sdk_never_sends_gold_answers(monkeypatch):
    """Prüft auch das bestehende SDK-Aufruflimit, ohne eine Verbindung aufzubauen."""
    from types import SimpleNamespace
    from app.services import rag_service

    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(
        status="completed", output_parsed=rag_service.MultiDocumentAnswer(
            answer="30 Urlaubstage.", sources=[rag_service.DocumentSource(document_id="doc", chunk_id=1, page_number=1)]
        ),
    )
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=client)
    constructor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(rag_service, "OpenAI", constructor)
    report = evaluation.run_evaluation([case()], mode="live", confirm_live=True)
    assert report["summary"]["completed_answers"] == 1
    assert constructor.call_args.kwargs["max_retries"] == 0
    request = client.responses.parse.call_args.kwargs
    assert request["max_output_tokens"] == 1000
    assert request["model"] == "gpt-4.1-mini"
    payload = json.loads(request["input"])
    assert set(payload) == {"query", "context"}
    assert "expected_answer" not in payload and "references" not in payload
    client.responses.parse.assert_called_once()
