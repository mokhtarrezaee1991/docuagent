"""Prüft Auswahl, Split-Isolation und Entwicklungs-Tuning ohne echte Modelle."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from app.services import rag_service, semantic_search
from scripts import evaluate_mmr_selection as evaluation
from scripts.evaluate_query_retry import mmr_context


@pytest.fixture
def model(monkeypatch):
    fake = Mock(device="cpu", max_seq_length=128)
    fake.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": list(range(len(text.split()) + 2))}
    fake.encode.side_effect = lambda texts, **kwargs: [[1.0, float(len(text) % 5) / 5] for text in texts]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: fake)
    monkeypatch.setattr(rag_service, "OpenAI", Mock(side_effect=AssertionError("Keine OpenAI-Aufrufe")))
    return fake


@pytest.mark.parametrize("value", [-1, 1.1, None, True, float("nan")])
def test_mmr_invalid_lambda(value):
    with pytest.raises(ValueError, match="lambda"):
        mmr_context([], {}, [1, 0], lambda_value=value)


def test_reusable_mmr_preserves_default_dedup_anchor_and_budget():
    hits = [{"document_id": "a", "chunk_id": index, "page_number": 1, "text": str(index)} for index in range(1, 5)]
    vectors = {("a", index, 1): vector for index, vector in enumerate(([0.95, 0.3], [0.9, 0.4], [0.8, 0.5], [0.8, -0.6]), 1)}
    default = mmr_context([hits[0], *hits], vectors, [1, 0], anchor=hits[0])
    assert default == mmr_context(hits, vectors, [1, 0], anchor=hits[0], lambda_value=0.5)
    assert default[0] == hits[0] and len(default) == 3
    assert hits[3] in default
    assert mmr_context([], vectors, [1, 0]) == []
    assert mmr_context(hits, vectors, [1, 0], anchor=hits[0], lambda_value=1.0) == hits[:3]


def test_multi_fact_evidence_and_missing_information():
    refs = [{"document_id": "a", "chunk_ids": [1, 2], "evidence": "Fakt A"},
            {"document_id": "a", "chunk_ids": [3], "evidence": "Fakt B"}]
    hits = [{"document_id": "a", "chunk_id": 2, "page_number": 1, "text": "Fakt A"}]
    result = evaluation.evidence_score(hits, refs)
    assert result["rank"] == 1 and result["fact_coverage"] == 0.5 and not result["complete_evidence"]
    hits.append({"document_id": "a", "chunk_id": 3, "page_number": 1, "text": "Fakt B"})
    assert evaluation.evidence_score(hits, refs)["complete_evidence"]
    assert evaluation.evidence_score(hits, [])["complete_evidence"] is None


def test_lambda_priority_is_predefined():
    candidates = {value: {"complete_evidence_rate": 0.5, "mean_fact_coverage": 0.7, "mrr_at_3": 0.9} for value in evaluation.LAMBDA_GRID}
    assert evaluation.choose_lambda(candidates) == 1.0
    candidates[0.25]["complete_evidence_rate"] = 1.0
    assert evaluation.choose_lambda(candidates) == 0.25


def test_full_local_chain_freezes_lambda_before_test_embeddings(model, monkeypatch):
    events = []
    original_prepare, original_choose = evaluation.prepare, evaluation.choose_lambda
    def prepare(data, split, model):
        events.append(split)
        return original_prepare(data, split, model)
    def choose(scores):
        events.append("freeze")
        assert set(scores) == set(evaluation.LAMBDA_GRID)
        return 0.75
    monkeypatch.setattr(evaluation, "prepare", prepare)
    monkeypatch.setattr(evaluation, "choose_lambda", choose)
    embed = Mock(wraps=semantic_search.embed_chunks)
    monkeypatch.setattr(semantic_search, "embed_chunks", embed)
    report = evaluation.evaluate(repeats=2)
    assert events == ["development", "freeze", "test"]
    assert report["selected_lambda"] == 0.75
    assert report["splits"]["test"]["evaluated_lambdas"] == (0.75,)
    assert report["splits"]["development"]["evaluated_lambdas"] == evaluation.LAMBDA_GRID
    assert embed.call_count == 4
    for split in report["splits"].values():
        assert split["question_count"] == 10
        for row in split["questions"]:
            for method in ("top3", "mmr"):
                value = row[method]
                assert len(value["hits"]) <= 3 and value["outside_scope_hits"] == 0
                if row["document_ids"] is not None:
                    assert all(hit["document_id"] in row["document_ids"] for hit in value["hits"])
            if row["top3"]["hits"]:
                assert row["top3"]["hits"][0] == row["mmr"]["hits"][0]
        assert split["summary"]["mmr"]["question_count"] == 7
        assert split["summary"]["mmr"]["timing"]["sample_count"] == 20
    assert report["openai_api_calls"] == 0
    rag_service.OpenAI.assert_not_called()


def test_document_isolation_empty_and_unknown_ids(model):
    data = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    collection, vectors = evaluation.prepare(data, "development", model)
    assert collection.search("Frage", document_ids=[]) == []
    for identifier in ("unknown", "test-boje"):
        with pytest.raises(ValueError):
            collection.search("Frage", document_ids=[identifier])
    hits = collection.search("Frage", top_k=6, document_ids=["dev-kurier"])
    selected = mmr_context(hits, vectors, [1, 0], anchor=hits[0])
    assert all(hit["document_id"] == "dev-kurier" for hit in selected)


@pytest.mark.parametrize("fault", ["family", "scope", "evidence"])
def test_invalid_dataset(fault):
    data = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    if fault == "family":
        data["questions"][-1]["family"] = data["questions"][0]["family"]
    elif fault == "scope":
        data["questions"][-1]["document_ids"] = ["dev-feld"]
    else:
        data["questions"][0]["references"][0]["evidence"] = "Nicht vorhanden"
    with pytest.raises(ValueError):
        evaluation.validate_dataset(data)


def test_new_questions_do_not_reuse_6d_queries():
    data = json.loads(evaluation.DATASET.read_text(encoding="utf-8"))
    old = json.loads((evaluation.ROOT / "data/evaluation/query_retry_cases.json").read_text(encoding="utf-8"))
    previous = {value for split in old["splits"] for key, value in split.items() if key.endswith("query")}
    assert not {q["query"] for q in data["questions"]} & previous


def test_report_does_not_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("Vorhandener Bericht", encoding="utf-8")
    run = Mock(side_effect=AssertionError("Nicht starten"))
    monkeypatch.setattr(evaluation, "evaluate", run)
    with pytest.raises(SystemExit):
        evaluation.main(["--output", str(output)])
    run.assert_not_called()
    assert output.read_text(encoding="utf-8") == "Vorhandener Bericht"
