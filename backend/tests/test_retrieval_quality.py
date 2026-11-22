"""Prüft konservative Strukturdiagnose ohne Modell oder Antwortgenerierung."""

from copy import deepcopy

import pytest

from app.workflows.retrieval_quality import diagnose_retrieval

DOCUMENTS = [{"document_id": "a", "filename": "gleich.pdf"}, {"document_id": "b", "filename": "gleich.pdf"}]


def hit(**changes):
    return {"document_id": "a", "filename": "gleich.pdf", "chunk_id": 1,
            "page_number": 1, "text": "Ein vorhandener Text.", "similarity_score": 0.1, **changes}


@pytest.mark.parametrize("chunks", [[], [hit(text="")], [hit(text=" \n")], [{"text": None}]])
def test_no_context_only_without_usable_text(chunks):
    result = diagnose_retrieval(chunks, documents=DOCUMENTS)
    assert result.status == "no_context" and result.reason == "no_usable_text"
    assert result.usable_text_count == 0


@pytest.mark.parametrize("score", [-1.0, 0, 1.0, float("nan"), None])
def test_existing_context_always_unknown_regardless_of_score(score):
    result = diagnose_retrieval([hit(similarity_score=score)], documents=DOCUMENTS)
    assert result.status == "unknown" and result.reason == "semantic_sufficiency_unassessed"


def test_duplicates_and_conflicting_text_are_reported_without_mutation():
    chunks = [hit(), hit(), hit(text="Anderer Text mit identischer Referenz.")]
    original = deepcopy(chunks)
    result = diagnose_retrieval(chunks, documents=DOCUMENTS)
    assert result.duplicate_count == 2 and result.usable_text_count == 3
    assert "duplicate_reference" in result.issues and "conflicting_duplicate_text" in result.issues
    assert chunks == original


def test_missing_metadata_does_not_destroy_text():
    result = diagnose_retrieval([{"text": "Ohne Zitiermetadaten"}], documents=DOCUMENTS)
    assert result.status == "unknown"
    assert "missing_or_invalid_document_id" in result.issues
    assert "missing_or_invalid_chunk_id" in result.issues


def test_empty_selection_remains_empty():
    assert diagnose_retrieval([], documents=DOCUMENTS, document_ids=[]).status == "no_context"
    with pytest.raises(ValueError, match="leeren"):
        diagnose_retrieval([hit()], documents=DOCUMENTS, document_ids=[])


def test_valid_scope_and_duplicate_ids():
    assert diagnose_retrieval([hit()], documents=DOCUMENTS, document_ids=["a", "a"]).status == "unknown"


@pytest.mark.parametrize("chunks,ids", [([hit()], ["b"]), ([{"text": "Ohne ID"}], ["a"]),
    ([hit(document_id="gelöscht")], None), ([], ["gelöscht"])])
def test_invalid_selection_or_scope_is_an_error(chunks, ids):
    with pytest.raises(ValueError):
        diagnose_retrieval(chunks, documents=DOCUMENTS, document_ids=ids)


def test_deleted_id_uses_current_metadata():
    with pytest.raises(ValueError, match="gelöscht"):
        diagnose_retrieval([hit()], documents=DOCUMENTS[1:], document_ids=["a"])


@pytest.mark.parametrize("chunks", [None, {}, [None]])
def test_invalid_hit_container(chunks):
    with pytest.raises(ValueError):
        diagnose_retrieval(chunks, documents=DOCUMENTS)


def test_bad_reference_fields_and_filename_are_diagnostic():
    result = diagnose_retrieval([hit(filename="falsch.pdf", page_number=True, chunk_id=0)], documents=DOCUMENTS)
    assert result.status == "unknown"
    assert {"filename_mismatch", "missing_or_invalid_page_number", "missing_or_invalid_chunk_id"} <= set(result.issues)
