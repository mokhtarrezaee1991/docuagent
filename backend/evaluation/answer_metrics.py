"""Deterministische Teilprüfungen, strikt getrennt von semantischen Bewertungen."""

import re

from evaluation.retrieval_metrics import matches_reference


def abstention_signal(answer: str) -> bool:
    """Erkennt typische Nichtwissen-Formulierungen; beweist keine korrekte Verweigerung."""
    text = " ".join(answer.casefold().split())
    return bool(re.search(r"nicht.{0,50}(gefunden|enthalten|angegeben|genannt|ersichtlich)|keine.{0,35}(information|angabe)", text))


def assess_deterministic(case: dict, result: dict, context: list[dict]) -> dict:
    """Prüft nur Muster und Referenzen, niemals vollständige semantische Richtigkeit."""
    answer = result["answer"]
    sources = result["sources"]
    key = lambda source: (source.get("document_id"), source.get("chunk_id"), source.get("page_number"))
    known = {key(chunk): chunk for chunk in context}
    valid = [source for source in sources if key(source) in known and source.get("filename") == known[key(source)].get("filename")]
    cited_chunks = [known[key(source)] for source in valid]
    pattern_matches = [bool(re.search(pattern, answer, re.IGNORECASE)) for pattern in case["required_patterns"]]
    signal = abstention_signal(answer)
    return {
        "correctness_fact_match_proxy": all(pattern_matches) and not signal if case["answerable"] else None,
        "required_pattern_matches": pattern_matches,
        "citation_reference_validity": len(valid) / len(sources) if sources else None,
        "citation_expected_source_hit_proxy": any(matches_reference(chunk, reference) for chunk in cited_chunks for reference in case["references"]) if case["answerable"] else None,
        "abstention_signal": signal,
        "correct_abstention_proxy": signal and not sources if not case["answerable"] else None,
        "unexpected_abstention": signal if case["answerable"] else None,
    }


def pending_review() -> dict:
    """Semantische Kriterien bleiben unbewertet statt aus Mustern abgeleitet zu werden."""
    return {"status": "pending_human_or_separately_authorized_llm_review", **{
        metric: {"score": None, "reason": None}
        for metric in ("answer_correctness", "faithfulness", "citation_accuracy", "abstention")
    }}


def summarize(rows: list[dict]) -> dict:
    """Gibt für jede Teilmetrik den Messwert und den tatsächlichen Nenner an."""
    summary = {
        "prepared_questions": len(rows),
        "completed_answers": sum(row["status"] == "completed" for row in rows),
        "failed_answers": sum(row["status"] == "error" for row in rows),
        "semantic_reviews_completed": 0,
    }
    for name in ("correctness_fact_match_proxy", "citation_reference_validity", "citation_expected_source_hit_proxy", "correct_abstention_proxy", "unexpected_abstention"):
        values = [row["deterministic"][name] for row in rows if row.get("deterministic") is not None and row["deterministic"][name] is not None]
        summary[name] = {"value": sum(values) / len(values) if values else None, "sample_count": len(values)}
    return summary
