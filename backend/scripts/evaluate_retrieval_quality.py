"""Offline-Strukturevaluation mit ausdrücklich kontrollierten Ranglisten-Fixtures."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.workflows.retrieval_quality import diagnose_retrieval

DATASET = ROOT / "data/evaluation/retrieval_quality_cases.json"


def build_cases(data):
    """Materialisiert getrennte Dokumentbestände und Fragefamilien, ohne Suche."""
    documents = data["documents"]
    if len({doc["document_id"] for doc in documents}) != len(documents):
        raise ValueError("Dokument-IDs müssen eindeutig sein.")
    splits = data["splits"]
    if {split["name"] for split in splits} != {"development", "test"} or len(splits) != 2:
        raise ValueError("Entwicklung und Test müssen getrennt vorliegen.")
    if len({split["family"] for split in splits}) != len(splits):
        raise ValueError("Fragefamilien dürfen Splits nicht überlappen.")
    templates = data["case_templates"]
    if len({template["id"] for template in templates}) != len(templates):
        raise ValueError("Fall-IDs müssen eindeutig sein.")
    cases = []
    kinds = ("relevant", "similar", "partial", "distractor", "blank", "missing_metadata")
    for split in splits:
        docs = [doc for doc in documents if doc["split"] == split["name"]]
        known = {doc["document_id"]: doc["filename"] for doc in docs}
        if split["primary_id"] not in known or split["other_id"] not in known:
            raise ValueError("Dokumentreferenzen verletzen den Split.")
        for template in templates:
            scope = template.get("scope")
            allowed = [] if scope == "empty" else [split["other_id"]] if scope == "other" else ["unknown"] if scope == "unknown" else [split["primary_id"]] if scope == "deleted" else None
            current_docs = [doc for doc in docs if scope != "deleted" or doc["document_id"] != split["primary_id"]]
            ranking = []
            for position, kind in enumerate(template["ranking"]):
                if kind not in kinds:
                    raise ValueError("Unbekannte Fixture-Art.")
                identifier = split["other_id"] if template["id"] == "excluded" else split["primary_id"]
                text = {"relevant": split["relevant_text"], "similar": split["similar_text"],
                        "partial": split["partial_text"], "distractor": "Ein synthetischer Abschnitt ohne gesuchte Angaben.",
                        "blank": " \n", "missing_metadata": split["partial_text"]}[kind]
                hit = {"text": text} if kind == "missing_metadata" else {
                    "document_id": identifier, "filename": known[identifier], "chunk_id": kinds.index(kind) + 1,
                    "page_number": 1, "text": text, "similarity_score": 0.9 - position * 0.1,
                }
                ranking.append(hit)
            rank = next((index for index, kind in enumerate(template["ranking"], 1) if kind == "relevant"), None)
            if not template.get("expected_error") and rank != template["expected_rank"]:
                raise ValueError("Fixture-Rang stimmt nicht mit Referenz überein.")
            cases.append({
                "id": f"{split['name']}-{template['id']}", "split": split["name"], "family": split["family"],
                "query": split["unanswerable_query"] if template["id"] == "unanswerable" else split["query"],
                "category": template["category"], "semantic_label": template["semantic_label"],
                "documents": current_docs, "document_ids": allowed, "full_ranking_fixture": ranking,
                "hits": ranking[:3], "full_relevant_rank": rank,
                "expected_status": template.get("expected_status"), "expected_error": template.get("expected_error", False),
            })
    return cases


def summarize(rows):
    """Trennt strukturelle Entscheidungen, erwartete Fehler und semantische Labels."""
    diagnosed = [row for row in rows if row["diagnosis"] is not None]
    usable = [row for row in diagnosed if any(isinstance(hit.get("text"), str) and hit["text"].strip() for hit in row["case"]["hits"])]
    violations = [row for row in rows if row["case"]["category"] == "scope_violation"]
    expected_errors = [row for row in rows if row["case"]["expected_error"]]
    false_empty = sum(row["diagnosis"]["status"] == "no_context" for row in usable)
    return {
        "case_count": len(rows), "correct_cases": sum(row["correct"] for row in rows),
        "category_coverage": dict(Counter(row["case"]["category"] for row in rows)),
        "diagnosis_counts": dict(Counter(row["diagnosis"]["status"] for row in diagnosed)),
        "expected_errors": len(expected_errors), "correctly_rejected_errors": sum(row["correct"] for row in expected_errors),
        "scope_violations_injected": len(violations), "scope_violations_rejected": sum(row["error"] is not None for row in violations),
        "false_no_context": false_empty, "usable_context_cases": len(usable),
        "false_no_context_rate": false_empty / len(usable) if usable else None,
        "unknown_by_semantic_label": dict(Counter(row["case"]["semantic_label"] for row in diagnosed if row["diagnosis"]["status"] == "unknown")),
        "sufficient_count": sum(row["diagnosis"]["status"] == "sufficient" for row in diagnosed),
        "diagnostic_ms": sum(row["elapsed_ms"] for row in rows),
    }


def evaluate(dataset_path=DATASET):
    raw = Path(dataset_path).read_bytes()
    cases = build_cases(json.loads(raw))
    rows = []
    for case in cases:
        tick = perf_counter()
        diagnosis, error = None, None
        try:
            diagnosis = diagnose_retrieval(case["hits"], documents=case["documents"], document_ids=case["document_ids"]).model_dump(mode="json")
        except ValueError:
            error = "invalid_selection_or_scope"
        correct = error is not None if case["expected_error"] else error is None and diagnosis["status"] == case["expected_status"]
        rows.append({"case": case, "diagnosis": diagnosis, "error": error, "correct": correct, "elapsed_ms": (perf_counter() - tick) * 1000})
    return {
        "dataset_sha256": sha256(raw).hexdigest(), "input_kind": "controlled_ranked_fixtures_not_model_retrieval",
        "top_k": 3, "llm_calls": 0, "openai_api_calls": 0, "retrieval_calls": 0,
        "summary": summarize(rows),
        "splits": {split: summarize([row for row in rows if row["case"]["split"] == split]) for split in ("development", "test")},
        "cases": rows,
        "limitations": "Reine Strukturdiagnostik auf kontrollierten synthetischen Ranglisten. Gold-Labels werden niemals an die Diagnose übergeben. Unknown umfasst relevante, irrelevante und unzureichende Texte. Kein validierter sufficient-Prüfer, keine Abstention-, Retry- oder Query-Entscheidung. Keine Aussage über semantische Beantwortbarkeit oder bessere Antwortqualität. Identische Fallkategorien mit getrennten Dokumenten und Familien sind Funktionstests, keine unabhängige statistische Generalisierung. Keine echten Modell-Rankings, PDFs, LLMs oder API-End-to-End-Tests. Laufzeiten sind lokale Einzelmessungen.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "docs" / f"retrieval_quality_offline_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}.json"
    if output.exists():
        parser.error("Bericht existiert bereits. Bitte neuen Dateinamen wählen.")
    report = evaluate()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Bericht: {output}; OpenAI-Aufrufe: 0")
    return 0 if report["summary"]["correct_cases"] == report["summary"]["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
