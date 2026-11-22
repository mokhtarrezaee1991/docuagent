"""Evaluiert die echte Policy-Retrieval-Kette mit festen synthetischen Vektoren."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services import semantic_search
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline
from app.workflows.search_policy import select_search

DATASET = ROOT / "data/evaluation/search_policy_cases.json"


class FixtureEncoder:
    """Ersetzt nur die Frage-Einbettung; Ranking und BM25 bleiben produktiver Code."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return [self.vectors[text] for text in texts]


def build_collection(documents):
    """Stellt ausschließlich die synthetischen Vektoren im Arbeitsspeicher bereit."""
    collection = DocumentCollection()
    for document in documents:
        collection.add_prepared(DocumentRetrievalPipeline.from_record({
            "metadata": {"document_id": document["document_id"], "filename": document["filename"], "page_count": 1, "chunk_count": 1},
            "chunks": [{"chunk_id": 1, "page_number": 1, "text": document["text"], "embedding": document["embedding"]}],
        }))
    return collection


def score_hits(hits, case):
    """Bewertet synthetische Referenzen nach Dokument, Seite und Chunk."""
    relevant = set(case.get("relevant_ids", []))
    rank = next((rank for rank, hit in enumerate(hits, 1) if hit["document_id"] in relevant and hit["chunk_id"] == 1 and hit["page_number"] == 1), None)
    allowed = case.get("expected_ids")
    outside = sum(hit["document_id"] not in allowed for hit in hits) if allowed is not None else 0
    return {"relevant_rank": rank, "outside_scope_hits": outside, "hit_count": len(hits)}


def summarize(rows):
    """Zeigt Nenner explizit; Fehler- und Leerauswahlfälle zählen nicht als Relevanzfragen."""
    summary = {"case_count": len(rows), "routing_correct": sum(row["routing_correct"] for row in rows)}
    summary["routing_accuracy"] = summary["routing_correct"] / len(rows) if rows else None
    errors = [row for row in rows if "expected_error" in row["case"]]
    summary["expected_errors"] = len(errors)
    summary["correct_error_handling"] = sum(row["routing_correct"] for row in errors)
    for method in ("baseline", "policy"):
        relevant = [row for row in rows if row["case"].get("relevant_ids")]
        scoped = [row for row in rows if "expected_error" not in row["case"]]
        ranks = [row[method]["relevant_rank"] for row in relevant]
        summary[method] = {
            "relevance_cases": len(ranks),
            "hit_rate_at_1": sum(rank == 1 for rank in ranks) / len(ranks) if ranks else None,
            "hit_rate_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks) if ranks else None,
            "mrr_at_3": sum(1 / rank if rank else 0 for rank in ranks) / len(ranks) if ranks else None,
            "scope_cases": len(scoped),
            "outside_scope_hits": sum(row[method]["outside_scope_hits"] for row in scoped),
            "returned_hits": sum(row[method]["hit_count"] for row in scoped),
            "total_ms": sum(row[method]["elapsed_ms"] for row in rows),
        }
    return summary


def evaluate(dataset_path=DATASET):
    """Führt Policy vor tatsächlicher Suche aus, niemals auf einem Top-3-Replay."""
    started = perf_counter()
    dataset_bytes = Path(dataset_path).read_bytes()
    dataset = json.loads(dataset_bytes)
    ids = [document["document_id"] for document in dataset["documents"]]
    if len(set(ids)) != len(ids) or len({case["id"] for case in dataset["cases"]}) != len(dataset["cases"]):
        raise ValueError("Dokument- und Fall-IDs müssen eindeutig sein.")
    for case in dataset["cases"]:
        if case["query"] not in dataset["query_vectors"] or not set(case.get("relevant_ids", [])).issubset(ids):
            raise ValueError("Ungültige Frage oder synthetische Referenz.")
    encoder = FixtureEncoder(dataset["query_vectors"])
    rows = []
    # Temporärer lokaler Encoder: keine Änderung am produktiven Modell oder Index.
    with patch.object(semantic_search, "_get_model", return_value=encoder):
        for case in dataset["cases"]:
            collection = build_collection(dataset["documents"])
            for identifier in case.get("remove_ids", []):
                collection.remove_document(identifier)
            tick = perf_counter()
            baseline = collection.search(case["query"], mode="semantic", top_k=3)
            baseline_ms = (perf_counter() - tick) * 1000
            tick = perf_counter()
            plan, error, hits = None, None, []
            try:
                plan = select_search(collection.list_documents(), **case["options"])
                hits = collection.search(
                    case["query"], top_k=3, mode=plan.mode,
                    document_ids=None if plan.document_ids is None else list(plan.document_ids),
                )
            except ValueError as caught:
                error = str(caught)
            elapsed = (perf_counter() - tick) * 1000
            if "expected_error" in case:
                correct = error is not None and case["expected_error"] in error
            else:
                correct = error is None and plan is not None and plan.mode == case["expected_mode"] and (
                    None if plan.document_ids is None else list(plan.document_ids)
                ) == case["expected_ids"]
            rows.append({
                "id": case["id"], "case": case, "routing_correct": correct,
                "plan": plan.model_dump(mode="json") if plan else None, "error": error,
                "baseline": {**score_hits(baseline, case), "hits": baseline, "elapsed_ms": baseline_ms},
                "policy": {**score_hits(hits, case), "hits": hits, "elapsed_ms": elapsed},
            })
    return {
        "dataset_sha256": sha256(dataset_bytes).hexdigest(), "dataset_version": dataset["version"],
        "versions": {name: version(name) for name in ("rank-bm25", "snowballstemmer")},
        "embedding_model": "fixed_synthetic_fixture_vectors", "top_k": 3,
        "llm_calls": 0, "openai_api_calls": 0, "external_api_calls": 0,
        "query_embedding_calls": encoder.calls, "elapsed_ms": (perf_counter() - started) * 1000,
        "summary": summarize(rows), "cases": rows,
        "limitations": "Kontrollierte Funktionsprüfung mit manuell festgelegten synthetischen Vektoren, kein Beleg für bessere echte semantische Suche oder LLM-Antworten. Explizite Vorgaben sind kein autonomes Routing. Hybrid-Modus und Dokumentfilter können gleichzeitig variieren. Baseline ignoriert Auswahlvorgaben absichtlich; Treffer außerhalb des Sollbereichs sind Vergleichswerte, keine Baseline-Sicherheitslücke. MRR ist hier auf Top-3 begrenzt. Laufzeiten sind ungefähre lokale Messungen ohne Warm-up/Lasttest. Keine PDFs, Extraktion, LangGraph- oder FastAPI-End-to-End-Evaluation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "docs" / f"search_policy_offline_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}.json"
    if output.exists():
        parser.error("Der Bericht existiert bereits. Bitte einen neuen Ausgabepfad wählen.")
    report = evaluate()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"LLM-Aufrufe: {report['llm_calls']}; OpenAI-Aufrufe: {report['openai_api_calls']}; Laufzeit: {report['elapsed_ms']:.3f} ms")
    print(f"Bericht: {output}")
    return 0 if report["summary"]["routing_correct"] == report["summary"]["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
