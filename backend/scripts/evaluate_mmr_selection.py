"""Separate MMR-Evaluation; Lambda nur auf Entwicklung auswählen und einfrieren."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services import semantic_search
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline
from scripts.evaluate_query_retry import mmr_context
from scripts.evaluate_retrieval_modes import metrics, timing_summary

DATASET = ROOT / "data/evaluation/mmr_selection_cases.json"
LAMBDA_GRID = (0.25, 0.5, 0.75, 1.0)


def validate_dataset(data):
    """Prüft Dokument-/Familientrennung und jeden alternativen Textbeleg."""
    docs = {doc["document_id"]: doc for doc in data["documents"]}
    questions = data["questions"]
    if len(docs) != len(data["documents"]) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Dokument- und Frage-IDs müssen eindeutig sein.")
    families = [{q["family"] for q in questions if q["split"] == split} for split in ("development", "test")]
    if not all(families) or families[0] & families[1]:
        raise ValueError("Entwicklung und Test benötigen getrennte Fragefamilien.")
    for q in questions:
        if q["split"] not in ("development", "test") or not q["query"].strip():
            raise ValueError("Ungültige Frage oder Split.")
        known = {key for key, doc in docs.items() if doc["split"] == q["split"]}
        scope = q["document_ids"]
        if scope is not None and (not isinstance(scope, list) or not set(scope) <= known):
            raise ValueError("Dokumentauswahl verletzt den Split.")
        if not q["references"] and not q.get("absence_rationale"):
            raise ValueError("Fehlende Informationen benötigen eine Begründung.")
        for ref in q["references"]:
            if ref["document_id"] not in known or (scope is not None and ref["document_id"] not in scope):
                raise ValueError("Referenz verletzt den Dokumentbereich.")
            texts = docs[ref["document_id"]]["chunks"]
            if not ref["chunk_ids"] or not ref["evidence"] or any(type(index) is not int or not 1 <= index <= len(texts) or ref["evidence"] not in texts[index - 1] for index in ref["chunk_ids"]):
                raise ValueError("Nicht belegte Referenz.")


def evidence_score(hits, references):
    """Einzelne Treffer und alle benötigten Fakten getrennt messen."""
    def supports(hit, ref):
        return hit["document_id"] == ref["document_id"] and hit["chunk_id"] in ref["chunk_ids"] and hit["page_number"] == 1 and ref["evidence"] in hit["text"]
    if not references:
        return {"rank": None, "fact_coverage": None, "complete_evidence": None}
    matched = [any(supports(hit, ref) for hit in hits) for ref in references]
    return {"rank": next((rank for rank, hit in enumerate(hits, 1) if any(supports(hit, ref) for ref in references)), None),
            "fact_coverage": sum(matched) / len(matched), "complete_evidence": all(matched)}


def summarize(rows, method):
    valid = [row for row in rows if row["references"]]
    values = [row[method] for row in valid]
    baseline = [row["top3"] for row in valid]
    return {**metrics([value["rank"] for value in values]),
        "complete_evidence_cases": sum(value["complete_evidence"] for value in values),
        "complete_evidence_rate": sum(value["complete_evidence"] for value in values) / len(values),
        "mean_fact_coverage": sum(value["fact_coverage"] for value in values) / len(values),
        "improved_fact_coverage_cases": sum(value["fact_coverage"] > old["fact_coverage"] for value, old in zip(values, baseline)),
        "worsened_fact_coverage_cases": sum(value["fact_coverage"] < old["fact_coverage"] for value, old in zip(values, baseline)),
        "rescued_complete_cases": sum(value["complete_evidence"] and not old["complete_evidence"] for value, old in zip(values, baseline)),
        "lost_complete_cases": sum(old["complete_evidence"] and not value["complete_evidence"] for value, old in zip(values, baseline)),
        "outside_scope_hits": sum(row[method]["outside_scope_hits"] for row in rows),
        "missing_information_cases": sum(not row["references"] for row in rows),
        "missing_information_cases_with_returned_context": sum(not row["references"] and bool(row[method]["hits"]) for row in rows)}


def choose_lambda(development_scores):
    """Vorab festgelegte Priorität; niemals Testmetriken an diese Funktion geben."""
    return max(development_scores, key=lambda value: (
        development_scores[value]["complete_evidence_rate"],
        development_scores[value]["mean_fact_coverage"],
        development_scores[value]["mrr_at_3"], value))


def prepare(data, split, model):
    collection, vectors = DocumentCollection(), {}
    for doc in data["documents"]:
        if doc["split"] != split:
            continue
        chunks = [{"chunk_id": index, "page_number": 1, "text": text} for index, text in enumerate(doc["chunks"], 1)]
        if any(len(model.tokenizer(chunk["text"], truncation=False, verbose=False)["input_ids"]) > model.max_seq_length for chunk in chunks):
            raise ValueError("Chunk überschreitet das Tokenlimit.")
        embedded = semantic_search.embed_chunks(chunks)
        for chunk in embedded:
            vectors[(doc["document_id"], chunk["chunk_id"], 1)] = chunk["embedding"]
        collection.add_prepared(DocumentRetrievalPipeline.from_record({
            "metadata": {"document_id": doc["document_id"], "filename": doc["filename"], "page_count": 1, "chunk_count": len(embedded)},
            "chunks": embedded,
        }))
    return collection, vectors


def evaluate(dataset_path=DATASET, *, repeats=5):
    if type(repeats) is not int or not 2 <= repeats <= 50:
        raise ValueError("repeats muss zwischen 2 und 50 liegen.")
    raw = Path(dataset_path).read_bytes()
    data = json.loads(raw)
    validate_dataset(data)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tick = perf_counter()
    model = semantic_search._get_model()
    report = {"dataset_sha256": sha256(raw).hexdigest(), "embedding_model": semantic_search.MODEL_NAME,
        "device": str(model.device), "max_seq_length": model.max_seq_length, "model_load_ms": (perf_counter() - tick) * 1000,
        "versions": {name: version(name) for name in ("sentence-transformers", "torch")},
        "lambda_grid_development_only": LAMBDA_GRID,
        "selection_rule": "Maximiere vollständige Evidenzrate, dann mittlere Faktenabdeckung, dann MRR@3; bei Gleichstand höheres Lambda.",
        "mode": "semantic", "final_top_k": 3, "candidate_top_k": 6, "repeats": repeats,
        "llm_calls": 0, "openai_api_calls": 0, "splits": {},
        "limitations": "Neue, aber kleine synthetische Dokumente mit gleichartiger Struktur. Keine repräsentative oder statistisch robuste Generalisierung. Lambda ausschließlich auf Entwicklung festgelegt, Test nur einmal mit diesem Wert ausgewertet. Vorab bekanntes Lambda-Raster und Optimierungsziel; keine Testoptimierung. Rang 1 geschützt: Hit@1 kann durch MMR nicht steigen. Fehlende Informationen können trotz Kontext nicht erkannt werden. Keine Antwortgenerierung, Abstention-, Retry- oder produktive MMR-Integration. Zusätzliche Originalfrage-Embeddings für MMR sind in der Messzeit enthalten. Ein Warm-up je Frage/Methode, rotierende Wiederholungen, lokale Orientierung ohne Lasttest."
    }
    selected_lambda = None
    for split in ("development", "test"):
        tick = perf_counter()
        collection, vectors = prepare(data, split, model)
        embedding_ms = (perf_counter() - tick) * 1000
        rows = []
        tested_lambdas = LAMBDA_GRID if split == "development" else (selected_lambda,)
        for question in [q for q in data["questions"] if q["split"] == split]:
            scope = question["document_ids"]
            pool = collection.search(question["query"], mode="semantic", document_ids=scope, top_k=6)
            query_vector = model.encode([question["query"]], convert_to_numpy=True)[0] if pool else None
            row = {**question, "candidates": pool, "candidate_evidence": evidence_score(pool, question["references"])}
            def measure_hits(hits):
                return {"hits": hits, **evidence_score(hits, question["references"]),
                        "outside_scope_hits": sum(hit["document_id"] not in scope for hit in hits) if scope is not None else 0}
            row["top3"] = measure_hits(pool[:3])
            for value in tested_lambdas:
                selected = mmr_context(pool, vectors, query_vector, anchor=pool[0] if pool else None, lambda_value=value)
                row[f"lambda_{value}"] = measure_hits(selected)
            rows.append(row)
        development_scores = {value: summarize(rows, f"lambda_{value}") for value in tested_lambdas}
        if split == "development":
            selected_lambda = choose_lambda(development_scores)
            report["selected_lambda"] = selected_lambda
            report["lambda_frozen_before_test_embeddings"] = True
            report["development_lambda_comparison"] = {str(value): score for value, score in development_scores.items()}
        for row in rows:
            row["mmr"] = row[f"lambda_{selected_lambda}"]
        # Benchmark erst nach Lambda-Festlegung, ohne erneute Optimierung.
        for row in rows:
            def run(method):
                tick = perf_counter()
                pool = collection.search(row["query"], mode="semantic", document_ids=row["document_ids"], top_k=3 if method == "top3" else 6)
                if method == "mmr" and pool:
                    vector = model.encode([row["query"]], convert_to_numpy=True)[0]
                    pool = mmr_context(pool, vectors, vector, anchor=pool[0], lambda_value=selected_lambda)
                return pool, (perf_counter() - tick) * 1000
            for method in ("top3", "mmr"):
                hits, _ = run(method)
                if hits != row[method]["hits"]:
                    raise ValueError("Warm-up weicht vom Qualitätslauf ab.")
                row[method]["samples_ms"] = []
            for iteration in range(repeats):
                for method in (("top3", "mmr") if iteration % 2 == 0 else ("mmr", "top3")):
                    hits, duration = run(method)
                    if hits != row[method]["hits"]:
                        raise ValueError("Messung liefert abweichende Treffer.")
                    row[method]["samples_ms"].append(duration)
        summary = {method: summarize(rows, method) for method in ("top3", "mmr")}
        for method in summary:
            summary[method]["timing"] = timing_summary([duration for row in rows for duration in row[method]["samples_ms"]])
            summary[method]["search_calls_per_benchmark_pass"] = len(rows)
            summary[method]["extra_query_embeddings_per_benchmark_pass"] = sum(bool(row["candidates"]) for row in rows) if method == "mmr" else 0
        report["splits"][split] = {"question_count": len(rows), "document_embedding_ms": embedding_ms,
            "evaluated_lambdas": tested_lambdas, "summary": summary, "questions": rows}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, choices=range(2, 51), default=5)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "docs" / f"mmr_selection_local_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}.json"
    if output.exists():
        parser.error("Bericht existiert bereits. Bitte einen neuen Pfad wählen.")
    report = evaluate(repeats=args.repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Entwicklungs-Lambda: {report['selected_lambda']}")
    for split, result in report["splits"].items():
        print(split, json.dumps(result["summary"], ensure_ascii=False))
    print(f"Bericht: {output}; OpenAI-Aufrufe: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
