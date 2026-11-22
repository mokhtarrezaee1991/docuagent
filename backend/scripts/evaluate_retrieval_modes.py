"""Unabhängiger Offline-Vergleich fester Modi und einer reinen Fallback-Policy."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from math import isfinite
from pathlib import Path
import sys
from time import perf_counter
from statistics import median
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services import semantic_search
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline
from app.workflows.retrieval_mode_policy import choose_mode

DATASET = ROOT / "data/evaluation/retrieval_mode_cases.json"
MODES = ("semantic", "hybrid", "hybrid_de")
SPLITS = ("development", "test")


def validate_dataset(data):
    """Prüft Split-Grenzen und Referenzen vor jedem Ranking."""
    documents = data["documents"]
    questions = data["questions"]
    by_id = {document["document_id"]: document for document in documents}
    if len(by_id) != len(documents) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Dokument- und Frage-IDs müssen eindeutig sein.")
    families = {split: {q["family"] for q in questions if q["split"] == split} for split in SPLITS}
    if not all(families.values()) or families["development"] & families["test"]:
        raise ValueError("Fragefamilien müssen vollständig zwischen Entwicklung und Test getrennt sein.")
    queries = {split: {q["query"] for q in questions if q["split"] == split} for split in SPLITS}
    if queries["development"] & queries["test"]:
        raise ValueError("Identische Fragen dürfen nicht beide Splits betreffen.")
    for document in documents:
        if document["split"] not in SPLITS:
            raise ValueError("Ungültiger Dokument-Split.")
        if len({chunk["chunk_id"] for chunk in document["chunks"]}) != len(document["chunks"]):
            raise ValueError("Chunk-IDs müssen innerhalb eines Dokuments eindeutig sein.")
    for question in questions:
        if question["split"] not in SPLITS or not question["query"].strip() or not question["references"]:
            raise ValueError("Ungültige Frage oder fehlende Referenz.")
        available = {key for key, doc in by_id.items() if doc["split"] == question["split"]}
        scope = question["document_ids"]
        if scope is not None and (not isinstance(scope, list) or not set(scope).issubset(available)):
            raise ValueError("Dokumentauswahl verletzt den Split.")
        for reference in question["references"]:
            if reference["document_id"] not in available or (scope is not None and reference["document_id"] not in scope):
                raise ValueError("Referenz verletzt Split oder Suchbereich.")
            chunks = by_id[reference["document_id"]]["chunks"]
            if not reference["evidence"] or not any(
                chunk["chunk_id"] == reference["chunk_id"] and chunk["page_number"] == reference["page_number"]
                and reference["evidence"] in chunk["text"] for chunk in chunks
            ):
                raise ValueError("Referenz ist im synthetischen Dokument nicht belegt.")
    vectors = [chunk["embedding"] for doc in documents for chunk in doc["chunks"]] + [q["vector"] for q in questions]
    if any(len(vector) != 3 or not all(type(value) in (int, float) and isfinite(value) for value in vector) or not any(vector) for vector in vectors):
        raise ValueError("Fixtures benötigen endliche, nichtleere dreidimensionale Vektoren.")


def rank_relevant(hits, references):
    """Verlangt Dokument, Chunk, Seite und einen vorhandenen Textbeleg."""
    return next((rank for rank, hit in enumerate(hits, 1) if any(
        all(hit[key] == reference[key] for key in ("document_id", "chunk_id", "page_number"))
        and reference["evidence"] in hit["text"] for reference in references
    )), None)


def metrics(ranks):
    """MRR@3 zählt fehlende und außerhalb Top-3 liegende Treffer als null."""
    if not ranks or any(rank is not None and (type(rank) is not int or rank <= 0) for rank in ranks):
        raise ValueError("Ränge müssen positive ganze Zahlen oder None sein.")
    return {
        "question_count": len(ranks), "hit_rate_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "hit_rate_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "mrr_at_3": sum(1 / rank if rank is not None and rank <= 3 else 0 for rank in ranks) / len(ranks),
    }


def compare_ranks(ranks):
    """Gleich gute Modi sind gleichermaßen akzeptabel, kein erzwungener Gewinner."""
    utility = {mode: 1 / rank if rank is not None and rank <= 3 else 0 for mode, rank in ranks.items()}
    best = max(utility.values())
    baseline = utility["semantic"]
    return {
        "best_modes": [mode for mode, value in utility.items() if value == best],
        "versus_semantic": {mode: "better" if value > baseline else "worse" if value < baseline else "tie" for mode, value in utility.items()},
    }


class FixtureEncoder:
    """Nur der lokale Frage-Encoder wird durch unveränderte Testvektoren ersetzt."""

    def __init__(self, questions):
        self.vectors = {q["query"]: q["vector"] for q in questions}

    def encode(self, texts, **kwargs):
        return [self.vectors[text] for text in texts]


def evaluate(dataset_path=DATASET, *, embedding_mode="fixtures", repeats=5):
    if embedding_mode == "local":
        return evaluate_local(dataset_path, repeats=repeats)
    if embedding_mode != "fixtures":
        raise ValueError("embedding_mode muss fixtures oder local sein.")
    raw = Path(dataset_path).read_bytes()
    data = json.loads(raw)
    validate_dataset(data)
    report = {
        "dataset_sha256": sha256(raw).hexdigest(), "dataset_version": data["version"],
        "embedding_model": "fixed_synthetic_fixture_vectors", "top_k": 3,
        "versions": {name: version(name) for name in ("rank-bm25", "snowballstemmer")},
        "llm_calls": 0, "openai_api_calls": 0, "splits": {},
        "limitations": "Manuelle synthetische Vektoren statt echter Modell-Embeddings; keine Aussage über reale semantische Qualität. Keine Regeln, kein Training, keine Auswahl anhand von Testreferenzen. Policy bleibt deaktiviert oder ohne validierte Regel beim festen Fallback und kann ihn nicht verbessern. Gleichstände beziehen sich auf relevanten Rang/MRR@3, nicht auf veränderte Rohscores. Bestmodi sind nur nachträgliche Diagnostik. Kleine Splits, keine statistische Generalisierung. Laufzeiten sind Einzelmessungen ohne Warm-up; keine belastbare Performance-Aussage. Kein PDF-, API- oder LLM-End-to-End-Test.",
    }
    for split in SPLITS:
        questions = [q for q in data["questions"] if q["split"] == split]
        collection = DocumentCollection()
        documents = [doc for doc in data["documents"] if doc["split"] == split]
        for doc in documents:
            collection.add_prepared(DocumentRetrievalPipeline.from_record({
                "metadata": {"document_id": doc["document_id"], "filename": doc["filename"], "page_count": 2, "chunk_count": len(doc["chunks"])},
                "chunks": doc["chunks"],
            }))
        rows = []
        with patch.object(semantic_search, "_get_model", return_value=FixtureEncoder(questions)):
            for question in questions:
                results = {}
                def search(mode):
                    tick = perf_counter()
                    hits = collection.search(question["query"], top_k=3, mode=mode, document_ids=question["document_ids"])
                    allowed = question["document_ids"]
                    return {"rank": rank_relevant(hits, question["references"]), "hits": hits,
                            "outside_scope_hits": sum(hit["document_id"] not in allowed for hit in hits) if allowed is not None else 0,
                            "elapsed_ms": (perf_counter() - tick) * 1000}
                for mode in MODES:
                    results[mode] = search(mode)
                comparisons = compare_ranks({mode: results[mode]["rank"] for mode in MODES})
                for enabled in (False, True):
                    for fallback in MODES:
                        tick = perf_counter()
                        decision = choose_mode(question["query"], fallback_mode=fallback, enabled=enabled)
                        routing_ms = (perf_counter() - tick) * 1000
                        result = search(decision.mode)
                        result.update(decision=decision.model_dump(), routing_ms=routing_ms,
                                      routing_correct=decision.mode == fallback and decision.used_fallback and decision.routing_reason == ("no_validated_rule" if enabled else "automation_disabled"),
                                      best_mode_selected=decision.mode in comparisons["best_modes"])
                        results[f"policy_{'enabled' if enabled else 'disabled'}_{fallback}"] = result
                rows.append({"id": question["id"], "query": question["query"], "family": question["family"],
                             "document_ids": question["document_ids"], "references": question["references"],
                             **comparisons, "methods": results})
        summary = {}
        for method in rows[0]["methods"]:
            values = [row["methods"][method] for row in rows]
            summary[method] = {**metrics([value["rank"] for value in values]),
                               "outside_scope_hits": sum(value["outside_scope_hits"] for value in values),
                               "search_ms": sum(value["elapsed_ms"] for value in values)}
            if method.startswith("policy_"):
                summary[method].update(routing_accuracy=sum(value["routing_correct"] for value in values) / len(values),
                    fallback_rate=sum(value["decision"]["used_fallback"] for value in values) / len(values),
                    best_mode_selection_rate=sum(value["best_mode_selected"] for value in values) / len(values),
                    routing_ms=sum(value["routing_ms"] for value in values))
        report["splits"][split] = {"document_ids": [doc["document_id"] for doc in documents], "question_count": len(rows), "summary": summary, "questions": rows}
    return report


def timing_summary(samples):
    """Median und empirisches P95; keine Annahme einer Normalverteilung."""
    ordered = sorted(samples)
    return {"sample_count": len(samples), "median_ms": median(samples),
            "p95_ms": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]}


def evaluate_local(dataset_path=DATASET, *, repeats=5):
    """Verwendet das vorhandene CPU-Modell ausschließlich aus dem lokalen Cache."""
    if type(repeats) is not int or not 2 <= repeats <= 50:
        raise ValueError("repeats muss eine ganze Zahl zwischen 2 und 50 sein.")
    raw = Path(dataset_path).read_bytes()
    data = json.loads(raw)
    validate_dataset(data)
    # Offline erzwingen, ohne .env oder andere Konfigurationsdateien zu öffnen.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tick = perf_counter()
    model = semantic_search._get_model()
    load_ms = (perf_counter() - tick) * 1000
    report = {
        "dataset_sha256": sha256(raw).hexdigest(), "dataset_version": data["version"],
        "embedding_mode": "local", "embedding_model": semantic_search.MODEL_NAME,
        "device": str(model.device), "max_seq_length": model.max_seq_length,
        "versions": {name: version(name) for name in ("sentence-transformers", "torch", "rank-bm25", "snowballstemmer")},
        "model_load_ms": load_ms, "top_k": 3, "repeats": repeats,
        "warmup_runs_per_question_method": 1, "llm_calls": 0, "openai_api_calls": 0,
        "splits": {},
        "limitations": "Echte lokale Embeddings, aber nur sechs Fragen je Split auf sehr kurzen synthetischen Chunks. Bekannte Fixtures sind kein unberührter Nachweis einer neuen Auswahlregel. Keine automatische Auswahl und keine Modellanpassung. Deaktivierte Policy muss ihrem Fallback entsprechen. Suchzeit enthält Frage-Embedding, Cosinus/BM25/RRF und gegebenenfalls Aufbau des gefilterten BM25-Index, keine Dokument-Embeddings. Ein Warm-up je Frage/Methode; rotierende Methodenreihenfolge. Median/P95 sind lokale Orientierung ohne Lasttest. Modellladezeit ist ein einzelner Aufruf des gecachten Loaders, kein garantierter Kaltstart. Kein PDF-, API- oder LLM-End-to-End-Test.",
    }
    for split in SPLITS:
        questions = [q for q in data["questions"] if q["split"] == split]
        documents = [doc for doc in data["documents"] if doc["split"] == split]
        collection = DocumentCollection()
        tick = perf_counter()
        for doc in documents:
            chunks = [{key: chunk[key] for key in ("chunk_id", "page_number", "text")} for chunk in doc["chunks"]]
            if any(len(model.tokenizer(chunk["text"], truncation=False, verbose=False)["input_ids"]) > model.max_seq_length for chunk in chunks):
                raise ValueError("Ein Evaluations-Chunk überschreitet das Tokenlimit; keine stille Kürzung erlaubt.")
            embedded = semantic_search.embed_chunks(chunks)
            collection.add_prepared(DocumentRetrievalPipeline.from_record({
                "metadata": {"document_id": doc["document_id"], "filename": doc["filename"], "page_count": 2, "chunk_count": len(embedded)},
                "chunks": embedded,
            }))
        preparation_ms = (perf_counter() - tick) * 1000
        rows = []
        methods = list(MODES) + [f"policy_disabled_{mode}" for mode in MODES]
        for question in questions:
            results = {}
            for method in methods:
                policy = method.startswith("policy_disabled_")
                mode = method.removeprefix("policy_disabled_") if policy else method
                tick = perf_counter()
                decision = choose_mode(question["query"], fallback_mode=mode) if policy else None
                routing_ms = (perf_counter() - tick) * 1000 if policy else 0
                selected = decision.mode if decision else mode
                def search():
                    return collection.search(question["query"], top_k=3, mode=selected, document_ids=question["document_ids"])
                # Der erste Suchlauf wärmt Modell und den jeweiligen BM25-Pfad auf.
                hits = search()
                allowed = question["document_ids"]
                results[method] = {
                    "mode": selected, "rank": rank_relevant(hits, question["references"]), "hits": hits,
                    "outside_scope_hits": sum(hit["document_id"] not in allowed for hit in hits) if allowed is not None else 0,
                    "search_samples_ms": [], "routing_ms": routing_ms,
                }
                if decision:
                    results[method]["decision"] = decision.model_dump()
            # Rotieren statt alle Wiederholungen eines Verfahrens hintereinander.
            for iteration in range(repeats):
                for method in methods[iteration % len(methods):] + methods[:iteration % len(methods)]:
                    tick = perf_counter()
                    hits = collection.search(question["query"], top_k=3, mode=results[method]["mode"], document_ids=question["document_ids"])
                    elapsed_ms = (perf_counter() - tick) * 1000
                    if hits != results[method]["hits"]:
                        raise ValueError("Suchergebnisse variieren zwischen Warm-up und Messläufen.")
                    results[method]["search_samples_ms"].append(elapsed_ms)
            for method, value in results.items():
                value["timing"] = timing_summary(value["search_samples_ms"])
                if method.startswith("policy_disabled_") and value["hits"] != results[value["mode"]]["hits"]:
                    raise ValueError("Deaktivierte Policy weicht von ihrem festen Fallback ab.")
            rows.append({"id": question["id"], "query": question["query"], "family": question["family"],
                         "document_ids": question["document_ids"], "references": question["references"],
                         **compare_ranks({mode: results[mode]["rank"] for mode in MODES}), "methods": results})
        summary = {}
        for method in methods:
            values = [row["methods"][method] for row in rows]
            summary[method] = {**metrics([value["rank"] for value in values]),
                               "outside_scope_hits": sum(value["outside_scope_hits"] for value in values),
                               "timing": timing_summary([sample for value in values for sample in value["search_samples_ms"]])}
            if method.startswith("policy_disabled_"):
                summary[method].update(fallback_rate=sum(value["decision"]["used_fallback"] for value in values) / len(values),
                    routing_accuracy=sum(value["decision"]["mode"] == value["mode"] and value["decision"]["routing_reason"] == "automation_disabled" for value in values) / len(values))
        report["splits"][split] = {"document_ids": [doc["document_id"] for doc in documents],
                                  "question_count": len(rows), "document_embedding_ms": preparation_ms,
                                  "summary": summary, "questions": rows}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--embeddings", choices=("fixtures", "local"), default="fixtures")
    parser.add_argument("--repeats", type=int, choices=range(2, 51), default=5)
    args = parser.parse_args(argv)
    prefix = "retrieval_modes_local" if args.embeddings == "local" else "retrieval_modes_offline"
    output = args.output or ROOT / "docs" / f"{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}.json"
    if output.exists():
        parser.error("Bericht existiert bereits. Bitte neuen Ausgabepfad wählen.")
    report = evaluate() if args.embeddings == "fixtures" else evaluate(embedding_mode="local", repeats=args.repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for split, result in report["splits"].items():
        print(split, json.dumps(result["summary"], ensure_ascii=False))
    print(f"LLM/OpenAI-Aufrufe: 0; Bericht: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
