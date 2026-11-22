"""Offline-Experiment: manuelle Alternativen, echte Suche, feste Testvektoren."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services import semantic_search
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline
from app.workflows.query_retry_experiment import normalize_query, run_query_retry, select_context
from scripts.evaluate_retrieval_modes import metrics
from scripts.evaluate_retrieval_modes import timing_summary

DATASET = ROOT / "data/evaluation/query_retry_cases.json"


def prepare_split(data, split):
    """Baut nur den synthetischen Dokumentbestand dieses Splits auf."""
    if len(split["texts"]) != len(data["vectors"]):
        raise ValueError("Text- und Vektoranzahl passen nicht zusammen.")
    collection = DocumentCollection()
    for identifier, filename, texts, vectors in [
        (split["document_id"], split["filename"], split["texts"], data["vectors"]),
        (split["excluded_id"], split["excluded_filename"], [split["excluded_text"]], [[1, 0]]),
    ]:
        collection.add_prepared(DocumentRetrievalPipeline.from_record({
            "metadata": {"document_id": identifier, "filename": filename, "page_count": 1, "chunk_count": len(texts)},
            "chunks": [{"chunk_id": index, "page_number": 1, "text": text, "embedding": vector}
                       for index, (text, vector) in enumerate(zip(texts, vectors), 1)],
        }))
    return collection


def score(hits, references, allowed):
    """Hit-Metriken: erster relevanter Chunk; Vollständigkeit separat ausweisen."""
    keys = {(hit["document_id"], hit["chunk_id"], hit["page_number"]) for hit in hits}
    relevant = set(references)
    rank = next((index for index, hit in enumerate(hits, 1) if (hit["document_id"], hit["chunk_id"], hit["page_number"]) in relevant), None)
    return {"rank": rank, "complete_evidence": bool(relevant) and relevant <= keys,
            "outside_scope_hits": sum(hit["document_id"] not in allowed for hit in hits)}


def summarize(rows):
    """Nenner trennen: Relevanz nur für Fälle mit erlaubten Goldreferenzen."""
    answerable = [row for row in rows if row["references"]]
    summary = {"case_count": len(rows), "relevance_cases": len(answerable),
               "retry_count": sum(row["retry"]["retry_count"] for row in rows),
               "rescued_failures": sum(not row["single_pass"]["complete_evidence"] and row["retry"]["complete_evidence"] for row in answerable),
               "degraded_successes": sum(row["single_pass"]["complete_evidence"] and not row["retry"]["complete_evidence"] for row in answerable),
               "unnecessary_retries": sum(row["retry"]["retry_count"] and (row["single_pass"]["complete_evidence"] or not row["references"]) for row in rows),
               "retry_reasons": dict(Counter(row["retry"]["reason"] for row in rows))}
    for method in ("single_pass", "single_pass_budget", "retry"):
        summary[method] = {
            **metrics([row[method]["rank"] for row in answerable]),
            "complete_evidence_cases": sum(row[method]["complete_evidence"] for row in answerable),
            "outside_scope_hits": sum(row[method]["outside_scope_hits"] for row in rows),
            "search_calls": sum(row[method]["search_calls"] for row in rows),
            "candidate_count": sum(row[method]["candidate_count"] for row in rows),
            "elapsed_ms": sum(row[method]["elapsed_ms"] for row in rows),
        }
    return summary


class Encoder:
    """Kontrollierte Fragenvektoren, keine neuronale Modellberechnung."""

    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts, **kwargs):
        return [self.mapping[normalize_query(text)] for text in texts]


def evaluate(dataset_path=DATASET, *, embedding_mode="fixtures", repeats=5):
    if embedding_mode == "local":
        return evaluate_local(dataset_path, repeats=repeats)
    if embedding_mode != "fixtures":
        raise ValueError("embedding_mode muss fixtures oder local sein.")
    raw = Path(dataset_path).read_bytes()
    data = json.loads(raw)
    splits = data["splits"]
    if len(splits) != 2 or {split["name"] for split in splits} != {"development", "test"} or len({split["family"] for split in splits}) != 2:
        raise ValueError("Splits und Fragefamilien müssen getrennt sein.")
    ids = [identifier for split in splits for identifier in (split["document_id"], split["excluded_id"])]
    if len(set(ids)) != len(ids):
        raise ValueError("Dokument-IDs müssen zwischen Splits getrennt sein.")
    if len({case["id"] for case in data["cases"]}) != len(data["cases"]):
        raise ValueError("Fall-IDs müssen eindeutig sein.")
    report = {"dataset_sha256": sha256(raw).hexdigest(), "input_kind": "controlled_vectors_with_actual_retrieval",
              "mode": "semantic", "top_k": 3, "max_retry": 1, "llm_calls": 0, "openai_api_calls": 0, "splits": {},
              "context_rule": "Original-Rang 1 bleibt erhalten; Rest gleiche RRF(60), Deduplikation und Gleichstand zugunsten Original. Auch Budget-Baseline nutzt dieselbe Auswahlregel mit nur einer Liste.",
              "limitations": "Reine kontrollierte Fixtures mit manuellen Vektoren und vorgegebenen Alternativen. Keine echten Modell-Embeddings, keine validierte Retry-Auslösung, kein Qualitätsgewinn realer Antworten nachgewiesen. Explizite Aktivierung verursacht auch unnötige Retrys. Hit-Metriken messen einzelne relevante Chunks; vollständige Evidenz wird separat gezählt. Rettung/Verschlechterung bezieht sich auf vollständige Evidenz, nicht generierte Antworten. Ohne Re-Ranker bleibt die Single-Pass-Budget-Baseline trotz sechs Kandidaten beim Original-Top-3; Budget allein liefert daher keine bessere finale Auswahl. Unbeantwortbare/ausgeschlossene Fälle sind nicht im Relevanznenner. Laufzeiten sind lokale Einzelmessungen ohne Warm-up; keine Performance-Rangfolge."}
    all_rows = []
    for split in splits:
        collection = prepare_split(data, split)
        queries = {"original": split["original_query"], "alternative": split["alternative_query"],
                   "unanswerable": split["unanswerable_query"], "excluded": split["excluded_query"],
                   "multipart": split["multipart_query"]}
        mapping = {normalize_query(text): data["alternative_vector"] if key == "alternative" else data["original_vector"] for key, text in queries.items()}
        rows = []
        with patch.object(semantic_search, "_get_model", return_value=Encoder(mapping)):
            for case in data["cases"]:
                if any(type(index) is not int or not 1 <= index <= len(split["texts"]) for index in case["reference_chunks"]):
                    raise ValueError("Ungültige Chunk-Referenz.")
                query = queries[case["query"]]
                alternative = None if case["alternative"] is None else "  " + query.upper() + "  " if case["alternative"] == "normalized_original" else queries[case["alternative"]]
                allowed = [] if case.get("empty_scope") else [split["document_id"]]
                references = [(split["document_id"], index, 1) for index in case["reference_chunks"]]
                row = {"id": f"{split['name']}-{case['id']}", "category": case["category"],
                       "family": split["family"], "original_query": query, "alternative_query": alternative,
                       "document_ids": allowed, "references": references,
                       "excluded_reference": {"document_id": split["excluded_id"], "chunk_id": 1, "page_number": 1, "text": split["excluded_text"]} if case.get("excluded_reference") else None}
                for method, budget, enabled in (("single_pass", 3, False), ("single_pass_budget", 6, False), ("retry", 3, True)):
                    tick = perf_counter()
                    result = run_query_retry(query, retrieve=collection.search, get_documents=collection.list_documents,
                                             alternative_query=alternative, enabled=enabled, mode="semantic", document_ids=allowed, top_k=budget)
                    # Gleicher finaler Kontext: maximal drei Chunks in allen Varianten.
                    hits = select_context(result["first_hits"], result["second_hits"], 3)
                    row[method] = {**result, "final_hits": hits, "top_k": 3, "candidate_top_k_per_search": budget,
                                   **score(hits, references, allowed),
                                   "candidate_count": len(result["first_hits"]) + len(result["second_hits"]),
                                   "elapsed_ms": (perf_counter() - tick) * 1000}
                rows.append(row)
        report["splits"][split["name"]] = {"summary": summarize(rows), "questions": rows}
        all_rows.extend(rows)
    report["summary"] = summarize(all_rows)
    return report


def mmr_context(candidates, vectors, query_vector, *, anchor=None, top_k=3, lambda_value=0.5):
    """MMR mit Standard 0,5, Original-Rang 1 geschützt, stabile Gleichstände.

    Nutzt nur Originalfrage und gespeicherte Kandidatenvektoren, keine Goldlabels.
    Die Referenzmetadaten und angezeigten Retrieval-Scores bleiben unverändert.
    """
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k muss eine positive ganze Zahl sein.")
    if type(lambda_value) not in (int, float) or not 0 <= lambda_value <= 1:
        raise ValueError("lambda_value muss zwischen 0 und 1 liegen.")
    def key(hit):
        return (hit["document_id"], hit["chunk_id"], hit["page_number"])
    unique = {}
    for hit in candidates:
        unique.setdefault(key(hit), hit)
    remaining = list(unique)
    selected = []
    if anchor is not None and key(anchor) in remaining:
        selected.append(key(anchor))
        remaining.remove(key(anchor))
    while remaining and len(selected) < top_k:
        def utility(identity):
            relevance = semantic_search._cosine_similarity(query_vector, vectors[identity])
            redundancy = max((semantic_search._cosine_similarity(vectors[identity], vectors[other]) for other in selected), default=0)
            return lambda_value * relevance - (1 - lambda_value) * redundancy
        best = max(remaining, key=utility)
        selected.append(best)
        remaining.remove(best)
    return [dict(unique[identity]) for identity in selected]


def prepare_local_split(split, model):
    """Embeddings jedes synthetischen Dokuments genau einmal vorbereiten."""
    collection, vectors = DocumentCollection(), {}
    for identifier, filename, texts in [
        (split["document_id"], split["filename"], split["texts"]),
        (split["excluded_id"], split["excluded_filename"], [split["excluded_text"]]),
    ]:
        chunks = [{"chunk_id": index, "page_number": 1, "text": text} for index, text in enumerate(texts, 1)]
        if any(len(model.tokenizer(chunk["text"], truncation=False, verbose=False)["input_ids"]) > model.max_seq_length for chunk in chunks):
            raise ValueError("Ein Evaluations-Chunk überschreitet das Tokenlimit.")
        chunks = semantic_search.embed_chunks(chunks)
        for chunk in chunks:
            vectors[(identifier, chunk["chunk_id"], chunk["page_number"])] = chunk["embedding"]
        collection.add_prepared(DocumentRetrievalPipeline.from_record({
            "metadata": {"document_id": identifier, "filename": filename, "page_count": 1, "chunk_count": len(chunks)},
            "chunks": chunks,
        }))
    return collection, vectors


def evaluate_local(dataset_path=DATASET, *, repeats=5):
    """Lokales Cache-Modell, Warm-up und rotierende Messungen ohne OpenAI."""
    from importlib.metadata import version
    if type(repeats) is not int or not 2 <= repeats <= 50:
        raise ValueError("repeats muss eine ganze Zahl zwischen 2 und 50 sein.")
    raw = Path(dataset_path).read_bytes()
    data = json.loads(raw)
    splits = data["splits"]
    ids = [identifier for split in splits for identifier in (split["document_id"], split["excluded_id"])]
    if len(splits) != 2 or {split["name"] for split in splits} != {"development", "test"} or len({split["family"] for split in splits}) != 2 or len(set(ids)) != len(ids):
        raise ValueError("Splits, Dokumente und Fragefamilien müssen getrennt sein.")
    if len({case["id"] for case in data["cases"]}) != len(data["cases"]):
        raise ValueError("Fall-IDs müssen eindeutig sein.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tick = perf_counter()
    model = semantic_search._get_model()
    load_ms = (perf_counter() - tick) * 1000
    methods = ("single_pass", "single_pass_budget", "retry", "retry_mmr_control")
    report = {
        "dataset_sha256": sha256(raw).hexdigest(), "input_kind": "actual_local_embeddings",
        "embedding_model": semantic_search.MODEL_NAME, "device": str(model.device), "max_seq_length": model.max_seq_length,
        "versions": {name: version(name) for name in ("sentence-transformers", "torch")},
        "model_load_ms": load_ms, "repeats": repeats, "warmup_per_case_method": 1,
        "top_k": 3, "mode": "semantic", "max_retry": 1, "llm_calls": 0, "openai_api_calls": 0, "splits": {},
        "context_rules": {"single_pass": "Original Top-3", "retry": "Unveränderte Original-Rang-1-Ankerregel und RRF(60)",
                          "single_pass_budget": "Original Top-6, danach MMR(lambda=0.5) mit Original-Rang-1-Anker und stabilem Gleichstand",
                          "retry_mmr_control": "Zusatzkontrolle: gleiche MMR-Auswahl aus der Vereinigung beider Top-3-Listen, Originalfrage als Relevanzmaß"},
        "limitations": "Nur acht bekannte synthetische Fälle je Split und manuell vorgegebene Alternativen, keine autonome Reformulierung. Fallnamen wie rank4 beschreiben Fixture-Szenarien, nicht notwendigerweise tatsächliche Modell-Ränge. Hit-Metriken messen einzelne Referenzen; vollständige Evidenz separat. Budget-Baseline diversifiziert statt blind Top-3 zu behalten; MMR kann relevante ähnliche Passagen verdrängen. Retry-vs-Budget unterscheidet sowohl Query als auch Selektor; die MMR-Zusatzkontrolle hält den Selektor gleich, aber Quellen und Überlappung der Kandidaten unterscheiden sich weiter. MMR benötigt ein zusätzliches Originalfrage-Embedding; dessen Kosten sind in der Laufzeit und separat in selection_query_embedding_calls erfasst. Warm-up und wiederholte lokale Messungen ohne Lasttest; kein robuster Qualitätsnachweis. Kein LLM-, API- oder PDF-End-to-End-Test.",
    }
    all_rows = []
    for split in splits:
        tick = perf_counter()
        collection, vectors = prepare_local_split(split, model)
        preparation_ms = (perf_counter() - tick) * 1000
        queries = {"original": split["original_query"], "alternative": split["alternative_query"],
                   "unanswerable": split["unanswerable_query"], "excluded": split["excluded_query"], "multipart": split["multipart_query"]}
        rows = []
        for case in data["cases"]:
            if any(type(index) is not int or not 1 <= index <= len(split["texts"]) for index in case["reference_chunks"]):
                raise ValueError("Ungültige Referenz.")
            query = queries[case["query"]]
            alternative = None if case["alternative"] is None else "  " + query.upper() + "  " if case["alternative"] == "normalized_original" else queries[case["alternative"]]
            allowed = [] if case.get("empty_scope") else [split["document_id"]]
            references = [(split["document_id"], index, 1) for index in case["reference_chunks"]]
            row = {"id": f"{split['name']}-{case['id']}", "category": case["category"], "family": split["family"],
                   "original_query": query, "alternative_query": alternative, "document_ids": allowed, "references": references}
            def execute(method):
                tick = perf_counter()
                result = run_query_retry(query, retrieve=collection.search, get_documents=collection.list_documents,
                    alternative_query=alternative, enabled=method in ("retry", "retry_mmr_control"),
                    mode="semantic", document_ids=allowed, top_k=6 if method == "single_pass_budget" else 3)
                candidates = result["first_hits"] + result["second_hits"]
                selection_calls = 0
                if method in ("single_pass_budget", "retry_mmr_control") and candidates:
                    query_vector = model.encode([query], convert_to_numpy=True)[0]
                    selection_calls = 1
                    hits = mmr_context(candidates, vectors, query_vector, anchor=result["first_hits"][0] if result["first_hits"] else None)
                else:
                    hits = result["final_hits"]
                return {**result, "final_hits": hits, "top_k": 3, "candidate_top_k_per_search": 6 if method == "single_pass_budget" else 3,
                        **score(hits, references, allowed), "candidate_count": len(candidates),
                        "selection_query_embedding_calls": selection_calls, "elapsed_ms": (perf_counter() - tick) * 1000}
            # Ein ungemessener Warm-up je Methode und Frage.
            for method in methods:
                row[method] = execute(method)
                row[method]["samples_ms"] = []
            for iteration in range(repeats):
                offset = iteration % len(methods)
                for method in methods[offset:] + methods[:offset]:
                    measured = execute(method)
                    if measured["final_hits"] != row[method]["final_hits"]:
                        raise ValueError("Nicht reproduzierbare Treffer zwischen Messläufen.")
                    row[method]["samples_ms"].append(measured["elapsed_ms"])
            for method in methods:
                row[method]["timing"] = timing_summary(row[method]["samples_ms"])
                row[method]["elapsed_ms"] = sum(row[method]["samples_ms"]) / repeats
            rows.append(row)
        summary = summarize(rows)
        # Zusatzkontrolle bewusst separat vom unveränderten Drei-Wege-Summary.
        control_rows = [{**row, "retry": row["retry_mmr_control"]} for row in rows]
        summary["retry_mmr_control"] = summarize(control_rows)["retry"]
        summary["retry_mmr_control"]["rescued_failures"] = summarize(control_rows)["rescued_failures"]
        summary["retry_mmr_control"]["degraded_successes"] = summarize(control_rows)["degraded_successes"]
        for method in methods:
            summary[method]["timing"] = timing_summary([value for row in rows for value in row[method]["samples_ms"]])
            summary[method]["selection_query_embedding_calls_per_pass"] = sum(row[method]["selection_query_embedding_calls"] for row in rows)
        summary["search_count_note"] = "Suchläufe/Kandidaten zählen einen vollständigen Durchgang; tatsächliche Benchmark-Arbeit = Warm-up plus repeats Durchgänge. elapsed_ms summiert mittlere Messzeiten nach Warm-up. selection_query_embedding_calls sind zusätzlich zu den Frage-Embeddings der Suchläufe."
        report["splits"][split["name"]] = {"document_embedding_ms": preparation_ms, "summary": summary, "questions": rows}
        all_rows.extend(rows)
    report["summary"] = summarize(all_rows)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--embeddings", choices=("fixtures", "local"), default="fixtures")
    parser.add_argument("--repeats", type=int, choices=range(2, 51), default=5)
    args = parser.parse_args(argv)
    prefix = "query_retry_local" if args.embeddings == "local" else "query_retry_offline"
    output = args.output or ROOT / "docs" / f"{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}.json"
    if output.exists():
        parser.error("Bericht existiert bereits. Bitte einen neuen Dateinamen wählen.")
    report = evaluate() if args.embeddings == "fixtures" else evaluate(embedding_mode="local", repeats=args.repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for split, result in report["splits"].items():
        print(split, json.dumps(result["summary"], ensure_ascii=False))
    print(f"Bericht: {output}; OpenAI-Aufrufe: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
