"""Bereitet Antwort-Evaluation vor; echte API-Aufrufe nur mit live und Bestätigung."""

import argparse
from hashlib import sha256
import json
from math import ceil
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.pdf_reader import extract_pdf_text
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline
from app.services.rag_service import generate_answer, SYSTEM_PROMPT, MultiDocumentAnswer
from app.workflows.document_qa import build_document_qa_workflow
from evaluation.answer_metrics import assess_deterministic, pending_review, summarize
from evaluation.retrieval_metrics import matches_reference
from scripts.evaluate_retrieval import validate_references


DATASET = ROOT / "data/evaluation/answer_questions.json"
RETRIEVAL_DATASET = ROOT / "data/evaluation/retrieval_questions.json"
RUBRIC = ROOT / "data/evaluation/answer_review_rubric.json"
PRICING = {
    "model": "gpt-4.1-mini", "currency": "USD", "input_per_million": 0.40, "output_per_million": 1.60,
    "checked_on": "2026-11-18", "source": "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
}
MAX_CALLS = 3


def load_cases(ids: list[str] | None = None) -> list[dict]:
    """Verknüpft Sollantworten mit stabilen Textreferenzen statt zufälligen IDs."""
    retrieval = {row["id"]: row for row in json.loads(RETRIEVAL_DATASET.read_text(encoding="utf-8"))["questions"]}
    rows = json.loads(DATASET.read_text(encoding="utf-8"))["questions"]
    cases = []
    for row in rows:
        reference = retrieval[row["retrieval_id"]] if row.get("retrieval_id") else row
        case = {**row, "query": reference["query"], "references": reference["references"]}
        if case["answerable"]:
            if not case["references"] or not case["required_patterns"]:
                raise ValueError("Beantwortbare Fragen benötigen Quellen und überprüfbare Teilmerkmale.")
            if not all(re.search(pattern, case["expected_answer"], re.IGNORECASE) for pattern in case["required_patterns"]):
                raise ValueError(f"Sollantwort passt nicht zu ihren Teilprüfungen: {case['id']}")
        elif case["references"] or not case.get("absence_rationale"):
            raise ValueError("Nichtwissen-Fragen benötigen eine Begründung und dürfen keine erfundenen Quellen haben.")
        cases.append(case)
    mapping = {row["id"]: row for row in cases}
    if len(mapping) != len(cases):
        raise ValueError("Frage-IDs müssen eindeutig sein.")
    if ids is not None:
        if not ids or len(set(ids)) != len(ids) or any(identifier not in mapping for identifier in ids):
            raise ValueError("Bitte gültige, eindeutige Frage-IDs angeben.")
        cases = [mapping[identifier] for identifier in ids]
    return cases


def prepare_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Lädt nur die beiden künstlichen PDFs in eine eigene Sammlung ohne SQLite."""
    filenames = ("DocuAgent_Testdokument.pdf", "DocuAgent_IT_Sicherheitsrichtlinie.pdf")
    documents = {}
    for filename in filenames:
        content = (ROOT / "data/sample_documents" / filename).read_bytes()
        documents[filename] = {"content": content, **extract_pdf_text(content)}
    answerable = [case for case in cases if case["answerable"]]
    if answerable:
        validate_references(answerable, documents)
    collection = DocumentCollection()
    metadata = []
    for filename, document in documents.items():
        pipeline = DocumentRetrievalPipeline()
        pipeline.filename = filename
        # Stabile Evaluations-IDs; kein Zugriff auf den produktiven Index.
        pipeline.document_id = "eval-" + sha256(filename.encode() + document["content"]).hexdigest()[:24]
        pipeline.load_pdf(document["content"])
        metadata.append({**collection.add_prepared(pipeline), "sha256": sha256(document["content"]).hexdigest()})
    prepared = []
    for case in cases:
        context = collection.search(case["query"], top_k=3, mode="semantic")
        prepared.append({
            **case, "context": context,
            "context_contains_expected_evidence": any(matches_reference(hit, ref) for hit in context for ref in case["references"]) if case["answerable"] else None,
        })
    return prepared, metadata


def estimate_cost(cases: list[dict]) -> dict:
    """Schätzt Kosten transparent, ohne API und ohne exakte Tokenzählung.

    UTF-8-Bytes/3 plus 512 Tokens pro Anfrage dienen als grobe Inputheuristik.
    Output wird mit dem bestehenden Service-Limit von 1.000 Tokens angesetzt.
    Das ist keine Preisgarantie und keine harte Dollar-Ausgabengrenze.
    """
    schema = json.dumps(MultiDocumentAnswer.model_json_schema(), ensure_ascii=False)
    inputs = sum(ceil(len((SYSTEM_PROMPT + schema + json.dumps({"query": case["query"], "context": case["context"]}, ensure_ascii=False)).encode("utf-8")) / 3) + 512 for case in cases)
    outputs = len(cases) * 1000
    return {
        **PRICING, "planned_question_ids": [case["id"] for case in cases], "planned_calls": len(cases),
        "estimated_input_tokens": inputs, "reserved_output_tokens": outputs,
        "estimated_usd": (inputs * PRICING["input_per_million"] + outputs * PRICING["output_per_million"]) / 1_000_000,
        "method": "Grobe UTF-8-Bytes/3-Heuristik einschließlich Prompt/Schema plus 512 Tokens pro Anfrage; 1000 Outputtokens pro Antwort. Keine Preisgarantie. Preise vor Live-Nutzung prüfen.",
        "actual_cost_usd": None,
    }


def run_evaluation(cases: list[dict], *, mode: str = "dry-run", workflow: str = "legacy", confirm_live: bool = False, max_calls: int = 3, generator=None) -> dict:
    """Startet standardmäßig keine Antwortgenerierung und niemals einen LLM-Judge."""
    if mode not in ("dry-run", "live") or type(max_calls) is not int or not 1 <= max_calls <= MAX_CALLS:
        raise ValueError("Modus muss dry-run/live und max_calls eine ganze Zahl zwischen 1 und 3 sein.")
    if workflow not in ("legacy", "langgraph"):
        raise ValueError("Workflow muss legacy oder langgraph sein.")
    if mode == "live" and not confirm_live:
        raise ValueError("Live benötigt die ausdrückliche Bestätigung --confirm-live.")
    planned = cases[:max_calls]
    cost = estimate_cost(planned)
    print(f"Geplante Live-Fragen: {', '.join(cost['planned_question_ids'])}; maximal {len(planned)} API-Aufrufe.", flush=True)
    print(f"Geschätzte Kosten: ca. {cost['estimated_usd']:.6f} USD bei bis zu 1.000 Outputtokens pro Frage; keine Preisgarantie.", flush=True)
    if mode == "live" and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY muss in der Prozessumgebung gesetzt sein. Es wird keine .env-Datei geladen.")
    rows = []
    attempted = 0
    stopped = False
    for index, case in enumerate(cases):
        row = {**case, "status": "dry_run", "result": None, "deterministic": None, "semantic_review": pending_review()}
        if mode == "live":
            if index >= max_calls or stopped:
                row["status"] = "skipped_call_limit_or_prior_error"
            else:
                try:
                    def generate(query, context):
                        nonlocal attempted
                        # Leerer Kontext benötigt auch im bisherigen Service keine API.
                        if any(chunk["text"].strip() for chunk in context):
                            attempted += 1
                        return (generator if generator is not None else generate_answer)(query, context)

                    if workflow == "langgraph":
                        def retrieve(query, *, top_k):
                            # Exakt der einmalig vorbereitete semantische Top-3-Kontext.
                            return case["context"]

                        graph = build_document_qa_workflow(retrieve, generate)
                        result = graph.invoke({"query": case["query"], "top_k": 3})["result"]
                    else:
                        result = generate(case["query"], case["context"])
                    row.update(status="completed", result=result, deterministic=assess_deterministic(case, result, case["context"]))
                except (RuntimeError, ValueError):
                    # Keine rohen API-Fehler oder Schlüssel in Berichte/Logs übernehmen.
                    row.update(status="error", error="Antwortgenerierung fehlgeschlagen; Lauf ohne Wiederholung beendet.")
                    stopped = True
        rows.append(row)
    answerable = [case for case in cases if case["answerable"]]
    return {
        "mode": mode, "workflow": workflow, "answer_model": "gpt-4.1-mini", "retrieval_mode": "semantic", "top_k": 3,
        "api_calls_attempted": attempted, "max_calls": max_calls, "automatic_retries": 0,
        "max_output_tokens_per_answer": 1000, "judge_api_calls": 0, "cost_estimate": cost,
        "summary": summarize(rows), "questions": rows,
        "retrieval_context_coverage": {"matched": sum(case["context_contains_expected_evidence"] for case in answerable), "answerable_questions": len(answerable)},
        "review_rubric": json.loads(RUBRIC.read_text(encoding="utf-8")),
        "workflow_comparison": "Beide Varianten verwenden denselben vorbereiteten Kontext aus einer separaten Sammlung, semantic und top_k=3. LangGraph erhält diese Treffer über einen Kontext-Retriever ohne zweite Suche. Bei leerer Trefferliste verzichtet LangGraph lokal, legacy verwendet den bestehenden Service ohne API-Aufruf. Dry-run bereitet nur Kontext vor und führt weder Antwortgenerator noch Graph aus. FastAPI, dessen Index und HTTP-Fehlerabbildung werden nicht evaluiert. Live benötigt in beiden Varianten einen Schlüssel in der Prozessumgebung.",
        "limitations": "Regex-Faktentreffer, gültige IDs und Verweigerungssignale beweisen keine vollständige Correctness, Faithfulness oder Citation Accuracy. Semantische Reviews sind unbewertet. LLM-Ausgaben können bei Wiederholung variieren. Teilmetriken enthalten nur abgeschlossene, anwendbare Fälle; Fehler und ausgelassene Fragen werden gesondert ausgewiesen.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--workflow", choices=("legacy", "langgraph"), default="legacy")
    parser.add_argument("--confirm-live", action="store_true", help="Erlaubt kostenpflichtige Antworten, keinen LLM-Judge.")
    parser.add_argument("--max-calls", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--ids", nargs="+", help="Frage-IDs in gewünschter Reihenfolge, z.B. A01 A11 A21")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "live" and not args.confirm_live:
        parser.error("Live benötigt --confirm-live. Ohne Freigabe ausschließlich dry-run verwenden.")
    default_name = f"answer_evaluation_{args.mode}.json" if args.workflow == "legacy" else f"answer_evaluation_langgraph_{args.mode}.json"
    output = args.output or ROOT / "docs" / default_name
    if output.exists():
        parser.error("Der Bericht existiert bereits. Bitte mit --output einen neuen Dateinamen wählen.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cases, documents = prepare_cases(load_cases(args.ids))
    report = run_evaluation(cases, mode=args.mode, workflow=args.workflow, confirm_live=args.confirm_live, max_calls=args.max_calls)
    report.update(documents=documents, fingerprints={
        path.name: sha256(path.read_bytes()).hexdigest() for path in (DATASET, RETRIEVAL_DATASET, RUBRIC)
    }, system_prompt_sha256=sha256(SYSTEM_PROMPT.encode()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exklusives Erstellen schützt auch vor einem gleichzeitig angelegten Bericht.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Workflow: {args.workflow}; Modus: {args.mode}; Fragen: {len(cases)}; API-Aufrufversuche: {report['api_calls_attempted']}; abgeschlossene Antworten: {report['summary']['completed_answers']}")
    print(f"Retrieval-Kontext mit Sollbeleg: {report['retrieval_context_coverage']}")
    print("Semantische Antwortbewertungen: nicht ausgeführt. Siehe Review-Rubrik.")
    print(f"JSON-Bericht: {output}")
    return 1 if report["summary"]["failed_answers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
