"""Evaluiert nur Retrieval mit künstlichen PDFs und dem lokalen Modell, ohne LLM."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.pdf_reader import extract_pdf_text
from app.services.retrieval_pipeline import DocumentCollection, DocumentRetrievalPipeline
from app.services.semantic_search import MODEL_NAME
from app.services.hybrid_search import RRF_K, BM25_PARAMETERS
from evaluation.retrieval_metrics import calculate_metrics, first_relevant_hit, matches_reference


DATASET = ROOT / "data/evaluation/retrieval_questions.json"
SAMPLES = ROOT / "data/sample_documents"


def validate_references(questions: list[dict], documents: dict) -> None:
    """Prüft alle Belegstellen gegen die Originalextraktion vor dem Bewerten."""
    if not questions or len({item["id"] for item in questions}) != len(questions):
        raise ValueError("Der Datensatz muss Fragen mit eindeutigen IDs enthalten.")
    for item in questions:
        if not item["query"].strip() or not item["references"]:
            raise ValueError(f"Ungültige Frage oder Referenz: {item['id']}")
        for reference in item["references"]:
            pages = documents.get(reference["filename"], {}).get("pages", [])
            if not any(matches_reference({"filename": reference["filename"], **page}, reference) for page in pages):
                raise ValueError(f"Referenz ist nicht im PDF-Text nachweisbar: {item['id']} / {reference['filename']}")


def evaluate(dataset_path: Path = DATASET, sample_directory: Path = SAMPLES, *, mode: str = "semantic") -> dict:
    """Indexiert jedes Testdokument einmal in einer unabhängigen Speichersammlung."""
    if mode not in ("semantic", "hybrid", "compare"):
        raise ValueError("Modus muss semantic, hybrid oder compare sein.")
    dataset_bytes = dataset_path.read_bytes()
    questions = json.loads(dataset_bytes)["questions"]
    filenames = list(dict.fromkeys(ref["filename"] for item in questions for ref in item["references"]))
    documents = {}
    for filename in filenames:
        # Referenzen dürfen ausschließlich Dateien im angegebenen Testordner wählen.
        if Path(filename).name != filename:
            raise ValueError("Referenzen müssen reine Dateinamen enthalten.")
        content = (sample_directory / filename).read_bytes()
        documents[filename] = {"content": content, **extract_pdf_text(content)}
    validate_references(questions, documents)

    collection = DocumentCollection()
    document_metadata = []
    for filename, document in documents.items():
        pipeline = DocumentRetrievalPipeline()
        pipeline.filename = filename
        pipeline.load_pdf(document["content"])
        metadata = collection.add_prepared(pipeline)
        document_metadata.append({**metadata, "sha256": sha256(document["content"]).hexdigest()})
    total_chunks = sum(document["chunk_count"] for document in document_metadata)
    common = {
        "model": MODEL_NAME, "sentence_transformers_version": version("sentence-transformers"),
        "rank_bm25_version": version("rank-bm25"),
        "dataset_sha256": sha256(dataset_bytes).hexdigest(), "documents": document_metadata,
        "method": "Erster Chunk mit korrektem Dateinamen, Seite und allen Textbelegen; MRR über alle Chunks. Keine LLM-Bewertung.",
        "parameters": {"rrf_k": RRF_K, "weights": [1, 1], "bm25": BM25_PARAMETERS,
                       "candidate_pool": "Alle semantischen Treffer; alle BM25-Treffer mit Wortübereinstimmung; stabile Rangfolge bei Gleichstand",
                       "tokenization": "Unicode-Wörter und Zahlen, casefold; kein Stemming, keine Stoppwortliste"},
    }
    reports = {}
    for selected in (("semantic", "hybrid") if mode == "compare" else (mode,)):
        rows = []
        for question in questions:
            # Beide Methoden nutzen dieselben einmalig vorbereiteten Dokumente.
            ranking = collection.search(question["query"], top_k=total_chunks, mode=selected)
            rank, hit = first_relevant_hit(ranking, question["references"])
            rows.append({
                **question, "rank": rank,
                "similarity_score": hit["similarity_score"] if hit else None,
                "rrf_score": hit.get("rrf_score") if hit else None,
                "bm25_score": hit.get("bm25_score") if hit else None,
                "matched_source": {key: hit[key] for key in ("document_id", "filename", "chunk_id", "page_number")} if hit else None,
                "ranking": [
                    {key: candidate[key] for key in ("document_id", "filename", "chunk_id", "page_number", "similarity_score", "rrf_score", "bm25_score", "semantic_rank", "bm25_rank") if key in candidate}
                    for candidate in ranking
                ],
            })
        reports[selected] = {**common, "retrieval_mode": selected,
                             "metrics": calculate_metrics([row["rank"] for row in rows]), "questions": rows}
    return {**common, "methods": reports} if mode == "compare" else reports[mode]


def comparison_markdown(report: dict) -> str:
    """Dokumentiert auch Verschlechterungen ohne Anpassung der RRF-Parameter."""
    semantic = report["methods"]["semantic"]
    hybrid = report["methods"]["hybrid"]
    lines = [
        "# Semantic Search und Hybrid Search – echter lokaler Vergleich", "",
        f"Modell: `{report['model']}`. Identischer Datensatz mit {semantic['metrics']['question_count']} Fragen, "
        f"{len(report['documents'])} PDFs und {sum(doc['chunk_count'] for doc in report['documents'])} Chunks.", "",
        "| Methode | Hit Rate@1 | Hit Rate@3 | MRR |", "|---|---:|---:|---:|",
    ]
    for label, result in (("Semantic", semantic), ("Hybrid", hybrid)):
        metrics = result["metrics"]
        lines.append(f"| {label} | {metrics['hit_rate_at_1']:.2%} | {metrics['hit_rate_at_3']:.2%} | {metrics['mrr']:.6f} |")
    lines += ["", "| ID | Frage | Erwartete Quelle | Rang Semantic | Rang Hybrid | Änderung |", "|---|---|---|---:|---:|---|"]
    changes = {"besser": 0, "schlechter": 0, "gleich": 0}
    for first, second in zip(semantic["questions"], hybrid["questions"]):
        old, new = first["rank"], second["rank"]
        old_value, new_value = old or float("inf"), new or float("inf")
        change = "besser" if new_value < old_value else "schlechter" if new_value > old_value else "gleich"
        changes[change] += 1
        source = "; ".join(f"{ref['filename']}, S. {ref['page_number']}: {ref['section']}" for ref in first["references"])
        lines.append(f"| {first['id']} | {first['query']} | {source} | {old or 'fehlt'} | {new or 'fehlt'} | {change} |")
    lines += [
        "", f"Verbessert: {changes['besser']}; verschlechtert: {changes['schlechter']}; unverändert: {changes['gleich']}.", "",
        "Die reine Wortzerlegung gleicht deutsche Flexionen wie Urlaubsantrag/Urlaubsanträge bzw. Passwort/Passwörter nicht ab. "
        "Ohne Wortübereinstimmung erhält ein Chunk keinen BM25-Rangbeitrag. "
        "Die vollständigen Rangbeiträge stehen als semantic_rank und bm25_rank im JSON-Bericht. "
        "Die Parameter bleiben unabhängig vom Ergebnis unverändert; semantic bleibt der Anwendungsstandard.", "",
        "## Vor der Evaluation festgelegtes Verfahren", "",
        "BM25Okapi aus rank-bm25 mit k1=1,5, b=0,75, epsilon=0,25; Unicode-Wörter/Zahlen und casefold. "
        "Kein Stemming und keine Stoppwortliste. Nur Chunks mit mindestens einer Wortübereinstimmung werden in die BM25-Rangliste aufgenommen. "
        "Die vollständige semantische Rangliste wird mit dieser Liste gleichgewichtet kombiniert: "
        "RRF = Summe 1 / (60 + Rang). Ränge beginnen bei 1. "
        "Bei Gleichstand bleibt die semantische Reihenfolge erhalten. Keine Parameter wurden anhand dieser 20 Fragen angepasst.", "",
        "`similarity_score` bleibt die unveränderte Kosinus-Ähnlichkeit. Hybrid sortiert nach dem separaten `rrf_score`. "
        "BM25-Score, RRF-Score und Kosinus-Ähnlichkeit sind verschiedene Größen und keine Wahrscheinlichkeiten. "
        "Alle Rohscores und Ranglisten stehen im JSON-Bericht.", "",
        "Die Dokument-Embeddings werden einmal erzeugt und für beide Methoden wiederverwendet. "
        "Der BM25-Index wird aus denselben Chunks aufgebaut. Keine Änderungen an PDFs, Referenzen, SQLite-Nutzerdaten oder .env; keine LLM-Aufrufe.", "",
        "Relevanz verlangt denselben Dateinamen, dieselbe Seite und alle Textbelege im Chunk; alternative belegte Quellen sind erlaubt. "
        "Beispielfragen ohne Antwort zählen nicht. MRR verwendet die vollständige Rangliste. "
        "Der kleine Testdatensatz erlaubt keine allgemeine Aussage, dass Hybrid immer besser ist. "
        "PDF-Beispielfragen können die Suche begünstigen. Die LLM-Antwortqualität wird nicht gemessen.", "",
        "Quellen: [rank-bm25](https://github.com/dorianbrown/rank_bm25), "
        "[RRF-Originalarbeit](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf).", "",
    ]
    return "\n".join(lines)


def markdown_report(report: dict) -> str:
    """Erstellt einen lesbaren Bericht mit allen Fragen und tatsächlichen Scores."""
    metrics = report["metrics"]
    lines = [
        "# Retrieval-Evaluation – tatsächlicher lokaler Modelllauf", "",
        f"Modell: `{report['model']}`; Sentence Transformers: `{report['sentence_transformers_version']}`.",
        f"Retrieval-Modus: `{report.get('retrieval_mode', 'semantic')}`.",
        f"Datensatz: {metrics['question_count']} Fragen; {len(report['documents'])} künstliche PDFs; "
        f"{sum(document['chunk_count'] for document in report['documents'])} Chunks.", "",
        f"- Hit Rate@1: **{metrics['hit_rate_at_1']:.2%}**",
        f"- Hit Rate@3: **{metrics['hit_rate_at_3']:.2%}**",
        f"- MRR (vollständige Rangliste): **{metrics['mrr']:.6f}**", "",
        "Dateikürzel: H = DocuAgent_Testdokument.pdf; IT = DocuAgent_IT_Sicherheitsrichtlinie.pdf.", "",
        "| ID | Frage | Erwartete Quelle | Rang | Score des ersten relevanten Chunks |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["questions"]:
        expected = "; ".join(
            f"{'H' if ref['filename'] == 'DocuAgent_Testdokument.pdf' else 'IT'}, S. {ref['page_number']}: {ref['section']}"
            for ref in row["references"]
        )
        rank = row["rank"] if row["rank"] is not None else "nicht gefunden"
        score = f"{row['similarity_score']:.6f}" if row["similarity_score"] is not None else "–"
        lines.append(f"| {row['id']} | {row['query']} | {expected} | {rank} | {score} |")
    lines += [
        "", "## Messverfahren und Grenzen", "",
        "Ein Treffer zählt nur, wenn Dateiname, Seite und sämtliche angegebenen Belegstellen im selben Chunk vorkommen. "
        "Reine Beispielfragen zählen nicht. Alternative gültige Quellen werden ausdrücklich berücksichtigt. "
        "Alle Referenzen werden vorab gegen den extrahierten PDF-Text geprüft; Zeichenumbrüche und Großschreibung werden normalisiert.", "",
        "Hit Rate@1/3 misst den Anteil der Fragen mit mindestens einem relevanten Treffer unter den ersten 1/3. "
        "MRR mittelt 1/Rang des ersten relevanten Treffers; fehlende Treffer zählen 0. "
        "Es wird über alle Chunks gesucht, nicht nur bis Rang 3. Scores und Rangfolgen werden nicht verändert.", "",
        "Der kleine, manuell erstellte Datensatz ist kein allgemeiner Qualitätsnachweis. "
        "Einzelne PDF-Beispielfragen ähneln Evaluationsfragen und können die Suche begünstigen. "
        "Strikte Textbelege können auf mehrere Chunks verteilte Antworten als Fehlschlag werten. "
        "Gemessen wird nur Retrieval, nicht die Richtigkeit einer generierten Antwort.", "",
        "Der JSON-Bericht enthält Belegstellen, vollständige Rangfolgen, Dokument-IDs und SHA-256-Fingerabdrücke "
        "von PDFs und Datensatz. Er wird bei erneutem Ausführen überschrieben. "
        "Die Evaluation verwendet eine eigene In-Memory-Sammlung und verändert keine SQLite-Nutzerdaten. "
        "Es erfolgen keine OpenAI-Aufrufe und kein Zugriff auf .env.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("semantic", "hybrid", "compare"), default="compare")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Ausschließlich das bereits lokal vorhandene Modell verwenden.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report = evaluate(mode=args.mode)
    rendered = comparison_markdown(report) if args.mode == "compare" else markdown_report(report)
    if args.output is None:
        args.output = ROOT / f"docs/{'hybrid_retrieval_comparison' if args.mode == 'compare' else args.mode + '_retrieval_evaluation'}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"\nJSON-Bericht: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
