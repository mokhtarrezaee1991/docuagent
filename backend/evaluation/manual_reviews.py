"""Wertet separate manuelle Reviews aus; benötigt nur die Python-Standardbibliothek."""

from hashlib import sha256
import json
from pathlib import Path


CRITERIA = {
    "answer_correctness": "Answer Correctness",
    "faithfulness": "Faithfulness",
    "citation_accuracy": "Citation Accuracy",
    "abstention": "Abstention",
}
STATUSES = {"unreviewed", "scored", "not_applicable"}


def load_inputs(report_path: Path, rubric_path: Path) -> tuple[dict, dict, str, str]:
    """Liest ausschließlich Bericht/Rubrik und prüft die verwendete Rubrik."""
    report_bytes = report_path.read_bytes()
    rubric_bytes = rubric_path.read_bytes()
    report, rubric = json.loads(report_bytes), json.loads(rubric_bytes)
    if any(criterion not in rubric for criterion in CRITERIA):
        raise ValueError("Die Rubrik enthält nicht alle vier Kriterien.")
    if "review_rubric" in report and report["review_rubric"] != rubric:
        raise ValueError("Bericht und gewählte Rubrik stimmen nicht überein.")
    return report, rubric, sha256(report_bytes).hexdigest(), sha256(rubric_bytes).hexdigest()


def generated_cases(report: dict) -> dict:
    """Lässt nur abgeschlossene Fälle mit einer tatsächlich vorhandenen Antwort zu."""
    cases = {}
    seen = set()
    for row in report["questions"]:
        identifier = row["id"]
        if not isinstance(identifier, str) or identifier in seen:
            raise ValueError("Der Quellbericht muss eindeutige Frage-IDs enthalten.")
        seen.add(identifier)
        if row.get("status") == "completed":
            result = row.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("answer"), str) or not result["answer"].strip():
                raise ValueError("Ein abgeschlossener Fall enthält keine gültige Antwort.")
            if type(row.get("answerable")) is not bool:
                raise ValueError("Ein abgeschlossener Fall benötigt answerable als Wahrheitswert.")
            cases[identifier] = row
    return cases


def template(report: dict, source_hash: str, rubric_hash: str, reviewer: str = "") -> dict:
    """Erzeugt eine Bewertungsdatei, ohne Noten oder Aussagen zu erfinden."""
    reviews = []
    for identifier, case in generated_cases(report).items():
        assessments = {}
        for criterion in CRITERIA:
            not_applicable = (criterion == "answer_correctness" and not case["answerable"]) or (criterion == "abstention" and case["answerable"])
            assessments[criterion] = {
                "status": "not_applicable" if not_applicable else "unreviewed", "score": None,
                "reason": "Gemäß Rubrik für diese Frage nicht anwendbar." if not_applicable else "",
            }
        reviews.append({"question_id": identifier, "assessments": assessments})
    return {"version": 1, "source_report_sha256": source_hash, "rubric_sha256": rubric_hash,
            "reviewer": reviewer, "reviews": reviews}


def aggregate_reviews(report: dict, reviews: dict, source_hash: str, rubric_hash: str) -> dict:
    """Berechnet pro Kriterium Mittelwert/Nenner; null und N/A bleiben ausgeschlossen."""
    if reviews.get("version") != 1 or type(reviews.get("version")) is not int:
        raise ValueError("Nicht unterstützte Review-Dateiversion.")
    if reviews.get("source_report_sha256") != source_hash or reviews.get("rubric_sha256") != rubric_hash:
        raise ValueError("Review-Datei gehört nicht zu diesem Bericht oder dieser Rubrik (SHA-256).")
    cases = generated_cases(report)
    by_id = {}
    if not isinstance(reviews.get("reviews"), list):
        raise ValueError("reviews muss eine Liste sein.")
    for entry in reviews["reviews"]:
        identifier = entry.get("question_id")
        if identifier not in cases or identifier in by_id:
            raise ValueError("Review enthält eine unbekannte, nicht generierte oder doppelte Frage-ID.")
        assessments = entry.get("assessments")
        if not isinstance(assessments, dict) or set(assessments) - set(CRITERIA):
            raise ValueError("Review enthält unbekannte Kriterien oder ungültige assessments.")
        by_id[identifier] = assessments

    metrics = {criterion: {"scores": [], "unreviewed_count": 0, "not_applicable_count": 0} for criterion in CRITERIA}
    rows = []
    reviewed_ids = set()
    for identifier, case in cases.items():
        assessments = {}
        for criterion in CRITERIA:
            assessment = by_id.get(identifier, {}).get(criterion, {"status": "unreviewed", "score": None, "reason": ""})
            if not isinstance(assessment, dict) or assessment.get("status") not in STATUSES:
                raise ValueError("Jede Bewertung benötigt einen gültigen Status.")
            status, score = assessment["status"], assessment.get("score")
            reason = assessment.get("reason", "")
            if not isinstance(reason, str):
                raise ValueError("Die Bewertungsbegründung muss Text sein.")
            if status == "scored":
                if type(score) is not int or score not in (0, 1, 2) or not reason.strip():
                    raise ValueError("Bewertete Kriterien benötigen 0, 1 oder 2 als ganze Zahl und eine Begründung.")
                if (criterion == "answer_correctness" and not case["answerable"]) or (criterion == "abstention" and case["answerable"]):
                    raise ValueError("Das Kriterium ist gemäß Rubrik für diese Frage nicht anwendbar.")
                metrics[criterion]["scores"].append(score)
                reviewed_ids.add(identifier)
            elif score is not None:
                raise ValueError("Unbewertete/nicht anwendbare Kriterien müssen score: null haben.")
            elif status == "not_applicable":
                if not reason.strip():
                    raise ValueError("Nicht anwendbare Kriterien benötigen eine Begründung.")
                metrics[criterion]["not_applicable_count"] += 1
            else:
                metrics[criterion]["unreviewed_count"] += 1
            assessments[criterion] = {"status": status, "score": score, "reason": reason}
        rows.append({"question_id": identifier, "query": case.get("query", ""), "assessments": assessments})
    if reviewed_ids and (not isinstance(reviews.get("reviewer"), str) or not reviews["reviewer"].strip()):
        raise ValueError("Bei vergebenen Noten muss ein Reviewer angegeben sein.")
    for values in metrics.values():
        scores = values.pop("scores")
        values.update(reviewed_count=len(scores), mean=sum(scores) / len(scores) if scores else None)
    return {"generated_cases": len(cases), "cases_with_scores": len(reviewed_ids), "reviewer": reviews.get("reviewer", ""),
            "source_report_sha256": source_hash, "rubric_sha256": rubric_hash, "criteria": metrics, "cases": rows}


def markdown_report(summary: dict, source_path: Path) -> str:
    """Zeigt Mittelwerte und Abdeckung getrennt sowie sämtliche Begründungen."""
    escape = lambda value: str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    lines = ["# Manuelle RAG-Antwortbewertung", "", f"Quellbericht: `{escape(source_path.name)}`",
             f"Reviewer: {escape(summary['reviewer']) or 'noch nicht angegeben'}", "",
             f"Generierte Antworten: **{summary['generated_cases']}**. Fälle mit mindestens einer Note: **{summary['cases_with_scores']}**.", "",
             "| Kriterium | Mittelwert (0–2) | Bewertet | Unbewertet | Nicht anwendbar |", "|---|---:|---:|---:|---:|"]
    for criterion, label in CRITERIA.items():
        values = summary["criteria"][criterion]
        mean = f"{values['mean']:.3f}" if values["mean"] is not None else "nicht bewertet"
        lines.append(f"| {label} | {mean} | {values['reviewed_count']} | {values['unreviewed_count']} | {values['not_applicable_count']} |")
    lines += ["", "Mittelwert = Summe der vergebenen Noten / Anzahl tatsächlich bewerteter Fälle pro Kriterium. "
              "Eine echte Note 0 zählt mit; unbewertet, ausgelassen und nicht anwendbar zählen nicht als 0. "
              "Ohne Noten bleibt der Mittelwert null bzw. nicht bewertet. Kein Gesamtwert über verschiedene Kriterien wird gebildet.", "",
              "## Einzelbewertungen", "", "| Frage | Kriterium | Status | Note | Begründung |", "|---|---|---|---:|---|"]
    for row in summary["cases"]:
        for criterion, assessment in row["assessments"].items():
            score = assessment["score"] if assessment["score"] is not None else "–"
            lines.append(f"| {escape(row['question_id'] + ': ' + row['query'])} | {CRITERIA[criterion]} | {assessment['status']} | {score} | {escape(assessment['reason'])} |")
    lines += ["", "## Zuordnung und Grenzen", "", f"Quellbericht SHA-256: `{summary['source_report_sha256']}`",
              f"Rubrik SHA-256: `{summary['rubric_sha256']}`", "",
              "Die Noten stammen ausschließlich aus der separaten manuellen Review-Datei. "
              "Dieser Lauf erzeugt weder Antworten noch automatische LLM-Bewertungen. "
              "Geringe Bewertungsabdeckung und subjektive Urteile begrenzen die Aussagekraft. "
              "Es erfolgt kein Zugriff auf OpenAI, Retrieval, .env oder SQLite.", ""]
    return "\n".join(lines)


def write_new(path: Path, text: str) -> None:
    """Erzeugt ausschließlich neue Dateien; vorhandene Berichte bleiben unverändert."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
