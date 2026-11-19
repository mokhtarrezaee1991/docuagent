"""Erstellt und aggregiert lokale manuelle Reviews; keinerlei Modell-/API-Aufrufe."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.manual_reviews import aggregate_reviews, load_inputs, markdown_report, template, write_new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--report", type=Path, required=True)
        command.add_argument("--reviews", type=Path, required=True)
        command.add_argument("--rubric", type=Path, default=ROOT / "data/evaluation/answer_review_rubric.json")
        if name == "init":
            command.add_argument("--reviewer", default="")
        else:
            command.add_argument("--output", type=Path, required=True, help="Neuer Markdown-Pfad; bestehende Dateien werden nie überschrieben.")
    args = parser.parse_args(argv)
    try:
        report, rubric, source_hash, rubric_hash = load_inputs(args.report, args.rubric)
        if args.command == "init":
            result = template(report, source_hash, rubric_hash, args.reviewer)
            write_new(args.reviews, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"Neue Review-Datei: {args.reviews}; generierte Antworten: {len(result['reviews'])}. Es wurden keine Noten vergeben.")
        else:
            reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
            result = aggregate_reviews(report, reviews, source_hash, rubric_hash)
            write_new(args.output, markdown_report(result, args.report))
            print(f"Generierte Antworten: {result['generated_cases']}; Fälle mit Noten: {result['cases_with_scores']}")
            for criterion, values in result["criteria"].items():
                mean = f"{values['mean']:.3f}" if values["mean"] is not None else "nicht bewertet"
                print(f"{criterion}: {mean}; bewertet: {values['reviewed_count']}; unbewertet: {values['unreviewed_count']}; nicht anwendbar: {values['not_applicable_count']}")
            print(f"Neuer Markdown-Bericht: {args.output}")
    except FileExistsError:
        print("Zieldatei existiert bereits. Bitte einen neuen Dateinamen wählen; nichts wurde überschrieben.", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Review-Auswertung abgebrochen: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
