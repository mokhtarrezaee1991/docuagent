"""Erstellt reine Offline-Berichte und optional eine unbewertete Review-Vorlage."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.answer_quality import Reviews, evaluate, load_inputs, review_template

DATASET = ROOT / "data/evaluation/answer_quality_cases.json"
FIXTURES = ROOT / "data/evaluation/answer_quality_fixtures.json"


def write_new(path, payload):
    """Exklusiv erstellen; niemals Berichte, Quellen oder Reviews überschreiben."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--run-artifact", type=Path, help="Gebundenes Live-Artefakt für echte v2-Modellantworten.")
    parser.add_argument("--init-review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "docs" / f"answer_quality_offline_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}.json"
    if args.reviews and args.init_review:
        parser.error("Review entweder laden oder neu vorbereiten.")
    if output.exists() or (args.init_review and args.init_review.exists()):
        parser.error("Ausgabedatei existiert bereits. Bitte neue Pfade wählen.")
    if args.init_review and args.init_review.resolve() == output.resolve():
        parser.error("Bericht und Review-Vorlage benötigen unterschiedliche Pfade.")
    dataset, fixtures, dataset_hash, fixtures_hash = load_inputs(args.dataset, args.fixtures, args.run_artifact)
    reviews = Reviews.model_validate_json(args.reviews.read_bytes()) if args.reviews else None
    report = evaluate(dataset, fixtures, dataset_hash, fixtures_hash, reviews)
    if args.init_review:
        write_new(args.init_review, review_template(dataset, fixtures, dataset_hash, fixtures_hash).model_dump(mode="json"))
    write_new(output, report)
    print(json.dumps({key: report[key] for key in ("snapshot_count", "coverage", "technical", "semantic", "pending_review_units", "snapshots_pending_review", "facet_outcomes")}, ensure_ascii=False, indent=2))
    print(f"Eingabe: {report['input_kind']}; Offline-Auswertung, OpenAI-Aufrufe: 0. Bericht: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
