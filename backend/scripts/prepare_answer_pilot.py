"""Exportiert Phase 7e-1 ausschließlich lokal; kein Live-Modus vorhanden."""

import argparse
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.answer_pilot import prepare_pilot
from scripts.evaluate_answer_quality import write_new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "docs" / f"answer_pilot_dry_run_{uuid4().hex}.json"
    if output.exists():
        parser.error("Ausgabedatei existiert bereits; neuen Pfad verwenden.")
    report = prepare_pilot(ROOT / "data/evaluation/answer_quality_cases.json")
    write_new(output, report)
    failed = sum(row["status"] != "prepared" for row in report["rows"])
    print(f"Run-ID: {report['run_id']}; Fälle: 6; Vorbereitungsfehler: {failed}; OpenAI-Aufrufe: 0.")
    print(f"Bericht: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
