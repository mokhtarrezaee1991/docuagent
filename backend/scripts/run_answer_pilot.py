"""Prüft einen eingefrorenen Pilot; Versand nur mit separater gebundener Freigabe."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.answer_pilot_live import PilotError, execute_live, validate_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--live", action="store_true", help="Nur nach separat erteilter Freigabe verwenden.")
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args(argv)
    try:
        # Ohne --live wird ausschließlich lokal mit MockTransport validiert.
        plan = validate_plan(args.artifact, args.run_id, args.artifact_sha256)
        if not args.live:
            print("Artefakt gültig. Live deaktiviert; kein Budget reserviert, keine Netzwerkaufrufe.")
            return 0
        result = execute_live(args.artifact, args.run_id, args.artifact_sha256.lower(), enabled=True, approval_path=args.approval)
        print(f"Laufstatus: {result['report']['status']}; Versuche: {result['report']['generation_attempts']}.")
        print(f"Artefakt: {result['artifact_path']}; Snapshots: {result['snapshot_path'] or 'keine parsebaren Antworten'}")
        return 0 if result["report"]["status"] == "completed" else 1
    except PilotError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        print("Pilot sicher abgebrochen; lokale Dateien und Sperre prüfen. Keine automatische Wiederholung.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
