"""Isolierter Pilot-Runner: Freigabebindung, Einmalstartsperre und dauerhaftes Budget."""

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import re
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx

from app.services import rag_service
from evaluation.answer_pilot import CASE_IDS, MAX_CALLS, RecordingResponses, capture_request, describe_request, fingerprint
from evaluation.answer_quality import Dataset, ModelSnapshot, ModelSnapshots, Source, validate_model_artifact

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/evaluation/answer_quality_cases.json"
# Ein einziges Budget für diesen Pilot, auch bei anderer Run-ID oder Artefaktkopie.
STATE_ROOT = ROOT / "data/storage/answer_pilot_7e"
REPORT_ROOT = ROOT / "docs"
ORIGIN = "https://api.openai.com/v1"


class PilotError(RuntimeError):
    """Nur feste, nicht vertrauliche Fehlermeldungen nach außen geben."""


@contextmanager
def _quiet_network():
    """Bibliotheks-Debuglogs im isolierten Prozess dürfen keine Header ausgeben."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate_plan(path: Path, run_id: str, digest: str) -> dict:
    """Prüft eingefrorene Daten und reproduziert den Request nur offline."""
    if path.suffix.lower() != ".json":
        raise PilotError("Nur eine explizite JSON-Artefaktdatei ist zulässig.")
    raw = path.read_bytes()
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest) or sha256(raw).hexdigest() != digest.lower():
        raise PilotError("Artefakt-SHA-256 stimmt nicht überein.")
    try:
        plan = json.loads(raw)
        if not isinstance(plan, dict) or plan.get("run_id") != run_id or not re.fullmatch(r"answer-pilot-[0-9a-f]{32}", run_id):
            raise PilotError("Run-ID stimmt nicht überein.")
        configuration = {"context_mode": "frozen", "retrieval_performed": False,
                         "max_generation_calls": 6, "automatic_retries": 0,
                         "judge_calls": 0, "workflow_comparison": False, "live_enabled": False}
        if type(plan.get("version")) is not int or plan["version"] != 1 or plan.get("mode") != "dry-run" or fingerprint(plan.get("configuration")) != fingerprint(configuration):
            raise PilotError("Unzulässige Pilot-Konfiguration.")
        if plan.get("case_ids") != list(CASE_IDS) or any(type(plan.get(key)) is not int or plan[key] != 0 for key in ("openai_api_calls", "generated_answers")):
            raise PilotError("Unzulässiger Fallumfang oder bereits verwendetes Artefakt.")
        if plan.get("software") != {"python": platform.python_version(), **{name: version(name) for name in ("openai", "httpx", "pydantic")}}:
            raise PilotError("Softwarestand hat sich geändert; neuen Dry-Run vorbereiten.")
        # Keine beliebigen Dateipfade aus einem Artefakt lesen, insbesondere keine .env.
        paths = [ROOT / name for name in (
            "backend/app/services/rag_service.py", "backend/evaluation/answer_pilot.py",
            "backend/evaluation/answer_quality.py", "backend/scripts/prepare_answer_pilot.py")]
        expected_code = {str(p.relative_to(ROOT)): _digest(p) for p in paths}
        if plan.get("code_sha256") != expected_code or plan.get("dataset_sha256") != _digest(DATASET):
            raise PilotError("Code- oder Datensatz-Fingerprint hat sich geändert.")
        dataset = Dataset.model_validate_json(DATASET.read_bytes())
        cases = {c.case_id: c for c in dataset.cases}
        names = {d.document_id: d.filename for d in dataset.documents}
        rows = plan.get("rows")
        if not isinstance(rows, list) or [row.get("case_id") for row in rows] != list(CASE_IDS):
            raise PilotError("Pilot-Fälle sind nicht vollständig und eindeutig geordnet.")
        for row in rows:
            case = cases[row["case_id"]]
            context = [c.model_dump() for c in case.context]
            service_context = [{**c, "filename": names[c["document_id"]]} for c in context]
            expected = {"query": case.query, "allowed_document_ids": case.allowed_document_ids,
                        "context": context, "service_context": service_context, "split": case.split,
                        "context_sha256": fingerprint(context), "case_sha256": fingerprint(case.model_dump()),
                        "context_origin": "answer_quality_cases.json: gespeicherter Kontext, kein Retrieval",
                        "status": "prepared", "result": None}
            if any(fingerprint(row.get(key)) != fingerprint(value) for key, value in expected.items()):
                raise PilotError("Frage, Kontext, Bereich oder Fallbindung weicht ab.")
            actual = capture_request(case.query, service_context)
            if fingerprint(row.get("planned_request")) != fingerprint(actual):
                raise PilotError("Die produktive Anfrage stimmt nicht mit dem Dry-Run überein.")
        return plan
    except PilotError:
        raise
    except Exception:
        raise PilotError("Dry-Run-Artefakt ist ungültig.") from None


def verify_approval(path: Path | None, plan: dict, digest: str):
    """Lokale explizite Erklärung, keine automatisch erteilte Freigabe."""
    if path is None:
        raise PilotError("Separate Freigabedatei fehlt; Live bleibt deaktiviert.")
    if path.suffix.lower() != ".json":
        raise PilotError("Nur eine explizite JSON-Freigabedatei ist zulässig.")
    try:
        grant = json.loads(path.read_bytes())
    except Exception:
        raise PilotError("Freigabedatei ist nicht lesbar.") from None
    expected = {"version": 1, "approved": True, "purpose": "docuagent_answer_pilot_7e2",
                "run_id": plan["run_id"], "dry_run_sha256": digest.lower(),
                "case_ids": list(CASE_IDS), "max_attempts": MAX_CALLS,
                "runner_sha256": _digest(Path(__file__)),
                "cli_sha256": _digest(ROOT / "backend/scripts/run_answer_pilot.py")}
    if grant != expected or type(grant.get("version")) is not int or type(grant.get("max_attempts")) is not int or grant.get("approved") is not True:
        raise PilotError("Freigabe ist unvollständig oder passt nicht zu Lauf, Budget und Runner.")


class SecretGuard:
    """Bei sensiblen Inhalten abbrechen statt Originalantworten still zu verändern."""

    def __init__(self, secret: str):
        self.secret = secret

    def check(self, value):
        def walk(item):
            if isinstance(item, dict):
                if any(str(k).casefold() in {"authorization", "proxy-authorization", "api_key", "access_token", "cookie", "set-cookie", "headers"} for k in item):
                    raise PilotError("Sensible Felder werden nicht gespeichert.")
                for child in item.values():
                    walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)
            elif isinstance(item, str):
                if self.secret in item or re.search(r"(?i)(?:bearer\s+\S+|sk-(?:proj-)?[a-z0-9_-]{12,}|authorization\s*:|set-cookie\s*:)", item):
                    raise PilotError("Verdächtiger Inhalt wird nicht gespeichert.")
        walk(value)


def write_new(path: Path, value: dict, guard: SecretGuard | None = None):
    """Exklusiv erstellen und vor Fortsetzung auf den Datenträger synchronisieren.

    Eine bei Absturz unvollständige Datei bleibt bestehen und sperrt sicher.
    Keine Ersetzung und keine Löschung bestehender Marker oder Berichte.
    """
    if guard is not None:
        guard.check(value)
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class RunLedger:
    """Dauerhafte Einmalstartsperre; auch ein Prozessabsturz gibt sie nicht frei."""

    def __init__(self, root: Path, run_id: str, digest: str, guard: SecretGuard):
        self.root, self.guard = root, guard
        self.attempts, self.events, self.cases = 0, 0, set()
        try:
            write_new(root / "pilot_claim.json", {"run_id": run_id, "dry_run_sha256": digest,
                "claimed_at": _now(), "max_attempts": MAX_CALLS, "resume_allowed": False}, guard)
        except FileExistsError:
            raise PilotError("Pilot bereits beansprucht; keine Fortsetzung oder Wiederholung erlaubt.") from None

    def reserve(self, case_id: str, request: dict) -> int:
        if self.attempts >= MAX_CALLS or case_id not in CASE_IDS or case_id in self.cases:
            raise PilotError("Budget erschöpft oder Fall bereits reserviert.")
        attempt = self.attempts + 1
        write_new(self.root / f"attempt_{attempt:02d}_reserved.json", {
            "attempt": attempt, "case_id": case_id, "reserved_at": _now(),
            "request": request, "outcome": "unknown_until_documented", "consumes_budget": True}, self.guard)
        self.attempts = attempt
        self.cases.add(case_id)
        return attempt

    def event(self, case_id: str, kind: str, data: dict):
        self.events += 1
        write_new(self.root / f"event_{self.events:04d}.json", {
            "event": self.events, "at": _now(), "case_id": case_id, "kind": kind, "data": data}, self.guard)


class _RecordingTransport(httpx.BaseTransport):
    """Genau ein Versand pro Reservierung, ohne Redirects oder Transport-Retry."""

    def __init__(self, delegate, row, ledger, guard):
        self.delegate, self.row, self.ledger, self.guard = delegate, row, ledger, guard
        self.sent = False

    def handle_request(self, request):
        if self.sent or not self.row.get("attempt"):
            raise PilotError("Kein weiterer HTTP-Versand erlaubt.")
        if request.method != "POST" or request.url.scheme != "https" or request.url.host != "api.openai.com" or request.url.port not in (None, 443) or request.url.path != "/v1/responses":
            raise PilotError("Unerwartetes API-Ziel.")
        body = json.loads(request.content)
        if fingerprint(body) != fingerprint(self.row["planned_request"]["wire_body"]):
            raise PilotError("Tatsächlicher Request weicht vom freigegebenen Plan ab.")
        self.guard.check(body)
        self.row["actual_request"] = {"wire_body": body, "sha256": fingerprint(body)}
        self.ledger.event(self.row["case_id"], "before_http_send", self.row["actual_request"])
        self.sent = True
        self.row["http_send_attempted"] = True
        try:
            response = self.delegate.handle_request(request)
            raw = response.read()
        except Exception:
            self.row["transport_error"] = True
            self.ledger.event(self.row["case_id"], "transport_error", {"code": "timeout_or_connection_error"})
            raise PilotError("Transport fehlgeschlagen; keine Wiederholung.") from None
        observation = {"status_code": response.status_code, "body_sha256": sha256(raw).hexdigest(), "byte_count": len(raw)}
        # HTTP-Fehlermeldungen können Schlüssel spiegeln: nur Status und Hash.
        if response.is_success:
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                observation["body_kind"] = "invalid_json_withheld"
            else:
                try:
                    # Kein arbitrary root-Metadatum; Antwortinhalt bleibt unverändert.
                    selected = {key: body[key] for key in ("id", "object", "created_at", "model", "status", "output", "usage", "incomplete_details") if key in body}
                    self.guard.check(selected)
                except Exception:
                    observation["body_kind"] = "sensitive_or_invalid_body_withheld"
                    self.row["http_response"] = observation
                    self.ledger.event(self.row["case_id"], "http_response", observation)
                    raise PilotError("Antwortinhalt wird aus Sicherheitsgründen nicht gespeichert.") from None
                observation.update(body_kind="response_fields", body=selected)
        else:
            observation["body_kind"] = "http_error_message_withheld"
        self.row["http_response"] = observation
        self.ledger.event(self.row["case_id"], "http_response_before_sdk_parse", observation)
        return response

    def close(self):
        self.delegate.close()


def execute_live(artifact_path: Path, run_id: str, digest: str, *, enabled=False, approval_path=None,
                 transport: httpx.MockTransport | None = None) -> dict:
    """Nur nach CLI-Freigabe; injizierbarer MockTransport für netzwerkfreie Tests.

    Keine produktive Pipeline importieren, keinen SQLite-Index öffnen. Der
    Freigabecheck erfolgt erneut, auch bei direktem Python-Aufruf.
    """
    if enabled is not True:
        raise PilotError("Live ist deaktiviert.")
    # Auch Python-Aufrufer können keine ungeprüften In-Memory-Pläne einschleusen.
    plan = validate_plan(artifact_path, run_id, digest)
    digest = digest.lower()
    verify_approval(approval_path, plan, digest)
    if transport is not None and not isinstance(transport, httpx.MockTransport):
        raise PilotError("Tests dürfen nur einen lokalen MockTransport injizieren.")
    secret = os.environ.get("OPENAI_API_KEY", "").strip()
    if not secret:
        raise PilotError("OPENAI_API_KEY muss in der Prozessumgebung gesetzt sein.")
    guard = SecretGuard(secret)
    # recording_plan.authorization enthält lediglich Anleitungstext; exportiert
    # werden nur die validierten Fallzeilen, keine allgemeinen Plan-Kommentare.
    guard.check(plan["rows"])
    ledger = RunLedger(STATE_ROOT, plan["run_id"], digest, guard)
    report = {"version": 1, "mode": "live", "run_id": plan["run_id"],
              "execution_id": uuid4().hex, "dry_run_sha256": digest,
              "dataset_sha256": plan["dataset_sha256"], "started_at": _now(),
              "execution_origin": "local_mock_transport" if transport is not None else "openai_api",
              "configuration": {"max_generation_calls": 6, "automatic_retries": 0, "judge_calls": 0,
                                "timeout_seconds": 30.0, "http_retries": 0, "follow_redirects": False},
              "runner_sha256": _digest(Path(__file__)), "rows": [], "status": "running"}
    active = None
    original_client = rag_service.OpenAI
    sdk_parse = None

    def delegate(**kwargs):
        return sdk_parse(**kwargs)

    def persist(record):
        nonlocal active
        if record["status"] == "attempt_reserved":
            if fingerprint(record["request"]) != fingerprint(active["planned_request"]["sdk_arguments"]):
                raise PilotError("SDK-Aufruf stimmt nicht mit dem Plan überein.")
            active["attempt"] = ledger.reserve(active["case_id"], record["request"])
        # Originale strukturierte Antwort noch vor Service-Quellenvalidierung.
        guard.check(record)
        if "raw_response" in record:
            active["raw_response"] = deepcopy(record["raw_response"])
        ledger.event(active["case_id"], record["status"], record)

    recorder = RecordingResponses(delegate, persist)

    def factory(**options):
        nonlocal sdk_parse
        if options.get("max_retries") != 0 or options.get("timeout") != 30.0:
            raise PilotError("Unzulässige SDK-Konfiguration.")
        base_transport = transport if transport is not None else httpx.HTTPTransport(retries=0)
        client = original_client(api_key=secret, base_url=ORIGIN, max_retries=0, timeout=30.0,
            http_client=httpx.Client(transport=_RecordingTransport(base_transport, active, ledger, guard),
                                    follow_redirects=False, trust_env=False, timeout=30.0))
        sdk_parse = client.responses.parse
        client.responses.parse = recorder.parse
        return client

    try:
        with _quiet_network(), patch.object(rag_service, "OpenAI", factory), patch.object(rag_service, "os", SimpleNamespace(environ={"OPENAI_API_KEY": secret})):
            for original_row in plan["rows"]:
                active = deepcopy(original_row)
                active.update(status="started", result=None)
                report["rows"].append(active)
                try:
                    active["result"] = rag_service.generate_answer(active["query"], active["service_context"])
                    active["status"] = "completed"
                    ledger.event(active["case_id"], "service_completed", {"result": active["result"]})
                except Exception:
                    active["status"] = _failure_status(active)
                    ledger.event(active["case_id"], "service_failed", {"status": active["status"]})
                    report["status"] = "stopped_on_error"
                    break
            else:
                report["status"] = "completed"
    except BaseException:
        # Auch KeyboardInterrupt/SystemExit erzeugen ein partielles Artefakt.
        # Bei hartem Prozessabbruch bleibt mindestens die dauerhafte Reservierung.
        report["status"] = "interrupted_or_persistence_error"
        if active is not None:
            active["status"] = "unknown_interrupted"
    report.update(finished_at=_now(), generation_attempts=ledger.attempts,
        http_send_attempts=sum(bool(r.get("http_send_attempted")) for r in report["rows"]),
        openai_api_calls=0 if transport is not None else sum(bool(r.get("http_send_attempted")) for r in report["rows"]),
        judge_api_calls=0, automatic_resume=False,
        skipped_case_ids=[identifier for identifier in CASE_IDS if identifier not in {row["case_id"] for row in report["rows"]}])
    artifact = REPORT_ROOT / f"answer_pilot_live_{report['execution_id']}.json"
    write_new(artifact, report, guard)
    ledger.event("pilot", "final_artifact", {"filename": artifact.name, "sha256": _digest(artifact), "status": report["status"]})
    snapshot_path = _export_snapshots(report, artifact, guard)
    return {"report": report, "artifact_path": str(artifact), "snapshot_path": str(snapshot_path) if snapshot_path else None}


def _failure_status(row):
    if row.get("transport_error"):
        return "transport_error"
    http = row.get("http_response", {})
    if http.get("status_code", 200) >= 300:
        return "http_error"
    if http.get("body_kind") == "sensitive_or_invalid_body_withheld":
        return "sensitive_response_withheld"
    raw = row.get("raw_response", {})
    if raw.get("structured") is not None:
        return "source_validation_or_service_error"
    return "sdk_parse_or_incomplete_response"


def _export_snapshots(report, artifact, guard):
    digest = _digest(artifact)
    snapshots = []
    for row in report["rows"]:
        raw = row.get("raw_response", {})
        parsed = raw.get("structured")
        if raw.get("status") != "completed" or parsed is None:
            continue
        # Keine Aussagen oder Zuordnungen erfinden; Inventar später annotieren.
        snapshots.append(ModelSnapshot(snapshot_id=f"{report['execution_id']}-{row['case_id']}",
            case_id=row["case_id"], provenance="openai_model_response", run_id=report["run_id"],
            run_artifact_sha256=digest, answer=parsed["answer"],
            sources=[Source.model_validate(source) for source in parsed["sources"]], claims=[]))
    if not snapshots:
        return None
    payload = ModelSnapshots(version=2, description="Aufgezeichnete Originalantworten; keine semantischen Bewertungen.",
        dataset_sha256=report["dataset_sha256"], run_id=report["run_id"], run_artifact_sha256=digest, snapshots=snapshots)
    dataset = Dataset.model_validate_json(DATASET.read_bytes())
    validate_model_artifact(dataset, payload, _digest(DATASET), artifact.read_bytes())
    path = REPORT_ROOT / f"answer_pilot_snapshots_{report['execution_id']}.json"
    write_new(path, payload.model_dump(mode="json"), guard)
    return path
