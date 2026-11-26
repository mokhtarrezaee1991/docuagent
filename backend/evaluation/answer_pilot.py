"""Isolierter Offline-Pilot: produktive Anfragebildung, ausschließlich lokaler Transport."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch
from uuid import uuid4

import httpx

from app.services import rag_service
from evaluation.answer_quality import Dataset

CASE_IDS = ("D01", "D02", "D04", "T01", "T05", "T07")
MAX_CALLS = 6


def fingerprint(value) -> str:
    """Bindet JSON-Inhalte unabhängig von Einrückung und Schlüsselreihenfolge."""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def describe_request(kwargs: dict) -> dict:
    """Nur explizite API-Argumente aufnehmen, niemals Client-Optionen oder Header."""
    required = {"model", "instructions", "input", "text_format", "max_output_tokens", "store"}
    if set(kwargs) != required:
        raise ValueError("Die produktive SDK-Schnittstelle hat sich geändert; Plan erneut prüfen.")
    schema = kwargs["text_format"].model_json_schema()
    return {**{key: value for key, value in kwargs.items() if key != "text_format"},
            "text_format_class": kwargs["text_format"].__name__, "answer_schema": schema}


class _RequestCaptured(BaseException):
    """Stoppt vor einer Antwort, auch vor den Fehler-/Retry-Handlern des SDK."""


def capture_request(query: str, chunks: list[dict]) -> dict:
    """Führt die echte Service-/SDK-Serialisierung ohne Netzwerk oder Modell aus.

    Nur im separaten Evaluationsprozess verwenden: die beiden Service-Symbole
    werden vorübergehend ersetzt. Die Prozessumgebung wird weder gelesen noch
    verändert. Insbesondere wird kein tatsächlicher Schlüssel benötigt.
    """
    captured = {}
    real_client = rag_service.OpenAI

    def transport(request: httpx.Request):
        # Nur der JSON-Body ist freigegeben; niemals Header oder URL speichern.
        captured["wire_body"] = json.loads(request.content)
        raise _RequestCaptured()

    def factory(**options):
        if options.get("max_retries") != 0 or options.get("timeout") != 30.0:
            raise ValueError("Unerwartete Timeout-/Retry-Konfiguration.")
        captured["client_config"] = {"timeout_seconds": options["timeout"], "max_retries": options["max_retries"]}
        client = real_client(
            api_key="offline-placeholder", base_url="https://offline.invalid/v1",
            timeout=options["timeout"], max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(transport), trust_env=False),
        )
        original_parse = client.responses.parse

        def parse(**kwargs):
            captured["sdk_arguments"] = describe_request(kwargs)
            return original_parse(**kwargs)

        client.responses.parse = parse
        return client

    # Der Platzhalter bleibt ausschließlich im Speicher und wird nicht exportiert.
    environment = SimpleNamespace(environ={"OPENAI_API_KEY": "offline-placeholder"})
    with patch.object(rag_service, "os", environment), patch.object(rag_service, "OpenAI", factory):
        try:
            rag_service.generate_answer(query, deepcopy(chunks))
        except _RequestCaptured:
            pass
    if set(captured) != {"client_config", "sdk_arguments", "wire_body"}:
        raise ValueError("Die vorgesehene Modellanfrage konnte nicht vollständig erfasst werden.")
    captured["request_sha256"] = fingerprint(captured["wire_body"])
    captured["verification"] = {
        "service_arguments": "verified_offline", "installed_sdk_serialization": "verified_offline",
        "live_transmission": "unconfirmed", "server_model_revision": "unconfirmed",
        "server_defaults": "unconfirmed", "token_usage": "unconfirmed",
        "transport": "httpx.MockTransport; kein Netzwerk", "sdk_response": None,
    }
    return captured


def prepare_pilot(dataset_path: Path) -> dict:
    """Verwendet ausschließlich die sechs festgelegten Fälle und gespeicherten Kontexte."""
    raw = dataset_path.read_bytes()
    dataset = Dataset.model_validate_json(raw)
    cases = {case.case_id: case for case in dataset.cases}
    if any(identifier not in cases for identifier in CASE_IDS):
        raise ValueError("Der Pilot benötigt alle sechs festgelegten Fall-IDs.")
    filenames = {doc.document_id: doc.filename for doc in dataset.documents}
    rows = []
    for identifier in CASE_IDS:
        case = cases[identifier]
        context = [chunk.model_dump() for chunk in case.context]
        # Nur der zugehörige vorhandene Dateiname ergänzt das Service-Metadatum.
        service_context = [{**chunk, "filename": filenames[chunk["document_id"]]} for chunk in context]
        row = {
            "case_id": identifier, "query": case.query, "split": case.split,
            "allowed_document_ids": list(case.allowed_document_ids), "context": context,
            "service_context": service_context, "context_sha256": fingerprint(context),
            "case_sha256": fingerprint(case.model_dump()),
            "context_origin": "answer_quality_cases.json: gespeicherter Kontext, kein Retrieval",
            "status": "prepared", "result": None,
        }
        try:
            row["planned_request"] = capture_request(case.query, service_context)
        except Exception:
            # Keine Exception-Texte, Tracebacks, Client-Objekte oder Umgebungswerte.
            row.update(status="preparation_error", error_code="request_capture_failed")
        rows.append(row)
    root = Path(__file__).resolve().parents[2]
    return {
        "version": 1, "run_id": "answer-pilot-" + uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(), "mode": "dry-run",
        "dataset_sha256": sha256(raw).hexdigest(), "case_ids": list(CASE_IDS),
        "configuration": {"context_mode": "frozen", "retrieval_performed": False,
                          "max_generation_calls": MAX_CALLS, "automatic_retries": 0,
                          "judge_calls": 0, "workflow_comparison": False, "live_enabled": False},
        "openai_api_calls": 0, "generated_answers": 0, "rows": rows,
        "software": {"python": platform.python_version(), **{name: version(name) for name in ("openai", "httpx", "pydantic")}},
        "code_sha256": {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in (
            root / "backend/app/services/rag_service.py", Path(__file__),
            root / "backend/evaluation/answer_quality.py", root / "backend/scripts/prepare_answer_pilot.py")},
        "recording_plan": {
            "state": "not_live_enabled", "request": "SDK-Argumente und tatsächlichen JSON-Body ohne Header erfassen",
            "response": "strukturierte Originalantwort vor Service-Quellenvalidierung, output_text, Status, Modell, ID, Zeit und Tokenverbrauch; für SDK-Parsefehler später auch freigegebenen HTTP-Antwortbody erfassen",
            "persistence": "vor jedem Versand Versuch reservieren und protokollieren; danach Antwort, dann Service-Ergebnis separat sichern",
            "failure": "nach erstem Fehler abbrechen; keine Wiederaufnahme oder automatischen Wiederholungen",
            "authorization": "Phase 7e-2 separat freigeben, dabei Run-ID und Artefakt-SHA-256 sowie maximal sechs Aufrufe nennen",
        },
        "limitations": ["Keine Modellantwort und keine semantische Qualitätsmessung.",
                        "SDK-Serialisierung nur mit lokalem MockTransport verifiziert; keine bestätigte Live-Übertragung.",
                        "Kein Kostenbetrag geschätzt; Preise vor separater Live-Freigabe prüfen.",
                        "Temporäre Service-Symbole nur im isolierten Evaluationsprozess ersetzen, niemals in laufender FastAPI."],
    }


class RecordingResponses:
    """Delegierende Aufzeichnungsgrenze für Tests und die spätere Live-Vorbereitung.

    Öffnet selbst keinen Client und führt ohne übergebenes Callable nichts aus.
    Es gibt in Phase 7e-1 keinen produktiven oder CLI-seitigen Live-Aufrufer.
    Ein Recorder gilt für den gesamten Pilot, nicht pro Frage neu erstellen.
    """

    def __init__(self, delegate: Callable, persist: Callable[[dict], None]):
        self.delegate, self.persist = delegate, persist
        self.attempts = 0
        self.failed = False
        self.records = []

    def parse(self, **kwargs):
        if self.failed or self.attempts >= MAX_CALLS:
            raise RuntimeError("Pilot beendet oder Aufruflimit erreicht.")
        request = describe_request(kwargs)
        self.attempts += 1
        record = {"attempt": self.attempts, "request": request, "status": "attempt_reserved"}
        self.records.append(record)
        try:
            # Fehler beim Persistieren verhindert den Versand.
            self.persist(deepcopy(record))
            response = self.delegate(**kwargs)
            parsed = response.output_parsed
            record["raw_response"] = {
                "structured": None if parsed is None else parsed.model_dump(mode="json"),
                "output_text": getattr(response, "output_text", None),
                **{field: getattr(response, field, None) for field in ("id", "model", "status", "created_at")},
                "usage": _usage(getattr(response, "usage", None)),
            }
            record["status"] = "response_recorded_before_source_validation"
            self.persist(deepcopy(record))
            return response
        except Exception:
            self.failed = True
            record.update(status="failed", error_code="generation_or_recording_failed")
            # Auch bei einem zweiten Schreibfehler keine vertraulichen Details ausgeben.
            try:
                self.persist(deepcopy(record))
            except Exception:
                pass
            raise RuntimeError("Pilot-Aufruf oder Aufzeichnung fehlgeschlagen; keine Wiederholung.") from None


def _usage(usage) -> dict | None:
    """Nur definierte Tokenzähler, niemals beliebige SDK-Metadaten übernehmen."""
    if usage is None:
        return None
    result = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        result[field] = value if type(value) is int and value >= 0 else None
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    result["cached_input_tokens"] = cached if type(cached) is int and cached >= 0 else None
    return result
