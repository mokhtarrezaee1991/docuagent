"""Sicherheits- und SDK-Vertragstests: ausschließlich lokale MockTransports."""

from copy import deepcopy
from hashlib import sha256
import json
import logging
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import httpx
import pytest

from app.services import rag_service
from evaluation import answer_pilot as pilot
from evaluation import answer_pilot_live as live
from evaluation import answer_quality as quality
from scripts import run_answer_pilot as cli

TEST_SECRET = "unit-test-credential-must-never-be-saved"


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    # Selbst bei fehlerhafter Freigabelogik darf kein echter Transport entstehen.
    monkeypatch.setattr(httpx, "HTTPTransport", Mock(side_effect=AssertionError("Echtes Netzwerk verboten")))


@pytest.fixture(scope="module")
def base_plan():
    return pilot.prepare_pilot(live.DATASET)


@pytest.fixture
def setup(tmp_path, monkeypatch, base_plan):
    monkeypatch.setattr(live, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(live, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setenv("OPENAI_API_KEY", TEST_SECRET)
    plan = deepcopy(base_plan)
    artifact = tmp_path / "dry-run.json"
    artifact.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    digest = sha256(artifact.read_bytes()).hexdigest()
    grant = {"version": 1, "approved": True, "purpose": "docuagent_answer_pilot_7e2",
        "run_id": plan["run_id"], "dry_run_sha256": digest, "case_ids": list(pilot.CASE_IDS), "max_attempts": 6,
        "runner_sha256": live._digest(Path(live.__file__)),
        "cli_sha256": live._digest(live.ROOT / "backend/scripts/run_answer_pilot.py")}
    approval = tmp_path / "test-only-approval.json"
    approval.write_text(json.dumps(grant), encoding="utf-8")
    return artifact, plan, digest, approval


def run(setup, handler):
    artifact, plan, digest, approval = setup
    return live.execute_live(artifact, plan["run_id"], digest, enabled=True,
                             approval_path=approval, transport=httpx.MockTransport(handler))


def provider_response(request, *, invalid_source=False, malformed=False, secret_in_text=False):
    sent = json.loads(request.content)
    source = sent["input"]
    context = json.loads(source)["context"]
    reference = {key: context[0][key] for key in ("document_id", "chunk_id", "page_number")}
    if invalid_source:
        reference["document_id"] = "excluded-document"
    text = json.dumps({"answer": TEST_SECRET if secret_in_text else "Unveränderte synthetische Mock-Antwort.", "sources": [reference]}, ensure_ascii=False)
    if malformed:
        text = '{"answer": 42, "sources": "kein Array"}'
    body = {"id": "resp_mock_only", "object": "response", "created_at": 123,
            "status": "completed", "model": "gpt-4.1-mini-test-mock",
            "output": [{"id": "msg_mock", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}
    return httpx.Response(200, json=body, headers={"set-cookie": TEST_SECRET, "x-private": TEST_SECRET})


def test_default_disabled_and_missing_approval_never_reserve_or_send(setup):
    artifact, plan, digest, _ = setup
    handler = Mock()
    with pytest.raises(live.PilotError, match="deaktiviert"):
        live.execute_live(artifact, plan["run_id"], digest, transport=httpx.MockTransport(handler))
    with pytest.raises(live.PilotError, match="Freigabedatei"):
        live.execute_live(artifact, plan["run_id"], digest, enabled=True, transport=httpx.MockTransport(handler))
    assert not live.STATE_ROOT.exists()
    handler.assert_not_called()


@pytest.mark.parametrize("change", ["run_id", "hash", "context", "instructions", "case_order", "code_path", "software"])
def test_invalid_plan_blocks_before_network(setup, change):
    artifact, plan, digest, approval = setup
    expected_run = plan["run_id"]
    if change == "run_id":
        expected_run = "answer-pilot-" + "0" * 32
    elif change == "hash":
        digest = "0" * 64
    else:
        if change == "context":
            plan["rows"][0]["service_context"][0]["text"] = "Fremdes Dokument"
        elif change == "instructions":
            plan["rows"][0]["planned_request"]["wire_body"]["instructions"] = "Andere Instructions"
        elif change == "case_order":
            plan["rows"].reverse()
        elif change == "code_path":
            plan["code_sha256"] = {"backend/.env": "0" * 64}
        else:
            plan["software"]["openai"] = "0.0"
        artifact.write_text(json.dumps(plan), encoding="utf-8")
        digest = live._digest(artifact)
    handler = Mock()
    with pytest.raises(live.PilotError):
        live.execute_live(artifact, expected_run, digest, enabled=True, approval_path=approval, transport=httpx.MockTransport(handler))
    handler.assert_not_called()
    assert not live.STATE_ROOT.exists()


@pytest.mark.parametrize("field,value", [("approved", False), ("run_id", "other"), ("max_attempts", 7),
                                         ("dry_run_sha256", "0" * 64), ("runner_sha256", "0" * 64),
                                         ("cli_sha256", "0" * 64), ("max_attempts", True)])
def test_bad_grants_never_send(setup, field, value):
    _, _, _, approval = setup
    grant = json.loads(approval.read_bytes())
    grant[field] = value
    approval.write_text(json.dumps(grant), encoding="utf-8")
    handler = Mock()
    with pytest.raises(live.PilotError, match="Freigabe"):
        run(setup, handler)
    handler.assert_not_called()
    assert not live.STATE_ROOT.exists()


def test_completed_run_exact_requests_six_durable_reservations_and_v2(setup):
    calls = []

    def handler(request):
        # Dauerhafte Reservierung existiert bereits vor der lokalen HTTP-Delegation.
        reservations = list(live.STATE_ROOT.glob("attempt_*_reserved.json"))
        assert len(reservations) == len(calls) + 1
        body = json.loads(request.content)
        calls.append(body)
        assert body == setup[1]["rows"][len(calls) - 1]["planned_request"]["wire_body"]
        assert "forbidden_claims" not in body["input"] and "expected_fact" not in body["input"]
        return provider_response(request)

    result = run(setup, handler)
    report = result["report"]
    assert report["status"] == "completed" and report["generation_attempts"] == len(calls) == 6
    assert report["http_send_attempts"] == 6 and report["openai_api_calls"] == 0
    assert report["execution_origin"] == "local_mock_transport"
    assert report["skipped_case_ids"] == []
    inputs = quality.load_inputs(live.DATASET, Path(result["snapshot_path"]), Path(result["artifact_path"]))
    assert len(inputs[1].snapshots) == 6
    assert all(snapshot.claims == [] for snapshot in inputs[1].snapshots)
    assert all(snapshot.answer == "Unveränderte synthetische Mock-Antwort." for snapshot in inputs[1].snapshots)
    assert report["rows"][0]["raw_response"]["usage"]["total_tokens"] == 40
    assert report["rows"][0]["actual_request"]["sha256"] == report["rows"][0]["planned_request"]["request_sha256"]
    for path in list(live.STATE_ROOT.glob("*.json")) + list(live.REPORT_ROOT.glob("*.json")):
        assert TEST_SECRET not in path.read_text(encoding="utf-8")
    # Derselbe Pilot ist auch nach vollständigem Erfolg unwiderruflich beansprucht.
    with pytest.raises(live.PilotError, match="bereits beansprucht"):
        run(setup, Mock())


def test_different_run_id_cannot_reset_global_budget(setup):
    run(setup, provider_response)
    artifact, plan, _, approval = setup
    plan["run_id"] = "answer-pilot-" + "a" * 32
    artifact.write_text(json.dumps(plan), encoding="utf-8")
    digest = live._digest(artifact)
    grant = json.loads(approval.read_bytes())
    grant.update(run_id=plan["run_id"], dry_run_sha256=digest)
    approval.write_text(json.dumps(grant), encoding="utf-8")
    with pytest.raises(live.PilotError, match="bereits beansprucht"):
        run((artifact, plan, digest, approval), Mock())


def test_budget_limit_and_duplicate_case_are_durable(tmp_path):
    guard = live.SecretGuard(TEST_SECRET)
    ledger = live.RunLedger(tmp_path, "run", "0" * 64, guard)
    for identifier in pilot.CASE_IDS:
        ledger.reserve(identifier, {"synthetic": True})
    with pytest.raises(live.PilotError, match="Budget"):
        ledger.reserve("D01", {})
    assert len(list(tmp_path.glob("attempt_*_reserved.json"))) == 6


def test_parallel_process_claims_only_one_winner(tmp_path):
    code = "from pathlib import Path; import sys; from evaluation.answer_pilot_live import RunLedger, SecretGuard, PilotError\ntry:\n RunLedger(Path(sys.argv[1]), 'test-run', '0'*64, SecretGuard('test-secret')); sys.exit(0)\nexcept PilotError:\n sys.exit(9)"
    command = [sys.executable, "-c", code, str(tmp_path / "state")]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    results = [p.communicate(timeout=30) for p in processes]
    assert sorted(p.returncode for p in processes) == [0, 9], results


def test_hard_crash_after_reservation_blocks_restart(tmp_path):
    code = "from pathlib import Path; import os,sys; from evaluation.answer_pilot_live import RunLedger, SecretGuard\nr=RunLedger(Path(sys.argv[1]), 'test-run', '0'*64, SecretGuard('test-secret')); r.reserve('D01', {'mock':True}); os._exit(7)"
    root = tmp_path / "state"
    result = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, timeout=30)
    assert result.returncode == 7
    reserved = json.loads((root / "attempt_01_reserved.json").read_bytes())
    assert reserved["consumes_budget"] and reserved["outcome"] == "unknown_until_documented"
    with pytest.raises(live.PilotError, match="bereits beansprucht"):
        live.RunLedger(root, "other-run", "1" * 64, live.SecretGuard(TEST_SECRET))


@pytest.mark.parametrize("scenario", ["timeout", "http_429", "http_500", "redirect", "invalid_json", "parse_error", "invalid_source", "secret"])
def test_errors_recorded_stop_without_retry(setup, scenario):
    calls = []

    def handler(request):
        calls.append(request)
        if scenario == "timeout":
            raise httpx.ReadTimeout(TEST_SECRET, request=request)
        if scenario.startswith("http_"):
            return httpx.Response(int(scenario.split("_")[1]), json={"error": {"message": TEST_SECRET}}, headers={"set-cookie": TEST_SECRET})
        if scenario == "redirect":
            return httpx.Response(307, headers={"location": "https://example.invalid/secret"})
        if scenario == "invalid_json":
            return httpx.Response(200, content=b'not json')
        return provider_response(request, malformed=scenario == "parse_error", invalid_source=scenario == "invalid_source", secret_in_text=scenario == "secret")

    result = run(setup, handler)
    report = result["report"]
    assert report["status"] == "stopped_on_error" and report["generation_attempts"] == len(calls) == 1
    assert len(report["skipped_case_ids"]) == 5
    row = report["rows"][0]
    if scenario == "timeout":
        assert row["status"] == "transport_error"
    elif scenario == "invalid_source":
        assert row["status"] == "source_validation_or_service_error"
        assert row["raw_response"]["structured"]["sources"][0]["document_id"] == "excluded-document"
        inputs = quality.load_inputs(live.DATASET, Path(result["snapshot_path"]), Path(result["artifact_path"]))
        assert quality.evaluate(*inputs)["technical"]["reference_validity"] == 0
    elif scenario == "parse_error":
        assert row["http_response"]["body"]["output"][0]["content"][0]["text"] == '{"answer": 42, "sources": "kein Array"}'
        assert result["snapshot_path"] is None
    elif scenario == "secret":
        assert row["status"] == "sensitive_response_withheld" and result["snapshot_path"] is None
    else:
        assert "body_sha256" in row["http_response"]
    for path in list(live.STATE_ROOT.glob("*.json")) + list(live.REPORT_ROOT.glob("*.json")):
        assert TEST_SECRET not in path.read_text(encoding="utf-8")


def test_partial_success_preserved_and_failure_not_resumed(setup):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 2:
            raise httpx.ReadTimeout("mock", request=request)
        return provider_response(request)

    result = run(setup, handler)
    assert result["report"]["generation_attempts"] == 2
    assert [r["status"] for r in result["report"]["rows"]] == ["completed", "transport_error"]
    inputs = quality.load_inputs(live.DATASET, Path(result["snapshot_path"]), Path(result["artifact_path"]))
    assert len(inputs[1].snapshots) == 1
    with pytest.raises(live.PilotError):
        run(setup, Mock())


def test_keyboard_interrupt_consumes_attempt_and_saves_partial_report(setup):
    def handler(request):
        raise KeyboardInterrupt()
    result = run(setup, handler)
    assert result["report"]["status"] == "interrupted_or_persistence_error"
    assert result["report"]["generation_attempts"] == 1
    assert result["report"]["rows"][0]["status"] == "unknown_interrupted"
    with pytest.raises(live.PilotError):
        run(setup, Mock())


def test_debug_logs_cannot_expose_credentials(setup, caplog):
    before = logging.root.manager.disable
    def handler(request):
        logging.getLogger("openai.test").critical(TEST_SECRET)
        return provider_response(request)
    run(setup, handler)
    assert TEST_SECRET not in caplog.text
    assert logging.root.manager.disable == before


def test_cli_default_checks_only_and_live_without_grant_fails(setup):
    artifact, plan, digest, _ = setup
    arguments = ["--artifact", str(artifact), "--run-id", plan["run_id"], "--artifact-sha256", digest]
    assert cli.main(arguments) == 0
    assert not live.STATE_ROOT.exists()
    assert cli.main([*arguments, "--live"]) == 2
    assert not live.STATE_ROOT.exists()


def test_existing_reports_never_overwritten(tmp_path):
    path = tmp_path / "existing.json"
    live.write_new(path, {"old": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        live.write_new(path, {"new": True})
    assert path.read_bytes() == original


def test_missing_environment_key_does_not_claim_pilot(setup, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    handler = Mock()
    with pytest.raises(live.PilotError, match="Prozessumgebung"):
        run(setup, handler)
    handler.assert_not_called()
    assert not live.STATE_ROOT.exists()


def test_reservation_write_failure_prevents_http_and_blocks_restart(setup, monkeypatch):
    original = live.write_new

    def broken(path, payload, guard=None):
        if path.name.startswith("attempt_"):
            raise OSError("simulierter Datenträgerfehler")
        return original(path, payload, guard)

    monkeypatch.setattr(live, "write_new", broken)
    handler = Mock()
    result = run(setup, handler)
    handler.assert_not_called()
    assert result["report"]["status"] == "stopped_on_error"
    assert (live.STATE_ROOT / "pilot_claim.json").exists()
    with pytest.raises(live.PilotError, match="bereits beansprucht"):
        run(setup, handler)


@pytest.mark.parametrize("change", ["body", "endpoint", "second_send"])
def test_transport_rejects_unapproved_request_before_delegate(setup, change):
    row = deepcopy(setup[1]["rows"][0])
    row["attempt"] = 1
    guard = live.SecretGuard(TEST_SECRET)
    ledger = live.RunLedger(live.STATE_ROOT, "unit-test", "0" * 64, guard)
    handler = Mock()
    transport = live._RecordingTransport(httpx.MockTransport(handler), row, ledger, guard)
    body = deepcopy(row["planned_request"]["wire_body"])
    if change == "body":
        body["instructions"] = "Nicht freigegebene Instruktion"
    if change == "second_send":
        transport.sent = True
    url = "https://example.invalid/v1/responses" if change == "endpoint" else "https://api.openai.com/v1/responses"
    with pytest.raises(live.PilotError):
        transport.handle_request(httpx.Request("POST", url, json=body))
    handler.assert_not_called()
