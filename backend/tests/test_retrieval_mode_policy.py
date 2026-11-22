"""Prüft ausschließlich deterministische Fallback-Entscheidungen."""

import pytest
from pydantic import ValidationError

from app.workflows.retrieval_mode_policy import ModeDecision, choose_mode


@pytest.mark.parametrize("mode", ["semantic", "hybrid", "hybrid_de"])
@pytest.mark.parametrize("enabled", [False, True])
def test_fallback_is_never_changed(mode, enabled):
    decision = choose_mode("Passwort 123 Frist und englische Begriffe", fallback_mode=mode, enabled=enabled)
    assert decision.mode == mode and decision.used_fallback
    assert decision.routing_reason == ("no_validated_rule" if enabled else "automation_disabled")


def test_default_is_disabled_and_environment_is_not_used(monkeypatch):
    monkeypatch.setenv("DOCUAGENT_RETRIEVAL_MODE", "hybrid_de")
    assert choose_mode("Frage", fallback_mode="hybrid") == ModeDecision(mode="hybrid", routing_reason="automation_disabled")


@pytest.mark.parametrize("mode", [None, "", "auto", "bm25", "SEMANTIC", 1, True, []])
def test_invalid_mode(mode):
    with pytest.raises(ValueError, match="fallback_mode"):
        choose_mode("Frage", fallback_mode=mode)


@pytest.mark.parametrize("query", ["", "  ", None, 1])
def test_invalid_query(query):
    with pytest.raises(ValueError):
        choose_mode(query, fallback_mode="semantic")


@pytest.mark.parametrize("enabled", [1, "true", None])
def test_enabled_is_strict(enabled):
    with pytest.raises(ValueError, match="enabled"):
        choose_mode("Frage", fallback_mode="semantic", enabled=enabled)


def test_decision_is_immutable_and_rejects_unknown_fields():
    decision = choose_mode("Frage", fallback_mode="semantic")
    with pytest.raises(ValidationError):
        decision.mode = "hybrid"
    with pytest.raises(ValidationError):
        ModeDecision(mode="bm25", routing_reason="automation_disabled")
    with pytest.raises(ValidationError):
        ModeDecision(mode="semantic", routing_reason="automation_disabled", document_ids=[])
