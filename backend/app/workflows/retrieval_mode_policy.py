"""Isolierte Modusentscheidung ohne Regeln, Modelle oder produktive Anbindung."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.workflows.search_policy import SearchMode


class ModeDecision(BaseModel):
    """Validiertes Ergebnis; eine Fallback-Entscheidung ist kein Qualitätsversprechen."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    mode: SearchMode
    routing_reason: Literal["automation_disabled", "no_validated_rule"]
    used_fallback: Literal[True] = True


def choose_mode(query: str, *, fallback_mode: str, enabled: bool = False) -> ModeDecision:
    """Behält ohne validierte Regel immer den konfigurierten Fallback bei.

    Explizite Modusvorgaben muss ein späterer Aufrufer vor dieser optionalen
    Automatik behandeln. fallback_mode bezeichnet ausschließlich Konfiguration,
    keine explizite Nutzeranweisung. Dokumentauswahl ist nicht Teil dieser Policy.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Die Frage muss ein nichtleerer Text sein.")
    if type(enabled) is not bool:
        raise ValueError("enabled muss ein boolescher Wert sein.")
    if not isinstance(fallback_mode, str) or fallback_mode not in ("semantic", "hybrid", "hybrid_de"):
        raise ValueError("fallback_mode muss semantic, hybrid oder hybrid_de sein.")
    return ModeDecision(mode=fallback_mode, routing_reason="no_validated_rule" if enabled else "automation_disabled")
