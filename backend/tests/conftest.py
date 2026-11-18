"""Gemeinsame kleine Tokenizer-Testdaten ohne Modell-Downloads."""

import pytest


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Jeder Test erhält eine eigene Datenbank und berührt keine Nutzerdokumente."""
    monkeypatch.setenv("DOCUAGENT_DB_PATH", str(tmp_path / "docuagent.sqlite3"))


class CharacterTokenizer:
    """Zählt pro Zeichen ein Token und fügt zwei spezielle Tokens hinzu."""

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=False):
        result = {"input_ids": list(range(len(text) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()
