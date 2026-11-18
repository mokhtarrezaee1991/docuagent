"""Prüft persistente Vektoren, Transaktionen und Laden in einem frischen Prozess."""

import json
import sqlite3
import subprocess
import sys

import pytest

from app.services.document_store import SQLiteDocumentStore


def record(document_id="eins", filename="gleich.pdf"):
    """Enthält ausschließlich künstliche Metadaten, Texte und Zahlen."""
    return {
        "metadata": {"document_id": document_id, "filename": filename, "page_count": 2, "chunk_count": 1},
        "chunks": [{"chunk_id": 1, "page_number": 2, "text": "Künstlicher Text: Grüße", "embedding": [0.25, -0.75]}],
    }


def test_save_reload_and_delete(tmp_path):
    path = tmp_path / "nested" / "documents.sqlite3"
    store = SQLiteDocumentStore(path)
    store.save_document(record())
    store.save_document(record("zwei"))
    loaded = SQLiteDocumentStore(path)
    assert loaded.load_documents() == ([record(), record("zwei")], False)
    assert loaded.delete_document("eins")
    assert not loaded.delete_document("fehlt")
    assert SQLiteDocumentStore(path).load_documents() == ([record("zwei")], False)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunks WHERE document_id = 'eins'").fetchone()[0] == 0


def test_replacement_rolls_back_on_invalid_chunks(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.save_document(record())
    invalid = record("zwei")
    invalid["chunks"] *= 2
    with pytest.raises(sqlite3.IntegrityError):
        store.save_document(invalid, replace_all=True)
    assert store.load_documents() == ([record()], False)
    store.save_document(record("drei"), replace_all=True)
    assert store.load_documents() == ([record("drei")], True)


def test_filename_is_parameterized(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    data = record(filename="test'); DROP TABLE documents; --.pdf")
    store.save_document(data)
    assert store.load_documents()[0] == [data]


def test_load_in_fresh_python_process_without_model(tmp_path):
    path = tmp_path / "documents.sqlite3"
    SQLiteDocumentStore(path).save_document(record())
    code = (
        "import json,sys; from pathlib import Path; "
        "from app.services.document_store import SQLiteDocumentStore; "
        "print(json.dumps(SQLiteDocumentStore(Path(sys.argv[1])).load_documents()[0])); "
        "assert 'sentence_transformers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code, str(path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [record()]
