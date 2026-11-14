"""Prüft die sichere Diagnose ohne Netz, Schlüsseldateien oder echte LLM-Aufrufe."""

from unittest.mock import Mock

import pymupdf

from app import main
from app.services import semantic_search
from scripts.diagnose_multi_document import diagnose


def test_diagnostic_captures_password_context(tmp_path, monkeypatch, tokenizer):
    model = Mock(tokenizer=tokenizer, max_seq_length=128)
    model.encode.side_effect = lambda texts, **kwargs: [
        [1.0, 0.0] if "14 Zeichen" in text or "Passwort" in text else [0.0, 1.0]
        for text in texts
    ]
    monkeypatch.setattr(semantic_search, "_get_model", lambda: model)
    texts = ["Künstliche Urlaubsregel.", "Fiktiver Vorspann.\n1. Passwörter\nMindestens 14 Zeichen verwenden."]
    paths = []
    for index, text in enumerate(texts):
        path = tmp_path / f"test_{index}.pdf"
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), text)
            path.write_bytes(document.tobytes())
        paths.append(path)
    previous = getattr(main.app.state, "retrieval_pipeline", None)
    report = diagnose(paths)
    assert report["real_openai_calls"] == 0
    assert all(document["text_preserved"] for document in report["documents"])
    assert "14 Zeichen" in report["ranking"][0]["text"]
    assert "14 Zeichen" in report["generate_answer_chunks"][0]["text"]
    assert "14 Zeichen" in report["llm_payload"]["context"][0]["text"]
    assert main.app.state.retrieval_pipeline is previous
