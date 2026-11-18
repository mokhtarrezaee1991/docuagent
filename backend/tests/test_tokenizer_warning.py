"""Prüft echte Tokenizer-Warnungen ohne Modell-Download oder Netzwerk."""

from unittest.mock import Mock

from app.services.text_chunker import chunk_pages_by_tokens


def test_long_analysis_is_quiet_but_final_chunks_stay_bounded(monkeypatch):
    """Lange Analyse-Eingaben gehen nicht ans Modell und sollen nicht warnen."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    from transformers import tokenization_utils_base

    words = [f"wort{index}" for index in range(40)]
    vocabulary = {word: index for index, word in enumerate(["[UNK]", "[CLS]", "[SEP]", *words])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", model_max_length=8
    )
    warning = Mock(wraps=tokenization_utils_base.logger.warning)
    monkeypatch.setattr(tokenization_utils_base.logger, "warning", warning)
    text = " ".join(words)
    chunks = chunk_pages_by_tokens([{"page_number": 1, "text": text}], tokenizer, 8, overlap=2)
    assert len(chunks) > 1
    warning.assert_not_called()
    assert all(len(tokenizer(chunk["text"], truncation=False)["input_ids"]) <= 8 for chunk in chunks)
    assert tokenizer.model_max_length == 8
    # Die Warnung bleibt für echte überlange Modell-Eingaben verfügbar.
    warning.reset_mock()
    tokenizer.deprecation_warnings.clear()
    tokenizer(text, truncation=False)
    assert any("Token indices sequence length" in call.args[0] for call in warning.call_args_list)
