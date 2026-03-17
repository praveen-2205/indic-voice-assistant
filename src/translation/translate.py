# src/translation/translate.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Optimized translation with:
#   1. Shared model from ModelManager (no reload per request)
#   2. LRU cache for repeated phrases
#   3. Per-request timing logs
# ─────────────────────────────────────────────────────────────────────────────

import time
from functools import lru_cache
from src.config.settings import (
    TRANSLATION_MODEL,
    TRANSLATION_MAX_LEN,
    TRANSLATION_BEAMS,
    TRANSLATION_CACHE_SIZE,
    ENABLE_TRANSLATION_CACHE,
)

_tokenizer = None
_model     = None
_processor = None


def _get_models():
    """Get models from ModelManager if loaded, else load directly."""
    global _tokenizer, _model, _processor

    try:
        from src.config.model_manager import ModelManager
        mm = ModelManager.get()
        if mm.is_loaded:
            return mm.translation_tokenizer, mm.translation_model, mm.indic_processor
    except ImportError:
        pass

    if _tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL, trust_remote_code=True)
        _model     = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL, trust_remote_code=True)

    if _processor is None:
        try:
            from IndicTransToolkit import IndicProcessor
            _processor = IndicProcessor(inference=True)
        except ImportError:
            _processor = None

    return _tokenizer, _model, _processor


@lru_cache(maxsize=TRANSLATION_CACHE_SIZE)
def _cached_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """LRU-cached translation core. Same input = instant cache hit."""
    import torch
    tokenizer, model, processor = _get_models()

    if processor is not None:
        batch  = processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=TRANSLATION_BEAMS,
                num_return_sequences=1,
                max_length=TRANSLATION_MAX_LEN,
            )
        raw    = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result = processor.postprocess_batch(raw, lang=tgt_lang)
        return result[0]
    else:
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        inputs     = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=TRANSLATION_MAX_LEN,
                forced_bos_token_id=forced_bos,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_text(text: str, src_lang: str = "tam_Taml", tgt_lang: str = "eng_Latn") -> str:
    """Translate with caching and timing."""
    text = text.strip()
    if not text:
        return ""

    t          = time.time()
    cache_before = _cached_translate.cache_info().hits
    result     = _cached_translate(text, src_lang, tgt_lang)
    cache_hit  = _cached_translate.cache_info().hits > cache_before
    elapsed    = time.time() - t

    status = "CACHE HIT ⚡" if cache_hit else f"{elapsed:.2f}s"
    print(f"  [Translate] {src_lang} → {tgt_lang}  [{status}]")
    print(f"  [Translate] {text[:60]} → {result[:60]}")
    return result


def get_cache_stats() -> dict:
    info = _cached_translate.cache_info()
    return {
        "hits":     info.hits,
        "misses":   info.misses,
        "currsize": info.currsize,
        "maxsize":  info.maxsize,
        "hit_rate": f"{info.hits / max(info.hits + info.misses, 1) * 100:.1f}%",
    }


def clear_cache():
    _cached_translate.cache_clear()