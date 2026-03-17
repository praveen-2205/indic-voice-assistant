# src/config/model_manager.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Singleton model manager
#
# Loads ALL models ONCE at server startup and reuses them.
# Before: models reloaded on every module import = slow
# After:  models loaded once, shared across all requests = fast
# ─────────────────────────────────────────────────────────────────────────────

import time
from src.config.settings import (
    WHISPER_MODEL_SIZE,
    TRANSLATION_MODEL,
)


class ModelManager:
    """
    Singleton that holds all loaded models.
    Call ModelManager.get() to access the shared instance.
    """
    _instance = None

    def __init__(self):
        self.whisper_model       = None
        self.translation_model   = None
        self.translation_tokenizer = None
        self.indic_processor     = None
        self._loaded             = False

    @classmethod
    def get(cls) -> "ModelManager":
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_all(self):
        """Load all models at startup. Call once from FastAPI lifespan."""
        if self._loaded:
            return

        total_start = time.time()
        print("\n" + "="*55)
        print("  Loading all models at startup...")
        print("="*55)

        # ── Whisper ───────────────────────────────────────────────
        print(f"\n  [1/3] Whisper {WHISPER_MODEL_SIZE}...")
        t = time.time()
        import whisper
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        print(f"        ✅ Loaded in {time.time()-t:.1f}s")

        # ── IndicTrans2 ───────────────────────────────────────────
        print(f"\n  [2/3] IndicTrans2 tokenizer...")
        t = time.time()
        from transformers import AutoTokenizer
        self.translation_tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL, trust_remote_code=True
        )
        print(f"        ✅ Loaded in {time.time()-t:.1f}s")

        print(f"\n  [3/3] IndicTrans2 model...")
        t = time.time()
        from transformers import AutoModelForSeq2SeqLM
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL, trust_remote_code=True
        )
        print(f"        ✅ Loaded in {time.time()-t:.1f}s")

        # ── IndicProcessor ────────────────────────────────────────
        try:
            from IndicTransToolkit import IndicProcessor
            self.indic_processor = IndicProcessor(inference=True)
            print(f"\n        ✅ IndicProcessor ready")
        except ImportError:
            print(f"\n  ⚠  IndicTransToolkit not found — install it for best results")
            self.indic_processor = None

        self._loaded = True
        total = time.time() - total_start
        print(f"\n{'='*55}")
        print(f"  All models loaded in {total:.1f}s — server ready ✅")
        print(f"{'='*55}\n")

    @property
    def is_loaded(self) -> bool:
        return self._loaded