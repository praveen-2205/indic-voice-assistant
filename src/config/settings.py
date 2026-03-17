# src/config/settings.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Central configuration for all models and performance settings
# ─────────────────────────────────────────────────────────────────────────────

# ── Whisper ASR ───────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE  = "medium"   # base | small | medium | large
WHISPER_BEAM_SIZE   = 5
WHISPER_TEMPERATURE = 0.0
WHISPER_FP16        = False      # CPU doesn't support fp16

# ── IndicTrans2 Translation ───────────────────────────────────────────────────
TRANSLATION_MODEL   = "ai4bharat/indictrans2-indic-en-1B"
TRANSLATION_MAX_LEN = 256
TRANSLATION_BEAMS   = 5

# ── Audio Preprocessing ───────────────────────────────────────────────────────
TARGET_SAMPLE_RATE  = 16000
SILENCE_TOP_DB      = 30
NOISE_PROP_DECREASE = 0.8

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_VOICE_MAP = {
    "en": "en-US-JennyNeural",
    "ta": "ta-IN-PallaviNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural",
}

# ── Performance ───────────────────────────────────────────────────────────────
TRANSLATION_CACHE_SIZE  = 256    # max number of cached translations
PREPROCESS_CACHE_SIZE   = 128    # max number of cached preprocessed audio paths
ENABLE_PREPROCESSING    = True
ENABLE_TRANSLATION_CACHE = True