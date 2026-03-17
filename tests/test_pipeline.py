# tests/test_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 Verification Script
#
# Usage (run from project ROOT):
#   python tests/test_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import traceback

# ── Fix: add project root to sys.path so 'src' is always findable ────────────
# Works whether you run from project root OR from tests/ subfolder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if detail and not condition:
        print(f"         {detail}")
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — WER computation
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 1 — Word Error Rate (WER)")

try:
    from src.evaluation.evaluate_asr import compute_wer

    r = compute_wer("I am going to the market today",
                    "I am going to the market today")
    check("Perfect match → WER = 0%", r["wer"] == 0.0, str(r))

    r = compute_wer("I am going to the market today",
                    "I am going market today")
    check("Deletion detected → WER > 0", r["wer"] > 0, str(r))
    check("Deletion count = 2", r["deletions"] == 2, str(r))

    r = compute_wer("hello world", "hello beautiful world")
    check("Insertion detected", r["insertions"] == 1, str(r))

    r = compute_wer("good morning", "good evening")
    check("Substitution detected", r["substitutions"] == 1, str(r))

    result = compute_wer("I am going to the market today", "I am going market today")
    print(f"\n  Sample WER: REF='I am going to the market today'")
    print(f"              HYP='I am going market today'")
    print(f"              → WER={result['wer_percent']}%  "
          f"S={result['substitutions']} D={result['deletions']} I={result['insertions']}")

except Exception as e:
    print(f"  {FAIL}  WER module error: {e}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — BLEU computation
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 2 — BLEU Score")

try:
    from src.evaluation.evaluate_asr import compute_bleu

    b = compute_bleu("I am going to the market today",
                     "I am going to the market today")
    check("Perfect match → BLEU = 100", b["bleu"] == 100.0, str(b))

    b = compute_bleu("I am going to the market today",
                     "I go market today")
    check("Partial match → 0 < BLEU < 100", 0 < b["bleu"] < 100, str(b))

    b = compute_bleu("the cat sat on the mat", "xyz xyz xyz")
    check("No match → BLEU ≈ 0", b["bleu"] < 5.0, str(b))

    b2 = compute_bleu("I am going to the market today",
                      "I will go to the market today")
    print(f"\n  Sample BLEU: REF='I am going to the market today'")
    print(f"               HYP='I will go to the market today'")
    print(f"               → BLEU={b2['bleu']}  precisions={b2['ngram_precisions']}")

except Exception as e:
    print(f"  {FAIL}  BLEU module error: {e}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Language detection
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 3 — Language Detection")

try:
    from src.utils.language_detector import detect_language

    cases = [
        ("Tamil script",  "நான் இன்று சந்தைக்கு போகிறேன்",   "ta", False),
        ("Hindi script",  "कल मीटिंग है लेकिन मैं व्यस्त हूँ", "hi", False),
        ("English",       "I am going to the market today",       "en", False),
        ("Tanglish",      "naan office ku late aa varen",          "ta", True),
        ("Hinglish",      "kal meeting hai but I am busy",         "hi", True),
    ]

    for label, text, expected_lang, expected_mixed in cases:
        result = detect_language(text)
        lang_ok  = result.primary_lang == expected_lang
        mixed_ok = result.is_code_mixed == expected_mixed
        detail = (f"got lang={result.primary_lang} (expected {expected_lang}), "
                  f"mixed={result.is_code_mixed} (expected {expected_mixed})")
        check(f"{label:<25} lang={expected_lang}  mixed={expected_mixed}",
              lang_ok and mixed_ok, detail)

    # Show code-mixed segmentation
    print(f"\n  Tanglish segment breakdown:")
    r = detect_language("naan office ku late aa varen")
    for seg in r.segments:
        print(f"    [{seg.lang_code}] \"{seg.text}\"  →  {seg.indictrans_tag}")

    print(f"\n  Hinglish segment breakdown:")
    r = detect_language("kal meeting hai but I am busy")
    for seg in r.segments:
        print(f"    [{seg.lang_code}] \"{seg.text}\"  →  {seg.indictrans_tag}")

except Exception as e:
    print(f"  {FAIL}  Language detector error: {e}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Translation module import
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 4 — Translation Module")

try:
    from src.translation.translate import translate_text
    check("translate_text imports successfully", True)

    try:
        result = translate_text(
            "I am going to the market",
            src_lang="eng_Latn",
            tgt_lang="eng_Latn"
        )
        check("translate_text runs without error", isinstance(result, str),
              f"got: {result}")
        print(f"  Output: \"{result}\"")
    except Exception as e:
        print(f"  ⚠  translate_text raised (model may need download): {e}")

except ImportError as e:
    print(f"  {FAIL}  Cannot import translate_text: {e}")
    print("          → Install missing package:")
    print("            pip install IndicTransToolkit")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — ASR module import
# ─────────────────────────────────────────────────────────────────────────────

section("TEST 5 — ASR Module")

try:
    from src.asr.speech_to_text import transcribe_audio
    check("transcribe_audio imports successfully", True)
    print("  ℹ  To test ASR: place a WAV in data/input_audio/ then run:")
    print("     GET /evaluate/phase1  (after starting the server)")

except ImportError as e:
    print(f"  {FAIL}  Cannot import transcribe_audio: {e}")
    print("          → Install missing package:")
    print("            pip install openai-whisper")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("PHASE 1 CHECKLIST — Next Steps")

print("""
  Once all 5 tests are green:

  1. Create the evaluation folder if it doesn't exist:
       mkdir src\\evaluation
       type nul > src\\evaluation\\__init__.py

  2. Place 5 audio samples in data/input_audio/:
       tamil_market.wav
       tanglish_office.wav
       hindi_meeting.wav
       hinglish_busy.wav
       english_baseline.wav

  3. Start server:
       uvicorn app.main:app --reload

  4. Run full evaluation:
       GET http://127.0.0.1:8000/evaluate/phase1

  5. Test auto language detection:
       POST http://127.0.0.1:8000/detect-language
       Body: {"text": "naan office ku late aa varen"}

  6. Paste your WER + BLEU scores → move to Phase 3 (TTS)
""")