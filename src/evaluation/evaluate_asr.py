# src/evaluation/evaluate_asr.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Step 2: Measure ASR accuracy using Word Error Rate (WER)
#
# WER = (Substitutions + Deletions + Insertions) / Total Reference Words
# Lower is better. 0.0 = perfect, 1.0 = completely wrong.
# ─────────────────────────────────────────────────────────────────────────────

import json
from dataclasses import dataclass, field
from typing import List


# ── Core WER calculation ─────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> dict:
    """
    Compute Word Error Rate between a reference and hypothesis sentence.

    Args:
        reference:  The ground-truth transcript.
        hypothesis: The ASR-predicted transcript.

    Returns:
        dict with keys: wer, substitutions, deletions, insertions,
                        ref_length, hyp_length
    """
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    r = len(ref_words)
    h = len(hyp_words)

    # Build edit distance matrix
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # deletion
                    dp[i][j - 1],       # insertion
                    dp[i - 1][j - 1],   # substitution
                )

    # Back-trace to count operation types
    i, j = r, h
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    total_errors = substitutions + deletions + insertions
    wer = total_errors / r if r > 0 else 0.0

    return {
        "wer": round(wer, 4),
        "wer_percent": round(wer * 100, 2),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": total_errors,
        "ref_length": r,
        "hyp_length": h,
    }


# ── BLEU score for translation quality ──────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> dict:
    """
    Compute corpus BLEU score (up to 4-gram) for translation quality.

    Args:
        reference:  Ground-truth translation.
        hypothesis: Model-predicted translation.
        max_n:      Maximum n-gram order (default 4).

    Returns:
        dict with bleu score and per-ngram precision.
    """
    import math

    ref_tokens  = reference.lower().strip().split()
    hyp_tokens  = hypothesis.lower().strip().split()

    if not hyp_tokens:
        return {"bleu": 0.0, "ngram_precisions": []}

    precisions = []

    for n in range(1, max_n + 1):
        # Count n-grams in reference
        ref_ngrams: dict = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        # Count clipped n-gram matches
        hyp_ngrams: dict = {}
        for i in range(len(hyp_tokens) - n + 1):
            ng = tuple(hyp_tokens[i:i + n])
            hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1

        clipped = sum(
            min(count, ref_ngrams.get(ng, 0))
            for ng, count in hyp_ngrams.items()
        )
        total = max(len(hyp_tokens) - n + 1, 0)
        precisions.append(clipped / total if total > 0 else 0.0)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if hyp_tokens else 0.0

    # Geometric mean of precisions (skip zero precisions gracefully)
    non_zero = [p for p in precisions if p > 0]
    if not non_zero:
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in non_zero) / max_n
        bleu = bp * math.exp(log_avg)

    return {
        "bleu": round(bleu * 100, 2),          # expressed as 0–100
        "brevity_penalty": round(bp, 4),
        "ngram_precisions": [round(p * 100, 2) for p in precisions],
    }


# ── Test cases ───────────────────────────────────────────────────────────────

@dataclass
class ASRTestCase:
    label: str
    language: str
    audio_file: str                          # path relative to data/input_audio/
    reference_transcript: str
    reference_translation: str = ""          # optional, for BLEU


# Standard Phase 1 test suite
PHASE1_TEST_CASES: List[ASRTestCase] = [

    ASRTestCase(
        label="Tamil — pure script",
        language="Tamil",
        audio_file="tamil_market.wav",
        reference_transcript="நான் இன்று சந்தைக்கு போகிறேன்",
        reference_translation="I am going to the market today",
    ),

    ASRTestCase(
        label="Tamil — romanised (Tanglish)",
        language="Tanglish",
        audio_file="tanglish_office.wav",
        reference_transcript="naan office ku late aa varen",
        reference_translation="I will come late to the office",
    ),

    ASRTestCase(
        label="Hindi — pure script",
        language="Hindi",
        audio_file="hindi_meeting.wav",
        reference_transcript="कल मीटिंग है लेकिन मैं व्यस्त हूँ",
        reference_translation="There is a meeting tomorrow but I am busy",
    ),

    ASRTestCase(
        label="Hinglish — code-mixed",
        language="Hinglish",
        audio_file="hinglish_busy.wav",
        reference_transcript="kal meeting hai but I am busy",
        reference_translation="There is a meeting tomorrow but I am busy",
    ),

    ASRTestCase(
        label="English — baseline",
        language="English",
        audio_file="english_baseline.wav",
        reference_transcript="I am going to the market today",
        reference_translation="I am going to the market today",
    ),
]


# ── Runner ───────────────────────────────────────────────────────────────────

def run_asr_evaluation(transcribe_fn, translate_fn=None, test_cases=None):
    """
    Run Phase 1 evaluation.

    Args:
        transcribe_fn:  Callable(audio_path) → str
        translate_fn:   Optional Callable(text, src_lang) → str
        test_cases:     List[ASRTestCase], defaults to PHASE1_TEST_CASES

    Returns:
        List of result dicts, one per test case.
    """
    if test_cases is None:
        test_cases = PHASE1_TEST_CASES

    results = []

    print("\n" + "=" * 70)
    print("  PHASE 1 — ASR & TRANSLATION EVALUATION")
    print("=" * 70)

    for tc in test_cases:
        print(f"\n▶  {tc.label}")
        print(f"   Audio : data/input_audio/{tc.audio_file}")

        audio_path = f"data/input_audio/{tc.audio_file}"

        # ── ASR ──────────────────────────────────────────────────────────────
        try:
            hypothesis = transcribe_fn(audio_path)
        except FileNotFoundError:
            print(f"   ⚠  Audio file not found — skipping")
            results.append({"label": tc.label, "status": "skipped — no audio"})
            continue
        except Exception as e:
            print(f"   ✗  ASR error: {e}")
            results.append({"label": tc.label, "status": f"ASR error: {e}"})
            continue

        wer_result = compute_wer(tc.reference_transcript, hypothesis)

        print(f"   Reference  : {tc.reference_transcript}")
        print(f"   Hypothesis : {hypothesis}")
        print(f"   WER        : {wer_result['wer_percent']}%  "
              f"(S={wer_result['substitutions']} D={wer_result['deletions']} "
              f"I={wer_result['insertions']})")

        entry = {
            "label":       tc.label,
            "language":    tc.language,
            "reference":   tc.reference_transcript,
            "hypothesis":  hypothesis,
            "wer":         wer_result,
            "status":      "ok",
        }

        # ── Translation + BLEU ───────────────────────────────────────────────
        if translate_fn and tc.reference_translation:
            try:
                translated = translate_fn(hypothesis)
                bleu_result = compute_bleu(tc.reference_translation, translated)

                print(f"   Translation: {translated}")
                print(f"   BLEU       : {bleu_result['bleu']}")

                entry["translation"]      = translated
                entry["ref_translation"]  = tc.reference_translation
                entry["bleu"]             = bleu_result

            except Exception as e:
                print(f"   ✗  Translation error: {e}")
                entry["translation_error"] = str(e)

        results.append(entry)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Test Case':<35} {'WER %':>7}  {'BLEU':>7}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*7}")

    for r in results:
        if r.get("status") != "ok":
            print(f"  {r['label']:<35} {'SKIP':>7}")
            continue
        wer_str  = f"{r['wer']['wer_percent']:>6.1f}%"
        bleu_str = f"{r['bleu']['bleu']:>6.1f}" if "bleu" in r else "   N/A"
        print(f"  {r['label']:<35} {wer_str}  {bleu_str}")

    print("=" * 70 + "\n")

    # Save results to JSON
    with open("data/phase1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("  Results saved → data/phase1_results.json\n")
    return results


# ── Quick standalone test (no audio needed) ──────────────────────────────────

if __name__ == "__main__":

    print("\n── WER Quick Test ──")
    cases = [
        ("I am going to the market today",  "I am going market today"),
        ("naan office ku late aa varen",     "naan office ku late varen"),
        ("kal meeting hai but I am busy",    "kal meeting hai but I busy"),
    ]
    for ref, hyp in cases:
        r = compute_wer(ref, hyp)
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}")
        print(f"  WER: {r['wer_percent']}%  "
              f"(S={r['substitutions']} D={r['deletions']} I={r['insertions']})\n")

    print("── BLEU Quick Test ──")
    bleu_cases = [
        ("I am going to the market today", "I am going to the market today"),
        ("I am going to the market today", "I go market today"),
        ("I will come late to the office", "I will come late to office"),
    ]
    for ref, hyp in bleu_cases:
        b = compute_bleu(ref, hyp)
        print(f"  REF : {ref}")
        print(f"  HYP : {hyp}")
        print(f"  BLEU: {b['bleu']}  (n-gram precisions: {b['ngram_precisions']})\n")