# src/utils/language_detector.py

import unicodedata
from dataclasses import dataclass
from typing import List, Optional


LANG_TO_INDICTRANS = {
    "ta":      "tam_Taml",
    "hi":      "hin_Deva",
    "te":      "tel_Telu",
    "bn":      "ben_Beng",
    "en":      "eng_Latn",
    "unknown": "hin_Deva",
}

SCRIPT_RANGES = {
    "tamil":      (0x0B80, 0x0BFF),
    "devanagari": (0x0900, 0x097F),
    "telugu":     (0x0C00, 0x0C7F),
    "bengali":    (0x0980, 0x09FF),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
}

SCRIPT_TO_LANG = {
    "tamil":      "ta",
    "devanagari": "hi",
    "telugu":     "te",
    "bengali":    "bn",
    "kannada":    "kn",
    "malayalam":  "ml",
}

# ── Marker words ──────────────────────────────────────────────────────────────
# RULE: Only include words that CANNOT appear in plain English sentences.
# Do NOT include: "time", "office", "meeting" — these are normal English words.

TANGLISH_MARKERS = {
    "naan", "naaan", "ivan", "avan", "ava", "enna", "inge", "anga",
    "eppadi", "paaru", "sollu", "varen", "vaaren", "vandhen",
    "inga", "enga", "yenna", "paakalam", "pogiren", "pogirom",
    "irukken", "irukku", "saapduven", "sappidu", "vandha", "vanthu",
    # Tanglish-specific suffixes used as standalone words
    "ku", "la", "da", "di",
}

HINGLISH_MARKERS = {
    # Core Hindi words that never appear in plain English
    "hai", "hain", "kal", "aaj", "kya", "nahi", "nahin", "kyun",
    "kaise", "kahan", "yaar", "bhai", "dost", "accha", "theek",
    "bilkul", "lekin", "aur", "mein", "se", "ko", "ka", "ki", "ke",
    "ho", "hoga", "tha", "thi", "raha", "rahi", "rahe",
    "baat", "kaam", "waqt",
    # NOTE: "the", "time", "office", "meeting" removed — valid English words
}

# English anchor words for reliable English detection
ENGLISH_ANCHOR_WORDS = {
    "i", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "the", "a",
    "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "through", "going",
    "today", "market", "hello", "good", "morning", "evening", "night",
    "please", "thank", "you", "my", "your", "his", "her", "their",
    "this", "that", "these", "those", "what", "when", "where", "how",
    "why", "who", "which", "not", "no", "yes", "ok", "okay", "time",
    "office", "meeting", "come", "late", "busy", "go", "going",
}


@dataclass
class LanguageSegment:
    text: str
    lang_code: str
    indictrans_tag: str
    script: str
    confidence: float


@dataclass
class DetectionResult:
    raw_text: str
    primary_lang: str
    indictrans_tag: str
    is_code_mixed: bool
    segments: List[LanguageSegment]
    mix_type: Optional[str]
    confidence: float


def _detect_script(char: str) -> str:
    cp = ord(char)
    for script, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return script
    if 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
        return "latin"
    return "other"


def _script_distribution(text: str) -> dict:
    counts: dict = {}
    total = 0
    for ch in text:
        if ch.strip() and not ch.isdigit() and not unicodedata.category(ch).startswith("P"):
            script = _detect_script(ch)
            if script != "other":
                counts[script] = counts.get(script, 0) + 1
                total += 1
    if total == 0:
        return {}
    return {s: c / total for s, c in counts.items()}


def _is_english_sentence(text: str) -> bool:
    """
    True if text has NO Indic markers AND has 2+ English anchor words.
    Runs before langdetect — much more reliable for short sentences.
    """
    words = set(text.lower().split())
    if words & (TANGLISH_MARKERS | HINGLISH_MARKERS):
        return False
    return len(words & ENGLISH_ANCHOR_WORDS) >= 2


def _detect_code_mix_type(text: str) -> Optional[str]:
    words_lower = set(text.lower().split())
    has_tamil   = bool(words_lower & TANGLISH_MARKERS)
    has_hindi   = bool(words_lower & HINGLISH_MARKERS)
    # English words = Latin words NOT in any Indic marker list
    has_english = any(
        w.isascii() and w.isalpha()
        and w not in TANGLISH_MARKERS
        and w not in HINGLISH_MARKERS
        for w in words_lower
    )
    if has_tamil and (has_english or has_hindi):
        return "Tanglish"
    if has_hindi and has_english:
        return "Hinglish"
    return None


def _get_word_lang(word: str) -> tuple:
    """Return (script, lang_code) for one word."""
    # Non-Latin script check first
    dist = _script_distribution(word)
    non_latin = {s: p for s, p in dist.items() if s != "latin"}
    if non_latin:
        dominant = max(non_latin, key=non_latin.get)
        return dominant, SCRIPT_TO_LANG.get(dominant, "en")

    w = word.lower()
    if w in TANGLISH_MARKERS:
        return "latin", "ta"
    if w in HINGLISH_MARKERS:
        return "latin", "hi"
    return "latin", "en"


def _split_into_segments(text: str) -> List[LanguageSegment]:
    tokens     = text.split()
    token_info = [_get_word_lang(t) + (t,) for t in tokens]
    # token_info items: (script, lang, word)

    segments: List[LanguageSegment] = []
    if not token_info:
        return segments

    cur_script, cur_lang, cur_word = token_info[0]
    cur_tokens = [cur_word]

    for script, lang, word in token_info[1:]:
        if lang == cur_lang:
            cur_tokens.append(word)
        else:
            segments.append(LanguageSegment(
                text=" ".join(cur_tokens),
                lang_code=cur_lang,
                indictrans_tag=LANG_TO_INDICTRANS.get(cur_lang, "eng_Latn"),
                script=cur_script,
                confidence=0.85,
            ))
            cur_tokens = [word]
            cur_script = script
            cur_lang   = lang

    segments.append(LanguageSegment(
        text=" ".join(cur_tokens),
        lang_code=cur_lang,
        indictrans_tag=LANG_TO_INDICTRANS.get(cur_lang, "eng_Latn"),
        script=cur_script,
        confidence=0.85,
    ))
    return segments


def detect_language(text: str) -> DetectionResult:
    text = text.strip()
    if not text:
        return DetectionResult(
            raw_text=text, primary_lang="unknown",
            indictrans_tag="hin_Deva", is_code_mixed=False,
            segments=[], mix_type=None, confidence=0.0,
        )

    dist      = _script_distribution(text)
    non_latin = {s: p for s, p in dist.items() if s != "latin"}

    # ── Non-Latin (Indic script) ──────────────────────────────────────────────
    if non_latin:
        dominant       = max(non_latin, key=non_latin.get)
        latin_share    = dist.get("latin", 0.0)
        is_mixed       = latin_share > 0.15
        primary_lang   = SCRIPT_TO_LANG.get(dominant, "unknown")
        indictrans_tag = LANG_TO_INDICTRANS.get(primary_lang, "hin_Deva")
        confidence     = non_latin[dominant]
        mix_type       = _detect_code_mix_type(text) if is_mixed else None
        segments       = _split_into_segments(text) if is_mixed else [
            LanguageSegment(text=text, lang_code=primary_lang,
                            indictrans_tag=indictrans_tag,
                            script=dominant, confidence=confidence)
        ]
        return DetectionResult(
            raw_text=text, primary_lang=primary_lang,
            indictrans_tag=indictrans_tag, is_code_mixed=is_mixed,
            segments=segments, mix_type=mix_type, confidence=confidence,
        )

    # ── Latin script ──────────────────────────────────────────────────────────

    # 1. Pure English check (reliable anchor-word method)
    if _is_english_sentence(text):
        return DetectionResult(
            raw_text=text, primary_lang="en", indictrans_tag="eng_Latn",
            is_code_mixed=False,
            segments=[LanguageSegment(text=text, lang_code="en",
                                      indictrans_tag="eng_Latn",
                                      script="latin", confidence=0.95)],
            mix_type=None, confidence=0.95,
        )

    # 2. Tanglish / Hinglish check
    mix_type = _detect_code_mix_type(text)
    if mix_type:
        if "Tanglish" in mix_type:
            primary_lang, indictrans_tag = "ta", "tam_Taml"
        else:
            primary_lang, indictrans_tag = "hi", "hin_Deva"
        segments = _split_into_segments(text)
        return DetectionResult(
            raw_text=text, primary_lang=primary_lang,
            indictrans_tag=indictrans_tag, is_code_mixed=True,
            segments=segments, mix_type=mix_type, confidence=0.85,
        )

    # 3. Fallback: langdetect
    try:
        from langdetect import detect as ld_detect, detect_langs
        detected     = ld_detect(text)
        lang_probs   = detect_langs(text)
        confidence   = lang_probs[0].prob if lang_probs else 0.5
        primary_lang = detected if detected in LANG_TO_INDICTRANS else "en"
    except Exception:
        primary_lang = "en"
        confidence   = 0.5

    indictrans_tag = LANG_TO_INDICTRANS.get(primary_lang, "eng_Latn")
    return DetectionResult(
        raw_text=text, primary_lang=primary_lang,
        indictrans_tag=indictrans_tag, is_code_mixed=False,
        segments=[LanguageSegment(text=text, lang_code=primary_lang,
                                  indictrans_tag=indictrans_tag,
                                  script="latin", confidence=confidence)],
        mix_type=None, confidence=confidence,
    )


def get_indictrans_tag(text: str) -> str:
    """Convenience wrapper — returns just the IndicTrans2 tag."""
    return detect_language(text).indictrans_tag