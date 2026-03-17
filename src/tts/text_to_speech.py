# src/tts/text_to_speech.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Text-to-Speech using Microsoft Edge TTS
#
# Fix: asyncio.run() cannot be called inside FastAPI's running event loop.
# Solution: use nest_asyncio + asyncio.get_event_loop().run_until_complete()
#           which works both standalone AND inside FastAPI.
#
# Install: pip install edge-tts nest_asyncio
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import os

import nest_asyncio
import edge_tts

# Patch the event loop so asyncio works inside FastAPI
nest_asyncio.apply()

# ── Voice map ─────────────────────────────────────────────────────────────────
VOICE_MAP = {
    "en": "en-US-JennyNeural",
    "ta": "ta-IN-PallaviNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "unknown": "en-US-JennyNeural",
}


def get_voice(lang_code: str) -> str:
    return VOICE_MAP.get(lang_code, VOICE_MAP["unknown"])


async def _synthesize(text: str, voice: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def text_to_speech(
    text: str,
    lang_code: str = "en",
    output_path: str = "data/output_audio/response.mp3",
) -> str:
    """
    Convert text to speech and save as MP3.
    Works both standalone and inside FastAPI (via nest_asyncio).

    Args:
        text:        Text to speak.
        lang_code:   "en", "ta", "hi", "te", "bn"
        output_path: Output MP3 path.

    Returns:
        Path to saved audio file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    voice = get_voice(lang_code)
    print(f"  [TTS] Language: {lang_code}  Voice: {voice}")
    print(f"  [TTS] Text: {text}")

    # Works in both standalone and FastAPI contexts
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_synthesize(text, voice, output_path))

    print(f"  [TTS] Saved → {output_path}")
    return output_path


def text_to_speech_multilang(
    segments: list,
    output_dir: str = "data/output_audio",
) -> list:
    """Synthesize multiple language segments separately."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, seg in enumerate(segments):
        path = os.path.join(output_dir, f"segment_{i}_{seg['lang_code']}.mp3")
        text_to_speech(seg["text"], lang_code=seg["lang_code"], output_path=path)
        paths.append(path)
    return paths


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("I am going to the market today",   "en", "data/output_audio/test_english.mp3"),
        ("நான் இன்று சந்தைக்கு போகிறேன்",    "ta", "data/output_audio/test_tamil.mp3"),
        ("मैं आज बाजार जा रहा हूँ",           "hi", "data/output_audio/test_hindi.mp3"),
    ]

    print("\n── TTS Self-Test ──\n")
    for text, lang, path in test_cases:
        try:
            result = text_to_speech(text, lang_code=lang, output_path=path)
            print(f"  ✅ {lang.upper()} → {result}\n")
        except Exception as e:
            print(f"  ❌ {lang.upper()} failed: {e}\n")