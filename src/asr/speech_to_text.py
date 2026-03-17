# src/asr/speech_to_text.py

import sys
import os

# Fix: add project root to path so 'src' is always findable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import whisper
from src.utils.audio_utils import preprocess_audio

MODEL_SIZE = "medium"

print(f"[ASR] Loading Whisper {MODEL_SIZE} model...")
model = whisper.load_model(MODEL_SIZE)
print(f"[ASR] Whisper {MODEL_SIZE} ready ✅")

WHISPER_LANG_MAP = {
    "ta": "ta", "hi": "hi", "te": "te",
    "bn": "bn", "en": "en", "ml": "ml",
    "kn": "kn", "mr": "mr",
}


def detect_audio_language(audio_path: str) -> str:
    """Detect spoken language from audio using Whisper."""
    audio    = whisper.load_audio(audio_path)
    audio    = whisper.pad_or_trim(audio)
    mel      = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected   = max(probs, key=probs.get)
    confidence = probs[detected]
    print(f"  [ASR] Detected language: {detected}  (confidence: {confidence:.2%})")
    return detected


def transcribe_audio(
    audio_path: str,
    language: str = None,
    preprocess: bool = True,
) -> str:
    """
    Transcribe audio to text.

    Args:
        audio_path:  Path to audio file.
        language:    Language hint e.g. "ta", "hi". Auto-detected if None.
        preprocess:  Run noise reduction + silence removal + normalization first.

    Returns:
        Transcribed text string.
    """
    if preprocess:
        audio_path = preprocess_audio(audio_path)

    if language is None:
        language = detect_audio_language(audio_path)

    whisper_lang = WHISPER_LANG_MAP.get(language, language)
    print(f"  [ASR] Transcribing with language='{whisper_lang}'...")

    result = model.transcribe(
        audio_path,
        language=whisper_lang,
        task="transcribe",
        fp16=False,
        beam_size=5,
        best_of=5,
        temperature=0.0,
    )

    text = result["text"].strip()
    print(f"  [ASR] Transcription: {text}")
    return text


def transcribe_to_english(audio_path: str, preprocess: bool = True) -> str:
    """Directly translate audio to English using Whisper's translate task."""
    if preprocess:
        audio_path = preprocess_audio(audio_path)

    print(f"  [ASR] Direct translate to English...")
    result = model.transcribe(
        audio_path,
        task="translate",
        fp16=False,
        beam_size=5,
        temperature=0.0,
    )
    text = result["text"].strip()
    print(f"  [ASR] English: {text}")
    return text


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"\n── ASR Test: {audio_file} ──\n")
        lang    = detect_audio_language(audio_file)
        text    = transcribe_audio(audio_file)
        english = transcribe_to_english(audio_file)
        print(f"\n  Language      : {lang}")
        print(f"  Transcription : {text}")
        print(f"  English       : {english}")
    else:
        print("Usage: python src/asr/speech_to_text.py path/to/audio.wav")