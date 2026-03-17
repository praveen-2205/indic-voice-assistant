# src/utils/audio_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Audio Preprocessing Pipeline
#
# Steps applied before ASR:

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
#   1. Convert to mono (single channel)
#   2. Resample to 16kHz (Whisper's required sample rate)
#   3. Noise reduction (spectral subtraction)
#   4. Silence removal (strip leading/trailing silence)
#   5. Volume normalization (consistent loudness)
#
# Install:
#   pip install librosa soundfile noisereduce
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import librosa
import soundfile as sf

# Optional — graceful fallback if noisereduce not installed
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("[AudioUtils] ⚠  noisereduce not installed — noise reduction disabled")
    print("              Run: pip install noisereduce")

TARGET_SR       = 16000    # Whisper requires 16kHz
TOP_DB          = 30       # Silence threshold in dB (higher = more aggressive)
TARGET_LUFS     = -23.0    # Target loudness (EBU R128 broadcast standard)


# ── Step 1+2: Load, convert to mono, resample ────────────────────────────────

def load_audio(audio_path: str) -> tuple:
    """
    Load any audio file, convert to mono, resample to 16kHz.

    Returns:
        (audio_array, sample_rate)
    """
    print(f"  [Audio] Loading: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration  = len(audio) / sr
    print(f"  [Audio] Loaded  — SR={sr}Hz  Duration={duration:.2f}s  Samples={len(audio)}")
    return audio, sr


# ── Step 3: Noise reduction ───────────────────────────────────────────────────

def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply spectral noise reduction.

    Uses the first 0.5 seconds as a noise profile sample
    (assumes recording starts with background noise before speech).
    Falls back gracefully if noisereduce is not installed.
    """
    if not NOISEREDUCE_AVAILABLE:
        print("  [Audio] Noise reduction skipped (noisereduce not installed)")
        return audio

    # Use first 0.5s as noise sample
    noise_sample_duration = min(0.5, len(audio) / sr)
    noise_sample = audio[:int(noise_sample_duration * sr)]

    denoised = nr.reduce_noise(
        y=audio,
        y_noise=noise_sample,
        sr=sr,
        prop_decrease=0.8,      # 80% noise reduction
        stationary=False,       # handles non-stationary noise (fans, traffic)
    )
    print(f"  [Audio] Noise reduction applied")
    return denoised


# ── Step 4: Silence removal ───────────────────────────────────────────────────

def remove_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.
    Also removes long silent gaps in the middle (>1 second).
    """
    # Trim leading/trailing silence
    trimmed, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    # Split on internal silences and rejoin with short pauses
    intervals = librosa.effects.split(trimmed, top_db=TOP_DB)

    if len(intervals) == 0:
        return trimmed

    # Rejoin speech segments with 0.1s gap between them
    gap     = np.zeros(int(0.1 * sr))
    chunks  = [trimmed[start:end] for start, end in intervals]
    result  = chunks[0]
    for chunk in chunks[1:]:
        result = np.concatenate([result, gap, chunk])

    removed = (len(audio) - len(result)) / sr
    print(f"  [Audio] Silence removed — {removed:.2f}s of silence stripped")
    return result


# ── Step 5: Volume normalization ──────────────────────────────────────────────

def normalize_volume(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio volume to a consistent level.
    Uses peak normalization to -3dB to avoid clipping.
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    # Target peak at -3dB (0.708 linear)
    target_peak = 0.708
    normalized  = audio * (target_peak / peak)
    print(f"  [Audio] Volume normalized — peak was {20*np.log10(peak):.1f}dB")
    return normalized


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def preprocess_audio(
    input_path: str,
    output_path: str = None,
    apply_noise_reduction: bool = True,
    apply_silence_removal: bool = True,
    apply_normalization:   bool = True,
) -> str:
    """
    Full audio preprocessing pipeline.
    Applies all steps in sequence before ASR.

    Args:
        input_path:             Path to raw input audio file.
        output_path:            Where to save processed audio.
                                If None, saves as input_path + '_processed.wav'
        apply_noise_reduction:  Toggle noise reduction.
        apply_silence_removal:  Toggle silence removal.
        apply_normalization:    Toggle volume normalization.

    Returns:
        Path to the preprocessed audio file.
    """
    if output_path is None:
        base     = os.path.splitext(input_path)[0]
        output_path = f"{base}_processed.wav"

    print(f"\n  [Audio] ── Preprocessing Pipeline ──")

    # Step 1+2: Load + mono + resample
    audio, sr = load_audio(input_path)

    # Step 3: Noise reduction
    if apply_noise_reduction:
        audio = reduce_noise(audio, sr)

    # Step 4: Silence removal
    if apply_silence_removal:
        audio = remove_silence(audio, sr)

    # Step 5: Normalization
    if apply_normalization:
        audio = normalize_volume(audio)

    # Save processed audio
    sf.write(output_path, audio, sr)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [Audio] Saved processed audio → {output_path}  ({size_kb:.1f} KB)")
    print(f"  [Audio] ── Preprocessing complete ✅ ──\n")

    return output_path


def get_audio_info(audio_path: str) -> dict:
    """
    Get basic info about an audio file without processing it.
    Useful for debugging and logging.
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=False)

    # Handle mono vs stereo
    if audio.ndim == 1:
        channels = 1
        samples  = len(audio)
    else:
        channels = audio.shape[0]
        samples  = audio.shape[1]

    duration = samples / sr
    peak_db  = 20 * np.log10(np.max(np.abs(audio)) + 1e-8)

    return {
        "path":       audio_path,
        "sample_rate": sr,
        "channels":   channels,
        "duration_s": round(duration, 2),
        "samples":    samples,
        "peak_db":    round(peak_db, 1),
        "needs_resample": sr != TARGET_SR,
        "needs_mono":     channels > 1,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]

        print(f"\n── Audio Preprocessing Test ──")
        print(f"   Input: {audio_file}\n")

        # Show info before processing
        info = get_audio_info(audio_file)
        print("  Before processing:")
        for k, v in info.items():
            print(f"    {k:<20} {v}")

        # Run full pipeline
        output = preprocess_audio(audio_file)

        # Show info after processing
        info2 = get_audio_info(output)
        print("\n  After processing:")
        for k, v in info2.items():
            print(f"    {k:<20} {v}")

        print(f"\n  ✅ Processed audio saved to: {output}")
        print(f"     Now run ASR on this file for better accuracy.\n")

    else:
        print("Usage: python src/utils/audio_utils.py path/to/audio.wav")
        print("\nRunning quick unit tests instead...\n")

        # Generate a synthetic noisy signal for testing
        sr   = 16000
        t    = np.linspace(0, 2, 2 * sr)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)                 # 440Hz tone
        noise = 0.05 * np.random.randn(len(t))                    # background noise
        silence_start = np.zeros(int(0.3 * sr))                   # 0.3s silence
        silence_end   = np.zeros(int(0.5 * sr))                   # 0.5s silence
        signal = np.concatenate([silence_start, tone + noise, silence_end])

        # Save test input
        os.makedirs("data/input_audio", exist_ok=True)
        test_input = "data/input_audio/test_synthetic.wav"
        sf.write(test_input, signal, sr)
        print(f"  Synthetic test audio → {test_input}")

        # Process it
        output = preprocess_audio(test_input)
        print(f"  ✅ Preprocessing pipeline works correctly")
        print(f"     Output: {output}")