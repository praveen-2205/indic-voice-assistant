# app/main.py
# Phase 6 — Added static file serving for Web UI and audio files

import os
import time
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.asr.speech_to_text import transcribe_audio
from src.utils.language_detector import detect_language
from src.translation.translate import translate_text, get_cache_stats
from src.tts.text_to_speech import text_to_speech
from src.config.model_manager import ModelManager

os.makedirs("data/input_audio",  exist_ok=True)
os.makedirs("data/output_audio", exist_ok=True)
os.makedirs("static",            exist_ok=True)

_request_times = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelManager.get().load_all()
    yield
    print("\n[Server] Shutting down...")


app = FastAPI(
    title="Indic Voice Assistant",
    version="0.6.0",
    lifespan=lifespan,
)

# ── CORS — allow browser requests from the UI ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve output audio files at /audio/<filename> ─────────────────────────────
app.mount("/audio", StaticFiles(directory="data/output_audio"), name="audio")

# ── Serve static UI files ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Timing middleware ─────────────────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = time.time() - start
    _request_times.append({
        "path":   request.url.path,
        "method": request.method,
        "time_s": round(elapsed, 3),
    })
    if len(_request_times) > 50:
        _request_times.pop(0)
    print(f"  [Timing] {request.method} {request.url.path} → {elapsed:.2f}s")
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    return response


# ── UI endpoint ───────────────────────────────────────────────────────────────
@app.get("/ui")
def serve_ui():
    """Serve the Web UI."""
    return FileResponse("static/index.html")


@app.get("/")
def home():
    return {
        "message": "Indic Voice Assistant — Phase 6",
        "version": "0.6.0",
        "ui":      "http://127.0.0.1:8000/ui",
        "docs":    "http://127.0.0.1:8000/docs",
        "models_loaded": ModelManager.get().is_loaded,
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": ModelManager.get().is_loaded}


@app.get("/stats")
def stats():
    cache  = get_cache_stats()
    recent = _request_times[-10:] if _request_times else []
    avg    = sum(r["time_s"] for r in recent) / len(recent) if recent else 0
    return {
        "translation_cache": cache,
        "recent_requests":   recent,
        "avg_response_time": f"{avg:.2f}s",
        "total_requests":    len(_request_times),
    }


# ── /transcribe ───────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    t_total = time.time()
    file_location = f"data/input_audio/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    t = time.time()
    try:
        text = transcribe_audio(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {str(e)}")
    asr_time = time.time() - t

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No speech detected")

    detection = detect_language(text)

    t = time.time()
    try:
        translated = _translate_with_detection(text, detection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    translate_time = time.time() - t

    return {
        "transcription": text,
        "language": {
            "code":           detection.primary_lang,
            "indictrans_tag": detection.indictrans_tag,
            "is_code_mixed":  detection.is_code_mixed,
            "mix_type":       detection.mix_type,
            "confidence":     detection.confidence,
            "segments": [
                {"text": s.text, "lang": s.lang_code, "tag": s.indictrans_tag}
                for s in detection.segments
            ] if detection.is_code_mixed else [],
        },
        "translation": translated,
        "timings": {
            "asr_s":       round(asr_time, 2),
            "translate_s": round(translate_time, 2),
            "total_s":     round(time.time() - t_total, 2),
        }
    }


# ── /voice ────────────────────────────────────────────────────────────────────
@app.post("/voice")
async def voice_pipeline(file: UploadFile = File(...)):
    file_location = f"data/input_audio/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = transcribe_audio(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {str(e)}")

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No speech detected")

    detection  = detect_language(text)
    translated = _translate_with_detection(text, detection)

    output_path = f"data/output_audio/response_{file.filename}.mp3"
    try:
        text_to_speech(translated, lang_code="en", output_path=output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

    return FileResponse(path=output_path, media_type="audio/mpeg", filename="response.mp3")


# ── /voice/info ───────────────────────────────────────────────────────────────
@app.post("/voice/info")
async def voice_pipeline_info(file: UploadFile = File(...)):
    t_total = time.time()
    file_location = f"data/input_audio/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    t    = time.time()
    text = transcribe_audio(file_location)
    asr_time = time.time() - t

    detection = detect_language(text)

    t          = time.time()
    translated = _translate_with_detection(text, detection)
    translate_time = time.time() - t

    output_path = f"data/output_audio/response_{file.filename}.mp3"
    t = time.time()
    text_to_speech(translated, lang_code="en", output_path=output_path)
    tts_time = time.time() - t

    return {
        "transcription":  text,
        "detected_lang":  detection.primary_lang,
        "mix_type":       detection.mix_type,
        "translation":    translated,
        "audio_saved_to": output_path,
        "timings": {
            "asr_s":       round(asr_time, 2),
            "translate_s": round(translate_time, 2),
            "tts_s":       round(tts_time, 2),
            "total_s":     round(time.time() - t_total, 2),
        },
        "status": "success ✅",
    }


# ── /speak ────────────────────────────────────────────────────────────────────
@app.post("/speak")
async def speak(payload: dict):
    text = payload.get("text", "").strip()
    lang = payload.get("lang", "en")
    if not text:
        raise HTTPException(status_code=422, detail="No text provided")
    output_path = f"data/output_audio/speak_{lang}.mp3"
    try:
        text_to_speech(text, lang_code=lang, output_path=output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    return FileResponse(path=output_path, media_type="audio/mpeg", filename=f"speech_{lang}.mp3")


# ── /detect-language ──────────────────────────────────────────────────────────
@app.post("/detect-language")
async def detect_lang_endpoint(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="No text provided")
    r = detect_language(text)
    return {
        "text":           text,
        "language":       r.primary_lang,
        "indictrans_tag": r.indictrans_tag,
        "is_code_mixed":  r.is_code_mixed,
        "mix_type":       r.mix_type,
        "confidence":     r.confidence,
        "segments": [
            {"text": s.text, "lang": s.lang_code, "tag": s.indictrans_tag}
            for s in r.segments
        ],
    }


# ── /translate ────────────────────────────────────────────────────────────────
@app.post("/translate")
async def translate_endpoint(payload: dict):
    text     = payload.get("text", "").strip()
    tgt_lang = payload.get("tgt_lang", "eng_Latn")
    if not text:
        raise HTTPException(status_code=422, detail="No text provided")
    src_lang = payload.get("src_lang") or detect_language(text).indictrans_tag
    try:
        translated = translate_text(text, src_lang=src_lang, tgt_lang=tgt_lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")
    return {"original": text, "src_lang": src_lang, "tgt_lang": tgt_lang, "translation": translated}


# ── /evaluate/phase1 ──────────────────────────────────────────────────────────
@app.get("/evaluate/phase1")
def run_phase1_evaluation():
    from src.evaluation.evaluate_asr import run_asr_evaluation, PHASE1_TEST_CASES
    results = run_asr_evaluation(
        transcribe_fn=transcribe_audio,
        translate_fn=lambda t: _translate_with_detection(t, detect_language(t)),
        test_cases=PHASE1_TEST_CASES,
    )
    return JSONResponse(content=results)


# ── Helper ────────────────────────────────────────────────────────────────────
def _translate_with_detection(text: str, detection) -> str:
    if detection.is_code_mixed and len(detection.segments) > 1:
        parts = []
        for seg in detection.segments:
            if seg.lang_code == "en":
                parts.append(seg.text)
            else:
                parts.append(translate_text(seg.text, src_lang=seg.indictrans_tag))
        return " ".join(parts)
    return translate_text(text, src_lang=detection.indictrans_tag)