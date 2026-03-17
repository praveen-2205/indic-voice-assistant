# 🎙️ Indic Multilingual Voice-to-Voice Assistant

> A complete AI pipeline that converts speech in Indian languages into translated spoken English responses — supporting Tamil, Hindi, Telugu, Bengali, Tanglish, and Hinglish.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Whisper](https://img.shields.io/badge/Whisper-medium-orange)
![IndicTrans2](https://img.shields.io/badge/IndicTrans2-1B-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What It Does

```
Your Voice (Tamil/Hindi/Tanglish)
        ↓
  Audio Preprocessing       — noise reduction, silence removal, normalisation
        ↓
  Whisper ASR (medium)      — speech → text
        ↓
  Language Detection        — auto-detects Tamil, Hindi, Tanglish, Hinglish, etc.
        ↓
  IndicTrans2 Translation   — Indic text → English
        ↓
  Edge TTS                  — English text → spoken audio
        ↓
  Voice Response (MP3)      ✅
```

---

## 🌐 Supported Languages

| Language   | Script       | Input | Translation | TTS Voice              |
|------------|--------------|-------|-------------|------------------------|
| Tamil      | தமிழ்        | ✅    | ✅          | ta-IN-PallaviNeural    |
| Hindi      | देवनागरी     | ✅    | ✅          | hi-IN-SwaraNeural      |
| Telugu     | తెలుగు       | ✅    | ✅          | te-IN-ShrutiNeural     |
| Bengali    | বাংলা        | ✅    | ✅          | bn-IN-TanishaaNeural   |
| English    | Latin        | ✅    | ✅          | en-US-JennyNeural      |
| Tanglish   | Latin (mixed)| ✅    | ✅          | —                      |
| Hinglish   | Latin (mixed)| ✅    | ✅          | —                      |

---

## 🚀 Quick Start

### Option 1 — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/indic-voice-assistant.git
cd indic-voice-assistant

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --reload

# 5. Open the Web UI
# http://127.0.0.1:8000/ui
```

### Option 2 — Run with Docker

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 🖥️ Web Interface

Open `http://127.0.0.1:8000/ui` in your browser.

Features:
- 📂 Drag and drop audio file upload
- 🎤 Record directly from microphone
- 📝 View transcription in original language
- 🌐 View English translation
- 🔊 Play the spoken English response
- ⏱️ See per-stage timing breakdown (ASR / Translate / TTS)

---

## 📡 API Endpoints

| Method | Endpoint          | Description                              |
|--------|-------------------|------------------------------------------|
| POST   | `/transcribe`     | Audio → JSON (transcription + translation) |
| POST   | `/voice`          | Audio → MP3 spoken response              |
| POST   | `/voice/info`     | Audio → JSON + saves MP3 to disk         |
| POST   | `/speak`          | Text → MP3 speech                        |
| POST   | `/translate`      | Text → translated text                   |
| POST   | `/detect-language`| Text → language info + segments          |
| GET    | `/stats`          | Cache stats + request timings            |
| GET    | `/evaluate/phase1`| Run WER + BLEU evaluation                |
| GET    | `/ui`             | Web interface                            |
| GET    | `/docs`           | Swagger API documentation                |

### Example — Transcribe and Translate

```bash
curl -X POST "http://127.0.0.1:8000/voice/info" \
     -F "file=@your_audio.wav"
```

Response:
```json
{
  "transcription": "நான் இன்று சந்தைக்கு போகிறேன்",
  "detected_lang": "ta",
  "mix_type": null,
  "translation": "I am going to the market today",
  "audio_saved_to": "data/output_audio/response.mp3",
  "timings": {
    "asr_s": 93.5,
    "translate_s": 7.3,
    "tts_s": 1.2,
    "total_s": 102.0
  },
  "status": "success ✅"
}
```

### Example — Detect Language

```bash
curl -X POST "http://127.0.0.1:8000/detect-language" \
     -H "Content-Type: application/json" \
     -d '{"text": "naan office ku late aa varen"}'
```

Response:
```json
{
  "language": "ta",
  "indictrans_tag": "tam_Taml",
  "is_code_mixed": true,
  "mix_type": "Tanglish",
  "segments": [
    {"text": "naan", "lang": "ta", "tag": "tam_Taml"},
    {"text": "office", "lang": "en", "tag": "eng_Latn"},
    {"text": "ku late aa", "lang": "ta", "tag": "tam_Taml"},
    {"text": "varen", "lang": "ta", "tag": "tam_Taml"}
  ]
}
```

---

## 📁 Project Structure

```
indic-voice-assistant/
│
├── app/
│   └── main.py                  # FastAPI app — all endpoints + lifespan model loading
│
├── src/
│   ├── asr/
│   │   └── speech_to_text.py    # Whisper ASR with language detection
│   │
│   ├── translation/
│   │   └── translate.py         # IndicTrans2 with LRU cache
│   │
│   ├── tts/
│   │   └── text_to_speech.py    # Edge TTS multilingual synthesis
│   │
│   ├── utils/
│   │   ├── audio_utils.py       # Noise reduction, silence removal, normalisation
│   │   └── language_detector.py # Unicode script + Tanglish/Hinglish detection
│   │
│   ├── evaluation/
│   │   └── evaluate_asr.py      # WER + BLEU metric computation
│   │
│   └── config/
│       ├── settings.py          # Centralised configuration
│       └── model_manager.py     # Singleton — loads all models once at startup
│
├── static/
│   └── index.html               # Web UI
│
├── data/
│   ├── input_audio/             # Uploaded audio files
│   └── output_audio/            # Generated MP3 responses
│
├── tests/
│   └── test_pipeline.py         # Phase 1 verification tests
│
├── docker/
│   └── Dockerfile               # Container definition
│
├── docker-compose.yml           # Multi-service orchestration
├── requirements.txt             # All Python dependencies
└── README.md
```

---

## 🧠 Models Used

| Model | Size | Purpose |
|-------|------|---------|
| [OpenAI Whisper medium](https://github.com/openai/whisper) | 769M params | Speech-to-text for Indic languages |
| [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) | 1B params | Indic → English translation |
| [Microsoft Edge TTS](https://github.com/rany2/edge-tts) | Cloud neural | Text-to-speech in 5 Indian languages |

---

## ⚡ Performance

Measured on CPU (no GPU):

| Stage | Time | Notes |
|-------|------|-------|
| Audio preprocessing | ~0.5s | Noise reduction + normalisation |
| ASR (Whisper medium) | ~90s | CPU-bound — use GPU for 5-10x speedup |
| Translation (first run) | ~7s | IndicTrans2 inference |
| Translation (cached) | <0.01s | LRU cache hit |
| TTS (Edge) | ~1.2s | Network call to Microsoft servers |

> **GPU note:** An NVIDIA GPU reduces ASR from ~90s to ~3-5s, bringing total response time under 10 seconds.

---

## 🔧 Configuration

Edit `src/config/settings.py` to tune the system:

```python
WHISPER_MODEL_SIZE   = "medium"  # base | small | medium | large
TRANSLATION_CACHE_SIZE = 256     # number of cached translations
ENABLE_PREPROCESSING = True      # toggle audio preprocessing
SILENCE_TOP_DB       = 30        # silence detection threshold
```

---

## 🧪 Testing

Run the Phase 1 verification tests (no audio files needed):

```bash
python tests/test_pipeline.py
```

Expected output:
```
TEST 1 — Word Error Rate (WER)    ✅ 5/5 PASS
TEST 2 — BLEU Score               ✅ 3/3 PASS
TEST 3 — Language Detection       ✅ 5/5 PASS
TEST 4 — Translation Module       ✅ PASS
TEST 5 — ASR Module               ✅ PASS
```

---

## 🗺️ Roadmap

- [ ] GPU inference support
- [ ] Real-time streaming ASR via WebSockets
- [ ] All 22 IndicTrans2 languages
- [ ] Indic TTS voices (Tamil/Hindi output, not just English)
- [ ] LLM integration for conversational responses
- [ ] Mobile app (React Native)
- [ ] Fine-tuned Whisper on Indic corpora

---

## 📦 Dependencies

```
fastapi          — REST API framework
uvicorn          — ASGI server
openai-whisper   — Speech recognition
transformers     — IndicTrans2 model
torch            — Deep learning inference
edge-tts         — Text-to-speech
librosa          — Audio processing
noisereduce      — Noise reduction
langdetect       — Language detection fallback
nest_asyncio     — Async compatibility fix
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🤝 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) — IndicTrans2 model
- [OpenAI](https://openai.com/research/whisper) — Whisper ASR
- [Microsoft](https://azure.microsoft.com/en-us/products/cognitive-services/text-to-speech/) — Edge TTS
- [HuggingFace](https://huggingface.co/) — Model hosting and transformers library

---

## 📄 License

MIT License — free to use, modify, and distribute.

---
