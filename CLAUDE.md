# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Server

**Windows requires `PYTHONIOENCODING=utf-8`** (cp1251 console can't handle Unicode symbols):

```bash
# Auto device detection (default)
PYTHONIOENCODING=utf-8 python whisperx_server.py

# Force GPU or CPU
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cuda
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cpu

# With speaker diarization (requires HUGGINGFACE_TOKEN env var)
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cuda --diarize

# Via environment variable
set DEVICE=cuda && set PYTHONIOENCODING=utf-8 && python whisperx_server.py
```

Server runs on `http://0.0.0.0:5001`. Web UI at `http://localhost:5001`.

## GPU Diagnostics

```bash
python gpu_info.py
```

## API Endpoints

- `POST /transcribe` — multipart form with `data` (audio file) and optional `language`; returns `{task_id, poll_url}` instantly (HTTP 202)
- `GET /tasks/<task_id>` — poll task status (`pending|processing|done|error`) + result when done
- `GET /tasks` — list all tasks (no result field)
- `GET /health` — server status, GPU info, `diarization_enabled`, `queue_depth`
- `GET /files` — list saved transcriptions
- `GET /files/<filename>` — download a `.txt` transcription file
- `GET /` — Web UI (HTML)

## Architecture

```
POST /transcribe → save file → enqueue → return {task_id} (202, instant)
                                  ↓
Background worker thread → process_task() → update tasks dict
                                  ↓
GET /tasks/<id> ← poll every 2s ← Browser
```

**`whisperx_server.py`** — single-file Flask server. At startup it:
1. Parses `--device` / `--diarize` args (or `DEVICE` / `HUGGINGFACE_TOKEN` env vars)
2. Loads the WhisperX `medium` model globally
3. Optionally loads `whisperx.DiarizationPipeline` if `--diarize` or `HUGGINGFACE_TOKEN` set
4. Starts a single daemon worker thread consuming a `queue.Queue`
5. Creates `transcriptions/` output directory

**`process_task()`** pipeline stages:
1. `loading audio` — `whisperx.load_audio()`
2. `transcribing` — `model.transcribe()` with `batch_size=4, chunk_size=15`
3. `aligning` — `whisperx.load_align_model()` + `whisperx.align()` (skipped silently on failure)
4. `diarizing` — `diarize_model(audio)` + `whisperx.assign_word_speakers()` (if enabled)
5. `saving` — writes `.txt` to `transcriptions/`

**`templates/index.html`** — dark-theme web UI, polls `/tasks/<id>` every 2s, shows progress bar with stage labels, segments table with optional Speaker column.

**`gpu_info.py`** — standalone diagnostic script (no server dependency). Checks PyTorch, CUDA availability, GPU memory, runs a 5000×5000 matrix multiply benchmark, and calls `nvidia-smi`.

## Model Configuration

By default, `whisperx.load_model("medium", ...)` downloads from Hugging Face. To use a local model, edit `whisperx_server.py` line 58-59:

```python
model_path = "C:/path/to/your/model"
model = whisperx.load_model(model_path, device, compute_type=compute_type)
```

## Known Issues

- `README.md` has an unresolved git merge conflict (lines 1–410).
- The `auto` device mode sets `compute_type = "int8"`; GPU typically benefits from `"float16"` or `"float32"` (cuda forced mode already uses `float32`).
- Single worker thread — concurrent transcription requests queue behind each other (by design, GPU memory is limited).
- Windows cp1251 console: always run with `PYTHONIOENCODING=utf-8`.
- pyannote version mismatch warnings at startup are cosmetic and do not affect transcription.
