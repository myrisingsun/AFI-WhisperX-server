from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
import re
import uuid
import threading
import queue
from datetime import datetime
import argparse

app = Flask(__name__)

print("=" * 50)
print("Запуск WhisperX сервера...")
print("=" * 50)

try:
    import whisperx
    import torch

    print("✓ WhisperX и PyTorch импортированы успешно")
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    sys.exit(1)

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["auto", "cuda", "cpu"],
                    default=os.getenv("DEVICE", "auto"),
                    help="Устройство для вычислений: auto, cuda, cpu")
parser.add_argument("--diarize", action="store_true",
                    help="Включить диаризацию спикеров (требует HUGGINGFACE_TOKEN)")
args = parser.parse_args()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
diarize_enabled = args.diarize or bool(HUGGINGFACE_TOKEN)

# Определение устройства
if args.device == "cuda":
    if not torch.cuda.is_available():
        print("=" * 50)
        print("✗ ОШИБКА: CUDA запрошена, но недоступна!")
        print("✗ Доступные опции: --device cpu или --device auto")
        print("=" * 50)
        sys.exit(1)
    device = "cuda"
    compute_type = "int8"
    print("🎮 Режим: Принудительно GPU")
elif args.device == "cpu":
    device = "cpu"
    compute_type = "int8"
    print("💻 Режим: Принудительно CPU")
else:  # auto
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8"
    print("🔄 Режим: Авто-определение")

print(f"✓ Устройство: {device}")
if device == "cuda":
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA версия: {torch.version.cuda}")
    print(f"✓ Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
print(f"✓ Тип вычислений: {compute_type}")
print("Загрузка модели WhisperX...")

try:
    ## model_path = "C:/Users/kpaha/.cache/huggingface/hub/models--Systran--faster-whisper-medium/snapshots/08e178d48790749d25932bbc082711ddcfdfbc4f"
    model = whisperx.load_model("medium", device, compute_type=compute_type)
    print("✓ Модель загружена успешно!")
except Exception as e:
    print(f"✗ Ошибка загрузки модели: {e}")
    sys.exit(1)

# Загрузка модели диаризации (если включена)
diarize_model = None
if diarize_enabled:
    try:
        print("Загрузка модели диаризации...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=HUGGINGFACE_TOKEN, device=device)
        print("✓ Модель диаризации загружена!")
    except Exception as e:
        print(f"⚠ Диаризация недоступна: {e}")
        diarize_model = None
        diarize_enabled = False

# Создаем папку для результатов
OUTPUT_DIR = "transcriptions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Task store ---
tasks = {}
tasks_lock = threading.Lock()
task_queue = queue.Queue()

# Alignment model cache: {language_code: (model_a, metadata)}
align_model_cache = {}
align_cache_lock = threading.Lock()


def _now():
    return datetime.utcnow().isoformat() + "Z"


def build_txt_content(filename, duration, language, full_text, segments, speakers_present):
    lines = []
    lines.append(f"Файл: {filename}")
    lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Длительность: {duration:.2f} сек")
    lines.append(f"Язык: {language}")
    lines.append(f"Устройство: {device}")
    lines.append(f"Сегментов: {len(segments)}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("ПОЛНАЯ ТРАНСКРИПЦИЯ:")
    lines.append("-" * 80)
    lines.append(full_text)
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    lines.append("С ВРЕМЕННЫМИ МЕТКАМИ:")
    lines.append("-" * 80)
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        if speakers_present:
            speaker = seg.get("speaker", "")
            lines.append(f"[{start:.2f}s - {end:.2f}s] [{speaker}] {text}")
        else:
            lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
    return "\n".join(lines)


def _update_task(task_id, **kwargs):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kwargs)
            tasks[task_id]["updated_at"] = _now()


def _trim_tasks():
    """Keep only the 200 most recent tasks."""
    with tasks_lock:
        if len(tasks) > 200:
            sorted_ids = sorted(tasks, key=lambda k: tasks[k]["created_at"])
            for old_id in sorted_ids[:len(tasks) - 200]:
                del tasks[old_id]


def process_task(job):
    task_id = job["task_id"]
    file_path = job["file_path"]
    language = job["language"]
    filename = job["filename"]

    try:
        _update_task(task_id, status="processing", progress="loading audio")
        print(f"\n{'=' * 60}")
        print(f"[{task_id[:8]}] Обработка: {filename}")

        # Загружаем аудио
        print(f"[{task_id[:8]}] Загрузка аудио...")
        audio = whisperx.load_audio(file_path)
        duration = len(audio) / 16000
        print(f"[{task_id[:8]}] Длительность: {duration:.2f} сек")

        if device == "cuda":
            print(f"[{task_id[:8]}] GPU память: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

        # Транскрибация
        _update_task(task_id, progress="transcribing")
        print(f"[{task_id[:8]}] Транскрибация (язык: {language or 'авто'})...")

        transcribe_options = {
            "batch_size": 4,
            "chunk_size": 15,
            "print_progress": True,
            "combined_progress": True,
        }
        if language:
            transcribe_options["language"] = language

        result = model.transcribe(audio, **transcribe_options)
        segments = result.get("segments", [])
        detected_language = result.get("language", "unknown")

        # Alignment (с кэшем модели по языку)
        _update_task(task_id, progress="aligning")
        try:
            print(f"[{task_id[:8]}] Alignment (язык: {detected_language})...")
            with align_cache_lock:
                if detected_language not in align_model_cache:
                    print(f"[{task_id[:8]}] Загрузка align-модели для '{detected_language}'...")
                    align_model_cache[detected_language] = whisperx.load_align_model(
                        language_code=detected_language, device=device)
                model_a, metadata = align_model_cache[detected_language]
            result_aligned = whisperx.align(
                segments, model_a, metadata, audio, device,
                return_char_alignments=False)
            segments = result_aligned["segments"]
            print(f"[{task_id[:8]}] ✓ Alignment выполнен")
        except Exception as e:
            print(f"[{task_id[:8]}] ⚠ Alignment пропущен: {e}")

        # Diarization
        speakers_present = False
        if diarize_model is not None:
            _update_task(task_id, progress="diarizing")
            try:
                print(f"[{task_id[:8]}] Диаризация...")
                diarize_segments = diarize_model(audio)
                result_diarized = whisperx.assign_word_speakers(
                    diarize_segments, {"segments": segments})
                segments = result_diarized["segments"]
                speakers_present = any(seg.get("speaker") for seg in segments)
                print(f"[{task_id[:8]}] ✓ Диаризация выполнена")
            except Exception as e:
                print(f"[{task_id[:8]}] ⚠ Диаризация пропущена: {e}")

        # Собираем текст
        all_texts = [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
        full_text = " ".join(all_texts)

        print(f"[{task_id[:8]}] Сегментов: {len(segments)}, символов: {len(full_text)}")

        # Сохраняем файл
        _update_task(task_id, progress="saving")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        txt_filename = f"{base_name}_{timestamp}.txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)

        txt_content = build_txt_content(filename, duration, detected_language,
                                        full_text, segments, speakers_present)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        print(f"[{task_id[:8]}] ✓ Сохранено: {txt_path}")

        result_payload = {
            "success": True,
            "text": full_text,
            "language": detected_language,
            "device": device,
            "segments_count": len(segments),
            "duration": duration,
            "output_file": txt_filename,
            "file_path": os.path.abspath(txt_path),
            "text_length": len(full_text),
            "diarization": speakers_present,
            "all_segments": [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip(),
                    "speaker": seg.get("speaker", ""),
                }
                for seg in segments
            ]
        }

        _update_task(task_id, status="done", progress="done", result=result_payload)

    except Exception as e:
        import traceback
        print(f"[{task_id[:8]}] ✗ ОШИБКА: {e}")
        traceback.print_exc()
        _update_task(task_id, status="error", progress="error", error=str(e))

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        _trim_tasks()


def worker_loop():
    while True:
        job = task_queue.get(block=True)
        process_task(job)
        task_queue.task_done()


worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()
print("✓ Фоновый worker запущен")


# --- Endpoints ---

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "data" not in request.files:
        return jsonify({"error": "Файл 'data' не найден"}), 400

    file = request.files["data"]
    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400

    language = request.form.get("language") or None
    task_id = str(uuid.uuid4())
    safe_name = re.sub(r"[^\w\-.]", "_", file.filename)
    file_path = f"temp_audio_{task_id}_{safe_name}"

    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Не удалось сохранить файл: {e}"}), 500

    now = _now()
    task = {
        "task_id": task_id,
        "status": "pending",
        "progress": "queued",
        "created_at": now,
        "updated_at": now,
        "filename": file.filename,
        "language": language,
        "result": None,
        "error": None,
    }
    with tasks_lock:
        tasks[task_id] = task

    task_queue.put({
        "task_id": task_id,
        "file_path": file_path,
        "language": language,
        "filename": file.filename,
    })

    return jsonify({
        "task_id": task_id,
        "status": "pending",
        "poll_url": f"/tasks/{task_id}",
    }), 202


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if task is None:
        return jsonify({"error": "Задача не найдена"}), 404
    return jsonify(task)


@app.route("/tasks", methods=["GET"])
def list_tasks():
    with tasks_lock:
        summary = [
            {k: v for k, v in t.items() if k != "result"}
            for t in tasks.values()
        ]
    summary.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify({"tasks": summary, "count": len(summary)})


@app.route("/files", methods=["GET"])
def list_files():
    try:
        files = []
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".txt"):
                path = os.path.join(OUTPUT_DIR, f)
                size = os.path.getsize(path)
                mtime = os.path.getmtime(path)
                files.append({
                    "filename": f,
                    "size": size,
                    "modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
        files.sort(key=lambda x: x["modified"], reverse=True)
        return jsonify({"files": files, "count": len(files)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files/<filename>", methods=["GET"])
def download_file(filename):
    return send_from_directory(os.path.abspath(OUTPUT_DIR), filename, as_attachment=True)


@app.route("/health", methods=["GET"])
def health():
    gpu_info = {}
    if device == "cuda":
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB",
        }

    return jsonify({
        "status": "ok",
        "device": device,
        "compute_type": compute_type,
        "model": "whisperx-medium",
        "diarization_enabled": diarize_enabled,
        "queue_depth": task_queue.qsize(),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        **gpu_info,
    })


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", diarization=diarize_enabled)


if __name__ == "__main__":
    print("=" * 50)
    print("Сервер запущен на http://0.0.0.0:5001")
    print(f"Папка результатов: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Режим устройства: {args.device}")
    print(f"Используется: {device} ({compute_type})")
    print(f"Диаризация: {'включена' if diarize_enabled else 'выключена'}")
    print("Для остановки: Ctrl+C")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
