# WhisperX Transcription Server

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
## English

### Description

A Flask-based REST API server for audio transcription using WhisperX. Supports GPU acceleration (CUDA) and CPU processing with automatic device detection, async task queue, web UI, and optional speaker diarization.

### Features

- GPU (CUDA) and CPU support
- Automatic device detection
- Async task queue — instant 202 response, result fetched by polling
- Web UI at `http://localhost:5001`
- Text transcription with timestamps
- Word-level alignment via wav2vec2
- Optional speaker diarization (pyannote.audio)
- Multi-language support
- Automatic saving of results as `.txt`
- Simple REST API

### Requirements

- Python 3.8+
- PyTorch
- WhisperX
- Flask
- CUDA (optional, for GPU acceleration)
- pyannote.audio (optional, for speaker diarization)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/whisperx-server.git
cd whisperx-server
```

2. **Install PyTorch:**

For GPU (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision torchaudio
```

3. **Install dependencies:**
```bash
pip install whisperx flask
```

4. **Check GPU availability (optional):**
```bash
python gpu_info.py
```

### Usage

> **Windows note:** always run with `PYTHONIOENCODING=utf-8` to avoid console encoding errors.

#### Starting the Server

```bash
# Auto device detection
PYTHONIOENCODING=utf-8 python whisperx_server.py

# Force GPU
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cuda

# Force CPU
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cpu

# With speaker diarization (requires HUGGINGFACE_TOKEN)
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cuda --diarize
```

Server runs on `http://0.0.0.0:5001`. Open `http://localhost:5001` for the web UI.

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/transcribe` | Submit audio file, returns `{task_id}` instantly (HTTP 202) |
| `GET` | `/tasks/<id>` | Poll task status and result |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/health` | Server status, GPU info, queue depth |
| `GET` | `/files` | List saved transcriptions |
| `GET` | `/files/<filename>` | Download a `.txt` transcription |
| `GET` | `/` | Web UI |

**Transcribe example:**
```bash
curl -X POST -F "data=@audio.mp3" -F "language=ru" http://localhost:5001/transcribe
# → {"task_id": "...", "poll_url": "/tasks/..."}

curl http://localhost:5001/tasks/<task_id>
# → {"status": "done", "result": {...}}
```

**Python example:**
```python
import requests, time

r = requests.post("http://localhost:5001/transcribe",
                  files={"data": open("audio.mp3", "rb")},
                  data={"language": "ru"})
task_id = r.json()["task_id"]

while True:
    task = requests.get(f"http://localhost:5001/tasks/{task_id}").json()
    if task["status"] == "done":
        print(task["result"]["text"])
        break
    time.sleep(2)
```

#### Speaker Diarization

Requires a one-time online setup:

1. Accept license for `pyannote/speaker-diarization-3.1` on HuggingFace
2. Accept license for `pyannote/segmentation-3.0` on HuggingFace
3. Create a read token at `huggingface.co → Settings → Access Tokens`
4. Run once with internet to download models (~500 MB cached locally):
```bash
set HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cuda --diarize
```
5. After the first download, runs fully offline — no token or internet needed.

### Configuration

#### Using a Local Model

By default the server downloads from Hugging Face. To use a local model, edit `whisperx_server.py`:
```python
model_path = "C:/path/to/your/model"
model = whisperx.load_model(model_path, device, compute_type=compute_type)
```

### Project Structure

```
whisperx-server/
├── whisperx_server.py      # Main server
├── gpu_info.py             # GPU diagnostics utility
├── templates/
│   └── index.html          # Web UI
├── transcriptions/         # Output directory (auto-created)
├── README.md
├── CHANGES.md              # Changelog
└── .gitignore
```

### Troubleshooting

**CUDA not available:**
- Install NVIDIA drivers and CUDA-enabled PyTorch
- Run `python gpu_info.py` for diagnostics

**Out of memory on GPU:**
- Reduce `batch_size` in `whisperx_server.py` (default: 4)
- Switch to CPU mode: `--device cpu`

**UnicodeEncodeError on Windows:**
- Run with `PYTHONIOENCODING=utf-8`

**Model download issues:**
- Check internet connection or use a local model path

### License

MIT

---

<a name="russian"></a>
## Русский

### Описание

REST API сервер на Flask для транскрибации аудио с использованием WhisperX. Поддерживает GPU (CUDA) и CPU, асинхронную очередь задач, веб-интерфейс и опциональную диаризацию спикеров.

### Возможности

- Поддержка GPU (CUDA) и CPU
- Автоматическое определение устройства
- Асинхронная очередь — мгновенный ответ 202, результат забирается поллингом
- Веб-интерфейс по адресу `http://localhost:5001`
- Транскрибация с временными метками
- Выравнивание на уровне слов (wav2vec2)
- Опциональная диаризация спикеров (pyannote.audio)
- Поддержка множества языков
- Автоматическое сохранение результатов в `.txt`
- Простой REST API

### Требования

- Python 3.8+
- PyTorch
- WhisperX
- Flask
- CUDA (опционально, для GPU)
- pyannote.audio (опционально, для диаризации)

### Установка

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/yourusername/whisperx-server.git
cd whisperx-server
```

2. **Установите PyTorch:**

Для GPU (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Только CPU:
```bash
pip install torch torchvision torchaudio
```

3. **Установите зависимости:**
```bash
pip install whisperx flask
```

4. **Проверьте доступность GPU (опционально):**
```bash
python gpu_info.py
```

### Использование

> **Windows:** всегда запускать с `PYTHONIOENCODING=utf-8` во избежание ошибок кодировки.

#### Запуск сервера

```bash
# Авто-определение устройства
PYTHONIOENCODING=utf-8 python whisperx_server.py

# Принудительно GPU
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cuda

# Принудительно CPU
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cpu

# С диаризацией спикеров (требует HUGGINGFACE_TOKEN)
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cuda --diarize
```

Сервер запускается на `http://0.0.0.0:5001`. Веб-интерфейс: `http://localhost:5001`.

#### API Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/transcribe` | Отправить аудио, получить `{task_id}` мгновенно (HTTP 202) |
| `GET` | `/tasks/<id>` | Статус и результат задачи |
| `GET` | `/tasks` | Список всех задач |
| `GET` | `/health` | Статус сервера, GPU, глубина очереди |
| `GET` | `/files` | Список сохранённых транскрипций |
| `GET` | `/files/<filename>` | Скачать `.txt` файл |
| `GET` | `/` | Веб-интерфейс |

**Пример транскрибации:**
```bash
curl -X POST -F "data=@audio.mp3" -F "language=ru" http://localhost:5001/transcribe
# → {"task_id": "...", "poll_url": "/tasks/..."}

curl http://localhost:5001/tasks/<task_id>
# → {"status": "done", "result": {...}}
```

#### Диаризация спикеров

Требует однократной онлайн-настройки:

1. Принять лицензию `pyannote/speaker-diarization-3.1` на HuggingFace
2. Принять лицензию `pyannote/segmentation-3.0` на HuggingFace
3. Создать токен: `huggingface.co → Settings → Access Tokens`
4. Запустить один раз с интернетом для скачивания моделей (~500 MB):
```bash
set HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
PYTHONIOENCODING=utf-8 python whisperx_server.py --device cuda --diarize
```
5. После скачивания работает полностью офлайн — токен и интернет не нужны.

### Конфигурация

#### Локальная модель

По умолчанию модель загружается с Hugging Face. Для локальной модели отредактируйте `whisperx_server.py`:
```python
model_path = "C:/path/to/your/model"
model = whisperx.load_model(model_path, device, compute_type=compute_type)
```

### Структура проекта

```
whisperx-server/
├── whisperx_server.py      # Основной сервер
├── gpu_info.py             # Утилита диагностики GPU
├── templates/
│   └── index.html          # Веб-интерфейс
├── transcriptions/         # Папка с результатами (создается автоматически)
├── README.md
├── CHANGES.md              # История изменений
└── .gitignore
```

### Решение проблем

**CUDA недоступна:**
- Установите драйверы NVIDIA и PyTorch с поддержкой CUDA
- Запустите `python gpu_info.py` для диагностики

**Нехватка памяти GPU:**
- Уменьшите `batch_size` в `whisperx_server.py` (по умолчанию: 4)
- Переключитесь на CPU: `--device cpu`

**UnicodeEncodeError на Windows:**
- Запускайте с `PYTHONIOENCODING=utf-8`

**Проблемы с загрузкой модели:**
- Проверьте интернет или используйте локальный путь к модели

### Лицензия

MIT
