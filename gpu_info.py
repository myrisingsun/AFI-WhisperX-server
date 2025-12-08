#!/usr/bin/env python3
"""
GPU Information Script
Показывает детальную информацию о GPU и CUDA
"""

import sys


def print_separator(char="=", length=70):
    print(char * length)


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    print_separator()
    print("  GPU INFORMATION UTILITY")
    print_separator()

    # ========== ПРОВЕРКА PYTORCH ==========
    print_section("1. PyTorch")
    try:
        import torch
        print(f"✓ PyTorch установлен")
        print(f"  Версия: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch не установлен")
        print("  Установка: pip install torch")
        sys.exit(1)

    # ========== CUDA ДОСТУПНОСТЬ ==========
    print_section("2. CUDA")
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print(f"✓ CUDA доступна")
        print(f"  Версия CUDA: {torch.version.cuda}")
        print(f"  cuDNN версия: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    else:
        print(f"✗ CUDA недоступна")
        print(f"\nВозможные причины:")
        print(f"  - Нет NVIDIA GPU")
        print(f"  - Драйверы NVIDIA не установлены")
        print(f"  - PyTorch установлен без CUDA поддержки")
        print(f"  - Несовместимость версий CUDA и драйверов")
        return

    # ========== ИНФОРМАЦИЯ О GPU ==========
    print_section("3. GPU Устройства")
    device_count = torch.cuda.device_count()
    print(f"Количество GPU: {device_count}")

    for i in range(device_count):
        print(f"\n--- GPU #{i} ---")
        props = torch.cuda.get_device_properties(i)

        print(f"  Название: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Общая память: {props.total_memory / 1024 ** 3:.2f} GB")
        print(f"  Мультипроцессоры: {props.multi_processor_count}")
        print(f"  CUDA Cores: ~{props.multi_processor_count * 128}")  # примерная оценка

        # Текущее использование памяти
        print(f"\n  Использование памяти:")
        print(f"    Занято: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"    Зарезервировано: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
        print(f"    Свободно: {(props.total_memory - torch.cuda.memory_reserved(i)) / 1024 ** 3:.2f} GB")

    # ========== ТЕКУЩЕЕ УСТРОЙСТВО ==========
    print_section("4. Текущее устройство")
    current_device = torch.cuda.current_device()
    print(f"Текущий GPU ID: {current_device}")
    print(f"Название: {torch.cuda.get_device_name(current_device)}")

    # ========== ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ==========
    print_section("5. Быстрый тест производительности")
    try:
        import time

        # CPU тест
        print("\nТест на CPU...")
        x_cpu = torch.randn(5000, 5000)
        start = time.time()
        y_cpu = torch.matmul(x_cpu, x_cpu)
        cpu_time = time.time() - start
        print(f"  Умножение матриц 5000x5000: {cpu_time:.4f} сек")

        # GPU тест
        print("\nТест на GPU...")
        x_gpu = torch.randn(5000, 5000).cuda()
        torch.cuda.synchronize()
        start = time.time()
        y_gpu = torch.matmul(x_gpu, x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  Умножение матриц 5000x5000: {gpu_time:.4f} сек")

        speedup = cpu_time / gpu_time
        print(f"\n  Ускорение GPU vs CPU: {speedup:.2f}x")

        # Очистка памяти
        del x_cpu, y_cpu, x_gpu, y_gpu
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Ошибка теста: {e}")

    # ========== РЕКОМЕНДАЦИИ ==========
    print_section("6. Рекомендации для WhisperX")

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    if total_memory >= 8:
        print("✓ Память GPU достаточна для модели 'large'")
        print("  Рекомендуется: large или medium")
    elif total_memory >= 4:
        print("✓ Память GPU достаточна для модели 'medium'")
        print("  Рекомендуется: medium или small")
    else:
        print("⚠ Ограниченная память GPU")
        print("  Рекомендуется: small или tiny")

    print(f"\nРекомендуемые настройки:")
    print(f"  device = 'cuda'")
    print(f"  compute_type = 'float16'")

    # ========== СИСТЕМНАЯ ИНФОРМАЦИЯ ==========
    print_section("7. Системная информация")
    try:
        import platform
        print(f"ОС: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"Архитектура: {platform.machine()}")
    except:
        pass

    # ========== NVIDIA-SMI (если доступна) ==========
    print_section("8. NVIDIA-SMI (дополнительно)")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("✗ nvidia-smi недоступна")
    except FileNotFoundError:
        print("✗ nvidia-smi не найдена в PATH")
    except Exception as e:
        print(f"✗ Ошибка запуска nvidia-smi: {e}")

    print_separator()
    print("  Проверка завершена")
    print_separator()


if __name__ == "__main__":
    main()