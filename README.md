# vector_benchmark

Сравнение CPU и GPU для скалярного умножения векторов: `c[i] = a[i] * b[i]`.

Измеряются 4 режима (мс): **CPU calc**, **CPU e2e**, **GPU kernel**, **GPU e2e**.  
На выходе: `benchmark_results.csv` и график `benchmark_results.png`.

## Требования

| | Windows | Linux |
|---|---------|--------|
| GPU | NVIDIA с поддержкой CUDA | то же |
| Драйвер | актуальный NVIDIA | `nvidia-driver` |
| Сборка | [CUDA Toolkit](https://developer.nvidia.com/cuda-download) (`nvcc` в PATH) | `cuda-toolkit`, `nvcc` |
| Компилятор | MSVC (Visual Studio, x64) — см. ниже | `g++` (`build-essential`) |
| График | Python 3, `matplotlib`, `numpy` | то же |

Полный прогон до **N = 200·10⁶** может занять много времени. Для одного N нужно порядка **2.4 ГБ RAM** (три вектора `float` на CPU) и столько же VRAM на GPU (три массива).

## Сборка

### Windows

Откройте **x64 Native Tools Command Prompt for VS** (нужны `cl.exe` и `nvcc`):

```cmd
cd vector_benchmark
nvcc -O3 -std=c++17 benchmark.cu -o benchmark.exe
```

### Linux

```bash
cd vector_benchmark
nvcc -O3 -std=c++17 benchmark.cu -o benchmark
```

## Запуск

### Windows

```cmd
benchmark.exe
benchmark.exe 5
benchmark.exe 5 results.csv
```

### Linux

```bash
./benchmark
./benchmark 5
./benchmark 5 results.csv
```

**Аргументы:**

1. число повторов для каждого N (по умолчанию `5`);
2. путь к CSV (по умолчанию `benchmark_results.csv`).

В консоли: таблица времён, аппроксимация `T ≈ a·N + b`, точки break-even.

## График

```bash
pip install matplotlib numpy
python plot_results.py benchmark_results.csv
```

Без аргумента скрипт ищет `benchmark_results.csv` в текущей папке.

## Файлы

| Файл | Описание |
|------|----------|
| `benchmark.cu` | бенчмарк CPU + CUDA |
| `plot_results.py` | график T(N), метки break-even |

## Частые проблемы

- **`nvcc` не найден** — установите CUDA Toolkit, добавьте `bin` в PATH.
- **Windows: `cl.exe` не найден** — собирайте из **x64 Native Tools Command Prompt**, не из обычного cmd.
- **Нет CUDA device** — проверьте драйвер: `nvidia-smi`.
- **Пропуск больших N (мало VRAM)** — уменьшите верхнюю границу в `defaultSizes()` в `benchmark.cu`.
