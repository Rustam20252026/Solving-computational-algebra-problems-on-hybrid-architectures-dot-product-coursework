/**
 * =============================================================================
 * БЕНЧМАРК: поэлементное умножение двух векторов float на CPU и GPU
 * =============================================================================
 *
 * Задача: c[i] = a[i] * b[i] для i = 0 .. N-1.
 *
 * Четыре измеряемых времени (в миллисекундах):
 *   cpu_ms         — CPU calc: только цикл умножения (векторы уже в ОЗУ).
 *   cpu_e2e_ms     — CPU e2e:  заполнение векторов + умножение (полный путь на CPU).
 *   gpu_kernel_ms  — GPU kernel: только CUDA-kernel (данные уже скопированы в VRAM).
 *   gpu_e2e_ms     — GPU e2e:   fill + cudaMalloc + копии + kernel + free (полный путь GPU).
 *
 * Две точки break-even (когда GPU становится быстрее CPU):
 *   1) cpu_ms         против gpu_kernel_ms  — сравнение «чистого» вычисления;
 *   2) cpu_e2e_ms     против gpu_e2e_ms    — сравнение полного цикла с нуля.
 *
 * Сборка:  nvcc -O3 benchmark.cu -o benchmark.exe
 * Запуск:   benchmark.exe
 *           benchmark.exe 5
 *           benchmark.exe 5 results.csv
 * График:   python plot_results.py benchmark_results.csv
 */

// --- Подключаемые заголовки ---
#include <algorithm>      // std::sort — для медианы
#include <chrono>         // high_resolution_clock — высокоточный таймер
#include <cmath>          // std::fabs — модуль числа
#include <cstdlib>        // std::atoi, EXIT_FAILURE
#include <cuda_runtime.h> // API CUDA: память, копирование, запуск kernel
#include <fstream>        // std::ofstream — запись CSV-файла
#include <iomanip>        // std::setw, setprecision — таблица в консоли
#include <iostream>       // std::cout, std::cerr
#include <numeric>        // std::accumulate — сумма элементов (для R²)
#include <random>         // std::mt19937 — генератор случайных чисел
#include <sstream>        // std::ostringstream — строка с формулой
#include <string>         // std::string
#include <vector>         // std::vector — динамические массивы

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN  // меньше заголовков Windows
#include <windows.h>         // SetConsoleOutputCP — UTF-8 в консоли
#endif

// =============================================================================
// Раздел 1. CUDA-kernel и вспомогательные функции
// =============================================================================

// __global__ — функция выполняется на GPU; вызывается с хоста как kernel<<<...>>>()
__global__ void vectorMultiplyKernel(const float* a, const float* b, float* c, size_t N) {
    // Глобальный индекс потока: какой элемент вектора обрабатывает этот поток
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // blockIdx.x — номер блока, blockDim.x — потоков в блоке, threadIdx.x — поток в блоке
    if (idx < N) {              // лишние потоки при N не кратном размеру сетки — не работают
        c[idx] = a[idx] * b[idx];  // одно поэлементное умножение
    }
}

// Проверка кода возврата CUDA; при ошибке — сообщение и выход из программы
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {   // cudaSuccess — единственный «успешный» код
        std::cerr << "CUDA error [" << msg << "]: " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// Включить UTF-8 в консоли Windows (кириллица в выводе)
void setupConsoleUtf8() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);  // кодовая страница вывода
    SetConsoleCP(65001);        // кодовая страница ввода
#endif
}

// Достаточно ли свободной VRAM для трёх массивов float размера N на GPU
bool hasEnoughGpuMemory(size_t N) {
    size_t free_bytes = 0, total_bytes = 0;
    checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "mem info");
    // 3 массива × N × 4 байта + запас 128 МиБ под служебные нужды драйвера
    const size_t need = N * sizeof(float) * 3 + 128ULL * 1024 * 1024;
    return free_bytes >= need;
}

std::string formatSci(double v) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(4) << v; 
    return os.str();
}

// =============================================================================
// Раздел 2. Код для CPU
// =============================================================================

// Последовательное умножение векторов в оперативной памяти (один поток CPU)
void cpuMultiply(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    const size_t N = a.size();           // число элементов
    for (size_t i = 0; i < N; ++i) {
        c[i] = a[i] * b[i];
    }
}

// Заполнение векторов a и b случайными числами от 0 до 1
void fillVectors(size_t N, std::vector<float>& a, std::vector<float>& b) {
    std::mt19937 rng(42);  // фиксированный seed — одинаковые данные при каждом запуске
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    a.resize(N);  // выделить/изменить размер под N элементов
    b.resize(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);  // случайное a[i]
        b[i] = dist(rng);  // случайное b[i]
    }
}

// =============================================================================
// Раздел 3. Структуры данных и замер времени
// =============================================================================

// Один ряд таблицы результатов: размер N и четыре времени (медианы по повторам)
struct Sample {
    size_t N = 0;
    double cpu_ms = 0.0;         // CPU calc
    double cpu_e2e_ms = 0.0;     // CPU end-to-end
    double gpu_kernel_ms = 0.0;  // только kernel на GPU
    double gpu_e2e_ms = 0.0;     // полный цикл на GPU
};

using Clock = std::chrono::high_resolution_clock;  // тип часов для замеров

// Медиана: средний элемент отсортированного массива (устойчива к выбросам)
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());   // сортировка по возрастанию
    return v[v.size() / 2];          // центральный элемент
}

// Разница двух моментов времени в миллисекундах
double elapsedMs(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Запуск CUDA-kernel с сеткой потоков и ожидание завершения на GPU
void launchKernel(float* d_a, float* d_b, float* d_c, size_t N) {
    const int threadsPerBlock = 256;  // потоков в одном блоке (типичное значение)
    // Число блоков: округление вверх, чтобы покрыть все N элементов
    const int blocksPerGrid = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);
    // <<<число блоков, потоков в блоке>>> — синтаксис вызова kernel с GPU
    vectorMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    checkCuda(cudaDeviceSynchronize(), "kernel sync");  // CPU ждёт, пока GPU закончит
}

// Главная функция замеров для одного значения N
Sample benchmarkSize(size_t N, int repeats, int warmup) {
    const size_t bytes = N * sizeof(float);  // объём одного вектора в байтах

    // h_ — префикс host: массивы в оперативной памяти (ОЗУ) компьютера
    std::vector<float> h_a, h_b, h_c(N);  // h_c сразу размера N
    fillVectors(N, h_a, h_b);  // начальное заполнение (для прогрева GPU)

    // d_ — префикс device: указатели на память на видеокарте (VRAM)
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    checkCuda(cudaMalloc(&d_a, bytes), "malloc a");  // выделить bytes на GPU
    checkCuda(cudaMalloc(&d_b, bytes), "malloc b");
    checkCuda(cudaMalloc(&d_c, bytes), "malloc c");

    // --- Прогрев GPU (не входит в результаты) ---
    // Убирает эффект «первого запуска»: разгон частот, кэши, инициализация CUDA
    for (int w = 0; w < warmup; ++w) {
        checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "warmup H2D");
        checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "warmup H2D");
        launchKernel(d_a, d_b, d_c, N);
        checkCuda(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "warmup D2H");
    }

    // Массивы для хранения времён каждого из repeats прогонов
    std::vector<double> t_cpu, t_cpu_e2e, t_gpu_kern, t_gpu_e2e;
    t_cpu.reserve(repeats);       // reserve — заранее выделить память без лишних копий
    t_cpu_e2e.reserve(repeats);
    t_gpu_kern.reserve(repeats);
    t_gpu_e2e.reserve(repeats);

    // --- Основной цикл: repeats раз повторяем все четыре замера ---
    for (int r = 0; r < repeats; ++r) {

        // [1] CPU e2e — секундомер включает генерацию данных и умножение
        {
            auto t0 = Clock::now();           // старт таймера
            fillVectors(N, h_a, h_b);         // создание/заполнение векторов в ОЗУ
            cpuMultiply(h_a, h_b, h_c);       // вычисление
            auto t1 = Clock::now();           // стоп таймера
            t_cpu_e2e.push_back(elapsedMs(t0, t1));  // сохранить время этого прогона
        }

        // [2] CPU calc — только умножение; h_a, h_b уже заполнены в блоке e2e выше
        {
            auto t0 = Clock::now();
            cpuMultiply(h_a, h_b, h_c);       // данные уже лежат в ОЗУ
            auto t1 = Clock::now();
            t_cpu.push_back(elapsedMs(t0, t1));
        }

        // [3] GPU kernel — в таймер входит ТОЛЬКО launchKernel
        // Копии H2D делаем до t0, D2H — после t1, чтобы измерить чистую работу GPU
        {
            checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "H2D");
            checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "H2D");
            auto t0 = Clock::now();           // старт: данные уже в VRAM
            launchKernel(d_a, d_b, d_c, N);   // работа видеокарты
            auto t1 = Clock::now();           // стоп: kernel завершён
            t_gpu_kern.push_back(elapsedMs(t0, t1));
            checkCuda(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "D2H");
        }

        // [4] GPU e2e: fill, malloc на GPU, копии, kernel, free
        {
            auto t0 = Clock::now();

            fillVectors(N, h_a, h_b);  // снова генерируем данные на CPU

            // Отдельные указатели ea, eb, ec — каждый repeat имитирует новую задачу
            float *ea = nullptr, *eb = nullptr, *ec = nullptr;
            checkCuda(cudaMalloc(&ea, bytes), "e2e malloc");
            checkCuda(cudaMalloc(&eb, bytes), "e2e malloc");
            checkCuda(cudaMalloc(&ec, bytes), "e2e malloc");
            checkCuda(cudaMemcpy(ea, h_a.data(), bytes, cudaMemcpyHostToDevice), "e2e H2D");
            checkCuda(cudaMemcpy(eb, h_b.data(), bytes, cudaMemcpyHostToDevice), "e2e H2D");
            launchKernel(ea, eb, ec, N);
            checkCuda(cudaMemcpy(h_c.data(), ec, bytes, cudaMemcpyDeviceToHost), "e2e D2H");
            cudaFree(ea);  // освободить VRAM после задачи
            cudaFree(eb);
            cudaFree(ec);

            auto t1 = Clock::now();
            t_gpu_e2e.push_back(elapsedMs(t0, t1));
        }

        checkCuda(cudaGetLastError(), "kernel");  // проверка асинхронных ошибок CUDA
    }

    // Освобождение GPU-памяти, выделенной на всё время benchmarkSize
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Собрать итог: медиана по repeats для каждого режима
    Sample s;
    s.N = N;
    s.cpu_ms = median(t_cpu);
    s.cpu_e2e_ms = median(t_cpu_e2e);
    s.gpu_kernel_ms = median(t_gpu_kern);
    s.gpu_e2e_ms = median(t_gpu_e2e);
    return s;  // вернуть одну строку результатов для этого N
}

// =============================================================================
// Раздел 4. Аппроксимация T(N) и break-even
// =============================================================================

double rSquared(const std::vector<double>& y, const std::vector<double>& y_hat) {
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        ss_tot += (y[i] - mean) * (y[i] - mean);
        ss_res += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
    }
    return (ss_tot > 1e-12) ? (1.0 - ss_res / ss_tot) : 0.0;
}

// T = a·N + b: наклон по всей сетке, b — накладные расходы у минимального N (b >= 0)
struct LinearFit {
    double a = 0;
    double b = 0;
    double r2 = 0;
};

LinearFit fitLinearAnchored(const std::vector<double>& N, const std::vector<double>& T) {
    LinearFit fit;
    const size_t n = N.size();
    if (n < 2) return fit;

    double sumN = 0, sumT = 0, sumNN = 0, sumNT = 0;
    for (size_t i = 0; i < n; ++i) {
        sumN += N[i];
        sumT += T[i];
        sumNN += N[i] * N[i];
        sumNT += N[i] * T[i];
    }
    const double denom = static_cast<double>(n) * sumNN - sumN * sumN;
    double a = 0, b = 0;
    if (std::fabs(denom) > 1e-12) {
        a = (static_cast<double>(n) * sumNT - sumN * sumT) / denom;
        b = (sumT - a * sumN) / static_cast<double>(n);
    }

    if (b >= 0) {
        fit.a = a;
        fit.b = b;
    } else {
        size_t i0 = 0;
        for (size_t i = 1; i < n; ++i)
            if (N[i] < N[i0]) i0 = i;

        const double N0 = N[i0];
        const double T0 = T[i0];
        double sumXX = 0, sumXY = 0;
        for (size_t i = 0; i < n; ++i) {
            const double x = N[i] - N0;
            const double y = T[i] - T0;
            sumXX += x * x;
            sumXY += x * y;
        }
        a = (sumXX > 1e-12) ? sumXY / sumXX : 0;
        b = std::max(0.0, T0 - a * N0);

        double sumNN = 0, sumNr = 0;
        for (size_t i = 0; i < n; ++i) {
            const double r = T[i] - b;
            sumNN += N[i] * N[i];
            sumNr += N[i] * r;
        }
        if (sumNN > 1e-12) a = sumNr / sumNN;
        fit.a = a;
        fit.b = b;
    }

    std::vector<double> y_hat(n);
    for (size_t i = 0; i < n; ++i) y_hat[i] = fit.a * N[i] + fit.b;
    fit.r2 = rSquared(T, y_hat);
    return fit;
}

void reportLinearFit(const std::vector<Sample>& data, double (Sample::*field), const char* label) {
    const size_t n = data.size();
    std::vector<double> N(n), T(n);
    for (size_t i = 0; i < n; ++i) {
        N[i] = static_cast<double>(data[i].N);
        T[i] = data[i].*field;
    }
    const LinearFit fit = fitLinearAnchored(N, T);
    std::cout << "  [" << label << "] R²=" << std::fixed << std::setprecision(4) << fit.r2
              << "  T ≈ " << formatSci(fit.a) << "·N + " << formatSci(fit.b) << " мс\n";
}

// Результат поиска точки break-even
struct BreakEven {
    bool found = false;       // найдено ли пересечение на сетке N
    double N_exact = 0.0;   // N с дробной частью (интерполяция)
    size_t N_round = 0;     // округлённое N
};

// Найти N, где кривая GPU догоняет и обгоняет кривую CPU
// cpu_field / gpu_field — какие столбцы сравниваем (например cpu_ms и gpu_kernel_ms)
BreakEven findBreakEven(const std::vector<Sample>& data,
                        double (Sample::*cpu_field),
                        double (Sample::*gpu_field)) {
    BreakEven be;
    for (size_t i = 1; i < data.size(); ++i) {
        // diff = T_cpu - T_gpu: если < 0, CPU быстрее; если > 0, GPU быстрее
        const double d0 = data[i - 1].*cpu_field - data[i - 1].*gpu_field;
        const double d1 = data[i].*cpu_field - data[i].*gpu_field;
        // Ищем смену знака: было CPU быстрее, стало GPU быстрее
        if (d0 < 0.0 && d1 > 0.0) {
            const double N0 = static_cast<double>(data[i - 1].N);
            const double N1 = static_cast<double>(data[i].N);
            const double alpha = d0 / (d0 - d1);  // доля отрезка [N0,N1] до пересечения
            be.N_exact = N0 + alpha * (N1 - N0);  // линейная интерполяция
            be.N_round = static_cast<size_t>(be.N_exact + 0.5); // округление до ближайшего целого
            be.found = true;
            return be;
        }
    }
    return be;  // found=false, если на всей сетке одна сторона всегда быстрее
}

// Вывести break-even в консоль
void printBreakEven(const char* title, const BreakEven& be) {
    std::cout << "\n  " << title << '\n';
    if (!be.found) {
        std::cout << "    Пересечение не найдено на сетке N.\n";
        return;
    }
    std::cout << "    N ≈ " << std::scientific << std::setprecision(4) << be.N_exact
              << "  (" << std::fixed << std::setprecision(0) << be.N_round << " элементов)\n";
}

// Записать все результаты в CSV для построения графика в Python
void writeCsv(const std::string& path, const std::vector<Sample>& data) {
    std::ofstream out(path);  // открыть файл на запись
    out << "N,cpu_ms,cpu_e2e_ms,gpu_kernel_ms,gpu_e2e_ms\n";  // заголовок столбцов
    out << std::fixed << std::setprecision(6);
    for (const auto& s : data) {
        out << s.N << ',' << s.cpu_ms << ',' << s.cpu_e2e_ms << ',' << s.gpu_kernel_ms << ','
            << s.gpu_e2e_ms << '\n';  // одна строка — один размер N
    }
    std::cout << "\nДанные: " << path << '\n';
}

// Список значений N, при которых проводим замеры (сетка для графика T(N))
std::vector<size_t> defaultSizes() {
    return {
        100'000, 200'000, 500'000,
        1'000'000, 5'000'000, 10'000'000, 20'000'000, 40'000'000,
        60'000'000, 80'000'000, 100'000'000, 120'000'000, 140'000'000,
        160'000'000, 180'000'000, 200'000'000,
    };
}

// =============================================================================
// Раздел 5. Точка входа программы
// =============================================================================

int main(int argc, char** argv) {
    setupConsoleUtf8();

    int repeats = 5;   // сколько раз повторять замеры для каждого N
    int warmup = 2;    // сколько прогревочных циклов GPU до замеров
    std::string csv_path = "benchmark_results.csv";

    if (argc > 1) repeats = std::max(1, std::atoi(argv[1]));  // 1-й аргумент: repeats
    if (argc > 2) csv_path = argv[2];                          // 2-й аргумент: имя CSV

    cudaDeviceProp prop{};  // структура со свойствами видеокарты
    checkCuda(cudaGetDeviceProperties(&prop, 0), "device props");  // GPU №0
    size_t free_bytes = 0, total_bytes = 0;
    checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "mem info");

    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "VRAM: " << (total_bytes / (1024 * 1024)) << " MiB (свободно "
              << (free_bytes / (1024 * 1024)) << " MiB)\n";
    std::cout << "Повторов: " << repeats << ", прогрев: " << warmup << "\n\n";

    const auto sizes = defaultSizes();  // все N для прогона
    std::vector<Sample> results;        // накопитель результатов

    // Заголовок таблицы в консоли
    std::cout << std::setw(12) << "N" << std::setw(10) << "CPU" << std::setw(10) << "CPUe2e"
              << std::setw(10) << "GPUkern" << std::setw(10) << "GPUe2e" << '\n';
    std::cout << std::string(52, '-') << '\n';

    // Цикл по всем размерам вектора
    for (size_t N : sizes) {
        if (!hasEnoughGpuMemory(N)) {  // пропустить, если не влезает в VRAM
            std::cout << "Пропуск N=" << N << " (мало VRAM)\n";
            continue;
        }
        std::cout << "N=" << N << " ... " << std::flush;  // flush — сразу показать в консоли
        results.push_back(benchmarkSize(N, repeats, warmup));  // замер для этого N
        const auto& s = results.back();  // ссылка на последний добавленный результат
        std::cout << "OK\n" << std::setw(12) << s.N << std::fixed << std::setprecision(2)
                  << std::setw(10) << s.cpu_ms << std::setw(10) << s.cpu_e2e_ms
                  << std::setw(10) << s.gpu_kernel_ms << std::setw(10) << s.gpu_e2e_ms << '\n';
    }

    if (results.empty()) {  // если все N были пропущены
        std::cerr << "Нет замеров.\n";
        return 1;
    }

    writeCsv(csv_path, results);  // сохранить для plot_results.py

    std::cout << "\n--- T(N) ~ a·N + b ---\n";
    reportLinearFit(results, &Sample::cpu_ms, "CPU calc");
    reportLinearFit(results, &Sample::cpu_e2e_ms, "CPU e2e");
    reportLinearFit(results, &Sample::gpu_kernel_ms, "GPU kernel");
    reportLinearFit(results, &Sample::gpu_e2e_ms, "GPU e2e");

    std::cout << "\n========== BREAK-EVEN ==========\n";
    printBreakEven("1) CPU calc  vs  GPU kernel",
                   findBreakEven(results, &Sample::cpu_ms, &Sample::gpu_kernel_ms));
    printBreakEven("2) CPU e2e  vs  GPU e2e",
                   findBreakEven(results, &Sample::cpu_e2e_ms, &Sample::gpu_e2e_ms));

    std::cout << "\nГрафик: python plot_results.py " << csv_path << '\n';
    return 0;
}
