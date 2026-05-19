#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""График T(N) из benchmark_results.csv. Запуск: python plot_results.py [csv]"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    N = np.asarray(data["N"], dtype=np.float64)
    return {
        "N": N,
        "cpu": np.asarray(data["cpu_ms"], dtype=np.float64),
        "cpu_e2e": np.asarray(data["cpu_e2e_ms"], dtype=np.float64),
        "gpu_kern": np.asarray(data["gpu_kernel_ms"], dtype=np.float64),
        "gpu_e2e": np.asarray(data["gpu_e2e_ms"], dtype=np.float64),
    }


def break_even(N, t_cpu, t_gpu):
    diff = t_cpu - t_gpu
    for i in range(1, len(N)):
        if diff[i - 1] < 0 and diff[i] > 0:
            a = diff[i - 1] / (diff[i - 1] - diff[i])
            Nx = N[i - 1] + a * (N[i] - N[i - 1])
            ty = t_cpu[i - 1] + a * (t_cpu[i] - t_cpu[i - 1])
            return float(Nx), float(ty)
    return None, None


def mark_be(ax, N, t_cpu, t_gpu, label, color):
    nx, ty = break_even(N, t_cpu, t_gpu)
    if nx is None:
        return
    xm = nx / 1e6
    ax.axvline(xm, color=color, ls="--", lw=1.2, alpha=0.85)
    ax.plot(xm, ty, "D", color=color, ms=8, zorder=5)
    ax.annotate(
        f"{label}\nN≈{xm:.2f} млн",
        (xm, ty),
        textcoords="offset points",
        xytext=(8, 10),
        fontsize=8,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.7),
    )


def main():
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv")
    if not csv_path.exists():
        print(f"Нет файла: {csv_path}")
        print("Сначала запустите benchmark.exe")
        sys.exit(1)

    d = load_csv(csv_path)
    N, N_mln = d["N"], d["N"] / 1e6

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(N_mln, d["cpu"], "o-", label="CPU calc", lw=2, ms=4)
    ax.plot(N_mln, d["cpu_e2e"], "v-", label="CPU e2e", lw=2, ms=4)
    ax.plot(N_mln, d["gpu_kern"], "^-", label="GPU kernel", lw=2, ms=4)
    ax.plot(N_mln, d["gpu_e2e"], "s-", label="GPU e2e", lw=2, ms=4)

    ax.set_xlabel("N, млн элементов")
    ax.set_ylabel("Время, мс")
    ax.set_title("CPU vs GPU: четыре режима")
    ax.grid(True, alpha=0.3)

    mark_be(ax, N, d["cpu"], d["gpu_kern"], "BE: calc/kernel", "darkgreen")
    mark_be(ax, N, d["cpu_e2e"], d["gpu_e2e"], "BE: e2e", "crimson")

    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    out = csv_path.with_suffix(".png")
    plt.savefig(out, dpi=150)
    print(f"График: {out}")

    be1, _ = break_even(N, d["cpu"], d["gpu_kern"])
    be2, _ = break_even(N, d["cpu_e2e"], d["gpu_e2e"])
    print("\nBreak-even:")
    print(f"  CPU calc vs GPU kernel: {be1 / 1e6:.3f} млн" if be1 else "  CPU calc vs GPU kernel: нет")
    print(f"  CPU e2e vs GPU e2e:     {be2 / 1e6:.3f} млн" if be2 else "  CPU e2e vs GPU e2e: нет")


if __name__ == "__main__":
    main()
