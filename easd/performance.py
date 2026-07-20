import gc
import io
import os
import math
import pstats
import cProfile
import threading
import time
from dataclasses import dataclass
from typing import Dict

import psutil


# ──────────────────────────────────────────────────────────────────────────────
# Result structure
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ResourceSummary:
    # CPU sampled by a background thread, preserving the previous behavior.
    cpu_mean_percent: float
    cpu_peak_percent: float
    # Absolute RAM, total process RSS, preserving the previous behavior.
    ram_mean_mb: float
    ram_peak_mb: float
    # Incremental RAM: peak RSS during the loop minus RSS before the loop.
    # Isolates memory allocated by the algorithm without Python's static footprint.
    ram_incremental_peak_mb: float
    ram_baseline_mb: float
    samples: int


# ──────────────────────────────────────────────────────────────────────────────
# Resource monitor (CPU + RAM), with no instrumentation overhead
# ──────────────────────────────────────────────────────────────────────────────


class ProcessResourceMonitor:
    """
    Monitors CPU and RAM for a process in a separate thread.

    RAM is reported in two ways:
      - Absolute (ram_mean_mb, ram_peak_mb): total process RSS, including
        Python's static footprint (~200 MB). Kept for compatibility with
        previous results.
      - Incremental (ram_incremental_peak_mb): peak RSS during the loop minus
        the baseline measured in start(). This isolates memory allocated by
        the algorithm without instrumentation overhead and without interpreter
        footprint noise.

    Does not use tracemalloc or cProfile, keeping overhead close to zero.

    Parameters
    ----------
    pid : int | None
        PID of the process to monitor. None uses the current process.
    interval : float
        Sampling interval, in seconds.
    include_children : bool
        If True, include child processes such as numpy/BLAS workers.
    """

    def __init__(
        self,
        pid: int | None = None,
        interval: float = 0.5,
        include_children: bool = True,
    ):
        self.pid = pid if pid is not None else os.getpid()
        self.interval = interval
        self.include_children = include_children

        self._process = psutil.Process(self.pid)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Welford, CPU.
        self._cpu_n: int = 0
        self._cpu_mean: float = 0.0
        self._cpu_peak: float = 0.0

        # Welford, absolute RAM.
        self._ram_n: int = 0
        self._ram_mean: float = 0.0
        self._ram_peak: float = 0.0

        # Incremental RAM, only two floats and no lists.
        self._ram_baseline_mb: float = 0.0
        self._ram_incremental_peak_mb: float = 0.0

    # ── leituras de processo ─────────────────────────────────────────────────

    def _get_rss_mb(self) -> float:
        try:
            rss = self._process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
        if self.include_children:
            try:
                for child in self._process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return rss / (1024**2)

    def _get_cpu_percent(self) -> float:
        try:
            cpu = self._process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
        if self.include_children:
            try:
                for child in self._process.children(recursive=True):
                    try:
                        cpu += child.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return cpu

    # ── Welford ─────────────────────────────────────────────────────────────

    def _update_cpu(self, v: float) -> None:
        self._cpu_n += 1
        d = v - self._cpu_mean
        self._cpu_mean += d / self._cpu_n
        if v > self._cpu_peak:
            self._cpu_peak = v

    def _update_ram(self, v: float) -> None:
        self._ram_n += 1
        d = v - self._ram_mean
        self._ram_mean += d / self._ram_n
        if v > self._ram_peak:
            self._ram_peak = v
        incremental = v - self._ram_baseline_mb
        if incremental > self._ram_incremental_peak_mb:
            self._ram_incremental_peak_mb = incremental

    # ── loop de amostragem ───────────────────────────────────────────────────

    def _sample_loop(self) -> None:
        # Warm-up: the first cpu_percent read is always 0.0.
        try:
            self._process.cpu_percent(interval=None)
            if self.include_children:
                for child in self._process.children(recursive=True):
                    try:
                        child.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

        while not self._stop_event.is_set():
            self._update_cpu(self._get_cpu_percent())
            self._update_ram(self._get_rss_mb())
            time.sleep(self.interval)

    # ── interface pública ────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Starts monitoring.

        Call immediately before the evolutionary loop, after initialization
        (Dataset, initial population, etc.) has completed. The RAM baseline is
        measured here after gc.collect() so pending objects from previous
        generations do not contaminate the reference.
        """
        gc.collect()
        self._ram_baseline_mb = self._get_rss_mb()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSummary:
        """Stops monitoring and returns the metric summary."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        num_cpus = psutil.cpu_count(logical=True) or 1

        if self._cpu_n == 0:
            return ResourceSummary(
                cpu_mean_percent=0.0,
                cpu_peak_percent=0.0,
                ram_mean_mb=0.0,
                ram_peak_mb=0.0,
                ram_incremental_peak_mb=0.0,
                ram_baseline_mb=self._ram_baseline_mb,
                samples=0,
            )

        return ResourceSummary(
            cpu_mean_percent=self._cpu_mean / num_cpus,
            cpu_peak_percent=self._cpu_peak / num_cpus,
            ram_mean_mb=self._ram_mean,
            ram_peak_mb=self._ram_peak,
            ram_incremental_peak_mb=self._ram_incremental_peak_mb,
            ram_baseline_mb=self._ram_baseline_mb,
            samples=self._cpu_n,
        )

    def get_current_snapshot(self) -> Dict[str, float]:
        """Point-in-time read without stopping the monitor. Useful for per-generation logging."""
        num_cpus = psutil.cpu_count(logical=True) or 1
        return {
            "cpu_mean_so_far": self._cpu_mean / num_cpus,
            "cpu_peak_so_far": self._cpu_peak / num_cpus,
            "ram_mean_mb_so_far": self._ram_mean,
            "ram_peak_mb_so_far": self._ram_peak,
            "ram_incremental_peak_mb": self._ram_incremental_peak_mb,
            "ram_baseline_mb": self._ram_baseline_mb,
            "samples_so_far": self._cpu_n,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Per-function CPU profiler (cProfile). Use in separate executions.
# ──────────────────────────────────────────────────────────────────────────────


class CPUProfiler:
    """
    cProfile wrapper for identifying per-function bottlenecks.

    Adds ~10-20% runtime overhead. Use in separate executions with a fixed
    seed for analysis, not in the 30 benchmark executions.

    Alternative without modifying the code from the command line:
        python -m cProfile -o result.prof main.py <args> --seed 42
        python -c "import pstats; pstats.Stats('result.prof').sort_stats('cumulative').print_stats(20)"

    Programmatic usage inside the loop in core.py:
        profiler = CPUProfiler()
        profiler.start()
        while self._check_stop(gen_count):
            ...
        profiler.stop_and_print(n_top=15, sort_by="cumulative")
        profiler.save("easd.prof")  # snakeviz easd.prof for visualization
    """

    def __init__(self):
        self._pr = cProfile.Profile()

    def start(self) -> None:
        self._pr.enable()

    def stop(self) -> pstats.Stats:
        self._pr.disable()
        return pstats.Stats(self._pr)

    def stop_and_print(self, n_top: int = 20, sort_by: str = "cumulative") -> None:
        """
        Stops and prints the n_top most expensive functions.

        Useful sort_by values:
          "cumulative" -- total time including subcalls. Good for seeing which
                          part of the code dominates runtime.
          "tottime"    -- time inside the function, excluding subcalls. Good
                          for identifying the specific slowest function.
          "ncalls"     -- number of calls. Good for detecting unnecessary
                          repeated calls.
        """
        self._pr.disable()
        stream = io.StringIO()
        ps = pstats.Stats(self._pr, stream=stream)
        ps.sort_stats(sort_by)
        ps.print_stats(n_top)
        print(stream.getvalue())

    def save(self, filepath: str) -> None:
        """
        Saves the profile to a .prof file for visualization with snakeviz:
            pip install snakeviz && snakeviz <filepath>
        """
        self._pr.dump_stats(filepath)
        print(f"Profile saved to: {filepath}")
