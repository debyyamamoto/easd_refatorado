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
# Estrutura de resultado
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ResourceSummary:
    # CPU (amostrado por thread em background — mesmo comportamento anterior)
    cpu_mean_percent: float
    cpu_peak_percent: float
    # RAM absoluta (RSS total do processo, mesmo comportamento anterior)
    ram_mean_mb: float
    ram_peak_mb: float
    # RAM incremental: pico de RSS durante o loop - RSS antes do loop.
    # Isola a memória alocada pelo algoritmo sem o footprint estático do Python.
    ram_incremental_peak_mb: float
    ram_baseline_mb: float
    samples: int


# ──────────────────────────────────────────────────────────────────────────────
# Monitor de recursos (CPU + RAM) — sem overhead de instrumentação
# ──────────────────────────────────────────────────────────────────────────────


class ProcessResourceMonitor:
    """
    Monitora CPU e RAM de um processo em uma thread separada.

    RAM é reportada de duas formas:
      - Absoluta (ram_mean_mb, ram_peak_mb): RSS total do processo,
        inclui o footprint estático do Python (~200 MB). Mantida para
        compatibilidade com os resultados anteriores.
      - Incremental (ram_incremental_peak_mb): pico de RSS durante o
        loop menos o baseline medido em start(). Isola a memória
        alocada pelo algoritmo sem overhead de instrumentação e sem
        o ruído do footprint estático do interpretador.

    Não usa tracemalloc nem cProfile — overhead próximo de zero.

    Parameters
    ----------
    pid : int | None
        PID do processo a monitorar. None usa o processo atual.
    interval : float
        Intervalo entre amostras em segundos.
    include_children : bool
        Se True, inclui processos filhos (workers de numpy/BLAS).
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

        # Welford — CPU
        self._cpu_n: int = 0
        self._cpu_mean: float = 0.0
        self._cpu_peak: float = 0.0

        # Welford — RAM absoluta
        self._ram_n: int = 0
        self._ram_mean: float = 0.0
        self._ram_peak: float = 0.0

        # RAM incremental — apenas dois floats, sem listas
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
        # warm-up: primeira leitura de cpu_percent é sempre 0.0
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
        Inicia o monitoramento.

        Chame imediatamente antes do loop evolutivo, após toda
        inicialização (Dataset, população inicial, etc.) estar concluída.
        O baseline de RAM é medido aqui, após um gc.collect() para
        garantir que objetos pendentes de gerações anteriores não
        contaminem a referência.
        """
        gc.collect()
        self._ram_baseline_mb = self._get_rss_mb()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSummary:
        """Para o monitoramento e retorna o resumo das métricas."""
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
        """Leitura pontual sem interromper o monitor. Útil para logging por geração."""
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
# Profiler de CPU por função (cProfile) — use em execuções separadas
# ──────────────────────────────────────────────────────────────────────────────


class CPUProfiler:
    """
    Wrapper de cProfile para identificar gargalos por função.

    Adiciona ~10-20% de overhead no runtime — use em execuções separadas
    com seed fixa para análise, não nas 30 execuções do benchmark.

    Alternativa sem modificar o código (linha de comando):
        python -m cProfile -o result.prof main.py <args> --seed 42
        python -c "import pstats; pstats.Stats('result.prof').sort_stats('cumulative').print_stats(20)"

    Uso programático dentro do loop em core.py:
        profiler = CPUProfiler()
        profiler.start()
        while self._check_stop(gen_count):
            ...
        profiler.stop_and_print(n_top=15, sort_by="cumulative")
        profiler.save("easd.prof")  # snakeviz easd.prof para visualização
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
        Para e imprime as n_top funções mais custosas.

        sort_by úteis:
          "cumulative" — tempo total incluindo subchamadas.
                         Bom para ver qual parte do código domina o runtime.
          "tottime"    — tempo dentro da função, excluindo subchamadas.
                         Bom para identificar a função específica mais lenta.
          "ncalls"     — número de chamadas.
                         Bom para detectar chamadas repetidas desnecessariamente.
        """
        self._pr.disable()
        stream = io.StringIO()
        ps = pstats.Stats(self._pr, stream=stream)
        ps.sort_stats(sort_by)
        ps.print_stats(n_top)
        print(stream.getvalue())

    def save(self, filepath: str) -> None:
        """
        Salva o perfil em arquivo .prof para visualização com snakeviz:
            pip install snakeviz && snakeviz <filepath>
        """
        self._pr.dump_stats(filepath)
        print(f"Perfil salvo em: {filepath}")
