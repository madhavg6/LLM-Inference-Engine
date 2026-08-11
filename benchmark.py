"""
Benchmark harness — imported by every phase.

Usage
-----
from benchmark import Benchmark

bm = Benchmark("phase3-naive-fp32")
bm.record(prompt_tokens=12, generated_tokens=50, elapsed_sec=4.2)
bm.record(...)
bm.report()         # prints table to stdout
bm.save()           # appends to benchmark_log.jsonl
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


LOG_PATH = Path(__file__).parent / "benchmark_log.jsonl"


@dataclass
class Run:
    phase: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_sec: float
    tokens_per_sec: float = field(init=False)
    note: str = ""

    def __post_init__(self):
        self.tokens_per_sec = self.generated_tokens / max(self.elapsed_sec, 1e-9)


class Benchmark:
    def __init__(self, phase: str):
        self.phase = phase
        self.runs: List[Run] = []

    def record(
        self,
        prompt_tokens: int,
        generated_tokens: int,
        elapsed_sec: float,
        note: str = "",
    ) -> Run:
        r = Run(self.phase, prompt_tokens, generated_tokens, elapsed_sec, note)
        self.runs.append(r)
        return r

    def report(self):
        if not self.runs:
            print(f"[{self.phase}] No runs recorded.")
            return
        print(f"\n{'='*60}")
        print(f"  Phase: {self.phase}")
        print(f"{'='*60}")
        print(f"  {'prompt_tok':>10} {'gen_tok':>8} {'elapsed':>10} {'tok/s':>10}  note")
        print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
        for r in self.runs:
            print(
                f"  {r.prompt_tokens:>10} {r.generated_tokens:>8} "
                f"{r.elapsed_sec:>9.2f}s {r.tokens_per_sec:>9.2f}  {r.note}"
            )
        avg_tps = sum(r.tokens_per_sec for r in self.runs) / len(self.runs)
        print(f"\n  avg tok/s: {avg_tps:.2f}")
        print(f"{'='*60}\n")

    def save(self):
        with LOG_PATH.open("a", encoding="utf-8") as f:
            for r in self.runs:
                f.write(json.dumps(asdict(r)) + "\n")

    def load_all() -> List[dict]:
        """Read every run ever saved (all phases)."""
        if not LOG_PATH.exists():
            return []
        rows = []
        with LOG_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def compare_phases():
        """Print a summary table across all recorded phases."""
        rows = Benchmark.load_all()
        if not rows:
            print("No benchmark data yet.")
            return
        by_phase = {}
        for r in rows:
            by_phase.setdefault(r["phase"], []).append(r["tokens_per_sec"])
        print(f"\n{'Phase':<30} {'runs':>5} {'avg tok/s':>12} {'best tok/s':>12}")
        print(f"{'-'*30} {'-'*5} {'-'*12} {'-'*12}")
        for phase, tpss in sorted(by_phase.items()):
            print(
                f"{phase:<30} {len(tpss):>5} "
                f"{sum(tpss)/len(tpss):>12.2f} {max(tpss):>12.2f}"
            )


class Timer:
    """Context manager for simple wall-clock timing."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000
