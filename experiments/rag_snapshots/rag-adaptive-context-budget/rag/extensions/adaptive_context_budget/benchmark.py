"""Measured synthetic controller cost only; no retrieval or answer-quality claims."""
from __future__ import annotations

from dataclasses import asdict
import platform
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

from .core import AdaptiveBudget, EvidenceUnit, FixedBudget, TokenCounter, apply_budget
from .experiment import write_csv, write_json


def synthetic_units(count):
    return [EvidenceUnit(
        f"synthetic-doc-{i % 5}",
        f"Synthetic evidence unit {i} describes a controlled systems measurement. "
        f"Its second sentence records item {i} without claiming any course answer quality.",
        unit_id=str(i),
    ) for i in range(count)]


def benchmark(output, counts=(1, 5, 20, 100, 500, 1000), repeats=30, warmups=5):
    if repeats < 2 or warmups < 0 or any(n < 1 for n in counts):
        raise ValueError("use repeats>=2, warmups>=0 and positive candidate counts")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counter = TokenCounter()
    rows, samples = [], []
    # Hold packing work constant to isolate decision overhead from budget size.
    configs = [("fixed", FixedBudget(1024)), ("adaptive", AdaptiveBudget(1024, 1024))]
    for count in counts:
        units = synthetic_units(count)
        for mode, policy in configs:
            for packing in ("skip", "sentence_prefix"):
                name = mode + "_" + packing
                scores = [0.9, 0.8, 0.79, 0.78, 0.77]
                for _ in range(warmups):
                    apply_budget(units, scores, policy, counter, packing)
                timings = []
                for repeat in range(repeats):
                    start = time.perf_counter_ns()
                    result = apply_budget(units, scores, policy, counter, packing)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    timings.append(elapsed)
                    samples.append({"configuration": name, "candidate_units": count,
                                    "repeat": repeat, "latency_ms": elapsed})
                tracemalloc.start()
                apply_budget(units, scores, policy, counter, packing)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                ordered = sorted(timings)
                rows.append({
                    "configuration": name, "candidate_units": count, "repeats": repeats,
                    "mean_ms": statistics.mean(timings), "median_ms": statistics.median(timings),
                    "sd_ms": statistics.stdev(timings), "p95_ms": ordered[min(repeats - 1, int(0.95 * repeats))],
                    "python_peak_allocation_bytes": peak, **result["metrics"],
                })
    write_csv(output / "overhead.csv", rows)
    write_csv(output / "overhead_samples.csv", samples)
    write_json(output / "benchmark_manifest.json", {
        "dataset": "synthetic complete-sentence evidence; NOT course documents or answer evaluation",
        "platform": platform.platform(), "python": sys.version, "counts": list(counts),
        "repeats": repeats, "warmups": warmups, "counter": counter.name,
        "policies": {name: asdict(policy) for name, policy in configs},
        "scores": [0.9, 0.8, 0.79, 0.78, 0.77],
        "control": "Both policies use 1024 tokens; adaptive still calculates its confidence statistic",
        "timing": "perf_counter_ns; full decision, counting, packing and metrics; input construction excluded",
        "memory": "separate tracemalloc measurement; additional Python allocations, not total process RSS",
    })
    return rows
