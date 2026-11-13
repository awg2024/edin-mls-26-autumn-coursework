"""Measured systems summaries and nine report figures; no invented course results."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from .io import write_csv, write_json_atomic


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(value):
    return None if value in (None, "") else float(value)


METRICS = (
    "initial_build_ms", "insert_wall_mean_ms", "update_wall_mean_ms", "delete_wall_mean_ms",
    "batch_wall_mean_ms", "amortised_update_ms", "updates_per_second", "total_mutation_ms",
    "total_staging_ms", "total_rebuild_ms", "total_auto_compaction_ms", "automatic_compactions",
    "query_mean_ms", "query_median_ms", "query_p95_ms", "query_before_mean_ms",
    "query_after_manual_compaction_mean_ms", "measured_workload_ms", "query_time_sum_ms",
    "manual_compaction_ms", "embedding_bytes_before_manual_compaction", "embedding_bytes_after_manual_compaction",
    "delta_rows_before_manual_compaction", "tombstones_before_manual_compaction",
    "maintenance_traced_peak_bytes", "exact_topk_agreement",
)


def summarise(output):
    output = Path(output)
    raw = _read(output/"all_summary_per_repeat.csv")
    groups = defaultdict(list)
    for row in raw:
        groups[(row["case"], row["strategy"])].append(row)
    summaries = []
    for (_, _), rows in groups.items():
        first = rows[0]
        summary = {key: first[key] for key in (
            "case", "family", "strategy", "dataset", "dimension", "corpus_size", "initial_active_records",
            "operations", "batch_size", "profile", "queries_per_update", "compaction_threshold", "update_mode")}
        summary["repetitions"] = len(rows)
        for metric in METRICS:
            values = [_number(row[metric]) for row in rows]
            values = [v for v in values if v is not None]
            summary[metric + "_mean"] = statistics.mean(values) if values else None
            summary[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
            summary[metric + "_n"] = len(values)
        summaries.append(summary)
    references = {s["case"]: s for s in summaries if s["strategy"] == "full_rebuild"}
    for row in summaries:
        reference = references[row["case"]]
        row["update_speedup_vs_rebuild"] = reference["total_mutation_ms_mean"] / row["total_mutation_ms_mean"]
        row["workload_speedup_vs_rebuild"] = reference["measured_workload_ms_mean"] / row["measured_workload_ms_mean"]
        row["query_latency_change_pct"] = (
            100 * (row["query_mean_ms_mean"] / reference["query_mean_ms_mean"] - 1)
            if row["query_mean_ms_mean"] is not None and reference["query_mean_ms_mean"] else None)
    write_csv(output/"summary.csv", summaries)
    write_json_atomic(output/"summary.json", summaries)
    checks = [row for path in output.glob("*/retrieval_checks.csv") for row in _read(path)]
    correctness = {
        "query_checkpoint_checks": len(checks),
        "topk_mismatches": sum(row["exact_topk_agreement"] != "True" for row in checks),
        "score_mismatches": sum(row["scores_within_tolerance"] != "True" for row in checks),
        "logical_state_mismatches": sum(row["logical_state_equal"] != "True" for row in checks),
        "inactive_or_stale_text_results": sum(row["all_active_and_current_text"] != "True" for row in checks),
        "max_score_error": max((_number(row["max_score_error"]) for row in checks), default=None),
        "qrels_evaluated_checks": sum(bool(row["active_qrels_recall_at10"]) for row in checks),
        "agreement_fraction": sum(row["exact_topk_agreement"] == "True" for row in checks) / len(checks) if checks else None,
    }
    write_json_atomic(output/"correctness_summary.json", correctness)
    principal = [row for row in summaries if row["family"] == "principal"]
    if principal:
        order = {row["strategy"]: row for row in principal}
        labels = ["full_rebuild", "incremental", "incremental_compact"]
        lines = ["# Measured principal comparison", "", "| Metric | Full rebuild | Incremental | Incremental + compaction |",
                 "|---|---:|---:|---:|"]
        for label, metric, scale in (
            ("Mean insert wall latency (ms)", "insert_wall_mean_ms_mean", 1),
            ("Mean update wall latency (ms)", "update_wall_mean_ms_mean", 1),
            ("Mean delete wall latency (ms)", "delete_wall_mean_ms_mean", 1),
            ("Mean scheduled query latency (ms)", "query_mean_ms_mean", 1),
            ("Mean per-repeat p95 query latency (ms)", "query_p95_ms_mean", 1),
            ("Embedding storage before manual compaction (MiB)", "embedding_bytes_before_manual_compaction_mean", 1/2**20),
            ("Total mutation + automatic maintenance (ms)", "total_mutation_ms_mean", 1),
            ("Measured index workload time (ms)", "measured_workload_ms_mean", 1),
            ("Exact top-k agreement", "exact_topk_agreement_mean", 1),
        ):
            values = [order[s][metric] for s in labels]
            lines.append("| " + label + " | " + " | ".join("Not measured" if v is None else f"{v*scale:.4f}" for v in values) + " |")
        lines.extend(["", "Values are means across repeats. Embedding bytes exclude text, metadata, old snapshots and process RSS.",
                      "The workload sum excludes validation, logging, initial construction and the final diagnostic manual compaction.", ""])
        (output/"principal_table.md").write_text("\n".join(lines))
    return summaries


def plot(output, destination=None):
    import os
    output = Path(output)
    destination = Path(destination) if destination else output/"figures"
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(destination/".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(destination/".cache"))
    (Path(os.environ["XDG_CACHE_HOME"])/"fontconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = summarise(output)
    manifest = json.loads((output/"experiment_manifest.json").read_text())
    title = manifest["dataset"] + "; " + str(manifest["cases"][0]["initial_fraction"] * 100).rstrip("0").rstrip(".") + "% initially active"
    colors = {"full_rebuild": "tab:blue", "incremental": "tab:orange", "incremental_compact": "tab:green"}
    made, skipped = [], {}

    def save(fig, ax, name):
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(destination/f"{name}.{extension}", dpi=250, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    def lines(family, xkey, ykey, name, xlabel, ylabel, log_x=False):
        selected = [row for row in summaries if row["family"] == family]
        if not selected:
            skipped[name] = "No measured cases for this ablation"
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        for strategy in colors:
            rows = sorted((r for r in selected if r["strategy"] == strategy), key=lambda r: float(r[xkey]))
            if rows:
                ax.errorbar([float(r[xkey]) for r in rows], [r[ykey + "_mean"] for r in rows],
                            yerr=[r[ykey + "_sd"] or 0 for r in rows], marker="o", capsize=3,
                            color=colors[strategy], label=strategy)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if log_x:
            ax.set_xscale("log")
        save(fig, ax, name)

    lines("batch_size", "batch_size", "amortised_update_ms", "update_latency_vs_batch_size",
          "Atomic batch size", "Amortised mutation + maintenance wall time (ms/op)", True)
    principal = [row for row in summaries if row["family"] == "principal"]
    if principal:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([r["strategy"] for r in principal], [r["total_mutation_ms_mean"] for r in principal],
               yerr=[r["total_mutation_ms_sd"] or 0 for r in principal],
               color=[colors[r["strategy"]] for r in principal], capsize=4)
        ax.set_ylabel("Total mutation + automatic maintenance wall time (ms)")
        for r in principal:
            ax.plot([], [], color=colors[r["strategy"]], label=r["strategy"])
        save(fig, ax, "maintenance_time_comparison")
    else:
        skipped["maintenance_time_comparison"] = "No principal case"

    histories = [row for path in output.glob("*/state_history.csv") for row in _read(path)
                 if row["family"] == "accumulation" and row["phase"] != "after_manual_compaction"]
    for key, name, ylabel, scale in (
        ("probe_query_mean_ms", "query_latency_vs_accumulated_updates", "Probe query wall latency (ms)", 1),
        ("embedding_bytes", "memory_vs_accumulated_updates", "Embedding matrix storage (MiB; excludes metadata/RSS)", 1/2**20),
    ):
        if not histories:
            skipped[name] = "No accumulation measurements"
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for strategy in colors:
            points = defaultdict(list)
            for row in histories:
                if row["strategy"] == strategy:
                    points[int(row["processed_operations"])].append(float(row[key]) * scale)
            xs = sorted(points)
            means = [statistics.mean(points[x]) for x in xs]
            sd = [statistics.stdev(points[x]) if len(points[x]) > 1 else 0 for x in xs]
            ax.plot(xs, means, color=colors[strategy], label=strategy)
            ax.fill_between(xs, [max(0, m - s) for m, s in zip(means, sd)], [m + s for m, s in zip(means, sd)],
                            color=colors[strategy], alpha=0.12)
        ax.set(xlabel="Completed mutations", ylabel=ylabel)
        save(fig, ax, name)

    threshold_rows = [r for r in summaries if r["family"] == "threshold"]
    if threshold_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        compact = sorted((r for r in threshold_rows if r["strategy"] == "incremental_compact"),
                         key=lambda r: float(r["compaction_threshold"]))
        no_auto = [r["query_mean_ms_mean"] for r in threshold_rows if r["strategy"] == "incremental"]
        labels = ["none"] + [f"{float(r['compaction_threshold'])*100:g}%" for r in compact]
        values = [statistics.mean(no_auto)] + [r["query_mean_ms_mean"] for r in compact]
        ax.plot(range(len(values)), values, marker="o", label="incremental; threshold rule in manifest")
        ax.set_xticks(range(len(values)), labels)
        ax.set(xlabel="Automatic delta/tombstone compaction threshold", ylabel="Mean scheduled query wall time (ms)")
        save(fig, ax, "query_latency_vs_compaction_threshold")
    else:
        skipped["query_latency_vs_compaction_threshold"] = "No threshold measurements"

    lines("query_ratio", "queries_per_update", "measured_workload_ms", "workload_time_vs_query_ratio",
          "Queries per mutation (1, 10, 100)", "Measured index-operation + query wall time (ms)", True)
    lines("corpus_size", "corpus_size", "total_mutation_ms", "maintenance_vs_corpus_size",
          "Source subset records (initial active fraction in title)", "Total mutation + automatic maintenance (ms)")

    checks = json.loads((output/"correctness_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped = defaultdict(list)
    for path in output.glob("*/retrieval_checks.csv"):
        for row in _read(path):
            grouped[row["strategy"]].append(row["exact_topk_agreement"] == "True")
    for i, strategy in enumerate(colors):
        if grouped[strategy]:
            ax.bar(i, statistics.mean(grouped[strategy]), color=colors[strategy], label=strategy)
    ax.set_xticks(range(3), list(colors))
    ax.set(ylabel="Exact ordered top-k agreement with rebuilt oracle", ylim=(0, 1.05))
    ax.text(0.02, 0.05, f"{checks['query_checkpoint_checks']} query/checkpoint checks", transform=ax.transAxes)
    save(fig, ax, "exact_topk_agreement")

    cumulative = [row for path in output.glob("*/batch_timings.csv") for row in _read(path) if row["family"] == "principal"]
    if cumulative:
        fig, ax = plt.subplots(figsize=(7, 4))
        for strategy in colors:
            series = defaultdict(list)
            for row in cumulative:
                if row["strategy"] == strategy:
                    series[row["repeat"]].append(row)
            points = defaultdict(list)
            for rows in series.values():
                total = 0
                for row in sorted(rows, key=lambda r: int(r["processed_operations"])):
                    total += float(row["mutation_wall_ms"])
                    points[int(row["processed_operations"])].append(total)
            xs = sorted(points)
            means = [statistics.mean(points[x]) for x in xs]
            sd = [statistics.stdev(points[x]) if len(points[x]) > 1 else 0 for x in xs]
            ax.plot(xs, means, label=strategy, color=colors[strategy])
            ax.fill_between(xs, [max(0, m - s) for m, s in zip(means, sd)], [m + s for m, s in zip(means, sd)], alpha=0.12)
        ax.set(xlabel="Completed mutations", ylabel="Cumulative mutation + automatic maintenance (ms)")
        save(fig, ax, "cumulative_maintenance_cost")
    else:
        skipped["cumulative_maintenance_cost"] = "No principal operation sequence"
    write_json_atomic(destination/"plot_manifest.json", {"generated": made, "skipped": skipped,
                                                        "dataset": manifest["dataset"], "variability": "sample SD across repeats"})
    return made
