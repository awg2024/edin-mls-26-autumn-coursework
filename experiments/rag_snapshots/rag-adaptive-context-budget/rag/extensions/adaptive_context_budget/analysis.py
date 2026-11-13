"""Join supplied judge outputs, compute paired uncertainty, and plot actual data."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import random
import statistics

from .experiment import read_jsonl, write_csv, write_json

QUALITY = ("correctness", "grounding", "completeness", "normalised_answer_score")
COST = (
    "evidence_tokens_before", "evidence_tokens_after", "estimated_prompt_text_tokens",
    "api_input_tokens", "api_output_tokens", "api_total_tokens", "configured_budget",
    "budget_utilisation", "documents_represented", "selected_units", "discarded_units",
    "token_reduction_pct", "duplicate_units", "oversized_units", "truncated_units",
    "relevant_document_retention", "relevant_document_fraction", "budget_controller_ms", "context_builder_ms",
    "generation_ms", "reconstructed_downstream_ms",
)


def numeric(row, key):
    value = row.get(key, row.get("timing", {}).get(key))
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def describe(values):
    values = [value for value in values if value is not None]
    return {
        "n": len(values), "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def paired_interval(differences, samples=2000):
    """Seeded percentile bootstrap over paired queries, not independent arms."""
    if len(differences) < 2:
        return None, None
    if min(differences) == max(differences):
        return differences[0], differences[0]
    rng = random.Random(2026)
    estimates = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def load_run(run_dir):
    import json

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    rows = {}
    for spec in manifest["conditions"]:
        name = spec["name"]
        path = run_dir / name / "metrics.jsonl"
        records = read_jsonl(path) if path.exists() else []
        by_id = {row["query_id"]: row for row in records}
        if len(by_id) != len(records) or not set(by_id) <= set(manifest["query_ids"]):
            raise ValueError("duplicate or unexpected metrics query IDs")
        judge_path = run_dir / name / "judge" / "answer_judge_per_query.jsonl"
        if judge_path.exists():
            judged = read_jsonl(judge_path)
            if len({r["query_id"] for r in judged}) != len(judged):
                raise ValueError("duplicate judge query IDs")
            for result in judged:
                if result["query_id"] not in by_id:
                    raise ValueError("judge output does not match this run's query IDs")
                for key in QUALITY:
                    score = float(result[key])
                    if not math.isfinite(score) or not 0 <= score <= (1 if key == QUALITY[-1] else 4):
                        raise ValueError("judge score outside rubric range")
                    by_id[result["query_id"]][key] = score
        rows[name] = list(by_id.values())
    return manifest, rows


def summarise(run_dir):
    manifest, conditions = load_run(run_dir)
    expected = len(manifest["query_ids"])
    summaries = []
    specs = {s["name"]: s for s in manifest["conditions"]}
    for name, rows in conditions.items():
        spec = specs[name]
        summary = {
            "condition": name, "selector": spec["selector"], "packing_policy": spec["packing_policy"],
            "policy": "none" if spec["policy"] is None else "adaptive" if "min_budget" in spec["policy"] else "fixed",
            "configured_fixed_budget": (spec["policy"] or {}).get("budget"),
            "queries": len(rows), "expected_queries": expected,
            "metrics_coverage": len(rows) / expected,
            "generated_queries": sum(r["generation_status"] == "generated" for r in rows),
            "quality_coverage": sum(numeric(r, QUALITY[-1]) is not None for r in rows) / expected,
            "matched_target_tokens": spec.get("matched_target_tokens"),
        }
        for key in COST + QUALITY:
            summary.update({f"{key}_{stat}": value for stat, value in describe([numeric(r, key) for r in rows]).items()})
        if summary["matched_target_tokens"] is not None:
            target = summary["matched_target_tokens"]
            actual = summary["evidence_tokens_after_mean"]
            summary["matched_cost_error_pct"] = 100 * abs(actual - target) / target if target else None
        summaries.append(summary)

    contrasts = set()
    for name, spec in specs.items():
        reference = spec["selector"] + "_none"
        if name != reference and reference in conditions:
            contrasts.add((reference, name, "budget_only"))
        if spec["selector"] != "simple" and "simple_none" in conditions:
            contrasts.add(("simple_none", name, "selector_and_budget_practical"))
        if "_adaptive_" in name:
            matched = name.replace("_adaptive_", "_matched_")
            if matched in conditions:
                contrasts.add((matched, name, "budget_only_matched_cost"))
    paired, details = [], []
    for baseline, treatment, comparison_type in sorted(contrasts):
        left = {r["query_id"]: r for r in conditions[baseline]}
        right = {r["query_id"]: r for r in conditions[treatment]}
        common = sorted(set(left) & set(right))
        # A changed ranking/selector cannot be called a budget-only comparison.
        for qid in common:
            keys = ("candidate_ranking_hash", "transcript")
            if comparison_type != "selector_and_budget_practical":
                keys += ("selector_evidence_hash",)
            for key in keys:
                if left[qid][key] != right[qid][key]:
                    raise ValueError(f"unmatched {key} in paired comparison")
        for key in COST + QUALITY:
            pairs = [(qid, numeric(left[qid], key), numeric(right[qid], key)) for qid in common]
            pairs = [(qid, a, b) for qid, a, b in pairs if a is not None and b is not None]
            delta = [b - a for _, a, b in pairs]
            low, high = paired_interval(delta)
            a_mean = statistics.mean(a for _, a, _ in pairs) if pairs else None
            b_mean = statistics.mean(b for _, _, b in pairs) if pairs else None
            paired.append({
                "baseline": baseline, "treatment": treatment, "metric": key,
                "comparison_type": comparison_type,
                "paired_queries": len(pairs), "expected_queries": expected,
                "paired_coverage": len(pairs) / expected,
                "baseline_mean": a_mean, "treatment_mean": b_mean,
                "mean_delta": statistics.mean(delta) if delta else None,
                "bootstrap_95_low": low, "bootstrap_95_high": high,
                "retention_pct": 100 * b_mean / a_mean if a_mean not in (None, 0) else None,
            })
            details.extend({"baseline": baseline, "treatment": treatment, "query_id": qid,
                            "comparison_type": comparison_type,
                            "metric": key, "baseline_value": a, "treatment_value": b, "delta": b - a,
                            "query_retrieval_confidence": right[qid].get("query_retrieval_confidence"),
                            "treatment_budget": right[qid]["configured_budget"]}
                           for qid, a, b in pairs)
    groups = []
    for name, rows in conditions.items():
        strata = {}
        for row in rows:
            confidence = row.get("query_retrieval_confidence")
            band = "missing" if confidence is None else str(min(3, int(confidence * 4))) + "/4"
            for kind, value in (("confidence_quartile", band), ("selected_budget", str(row.get("configured_budget")))):
                strata.setdefault((kind, value), []).append(row)
        for (kind, value), members in strata.items():
            group = {"condition": name, "stratum": kind, "value": value, "queries": len(members)}
            for key in ("evidence_tokens_after", "normalised_answer_score", "generation_ms", "relevant_document_retention"):
                group.update({f"{key}_{stat}": val for stat, val in describe([numeric(r, key) for r in members]).items()})
            groups.append(group)
    run_dir = Path(run_dir)
    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "paired_summary.csv", paired)
    write_csv(run_dir / "paired_per_query.csv", details)
    write_csv(run_dir / "query_groups.csv", groups)
    paired_groups = {}
    for row in details:
        if row["metric"] == "normalised_answer_score":
            confidence = row["query_retrieval_confidence"]
            band = "missing" if confidence is None else str(min(3, int(confidence * 4))) + "/4"
            for kind, value in (("confidence_quartile", band), ("selected_budget", str(row["treatment_budget"]))):
                paired_groups.setdefault((row["baseline"], row["treatment"], kind, value), []).append(row["delta"])
    write_csv(run_dir / "paired_groups.csv", [
        {"baseline": base, "treatment": treatment, "stratum": kind, "value": value,
         "paired_queries": len(values), "mean_quality_delta": statistics.mean(values),
         "bootstrap_95_low": paired_interval(values)[0], "bootstrap_95_high": paired_interval(values)[1]}
        for (base, treatment, kind, value), values in paired_groups.items()
    ])
    write_json(run_dir / "analysis_notes.json", {
        "quality_missing": not any(s["normalised_answer_score_n"] for s in summaries),
        "confidence_interval": "2000-resample paired-query percentile bootstrap, seed=2026; unavailable for n<2",
        "coverage": "Both full planned-query and actually evaluated paired-query counts are reported",
        "qrels_limitation": "Document-level relevance cannot establish that a selected sentence answers the question",
        "token_limit": "Bound applies to selected counter; default is a four-UTF8-bytes-per-token estimate, not API tokenization",
    })
    return summaries


def plot_results(output, run_dir=None, overhead_csv=None):
    import os
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    (Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made, skipped = [], {}

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=250, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    def scatter(rows, xkey, ykey, name, xlabel, ylabel, controls=()):
        usable = [r for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
        if not usable:
            skipped[name] = "No actual generation/judge measurements for these axes"
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        for label in dict.fromkeys(r["condition"] for r in usable):
            group = [r for r in usable if r["condition"] == label]
            ax.scatter([r[xkey] for r in group], [r[ykey] for r in group], label=label, s=28)
        for control in controls:
            score = control.get("normalised_answer_score_mean")
            if score is not None:
                ax.axhline(score, linestyle="--", linewidth=1, label=control["condition"] + " control")
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
        save(fig, name)

    if run_dir is not None:
        summaries = summarise(run_dir)
        _, conditions = load_run(run_dir)
        fixed = [s for s in summaries if s["policy"] == "fixed"]
        scatter(fixed, "configured_fixed_budget", "normalised_answer_score_mean", "quality_vs_budget",
                "Configured evidence-token budget", "Mean normalised answer score",
                controls=[s for s in summaries if s["policy"] == "none"])
        generated = [r for rows in conditions.values() for r in rows if numeric(r, "generation_ms") is not None]
        api_tokens_available = bool(generated) and all(r["api_input_tokens"] is not None for r in generated)
        points = [{"condition": name,
                   "tokens": r["api_input_tokens"] if api_tokens_available else r["estimated_prompt_text_tokens"],
                   "latency": numeric(r, "generation_ms")}
                  for name, rows in conditions.items() for r in rows]
        scatter(points, "tokens", "latency", "prompt_tokens_vs_generation_latency",
                "API input tokens" if api_tokens_available else "Estimated prompt message-text tokens (not API usage)",
                "Generation latency (ms)")
        scatter(summaries, "evidence_tokens_after_mean", "normalised_answer_score_mean", "quality_vs_tokens",
                "Mean estimated evidence tokens used", "Mean normalised answer score")
        chosen = [s for s in summaries if s["normalised_answer_score_mean"] is not None]
        if chosen:
            fig, ax = plt.subplots(figsize=(max(7, len(chosen) * 0.7), 4))
            ax.bar(range(len(chosen)), [s["normalised_answer_score_mean"] for s in chosen])
            labels = [f"{s['condition']}\n{s['evidence_tokens_after_mean']:.0f} evidence tokens" for s in chosen]
            ax.set_xticks(range(len(chosen)), labels, rotation=50, ha="right", fontsize=7)
            ax.set(ylabel="Mean normalised answer score", ylim=(0, 1))
            save(fig, "fixed_vs_adaptive")
        else:
            skipped["fixed_vs_adaptive"] = "No judge scores"
        controls = {s["selector"]: s for s in summaries if s["policy"] == "none"}
        points = []
        for row in summaries:
            control = controls.get(row["selector"])
            if not control:
                continue
            quality, tokens = control["normalised_answer_score_mean"], control["evidence_tokens_after_mean"]
            if quality not in (None, 0) and tokens and row["normalised_answer_score_mean"] is not None:
                points.append({"condition": row["condition"],
                               "tokens_removed": 100 * (1 - row["evidence_tokens_after_mean"] / tokens),
                               "quality_removed": 100 * (1 - row["normalised_answer_score_mean"] / quality)})
        scatter(points, "tokens_removed", "quality_removed", "token_vs_quality_reduction",
                "Evidence-token reduction vs same-selector control (%)", "Relative answer-score reduction (%)")
    else:
        for name in ("quality_vs_budget", "prompt_tokens_vs_generation_latency", "quality_vs_tokens",
                     "fixed_vs_adaptive", "token_vs_quality_reduction"):
            skipped[name] = "No course generation/judge run provided"
    if overhead_csv is not None:
        with Path(overhead_csv).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        fig, ax = plt.subplots(figsize=(7, 4))
        for label in dict.fromkeys(r["configuration"] for r in rows):
            group = sorted((r for r in rows if r["configuration"] == label), key=lambda r: int(r["candidate_units"]))
            xs = [int(r["candidate_units"]) for r in group]
            means = [float(r["mean_ms"]) for r in group]
            sd = [float(r["sd_ms"]) for r in group]
            ax.plot(xs, means, marker="o", label=label)
            ax.fill_between(xs, [max(0, m - s) for m, s in zip(means, sd)],
                            [m + s for m, s in zip(means, sd)], alpha=0.12)
        ax.set(xlabel="Synthetic ranked evidence units (not retrieval K)",
               ylabel="Controller latency (ms; mean ± sample SD)", xscale="log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        save(fig, "overhead_vs_candidates")
    else:
        skipped["overhead_vs_candidates"] = "No measured overhead CSV provided"
    write_json(output / "plot_manifest.json", {"generated": made, "skipped": skipped})
    return made
