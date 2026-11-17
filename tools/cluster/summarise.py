"""Export recorded cluster outputs to Markdown and Excel-compatible UTF-8 CSV.

Missing measurements remain empty. No inference, tests, or API calls run here.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def mean(values):
    values = [v for v in (number(v) for v in values) if v is not None]
    return statistics.mean(values) if values else None


def save(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(lines, rows, columns):
    if not rows:
        lines.extend(["No recorded results yet.", ""])
        return
    def display(value):
        if value is None or value == "":
            return "Not measured"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(display(row.get(key)) for key in columns) + " |")
    lines.append("")


def asr(root, destination, lines):
    records = [json.loads(line) for line in (root/"commands.jsonl").read_text().splitlines()] if (root/"commands.jsonl").exists() else []
    codes = {row["name"]: row["exit_code"] for row in records}
    manual = (root/"manual_asr_validation.json").exists()
    text_path = root/"asr/asr_tests.log"
    text = text_path.read_text(errors="replace") if text_path.exists() else ""
    case_pattern = re.compile(r"^(.+?)\s+(PASS|FAIL)\s+max_err=([^\s]+)(?:\s+([\d.]+) ms\s*(.*))?$")
    finish_pattern = re.compile(r"^Task\s+([1-4]).*:\s+(ALL PASS|SOME CASES FAILED)$")
    cases, pending, results = [], [], {}
    for line in text.splitlines():
        match = case_pattern.match(line)
        if match:
            case, status, error, ms, speed = match.groups()
            rate = re.search(r"([\d.]+)\s+(GB/s|GFLOP/s)", speed or "")
            pending.append({"case": case.strip(), "status": status, "max_error": error,
                            "kernel_ms": number(ms), "rate": number(rate[1]) if rate else None,
                            "rate_unit": rate[2] if rate else ""})
        elif "for comparison:" in line and pending:
            match = re.search(r"naive\s+([\d.]+) GFLOP/s, cuBLAS\s+([\d.]+) GFLOP/s", line)
            if match:
                pending[-1].update(naive_GFLOPs=float(match[1]), cuBLAS_GFLOPs=float(match[2]))
        match = finish_pattern.match(line)
        if match:
            task = int(match[1])
            results[task] = match[2]
            cases.extend({"task": task, **row} for row in pending)
            pending = []
    cases.extend({"task": "unreported_after_abort", **row} for row in pending)
    checks = [{"check": "Clean build", "status": "PASS" if codes.get("asr_clean") == codes.get("asr_build") == 0 else
               "FAIL" if "asr_build" in codes or "asr_clean" in codes else "NOT RECORDED" if manual else "NOT RUN"}]
    for task in range(1, 5):
        group = [row for row in cases if row["task"] == task]
        errors = [float(row["max_error"]) for row in group]
        checks.append({"check": f"Task {task}", "status": "PASS" if results.get(task) == "ALL PASS" else
                       "FAIL" if task in results else "NOT REPORTED", "cases": len(group),
                       "failed_cases": sum(row["status"] == "FAIL" for row in group),
                       "max_error": max(errors) if errors else None})
    models, profiles = [], []
    for mode in ("none", "all"):
        path = root/f"asr/asr_model_{mode}.log"
        output = path.read_text(errors="replace") if path.exists() else ""
        status = re.search(r"^Status: (PASS|FAIL)$", output, re.M)
        successful = status is not None and status[1] == "PASS" and (
            codes.get("asr_model_"+mode) == 0 or manual and "asr_model_"+mode not in codes)
        checks.append({"check": "Model "+mode, "status": "PASS" if successful else "FAIL" if path.exists() else "NOT RUN"})
        timing = re.search(r"Time: ([\d.]+) ms per transcription \(\+/- ([\d.]+) ms\), ([\d.]+) ms per token", output)
        memory = re.search(r"Peak GPU memory: ([\d.]+) GiB", output)
        accuracy = re.search(r"^Accuracy: ([\d.]+)%", output, re.M)
        transcription = re.search(r"^Transcription: (.*)$", output, re.M)
        models.append({"mode": mode, "status": checks[-1]["status"],
                       "transcription_mean_ms": float(timing[1]) if timing else None,
                       "transcription_sd_ms": float(timing[2]) if timing else None,
                       "ms_per_token": float(timing[3]) if timing else None,
                       "peak_allocated_GiB": float(memory[1]) if memory else None,
                       "expected_word_set_coverage_pct": float(accuracy[1]) if accuracy else None,
                       "transcription": transcription[1] if transcription else ""})
        for op, backend, calls, ms, share in re.findall(r"^\s+(\w+)\s+(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%\s*$", output, re.M):
            profiles.append({"mode": mode, "operation": op, "backend": backend, "calls": int(calls),
                             "total_kernel_ms": float(ms), "kernel_share_pct": float(share)})
    speedup = (models[0]["transcription_mean_ms"]/models[1]["transcription_mean_ms"]
               if all(row["status"] == "PASS" and row["transcription_mean_ms"] for row in models) else None)
    for row in models:
        row["CUDA_all_speedup_vs_PyTorch"] = speedup
    save(destination/"asr_checks.csv", checks)
    save(destination/"asr_kernel_cases.csv", cases)
    save(destination/"asr_models.csv", models)
    save(destination/"asr_profiles.csv", profiles)
    lines.extend(["## ASR validation", ""])
    table(lines, checks, ["check", "status", "cases", "failed_cases", "max_error"])
    table(lines, models, ["mode", "status", "transcription_mean_ms", "peak_allocated_GiB", "expected_word_set_coverage_pct"])
    lines.extend(["Model 'accuracy' is the supplied benchmark's expected-word-set coverage on one audio clip; it is not WER.",
                  "Both kernel tests and the model run must pass. Profiling kernel-time totals are separate from wall-clock latency.", ""])
    if manual:
        lines.extend(["ASR results above were imported from the existing manual logs. Shell exit codes, clean-build output and historical source hashes were not recorded in those logs; no ASR validation was rerun.", ""])


def rag(root, destination, lines):
    dynamic = []
    for row in read_csv(root/"dynamic_course/summary.csv"):
        item = {key: row[key] for key in ("case", "family", "strategy", "dataset", "dimension", "corpus_size",
                "initial_active_records", "operations", "batch_size", "profile", "queries_per_update", "compaction_threshold", "repetitions")}
        for target, source in {
            "insert_ms": "insert_wall_mean_ms_mean", "update_ms": "update_wall_mean_ms_mean", "delete_ms": "delete_wall_mean_ms_mean",
            "maintenance_ms": "total_mutation_ms_mean", "maintenance_sd_ms": "total_mutation_ms_sd",
            "amortised_update_ms": "amortised_update_ms_mean", "updates_per_second": "updates_per_second_mean",
            "staging_ms": "total_staging_ms_mean", "rebuild_ms": "total_rebuild_ms_mean",
            "auto_compaction_ms": "total_auto_compaction_ms_mean", "automatic_compactions": "automatic_compactions_mean",
            "query_mean_ms": "query_mean_ms_mean", "mean_per_repeat_p95_ms": "query_p95_ms_mean",
            "query_median_ms": "query_median_ms_mean", "query_before_ms": "query_before_mean_ms_mean",
            "query_after_manual_compaction_ms": "query_after_manual_compaction_mean_ms_mean",
            "workload_ms": "measured_workload_ms_mean", "mutation_speedup": "update_speedup_vs_rebuild",
            "workload_speedup": "workload_speedup_vs_rebuild", "query_latency_change_pct": "query_latency_change_pct",
            "topk_agreement_fraction": "exact_topk_agreement_mean"}.items():
            item[target] = number(row.get(source))
        raw_bytes = number(row.get("embedding_bytes_before_manual_compaction_mean"))
        item["current_embedding_MiB"] = raw_bytes/2**20 if raw_bytes is not None else None
        dynamic.append(item)
    save(destination/"dynamic_systems.csv", dynamic)
    final = defaultdict(list)
    for path in (root/"dynamic_course").glob("*/retrieval_checks.csv"):
        config = json.loads((path.parent/"config.json").read_text())
        for row in read_csv(path):
            if row["phase"] == "checkpoint" and int(row["processed_operations"]) == config["operations"]:
                final[(row["case"], row["strategy"])].append(row)
    retrieval = []
    for (case, strategy), rows in final.items():
        item = {"case": case, "strategy": strategy, "scope": "final active state before diagnostic manual compaction",
                "unique_queries": len({r["query_id"] for r in rows}), "repetitions": len({r["repeat"] for r in rows}),
                "query_checkpoint_checks": len(rows), "quality_scope": "; ".join(sorted({r["quality_scope"] for r in rows}))}
        item["active_qrels_eligible_unique_queries"] = len({r["query_id"] for r in rows if number(r.get("active_qrels_recall_at10")) is not None})
        for key in ("original_qrels_recall_at10", "active_qrels_recall_at10", "active_qrels_mrr_at10",
                    "active_qrels_ndcg_at10", "relevant_document_availability"):
            item[key] = mean(r.get(key) for r in rows)
        retrieval.append(item)
    save(destination/"dynamic_retrieval.csv", retrieval)
    budget = []
    for path in sorted(root.glob("budget_*/summary.csv")):
        for row in read_csv(path):
            item = {"run": path.parent.name, **{key: row.get(key) for key in ("condition", "selector", "policy",
                "packing_policy", "queries", "expected_queries", "generated_queries", "quality_coverage", "configured_fixed_budget")}}
            for target, source in {"mean_allocated_budget": "configured_budget_mean", "evidence_tokens_before": "evidence_tokens_before_mean",
                "evidence_tokens_after": "evidence_tokens_after_mean", "token_reduction_pct": "token_reduction_pct_mean",
                "API_input_tokens": "api_input_tokens_mean", "API_output_tokens": "api_output_tokens_mean",
                "generation_ms": "generation_ms_mean", "controller_ms": "budget_controller_ms_mean",
                "correctness_0_to_4": "correctness_mean", "grounding_0_to_4": "grounding_mean", "completeness_0_to_4": "completeness_mean",
                "answer_score_fraction": "normalised_answer_score_mean", "judged_queries": "normalised_answer_score_n",
                "relevant_document_retention_fraction": "relevant_document_retention_mean", "matched_cost_error_pct": "matched_cost_error_pct"}.items():
                item[target] = number(row.get(source))
            budget.append(item)
    save(destination/"budget_conditions.csv", budget)
    paired = read_csv(root/"budget_pilot/paired_summary.csv")
    save(destination/"budget_paired.csv", paired)
    answers = read_csv(root/"dynamic_trial/rag_comparison.csv")
    grouped = defaultdict(list)
    for row in answers:
        grouped[row["condition"]].append(row)
    trial = []
    for condition, rows in grouped.items():
        item = {"condition": condition, "queries": len(rows), "generation_cache_hits": sum(r["generation_cache_hit"] == "True" for r in rows),
                "judged_queries": sum(number(r.get("normalised_answer_score")) is not None for r in rows)}
        for key in ("retrieval_ms", "downstream_pipeline_wall_ms", "correctness", "grounding", "completeness", "normalised_answer_score"):
            item[key+"_mean"] = mean(r.get(key) for r in rows)
        item["uncached_generation_ms_mean"] = mean(r.get("original_generation_ms") for r in rows if r["generation_cache_hit"] == "False")
        trial.append(item)
    save(destination/"dynamic_answers.csv", trial)
    lines.extend(["## Dynamic corpus: principal systems comparison", ""])
    table(lines, [r for r in dynamic if r["family"] == "principal"],
          ["strategy", "maintenance_ms", "query_mean_ms", "workload_ms", "current_embedding_MiB", "topk_agreement_fraction"])
    lines.extend(["Matrix storage excludes text, metadata, reader-held snapshots and RSS. Workload sums exclude validation/IO and diagnostic manual compaction.", ""])
    correctness = root/"dynamic_course/correctness_summary.json"
    if correctness.exists():
        lines.extend(["Recorded exactness checks:", "", "```json", correctness.read_text().strip(), "```", ""])
    lines.extend(["## Budget: real ELM pilot", ""])
    table(lines, [r for r in budget if r["run"] == "budget_pilot"],
          ["condition", "generated_queries", "judged_queries", "evidence_tokens_after", "API_input_tokens", "generation_ms", "answer_score_fraction"])
    lines.extend(["## Dynamic corpus: real ELM trial", ""])
    table(lines, trial, ["condition", "queries", "generation_cache_hits", "judged_queries", "retrieval_ms_mean", "normalised_answer_score_mean"])


def export(root):
    root = Path(root)
    destination = root/"excel"
    destination.mkdir(parents=True, exist_ok=True)
    lines = ["# Cluster validation summary", "", "Generated from recorded outputs only. Missing values mean not measured/not run.",
             "RAG is a five-practice-query pilot. It is not the teammate's 350-query K sweep.", ""]
    if (root/"stage_failures.jsonl").exists():
        lines.extend(["## Recorded stage failures", ""])
        table(lines, [json.loads(line) for line in (root/"stage_failures.jsonl").read_text().splitlines()], ["stage", "error"])
    asr(root, destination, lines)
    rag(root, destination, lines)
    lines.extend(["## Interpretation", "", "Fraction columns range from 0 to 1; use Excel Percentage formatting.",
        "Columns ending in _pct already range from 0 to 100; use Number formatting.",
        "Budget limits use the configured counter; defaults estimate UTF-8 bytes/4. API token counts are separate measurements.",
        "RAG experiments reuse ASR/encoder inputs; downstream pipeline timings are not fresh audio-to-answer wall times.",
        "Cached generation is marked explicitly. Blank answer scores are not zeros. Original gold has limitations after removing relevant documents.", ""])
    (root/"SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
