"""Cluster-only orchestration of the supplied tests and existing RAG runners.

No tests, inference, or remote calls run on import. Invoke each stage explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from summarise import export

BUDGET_ARMS = ["simple_none", "tfidf_none", "mmr_none", "tfidf_fixed_512_skip",
               "mmr_fixed_512_skip", "mmr_adaptive_skip", "mmr_matched_skip"]
DYNAMIC_ARMS = ["original_static", "dynamic_before", "dynamic_after", "freshly_rebuilt_after"]


class FailedStage(RuntimeError):
    pass


class Run:
    def __init__(self, args):
        self.args = args
        self.repo, self.out, self.assets = args.repo.resolve(), args.run_dir.resolve(), args.assets_root.resolve()
        self.budget = self.repo/"experiments/rag_snapshots/rag-adaptive-context-budget"
        self.dynamic = self.repo/"experiments/rag_snapshots/rag-dynamic-corpus"
        (self.out/"logs").mkdir(parents=True, exist_ok=True)
        (self.out/"inputs").mkdir(exist_ok=True)
        self.status = self.out/"commands.jsonl"
        self.asr_out = self.out/"asr"
        self.asr_out.mkdir(exist_ok=True)

    def env(self, project=None, api=False):
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", RAG_ASR_ROOT=str(self.repo/"hw1-asr"), ASR_CUDA=self.args.asr_cuda,
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   MPLBACKEND="Agg", MPLCONFIGDIR=str(self.out/"matplotlib_cache"))
        if project:
            env["PYTHONPATH"] = str(project/"rag")
            if project == self.dynamic:
                env["PYTHONPATH"] += os.pathsep + str(self.budget/"rag")
        if api:
            from dotenv import dotenv_values
            for key, value in dotenv_values(self.assets/".env").items():
                if value is not None:
                    env.setdefault(key, value)
            if not (env.get("ELM_API_KEY") or env.get("OPENAI_API_KEY")):
                raise FailedStage(f"Configure ELM_API_KEY in {self.assets/'.env'}")
        return env

    def command(self, name, argv, cwd=None, project=None, api=False, required=True):
        log = (self.asr_out if name.startswith("asr_") else self.out/"logs")/(name+".log")
        if log.exists():
            raise FailedStage(f"Log already exists; use a new run directory or preserve/rename the previous log: {log}")
        print(f"\nRunning {name}; log: {log}", flush=True)
        started = time.perf_counter()
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen([str(arg) for arg in argv], cwd=cwd or self.repo,
                                       env=self.env(project, api), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        with self.status.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"name": name, "argv": [str(v) for v in argv], "cwd": str(cwd or self.repo),
                                     "exit_code": code, "wall_seconds": time.perf_counter()-started,
                                     "log": str(log.relative_to(self.out))})+"\n")
        if code and required:
            raise FailedStage(f"{name} exited {code}; see {log}")
        return code

    def module(self, name, project, module, arguments, api=False):
        return self.command(name, [sys.executable, "-m", "extensions."+module, *arguments],
                            cwd=project, project=project, api=api)

    def record_sources(self):
        files = list((self.repo/"hw1-asr/csrc").glob("*")) + list((self.repo/"hw1-asr/tests").glob("*"))
        files += [self.repo/"hw1-asr/benchmark.py", self.repo/"hw1-asr/run.sh", self.repo/"hw1-asr/Makefile"]
        files += list((self.repo/"hw1-asr/glm_asr_cuda").glob("*.py"))
        files += list((self.repo/"tools/cluster").glob("*.py"))
        for project in (self.budget, self.dynamic):
            for folder in ("src", "scripts", "tests", "extensions"):
                files += list((project/"rag"/folder).rglob("*.py"))
        hashes = {str(p.relative_to(self.repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
        path = self.out/"source_sha256.json"
        if path.exists() and json.loads(path.read_text()) != hashes:
            raise FailedStage("Source files changed during this run; start a new run directory")
        path.write_text(json.dumps(hashes, indent=2)+"\n")

    def asr(self):
        root = self.repo/"hw1-asr"
        self.command("asr_environment", [sys.executable, "-c",
            "import json,platform,sys,torch,numpy; print(json.dumps({'platform':platform.platform(), 'python':sys.version, 'torch':torch.__version__, 'numpy':numpy.__version__, 'torch_cuda':torch.version.cuda},indent=2))"])
        self.command("asr_gpu", ["nvidia-smi", "-L"])
        self.command("asr_nvcc", ["nvcc", "--version"])
        self.command("asr_clean", ["make", "clean"], root)
        self.command("asr_build", ["make"], root)
        tests = self.command("asr_tests", ["make", "test"], root, required=False)
        # Keep an independent environment/control result even when kernels fail.
        baseline = self.command("asr_model_none", ["bash", "run.sh", "--cuda", "none", "--profile"], root, required=False)
        if tests == 0 and baseline == 0:
            self.command("asr_model_all", ["bash", "run.sh", "--cuda", "all", "--profile"], root)
        else:
            raise FailedStage("Kernel suite or PyTorch reference failed; CUDA-all model run was not attempted")

    def import_asr_logs(self):
        directory = self.args.validated_asr_logs.resolve()
        mapping = {"validation_tests.log": "asr_tests.log", "validation_pytorch.log": "asr_model_none.log",
                   "validation_cuda_all.log": "asr_model_all.log"}
        texts = {name: (directory/name).read_text() for name in mapping}
        suites = re.findall(r"^Task\s+([1-4]).*:\s+(ALL PASS|SOME CASES FAILED)\s*$", texts["validation_tests.log"], re.M)
        if sorted(suites) != [(str(task), "ALL PASS") for task in range(1, 5)]:
            raise FailedStage("Manual logs must contain exactly four ALL PASS task-suite summaries")
        for name, mode in (("validation_pytorch.log", "torch"), ("validation_cuda_all.log", "cuda")):
            body = texts[name]
            statuses = re.findall(r"^Status:\s*(PASS|FAIL)\s*$", body, re.M)
            if statuses != ["PASS"]:
                raise FailedStage(f"{name} must contain exactly one passing model result")
            settings = re.search(r"^Kernels:\s*(.*)$", body, re.M)
            if settings is None or any(f"{group}={mode}" not in settings[1].split()
                                       for group in ("rope", "reduce", "gemm", "attention", "provided")):
                raise FailedStage(f"Unexpected kernel configuration in {name}")
        if any((self.asr_out/name).exists() for name in mapping.values()):
            raise FailedStage("Use a fresh run directory to import the manual ASR evidence")
        for name, target in mapping.items():
            shutil.copy2(directory/name, self.asr_out/target)
        (self.out/"manual_asr_validation.json").write_text(json.dumps({
            "sources": {str(directory/name): hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in mapping},
            "validation": "Recorded ALL PASS/PASS markers and kernel configuration checked; no ASR tests rerun",
            "shell_exit_codes": "not recorded in manual logs", "clean_build": "not recorded in imported logs",
            "historical_execution_source_hash": "not recorded; source_sha256.json identifies current experiment code"}, indent=2)+"\n")

    def prepare(self):
        if self.args.validated_asr_logs:
            self.import_asr_logs()
        elif self.args.asr_cuda == "all":
            summary = self.out/"excel/asr_checks.csv"
            if not summary.exists():
                raise FailedStage("Run the ASR stage successfully first, or explicitly choose --asr-cuda none")
            import csv
            checks = list(csv.DictReader(summary.open()))
            if not all(row["status"] == "PASS" for row in checks):
                raise FailedStage("ASR validation is not all PASS; fix it or explicitly choose --asr-cuda none")
        if not os.environ.get("RAG_ASR_PYTHON"):
            raise FailedStage("Export RAG_ASR_PYTHON from the mls-asr interpreter before activating the RAG venv")
        for name, project in (("budget", self.budget), ("dynamic", self.dynamic)):
            self.command("test_"+name, [sys.executable, "-m", "unittest", "discover", "-s", "rag/tests", "-v"],
                         cwd=project, project=project)
        self.module("prepare_queries", self.dynamic, "dynamic_corpus", [
            "prepare-queries", "--data-dir", self.assets/"data/rag_corpus", "--model-dir", self.assets/"models/enc_a",
            "--split", "practice", "--limit", "5", "--device", "cpu", "--transcribe-missing",
            "--transcript-cache", self.out/"inputs/practice5_transcripts.jsonl",
            "--output", self.out/"inputs/practice5_queries.npz"])
        self.module("prepare_budget", self.budget, "adaptive_context_budget", [
            "prepare", "--data-dir", self.assets/"data/rag_corpus", "--encoder", self.assets/"models/enc_a",
            "--split", "practice", "--limit", "5", "--k", "5", "--device", "cpu",
            "--transcript-cache", self.out/"inputs/practice5_transcripts.jsonl",
            "--output", self.out/"inputs/budget_practice5.jsonl"])
        (self.out/"preparation_scope.json").write_text(json.dumps({"queries": 5, "split": "practice",
            "asr_cuda": self.args.asr_cuda, "asr_python": os.environ["RAG_ASR_PYTHON"],
            "timing_scope": "Subsequent experiments reuse these ASR transcripts and query inputs"}, indent=2)+"\n")

    def offline(self):
        self.module("dynamic_course", self.dynamic, "dynamic_corpus", ["ablate", "--dataset", "course",
            "--data-dir", self.assets/"data/rag_corpus", "--queries", self.out/"inputs/practice5_queries.npz",
            "--repeats", "3", "--output", self.out/"dynamic_course", "--plot"])
        common = ["run", "--snapshot", self.out/"inputs/budget_practice5.jsonl", "--match-adaptive",
                  "--qrels", self.assets/"data/rag_corpus/qrels_practice.tsv"]
        for name, extra in (("budget_offline", ["--budgets", "256", "512", "1024", "2048",
                            "--packing-policies", "skip", "sentence_prefix"]),
                            ("budget_adaptive128_1024", ["--min-budget", "128", "--max-budget", "1024"]),
                            ("budget_adaptive512_2048", ["--min-budget", "512", "--max-budget", "2048"])):
            self.module(name, self.budget, "adaptive_context_budget", [*common, *extra, "--output", self.out/name])
        self.module("budget_overhead", self.budget, "adaptive_context_budget", ["overhead", "--counts", "1", "5", "20",
            "100", "500", "1000", "--repeats", "30", "--output", self.out/"budget_overhead"])

    def generate(self):
        arms = BUDGET_ARMS + (["mmr_fixed_256_skip", "mmr_fixed_1024_skip", "mmr_fixed_2048_skip"]
                              if self.args.budget_sweep else [])
        self.module("budget_generate", self.budget, "adaptive_context_budget", ["run", "--snapshot",
            self.out/"inputs/budget_practice5.jsonl", "--output", self.out/"budget_pilot", "--generate", "--limit", "5",
            "--match-adaptive", "--qrels", self.assets/"data/rag_corpus/qrels_practice.tsv", "--conditions", *arms], api=True)
        # The budget runner records per-request API failures instead of raising.
        for arm in arms:
            rows = [json.loads(line) for line in (self.out/"budget_pilot"/arm/"metrics.jsonl").read_text().splitlines() if line.strip()]
            if len(rows) != 5 or any(row["generation_status"] != "generated" for row in rows):
                raise FailedStage(f"Incomplete/failed API generation in {arm}; inspect metrics.jsonl and preserve this run")
        self.module("dynamic_generate", self.dynamic, "dynamic_corpus", ["rag-trial", "--dataset", "course", "--generate",
            "--data-dir", self.assets/"data/rag_corpus", "--queries", self.out/"inputs/practice5_queries.npz",
            "--limit", "5", "--output", self.out/"dynamic_trial"], api=True)

    def judge(self):
        budget_arms = [arm["name"] for arm in json.loads((self.out/"budget_pilot/run_manifest.json").read_text())["conditions"]]
        for kind, project, arms in (("budget", self.budget, budget_arms), ("dynamic", self.dynamic, DYNAMIC_ARMS)):
            for arm in arms:
                candidate = (self.out/"budget_pilot"/arm/"candidates.jsonl" if kind == "budget" else
                             self.out/"dynamic_trial"/(arm+".jsonl"))
                args = [sys.executable, "rag/scripts/evaluate_answers.py", "--split", "practice", "--allow-subset",
                        "--data-dir", self.assets/"data/rag_corpus", "--candidates", candidate]
                self.command(f"{kind}_{arm}_validate", [*args, "--dry-run"], project, project)
                self.command(f"{kind}_{arm}_judge", [*args, "--out-dir", self.out/"shared_judge"], project, project, api=True)
                destination = (self.out/"budget_pilot"/arm/"judge" if kind == "budget" else self.out/"dynamic_trial"/("judge_"+arm))
                destination.mkdir(parents=True, exist_ok=True)
                for filename in ("answer_judge_per_query.jsonl", "answer_judge_summary.json"):
                    shutil.copy2(self.out/"shared_judge"/filename, destination/filename)
        self.module("budget_summary", self.budget, "adaptive_context_budget", ["summarise", "--run-dir", self.out/"budget_pilot"])
        plot_args = ["plot", "--run-dir", self.out/"budget_pilot", "--output", self.out/"budget_pilot/figures"]
        if (self.out/"budget_overhead/overhead.csv").exists():
            plot_args += ["--overhead-csv", self.out/"budget_overhead/overhead.csv"]
        self.module("budget_plot", self.budget, "adaptive_context_budget", plot_args)
        self.module("dynamic_summary", self.dynamic, "dynamic_corpus", ["summarise-trial", "--output", self.out/"dynamic_trial"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("asr", "prepare", "offline", "generate", "judge", "summary"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path, default=os.getenv("CW_RUN"))
    parser.add_argument("--assets-root", type=Path, default=os.getenv("CW_ASSETS"))
    parser.add_argument("--asr-cuda", choices=("all", "none"), default="all")
    parser.add_argument("--validated-asr-logs", type=Path, help="For prepare: import the three passing manual validation_*.log files instead of rerunning ASR validation")
    parser.add_argument("--budget-sweep", action="store_true", help="Explicitly add 15 real generation calls for MMR budgets 256/1024/2048")
    args = parser.parse_args()
    if args.run_dir is None:
        parser.error("set CW_RUN or supply --run-dir")
    args.run_dir = Path(args.run_dir)
    args.assets_root = Path(args.assets_root) if args.assets_root else args.repo/"rag"
    run = Run(args)
    try:
        if args.stage != "summary":
            run.record_sources()
            getattr(run, args.stage)()
    except (FailedStage, OSError, ImportError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        with (run.out/"stage_failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": args.stage, "error": str(error)})+"\n")
        return 1
    finally:
        export(run.out)
    print(f"Summary: {run.out/'SUMMARY.md'}; Excel CSVs: {run.out/'excel'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
