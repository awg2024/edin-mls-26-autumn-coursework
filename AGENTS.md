# Repository Guidelines

## Project Structure & Module Organization

- `cuda-tutorial/`: numbered CUDA examples and shared error-checking helpers.
- `hw1-asr/csrc/`: coursework kernels; `tests/`: standalone CUDA tests; `glm_asr_cuda/`: Python model integration; `docs/`: task requirements.
- `hw1-asr/experiments/`: configuration records, raw benchmark logs and interpreted results.
- `rag/src/`, `rag/scripts/`, `rag/tests/`: retrieval pipeline, command-line tools and adapter tests. ASR and RAG use separate Python environments.
- `report/`: LaTeX template and PDF format checker.
- `rag-assets.zip`: Git LFS archive supplying corpus data and encoder assets.

## Build, Test, and Development Commands

Reference commands for the user to run from the repository root on the university GPU cluster:

```bash
make -C cuda-tutorial                  # Build lecture examples
make -C hw1-asr                        # Build kernels and test executables
make -C hw1-asr test3                  # Run only tiled GEMM tests
make -C hw1-asr test                   # Run all four kernel suites
python -m unittest discover -s rag/tests -v
python report/format_checker/check_report_format.py path/to/report.pdf
```

CUDA commands require `nvcc` and a GPU allocation. Set `PATH=/opt/cuda-12.8.0/bin:$PATH` there. ASR builds reside under `hw1-asr/build/sm_XX/`. Follow module READMEs for setup. The user can run `./run.sh --cuda none` inside `hw1-asr/` on the cluster for the PyTorch baseline.

## Coding Style & Naming Conventions

Follow surrounding code: two-space indentation for CUDA/C++17 and four spaces for Python. Use descriptive `snake_case` functions and variables, `PascalCase` Python classes and uppercase tuning constants. Preserve public `launch_*` signatures and supplied model interfaces. Implement coursework kernels directly in CUDA. No repository-wide formatter or linter is configured; avoid unrelated reformatting.

## Testing Guidelines

**Do not run any tests locally. The user runs all tests on the university GPU cluster.** This includes CUDA, Python/CPU and integration tests, benchmarks and sanitizer diagnostics. Provide exact commands; mark validation pending until the user supplies results.

CUDA tests follow `tests/test_taskN_*.cu`; RAG uses Python `unittest` and `test_*.py`. Preserve authoritative tests. No numerical coverage threshold is configured. Keep raw logs intact and distinguish intentionally failing diagnostics from production results.

## Commit & Pull Request Guidelines

**Do not push changes autonomously. The user handles all pushes; provide commands instead of running `git push`.**

Use short, action-oriented commit subjects, such as `Implement Task 3 tiled GEMM` or `Fix formatting in README`. Use focused branches, for example `feature/task3_gemm`. PRs should describe changes, link issues and include validation evidence or experiment records. Keep unrelated tasks outside the diff.

## Security & Assets

Keep ELM credentials in ignored `rag/.env`; never commit keys. Retrieve assets with `git lfs pull --include="rag-assets.zip"`. Exclude generated binaries, environments and extracted datasets from commits.
