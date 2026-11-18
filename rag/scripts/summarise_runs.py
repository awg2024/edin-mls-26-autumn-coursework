
"""Convert RAG experiment JSONL outputs into analysis-ready CSV."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

FIELDS = [
    "query_id",
    "experiment",
    "split",
    "candidate_k",
    "final_k",
    "context_strategy",
    "reranker_enabled",
    "context_chars",
    "asr_ms",
    "retrieval_ms",
    "rerank_ms",
    "context_ms",
    "generation_ms",
    "total_ms",
    "final_retrieved_record_ids",
    "final_retrieval_scores",
    "selected_evidence",
    "answer",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    records = []
    with args.input.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError("No query records found")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for record in records:
            row = {}
            for field in FIELDS:
                value = record.get(field)

                if field == "experiment" and args.experiment:
                    value = args.experiment

                if isinstance(value, (list, dict)):
                    value = json.dumps(value)

                row[field] = value

            writer.writerow(row)

    print(f"Saved {len(records)} queries to {args.output}")

    for field in (
        "asr_ms",
        "retrieval_ms",
        "rerank_ms",
        "context_ms",
        "generation_ms",
        "total_ms",
        "context_chars",
    ):
        values = [
            record.get(field)
            for record in records
            if isinstance(record.get(field), (int, float))
            and not isinstance(record.get(field), bool)
        ]

        if values:
            print(
                f"{field}: mean={mean(values):.3f}, "
                f"coverage={len(values)}/{len(records)}"
            )
        else:
            print(f"{field}: NOT RECORDED")


if __name__ == "__main__":
    main()

