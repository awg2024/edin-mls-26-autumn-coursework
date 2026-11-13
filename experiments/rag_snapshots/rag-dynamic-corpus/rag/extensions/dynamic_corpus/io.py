"""Experiment serialization; reuse the coursework's atomic JSON writers."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.evaluation.answer_judge import load_jsonl, write_json_atomic, write_jsonl_atomic


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
                             for key, value in row.items()})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


__all__ = ["write_csv", "fingerprint", "load_jsonl", "write_json_atomic", "write_jsonl_atomic"]
