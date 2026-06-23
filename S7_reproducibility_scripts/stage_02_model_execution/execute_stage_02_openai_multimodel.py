from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\Auditing Explainability-corto\auditoria")
RAW_DIR = ROOT / "03_model_outputs" / "raw" / "openai_multimodel"
OUTPUT_PATH = RAW_DIR / "openai_multimodel_raw_outputs_v1.csv"
LOG_PATH = ROOT / "07_logs" / "run_logs" / "stage_02_openai_multimodel_execution_log.json"
MANIFEST_PATH = ROOT / "07_logs" / "manifests" / "stage_02_openai_multimodel_execution_manifest.json"


RAW_COLUMNS = [
    "run_id",
    "case_id",
    "source_dataset",
    "task_family",
    "expected_output",
    "model_slot",
    "provider",
    "model_id",
    "repetition",
    "temperature",
    "max_output_tokens",
    "request_timestamp",
    "response_timestamp",
    "raw_response",
    "finish_reason",
    "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def call_openai(model_id: str, api_key: str, prompt: str, temperature: float, max_tokens: int, timeout: int) -> tuple[str, str]:
    payload = {
        "model": model_id,
        "instructions": "You are participating in an explanation consistency audit. Follow the output constraints exactly.",
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    response = post_json(
        "https://api.openai.com/v1/responses",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
        timeout,
    )
    text = response.get("output_text", "")
    if not text:
        parts = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    parts.append(content.get("text", ""))
        text = "\n".join(parts)
    return text, response.get("status", "")


def completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["run_id", "raw_response", "error", "finish_reason"])
    except pd.errors.EmptyDataError:
        return set()
    has_response = df["raw_response"].fillna("").astype(str).str.len() > 0
    no_error = df["error"].fillna("").astype(str).str.len() == 0
    not_truncated = ~df["finish_reason"].fillna("").astype(str).str.lower().isin({"max_tokens", "incomplete"})
    return set(df[has_response & no_error & not_truncated]["run_id"].astype(str))


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RAW_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute OpenAI multi-model audit calls.")
    parser.add_argument("--run-matrix", required=True, help="Run matrix CSV.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")

    matrix_path = Path(args.run_matrix)
    matrix = pd.read_csv(matrix_path)
    completed = completed_run_ids(OUTPUT_PATH)
    pending = matrix[~matrix["run_id"].astype(str).isin(completed)].copy()
    if args.limit is not None:
        pending = pending.head(args.limit)

    log = {
        "stage": "02_openai_multimodel_execution",
        "timestamp": utc_now(),
        "dry_run": bool(args.dry_run),
        "run_matrix": str(matrix_path),
        "already_completed": len(completed),
        "pending_selected": int(len(pending)),
        "output_path": str(OUTPUT_PATH),
    }

    if args.dry_run:
        log["status"] = "dry_run_ok"
    else:
        errors = 0
        written = 0
        for _, row in pending.iterrows():
            request_timestamp = utc_now()
            raw_response = ""
            finish_reason = ""
            error = ""
            try:
                raw_response, finish_reason = call_openai(
                    row["model_id"],
                    api_key,
                    row["final_prompt"],
                    float(row["temperature"]),
                    int(row["max_output_tokens"]),
                    args.timeout,
                )
            except urllib.error.HTTPError as exc:
                errors += 1
                detail = exc.read().decode("utf-8", errors="replace")
                error = f"{type(exc).__name__}: {exc.code}: {detail[:500]}"
            except Exception as exc:
                errors += 1
                error = f"{type(exc).__name__}: {exc}"
            append_row(
                OUTPUT_PATH,
                {
                    **row.to_dict(),
                    "request_timestamp": request_timestamp,
                    "response_timestamp": utc_now(),
                    "raw_response": raw_response,
                    "finish_reason": finish_reason,
                    "error": error,
                },
            )
            written += 1
            time.sleep(args.sleep)
        log["status"] = "executed"
        log["rows_written"] = written
        log["errors"] = errors

    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
