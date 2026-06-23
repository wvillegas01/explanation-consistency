from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\Auditing Explainability-corto\auditoria")
RAW_PATH = ROOT / "03_model_outputs" / "raw" / "openai_multimodel" / "openai_multimodel_raw_outputs_v1.csv"
OUT_DIR = ROOT / "03_model_outputs" / "normalized" / "openai_multimodel"
NORMALIZED_PATH = OUT_DIR / "openai_multimodel_full_normalized_outputs_v1.csv"
REPORT_PATH = ROOT / "06_reports" / "stage_reports" / "stage_04_openai_full_normalization_report.md"
LOG_PATH = ROOT / "07_logs" / "run_logs" / "stage_04_openai_full_normalization_log.json"
MANIFEST_PATH = ROOT / "07_logs" / "manifests" / "stage_04_openai_full_normalization_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


def split_response(raw: str) -> tuple[str, str, str]:
    raw = clean_text(raw)
    pattern = r"final answer\s*[:\-]\s*(?P<answer>.*?)(?:explanation)\s*[:\-]\s*(?P<explanation>.*)$"
    match = re.search(pattern, raw, flags=re.IGNORECASE)
    if match:
        return clean_text(match.group("answer")), clean_text(match.group("explanation")), "label_split"
    parts = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)
    if len(parts) == 2:
        return clean_text(parts[0]), clean_text(parts[1]), "sentence_fallback"
    return raw, raw, "single_text_fallback"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_PATH)
    raw = raw[raw["run_id"].astype(str).str.startswith("full__")].copy()
    raw["_row_order"] = range(len(raw))
    latest = raw.sort_values("_row_order").groupby("run_id", as_index=False).tail(1).copy()
    finish = latest["finish_reason"].fillna("").astype(str).str.lower()
    usable = latest[
        latest["raw_response"].fillna("").astype(str).str.len().gt(0)
        & latest["error"].fillna("").astype(str).str.len().eq(0)
        & ~finish.isin(["max_tokens", "incomplete"])
    ].copy()

    rows = []
    for _, row in usable.iterrows():
        final_answer, explanation, parse_status = split_response(row["raw_response"])
        rows.append(
            {
                "run_id": row["run_id"],
                "case_id": row["case_id"],
                "source_dataset": row["source_dataset"],
                "task_family": row["task_family"],
                "expected_output": row["expected_output"],
                "model_slot": row["model_slot"],
                "provider": row["provider"],
                "model_id": row["model_id"],
                "repetition": row["repetition"],
                "final_answer": final_answer,
                "explanation": explanation,
                "raw_response": clean_text(row["raw_response"]),
                "parse_status": parse_status,
                "finish_reason": row["finish_reason"],
            }
        )
    normalized = pd.DataFrame(rows)
    normalized.to_csv(NORMALIZED_PATH, index=False, encoding="utf-8")

    parse_summary = normalized.groupby(["model_id", "parse_status"]).size().reset_index(name="n")
    lines = [
        "# Stage 04 OpenAI Full Normalization Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Full raw latest run IDs: {latest['run_id'].nunique()}",
        f"Full usable rows normalized: {len(normalized)}",
        "",
        "| model_id | parse_status | n |",
        "| --- | --- | --- |",
    ]
    for _, item in parse_summary.iterrows():
        lines.append(f"| {item['model_id']} | {item['parse_status']} | {item['n']} |")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    log = {
        "stage": "04_openai_full_normalization",
        "timestamp": utc_now(),
        "full_raw_latest_run_ids": int(latest["run_id"].nunique()),
        "normalized_rows": int(len(normalized)),
        "outputs": [str(NORMALIZED_PATH), str(REPORT_PATH), str(MANIFEST_PATH)],
    }
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
