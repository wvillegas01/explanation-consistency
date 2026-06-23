from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\Auditing Explainability-corto\auditoria")
DATA_ROOT = Path(r"C:\Users\wilop\Documents\Datos-generales")
RANDOM_STATE = 20260615


OUTPUT_SCHEMA = [
    "case_id",
    "source_dataset",
    "source_case_id",
    "task_family",
    "input_context",
    "expected_output",
    "prompt_template",
    "final_prompt",
]


def clean_text(value: object, max_chars: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def open_reasoning_template(prompt: str) -> str:
    return (
        "Answer the following user request. Provide the final answer first, then provide a concise explanation "
        "of the main reasoning that supports your answer.\n\n"
        f"User request:\n{prompt}"
    )


def bbc_template(article: str) -> str:
    return (
        "Classify the following news article into exactly one category: business, entertainment, politics, "
        "sport, or tech. Provide the category first, then explain the main textual evidence supporting your classification.\n\n"
        f"Article:\n{article}"
    )


def oulad_template(profile: str) -> str:
    return (
        "Given the following student profile, classify the likely academic outcome as one of: Distinction, Pass, "
        "Fail, or Withdrawn. Provide the outcome first, then explain the main factors supporting your classification.\n\n"
        f"Student profile:\n{profile}"
    )


def select_ultrachat(n: int) -> pd.DataFrame:
    path = DATA_ROOT / r"Comunicacion\UltraChat\test_sft.csv"
    df = pd.read_csv(path)
    df["prompt_clean"] = df["prompt"].map(lambda x: clean_text(x, 1200))
    mask = df["prompt_clean"].str.len().between(80, 1200)
    mask &= df["prompt_clean"].str.contains(
        r"why|how|explain|reason|justify|compare|what|which|should",
        case=False,
        na=False,
        regex=True,
    )
    selected = df[mask].drop_duplicates("prompt_clean").sample(n=n, random_state=RANDOM_STATE)
    selected = selected.reset_index(drop=True)
    selected["source_case_id"] = selected["prompt_id"].astype(str)
    selected["input_context"] = selected["prompt_clean"]
    return selected[["source_case_id", "input_context"]]


def select_wildchat(n: int) -> pd.DataFrame:
    path = DATA_ROOT / r"Comunicacion\comunicacion2\06_canonical\canonical_turns.parquet"
    cols = ["dataset", "conversation_id", "turn_index", "speaker_role", "text", "model", "language"]
    df = pd.read_parquet(path, columns=cols)
    df = df[
        (df["dataset"] == "WildChat-1M")
        & (df["speaker_role"] == "user")
        & (df["language"] == "English")
    ].copy()
    df["text_clean"] = df["text"].map(lambda x: clean_text(x, 1200))
    mask = df["text_clean"].str.len().between(80, 1200)
    mask &= df["text_clean"].str.contains(
        r"why|how|explain|reason|justify|compare|which|should|decision|classify",
        case=False,
        na=False,
        regex=True,
    )
    selected = df[mask].drop_duplicates("text_clean").sample(n=n, random_state=RANDOM_STATE + 1)
    selected = selected.reset_index(drop=True)
    selected["source_case_id"] = (
        selected["conversation_id"].astype(str) + "_turn_" + selected["turn_index"].astype(str)
    )
    selected["input_context"] = selected["text_clean"]
    return selected[["source_case_id", "input_context"]]


def select_bbc(n: int) -> pd.DataFrame:
    path = DATA_ROOT / r"Multilingue\BBC_Forum\bbc_news_text_complexity_summarization.csv"
    df = pd.read_csv(path)
    df["article_clean"] = df["text"].map(lambda x: clean_text(x, 1600))
    rows = []
    per_label = n // df["labels"].nunique()
    remainder = n - per_label * df["labels"].nunique()
    labels = sorted(df["labels"].dropna().unique())
    for idx, label in enumerate(labels):
        take = per_label + (1 if idx < remainder else 0)
        subset = df[df["labels"] == label].sample(n=take, random_state=RANDOM_STATE + idx)
        rows.append(subset)
    selected = pd.concat(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    selected["source_case_id"] = selected.index.map(lambda i: f"bbc_{i:04d}")
    selected["input_context"] = selected["article_clean"]
    selected["expected_output"] = selected["labels"].astype(str)
    return selected[["source_case_id", "input_context", "expected_output"]]


def build_oulad_profile(row: pd.Series) -> str:
    fields = [
        ("module", row.get("code_module")),
        ("presentation", row.get("code_presentation")),
        ("gender", row.get("gender")),
        ("region", row.get("region")),
        ("highest education", row.get("highest_education")),
        ("deprivation band", row.get("imd_band")),
        ("age band", row.get("age_band")),
        ("previous attempts", row.get("num_of_prev_attempts")),
        ("studied credits", row.get("studied_credits")),
        ("disability", row.get("disability")),
    ]
    return "; ".join(f"{name}: {clean_text(value)}" for name, value in fields)


def select_oulad(n: int) -> pd.DataFrame:
    path = DATA_ROOT / r"Educativos\OULAD\studentInfo.csv"
    df = pd.read_csv(path)
    labels = ["Distinction", "Pass", "Fail", "Withdrawn"]
    base = n // len(labels)
    remainder = n - base * len(labels)
    rows = []
    for idx, label in enumerate(labels):
        take = base + (1 if idx < remainder else 0)
        subset = df[df["final_result"] == label].sample(n=take, random_state=RANDOM_STATE + 20 + idx)
        rows.append(subset)
    selected = pd.concat(rows).sample(frac=1, random_state=RANDOM_STATE + 2).reset_index(drop=True)
    selected["source_case_id"] = selected["id_student"].astype(str)
    selected["input_context"] = selected.apply(build_oulad_profile, axis=1)
    selected["expected_output"] = selected["final_result"].astype(str)
    return selected[["source_case_id", "input_context", "expected_output"]]


def make_records() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ultrachat = select_ultrachat(18)
    wildchat = select_wildchat(17)
    open_pool = pd.concat(
        [
            ultrachat.assign(source_dataset="UltraChat", expected_output="not_applicable"),
            wildchat.assign(source_dataset="WildChat-1M", expected_output="not_applicable"),
        ],
        ignore_index=True,
    )
    bbc = select_bbc(35).assign(source_dataset="BBC")
    oulad = select_oulad(30).assign(source_dataset="OULAD")

    pools = {"open_reasoning": open_pool, "bbc": bbc, "oulad": oulad}
    records = []
    counters = {"open_ended_reasoning": 0, "text_classification": 0, "structured_decision_support": 0}

    for _, row in open_pool.iterrows():
        task = "open_ended_reasoning"
        counters[task] += 1
        case_id = f"OE-{counters[task]:03d}"
        prompt = open_reasoning_template(row["input_context"])
        records.append(
            {
                "case_id": case_id,
                "source_dataset": row["source_dataset"],
                "source_case_id": row["source_case_id"],
                "task_family": task,
                "input_context": row["input_context"],
                "expected_output": row["expected_output"],
                "prompt_template": "open_reasoning_v1",
                "final_prompt": prompt,
            }
        )

    for _, row in bbc.iterrows():
        task = "text_classification"
        counters[task] += 1
        case_id = f"TC-{counters[task]:03d}"
        prompt = bbc_template(row["input_context"])
        records.append(
            {
                "case_id": case_id,
                "source_dataset": row["source_dataset"],
                "source_case_id": row["source_case_id"],
                "task_family": task,
                "input_context": row["input_context"],
                "expected_output": row["expected_output"],
                "prompt_template": "bbc_classification_v1",
                "final_prompt": prompt,
            }
        )

    for _, row in oulad.iterrows():
        task = "structured_decision_support"
        counters[task] += 1
        case_id = f"SD-{counters[task]:03d}"
        prompt = oulad_template(row["input_context"])
        records.append(
            {
                "case_id": case_id,
                "source_dataset": row["source_dataset"],
                "source_case_id": row["source_case_id"],
                "task_family": task,
                "input_context": row["input_context"],
                "expected_output": row["expected_output"],
                "prompt_template": "oulad_decision_v1",
                "final_prompt": prompt,
            }
        )

    benchmark = pd.DataFrame(records, columns=OUTPUT_SCHEMA)
    return benchmark, pools


def write_templates(out_path: Path) -> None:
    text = """# Prompt Templates

## open_reasoning_v1

Answer the following user request. Provide the final answer first, then provide a concise explanation of the main reasoning that supports your answer.

## bbc_classification_v1

Classify the following news article into exactly one category: business, entertainment, politics, sport, or tech. Provide the category first, then explain the main textual evidence supporting your classification.

## oulad_decision_v1

Given the following student profile, classify the likely academic outcome as one of: Distinction, Pass, Fail, or Withdrawn. Provide the outcome first, then explain the main factors supporting your classification.
"""
    out_path.write_text(text, encoding="utf-8")


def simple_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [clean_text(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(benchmark: pd.DataFrame, out_path: Path) -> None:
    composition = benchmark.groupby(["task_family", "source_dataset"]).size().reset_index(name="n")
    labels = benchmark.groupby(["task_family", "expected_output"]).size().reset_index(name="n")
    lines = [
        "# Stage 01 Prompt Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Benchmark Composition",
        "",
        simple_markdown_table(composition),
        "",
        "## Label Distribution",
        "",
        simple_markdown_table(labels),
        "",
        "## Methodological Rationale",
        "",
        "The benchmark is intentionally heterogeneous. It evaluates whether explanation consistency changes across open-ended reasoning, controlled text classification, and structured decision-support tasks.",
        "",
        "The datasets are not merged as raw data. Instead, they are normalized into a common prompt schema with source traceability and a shared answer-plus-explanation instruction.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    final_dir = ROOT / "02_prompt_benchmark" / "final"
    pools_dir = ROOT / "02_prompt_benchmark" / "candidate_pools"
    templates_dir = ROOT / "02_prompt_benchmark" / "templates"
    reports_dir = ROOT / "06_reports" / "stage_reports"
    logs_dir = ROOT / "07_logs" / "run_logs"
    manifest_dir = ROOT / "07_logs" / "manifests"
    for folder in [final_dir, pools_dir, templates_dir, reports_dir, logs_dir, manifest_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    benchmark, pools = make_records()

    csv_path = final_dir / "explanation_consistency_prompt_benchmark_v1.csv"
    jsonl_path = final_dir / "explanation_consistency_prompt_benchmark_v1.jsonl"
    json_path = final_dir / "explanation_consistency_prompt_benchmark_v1.json"
    template_path = templates_dir / "prompt_templates_v1.md"
    report_path = reports_dir / "stage_01_prompt_benchmark_report.md"
    log_path = logs_dir / "stage_01_prompt_benchmark_log.json"
    manifest_path = manifest_dir / "stage_01_prompt_benchmark_manifest.json"

    benchmark.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps(benchmark.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in benchmark.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for name, pool in pools.items():
        pool.to_csv(pools_dir / f"{name}_candidate_pool_v1.csv", index=False, encoding="utf-8")

    write_templates(template_path)
    write_report(benchmark, report_path)

    log = {
        "stage": "01_prompt_benchmark",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": int(len(benchmark)),
        "task_family_counts": benchmark["task_family"].value_counts().to_dict(),
        "source_counts": benchmark["source_dataset"].value_counts().to_dict(),
        "outputs": [
            str(csv_path),
            str(json_path),
            str(jsonl_path),
            str(template_path),
            str(report_path),
            str(manifest_path),
        ],
    }
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
