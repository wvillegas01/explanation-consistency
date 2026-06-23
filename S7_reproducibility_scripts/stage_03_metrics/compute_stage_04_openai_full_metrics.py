from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\Auditing Explainability-corto\auditoria")
NORMALIZED_PATH = ROOT / "03_model_outputs" / "normalized" / "openai_multimodel" / "openai_multimodel_full_normalized_outputs_v1.csv"
METRICS_DIR = ROOT / "04_metrics" / "metric_tables" / "openai_multimodel"
TABLES_DIR = ROOT / "05_figures_tables" / "tables"
CASE_METRICS_PATH = METRICS_DIR / "openai_multimodel_full_case_metrics_v1.csv"
AGG_METRICS_PATH = METRICS_DIR / "openai_multimodel_full_aggregate_metrics_v1.csv"
COMPARISON_PATH = TABLES_DIR / "table_openai_multimodel_full_comparison_v1.csv"
REPORT_PATH = ROOT / "06_reports" / "stage_reports" / "stage_04_openai_full_metrics_report.md"
LOG_PATH = ROOT / "07_logs" / "run_logs" / "stage_04_openai_full_metrics_log.json"
MANIFEST_PATH = ROOT / "07_logs" / "manifests" / "stage_04_openai_full_metrics_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mean_pairwise(sim: np.ndarray) -> float:
    values = [sim[i, j] for i, j in itertools.combinations(range(sim.shape[0]), 2)]
    return float(np.mean(values)) if values else float("nan")


def mean_consecutive_distance(sim: np.ndarray) -> float:
    values = [1.0 - sim[i, i + 1] for i in range(sim.shape[0] - 1)]
    return float(np.mean(values)) if values else float("nan")


def mean_first_drift(sim: np.ndarray) -> float:
    values = [1.0 - sim[0, i] for i in range(1, sim.shape[0])]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(NORMALIZED_PATH)
    df["explanation"] = df["explanation"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=10000)
    matrix = vectorizer.fit_transform(df["explanation"])
    row_pos = {idx: pos for pos, idx in enumerate(df.index)}

    rows = []
    group_cols = ["case_id", "task_family", "source_dataset", "expected_output", "model_slot", "provider", "model_id"]
    for key, group in df.sort_values("repetition").groupby(group_cols, dropna=False):
        if len(group) < 5:
            continue
        positions = [row_pos[idx] for idx in group.index]
        sim = cosine_similarity(matrix[positions])
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_repetitions": int(len(group)),
                "ECS": mean_pairwise(sim),
                "EVR": mean_consecutive_distance(sim),
                "EDI": mean_first_drift(sim),
            }
        )
    case_metrics = pd.DataFrame(rows)
    case_metrics.to_csv(CASE_METRICS_PATH, index=False, encoding="utf-8")

    aggregate = (
        case_metrics.groupby(["model_id", "task_family"])
        .agg(
            n_cases=("case_id", "count"),
            ECS_mean=("ECS", "mean"),
            ECS_std=("ECS", "std"),
            EVR_mean=("EVR", "mean"),
            EVR_std=("EVR", "std"),
            EDI_mean=("EDI", "mean"),
            EDI_std=("EDI", "std"),
        )
        .reset_index()
    )
    aggregate.to_csv(AGG_METRICS_PATH, index=False, encoding="utf-8")

    comparison = aggregate.copy()
    for col in ["ECS_mean", "ECS_std", "EVR_mean", "EVR_std", "EDI_mean", "EDI_std"]:
        comparison[col] = comparison[col].round(4)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8")

    lines = [
        "# Stage 04 OpenAI Full Metrics Report",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Normalized rows: {len(df)}",
        f"Case-level metric rows: {len(case_metrics)}",
        "",
        "| model_id | task_family | n_cases | ECS_mean | EVR_mean | EDI_mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row['model_id']} | {row['task_family']} | {row['n_cases']} | "
            f"{row['ECS_mean']:.4f} | {row['EVR_mean']:.4f} | {row['EDI_mean']:.4f} |"
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    log = {
        "stage": "04_openai_full_metrics",
        "timestamp": utc_now(),
        "normalized_rows": int(len(df)),
        "case_metric_rows": int(len(case_metrics)),
        "outputs": [
            str(CASE_METRICS_PATH),
            str(AGG_METRICS_PATH),
            str(COMPARISON_PATH),
            str(REPORT_PATH),
            str(MANIFEST_PATH),
        ],
    }
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
