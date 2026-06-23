from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\Auditing Explainability-corto\auditoria")
BENCHMARK_PATH = ROOT / "02_prompt_benchmark" / "final" / "explanation_consistency_prompt_benchmark_v1.csv"
RAW_DIR = ROOT / "03_model_outputs" / "raw" / "openai_multimodel"
REPORT_PATH = ROOT / "06_reports" / "stage_reports" / "stage_02_openai_multimodel_plan.md"
LOG_PATH = ROOT / "07_logs" / "run_logs" / "stage_02_openai_multimodel_plan_log.json"
MANIFEST_PATH = ROOT / "07_logs" / "manifests" / "stage_02_openai_multimodel_plan_manifest.json"


MODELS = [
    {
        "model_slot": "gpt54",
        "provider": "openai",
        "model_id": "gpt-5.4",
        "api_key_env": "OPENAI_API_KEY",
        "enabled": True,
        "rationale": "More capable and more expensive OpenAI frontier model variant.",
    },
    {
        "model_slot": "gpt54mini",
        "provider": "openai",
        "model_id": "gpt-5.4-mini",
        "api_key_env": "OPENAI_API_KEY",
        "enabled": True,
        "rationale": "Lower-cost, lower-latency OpenAI model variant.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compact_prompt(prompt: str) -> str:
    return (
        "Output constraints for this explanation-consistency audit:\n"
        "- Start with 'Final answer:' followed by one concise answer.\n"
        "- Then write 'Explanation:' followed by a concise explanation.\n"
        "- Keep the entire response between 120 and 180 words.\n"
        "- Do not use tables, markdown headings, or long step-by-step lists.\n\n"
        "Task:\n"
        f"{prompt}"
    )


def select_pilot_cases(benchmark: pd.DataFrame) -> pd.DataFrame:
    open_cases = benchmark[benchmark["task_family"] == "open_ended_reasoning"].head(4)
    text_cases = benchmark[benchmark["task_family"] == "text_classification"].head(3)
    structured_cases = benchmark[benchmark["task_family"] == "structured_decision_support"].head(3)
    return pd.concat([open_cases, text_cases, structured_cases], ignore_index=True)


def build_matrix(cases: pd.DataFrame, repetitions: int, scope: str) -> pd.DataFrame:
    rows = []
    for _, case in cases.iterrows():
        for model in MODELS:
            for repetition in range(1, repetitions + 1):
                run_id = f"{scope}__{case['case_id']}__{model['model_slot']}__r{repetition:02d}"
                rows.append(
                    {
                        "run_id": run_id,
                        "case_id": case["case_id"],
                        "source_dataset": case["source_dataset"],
                        "task_family": case["task_family"],
                        "expected_output": case["expected_output"],
                        "model_slot": model["model_slot"],
                        "provider": model["provider"],
                        "model_id": model["model_id"],
                        "repetition": repetition,
                        "temperature": 0.7,
                        "max_output_tokens": 700,
                        "final_prompt": compact_prompt(case["final_prompt"]),
                        "status": "pending_openai_multimodel",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(BENCHMARK_PATH)
    pilot_cases = select_pilot_cases(benchmark)
    pilot_matrix = build_matrix(pilot_cases, repetitions=3, scope="pilot")
    full_matrix = build_matrix(benchmark, repetitions=5, scope="full")

    config_path = RAW_DIR / "openai_multimodel_config_v1.json"
    pilot_cases_path = RAW_DIR / "openai_multimodel_pilot_cases_v1.csv"
    pilot_matrix_path = RAW_DIR / "openai_multimodel_pilot_run_matrix_v1.csv"
    full_matrix_path = RAW_DIR / "openai_multimodel_full_run_matrix_v1.csv"

    config_path.write_text(json.dumps({"models": MODELS}, indent=2, ensure_ascii=False), encoding="utf-8")
    pilot_cases.to_csv(pilot_cases_path, index=False, encoding="utf-8")
    pilot_matrix.to_csv(pilot_matrix_path, index=False, encoding="utf-8")
    full_matrix.to_csv(full_matrix_path, index=False, encoding="utf-8")

    report = f"""# Stage 02 OpenAI Multi-Model Plan

Generated: {utc_now()}

## Methodological Decision

The short paper will use an OpenAI-only multi-model comparison. This avoids cross-provider operational instability observed with Gemini while preserving a comparative design suitable for an 8-10 page article.

## Models

- `gpt-5.4`
- `gpt-5.4-mini`

These model IDs are listed in the official OpenAI models documentation, and current OpenAI models are available through the Responses API.

## Pilot Matrix

- Cases: {len(pilot_cases)}
- Models: {len(MODELS)}
- Repetitions: 3
- Planned calls: {len(pilot_matrix)}

## Full Matrix

- Cases: {len(benchmark)}
- Models: {len(MODELS)}
- Repetitions: 5
- Planned calls: {len(full_matrix)}

## Prompt Control

All prompts use a compact output constraint requiring `Final answer:` and `Explanation:` in 120-180 words.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    log = {
        "stage": "02_openai_multimodel_plan",
        "timestamp": utc_now(),
        "pilot_calls": int(len(pilot_matrix)),
        "full_calls": int(len(full_matrix)),
        "models": [m["model_id"] for m in MODELS],
        "outputs": [
            str(config_path),
            str(pilot_cases_path),
            str(pilot_matrix_path),
            str(full_matrix_path),
            str(REPORT_PATH),
            str(MANIFEST_PATH),
        ],
    }
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
