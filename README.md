# Task-Specific Audit of Explanation Consistency in Large Language Models Using Repeated Queries

This repository contains the reproducibility package for the manuscript **Task-Specific Audit of Explanation Consistency in Large Language Models Using Repeated Queries**.

## Description

The study evaluates whether large language models produce semantically stable explanations when the same prompt is queried repeatedly under controlled conditions. The protocol uses three task families: open-ended reasoning, BBC text classification, and OULAD-based structured decision support. Two OpenAI model variants were queried five times per prompt, producing 1,000 normalized model responses and 200 case-model metric observations.

The repository is designed as supplementary material for PeerJ. It provides the benchmark, normalized outputs, case-level metrics, aggregate metrics, statistical analyses, representative cases, figures/tables, scripts, manifests, and references required to reproduce the reported results.

## Repository Structure

```text
S1_prompt_benchmark/          Final benchmark and prompt templates
S2_normalized_model_outputs/  Normalized model answers and explanations
S3_case_level_metrics/        ECS, EVR, and EDI metrics by prompt-model case
S4_figures_and_tables/        Paper-ready figures and table data
S5_representative_cases/      High-consistency, intermediate, and high-drift cases
S6_statistical_analysis/      Inferential statistics, effect sizes, and confidence intervals
S7_reproducibility_scripts/   Python scripts used across experimental stages
S8_experimental_manifest/     Machine-readable run manifests
references/                   BibTeX bibliography used in the manuscript
```

## Dataset Information

The benchmark includes 100 prompts distributed as follows:

- 35 open-ended reasoning prompts derived from public conversational/instructional data sources.
- 35 text classification prompts based on BBC news categories.
- 30 structured decision-support prompts derived from OULAD-style academic records.

Each benchmark record includes a case identifier, source dataset label, task family, expected output when applicable, prompt template, and final prompt sent to the model.

The file `S2_normalized_model_outputs/openai_multimodel_full_normalized_outputs_v1.csv` contains 1,000 normalized responses with the following core fields:

- `case_id`
- `task_family`
- `model_slot`
- `model_id`
- `run`
- `final_answer`
- `explanation`

The file `S3_case_level_metrics/openai_multimodel_full_case_metrics_v1.csv` contains 200 case-model observations with the metrics used in the manuscript.

## Code Information

The scripts in `S7_reproducibility_scripts/` implement the main stages:

1. Build the prompt benchmark.
2. Prepare model execution matrices.
3. Execute model calls through the OpenAI API.
4. Normalize raw model outputs into answer/explanation fields.
5. Compute explanation consistency metrics.
6. Run inferential statistical analysis.

The repository does not include API keys. Model execution scripts expect credentials through environment variables when rerunning API calls.

## Requirements

The analysis was implemented in Python. Main libraries:

- Python 3.12
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib

Optional for document/table workflows:

- python-docx
- openpyxl

Install a minimal environment with:

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

## Usage Instructions

To inspect the benchmark:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('S1_prompt_benchmark/explanation_consistency_prompt_benchmark_v1.csv')
print(df['task_family'].value_counts())
print(df.head())
PY
```

To inspect the normalized outputs:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('S2_normalized_model_outputs/openai_multimodel_full_normalized_outputs_v1.csv')
print(df.shape)
print(df[['case_id', 'task_family', 'model_id', 'run']].head())
PY
```

To inspect case-level metrics:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('S3_case_level_metrics/openai_multimodel_full_case_metrics_v1.csv')
print(df[['task_family', 'model_id', 'ECS', 'EVR', 'EDI']].head())
print(df.groupby(['task_family', 'model_id'])[['ECS', 'EVR', 'EDI']].mean())
PY
```

## Methodology Summary

For each prompt-model pair, five explanations were generated. Each explanation was normalized and represented using TF-IDF with lowercase preprocessing, English stop-word removal, L2 normalization, and a maximum feature budget of 10,000. The effective vocabulary size in the full experiment was 4,652 features. Cosine similarity between repeated explanations was used to compute:

- **Explanation Consistency Score (ECS):** mean pairwise similarity across all repetitions.
- **Explanation Variability Rate (EVR):** mean dissimilarity between consecutive repetitions.
- **Explanation Drift Index (EDI):** mean dissimilarity between the first explanation and subsequent explanations.

Statistical validation used nonparametric tests, including Kruskal-Wallis tests and paired Wilcoxon tests, with effect-size reporting where applicable.

## Model and Execution Information

The final experiment evaluated two OpenAI model identifiers recorded in the experimental artifacts:

- `gpt-5.4`
- `gpt-5.4-mini`

Inference parameters:

- repetitions per prompt-model pair: 5
- temperature: 0.7
- maximum output tokens: 700
- full normalized responses: 1,000
- case-model metric observations: 200

## Data Availability Statement for Manuscript

Suggested manuscript text:

> The prompt benchmark, normalized model outputs, case-level metrics, inferential analysis tables, representative cases, figures, scripts, and experimental manifests are available in the associated GitHub repository: [repository URL to be inserted after publication]. The repository contains all non-sensitive artifacts required to reproduce the reported ECS, EVR, and EDI results. API keys are not included.

## Code Availability Statement for Manuscript

Suggested manuscript text:

> The Python scripts used to construct the benchmark, normalize model outputs, compute the ECS, EVR, and EDI metrics, and run the inferential analyses are provided in the supplementary GitHub repository: [repository URL to be inserted after publication].

## Use of Artificial Intelligence

Suggested manuscript/acknowledgement text:

> Generative AI tools were used as editorial and programming support during manuscript preparation and reproducibility packaging. The tools assisted with drafting, grammar revision, code organization, and documentation. All scientific decisions, experimental design choices, data interpretation, statistical analyses, and final manuscript content were reviewed and approved by the authors.

Suggested confidential note for PeerJ staff:

> AI-assisted tools were used to support language editing, code organization, documentation drafting, and reproducibility-package preparation. The tools included OpenAI ChatGPT/Codex through the OpenAI platform (https://openai.com/). The authors reviewed and validated all generated text, code, analyses, and interpretations. AI tools were not treated as authors and did not make independent scientific decisions.

## Citations

If using this repository, cite the associated manuscript:

> Task-Specific Audit of Explanation Consistency in Large Language Models Using Repeated Queries.

The manuscript bibliography is provided in `references/explanation_consistency_published_verified_ieee_v5.bib`.

## License and Reuse

The code in this repository is released for scholarly reproducibility. Source datasets retain their original licenses and terms of use. Model outputs and derived metrics are provided as supplementary research artifacts for verification of the manuscript results.

See `DATA_LICENSES.md` for dataset-specific notes.
