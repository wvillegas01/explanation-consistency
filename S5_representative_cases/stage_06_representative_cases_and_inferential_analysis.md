# Representative Cases and Inferential Analysis

## Representative cases

| Case | Type | Task family | Model | ECS | EVR | EDI |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A | High explanation consistency | open_ended_reasoning | gpt-5.4 | 0.8957 | 0.1004 | 0.0953 |
| B | Intermediate consistency | structured_decision_support | gpt-5.4 | 0.5005 | 0.4904 | 0.4851 |
| C | High explanation drift | open_ended_reasoning | gpt-5.4-mini | 0.1262 | 0.9188 | 0.8832 |

## Significant paired model comparisons (Wilcoxon p < .05)
| Task family | Metric | Mean diff | p | Rank-biserial |
| --- | --- | ---: | ---: | ---: |
| open_ended_reasoning | ECS | -0.0607 | 0.0008382 | -0.6254 |
| open_ended_reasoning | EVR | 0.0685 | 0.001253 | 0.6063 |
| open_ended_reasoning | EDI | 0.0580 | 0.0124 | 0.4794 |
| structured_decision_support | ECS | -0.0435 | 0.0001529 | -0.7462 |
| structured_decision_support | EVR | 0.0378 | 0.003744 | 0.5914 |
| structured_decision_support | EDI | 0.0454 | 0.00664 | 0.5570 |
| text_classification | ECS | 0.0293 | 0.03414 | 0.4095 |
| text_classification | EDI | -0.0366 | 0.0166 | -0.4603 |

## Kruskal-Wallis task-family tests
| Model | Metric | H | p |
| --- | --- | ---: | ---: |
| gpt-5.4 | ECS | 41.6610 | 8.983e-10 |
| gpt-5.4 | EVR | 44.3512 | 2.34e-10 |
| gpt-5.4 | EDI | 34.6565 | 2.981e-08 |
| gpt-5.4-mini | ECS | 57.0468 | 4.097e-13 |
| gpt-5.4-mini | EVR | 55.3679 | 9.484e-13 |
| gpt-5.4-mini | EDI | 52.9147 | 3.234e-12 |
| all_models | ECS | 100.2665 | 1.688e-22 |
| all_models | EVR | 100.6804 | 1.373e-22 |
| all_models | EDI | 88.1807 | 7.109e-20 |