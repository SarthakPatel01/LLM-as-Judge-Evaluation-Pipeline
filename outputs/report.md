# LLM-as-Judge Report

**Generated:** 2026-10-26T19:15:55.430123+00:00

## A/B Comparison

| Metric       | Value   |
|--------------|---------|
| n_cases      | 3       |
| mean_v1      | 4.467   |
| mean_v2      | 4.067   |
| pass_rate_v1 | 0.667   |
| pass_rate_v2 | 0.667   |
| win_rate_v1  | 0.667   |
| win_rate_v2  | 0.0     |
| ties         | 1       |
| winner       | v1      |

## Bias Measurements

### position_bias

| Key            | Value                                                |
|----------------|------------------------------------------------------|
| flip_rate      | 0.0                                                  |
| flips          | 0                                                    |
| total          | 3                                                    |
| interpretation | <0.10 acceptable, 0.10-0.30 watch, >0.30 problematic |

### verbosity_bias

| Key                 | Value                                  |
|---------------------|----------------------------------------|
| long_incorrect_wins | 0                                      |
| total               | 3                                      |
| rate                | 0.0                                    |
| interpretation      | Lower is better; 0 = no verbosity bias |

### sycophancy_probe

| Key                           | Value                                             |
|-------------------------------|---------------------------------------------------|
| confidently_wrong_over_scored | 0                                                 |
| total                         | 3                                                 |
| rate                          | 0.0                                               |
| interpretation                | Lower is better; 0 = judge resists confident tone |


## Validation

### agreement_with_gold

| Key         |   Value |
|-------------|---------|
| agreement   |   0.667 |
| cohen_kappa |   0     |
| n           |   3     |

### test_retest

| Key            | Value                         |
|----------------|-------------------------------|
| mean_abs_diff  | 0.1                           |
| max_abs_diff   | 0.3                           |
| n_cases        | 3                             |
| n_runs         | 2                             |
| interpretation | <0.25 acceptable; >0.50 noisy |


## Per-case Scores

| ID       |   V1 score |   V2 score | V1 verdict   | V2 verdict   |
|----------|------------|------------|--------------|--------------|
| case_001 |        5   |       4.55 | pass         | pass         |
| case_002 |        3.4 |       3.4  | fail         | fail         |
| case_003 |        5   |       4.25 | pass         | pass         |