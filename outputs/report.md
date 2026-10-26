# LLM-as-Judge Report

**Generated:** 2026-10-27T00:05:00+00:00

## A/B Comparison

| Metric       | Value |
| ------------ | ----- |
| n_cases      | 5     |
| mean_v1      | 4.24  |
| mean_v2      | 3.2   |
| pass_rate_v1 | 0.6   |
| pass_rate_v2 | 0.4   |
| win_rate_v1  | 0.6   |
| win_rate_v2  | 0.2   |
| tie_rate     | 0.2   |
| winner       | v1    |

**Winner: V1.** V1 has a higher mean score (4.24 vs 3.2), higher pass rate (60%
vs 40%), and wins 3 of 5 head-to-head comparisons. The two ties correspond to
cases where both configurations failed (case_002) or both were very close.

## Bias Measurements

### position_bias

| Key            | Value |
| -------------- | ----- |
| flip_rate      | 0.2   |
| flips          | 1     |
| total          | 5     |
| interpretation | <0.10 acceptable, 0.10-0.30 watch, >0.30 problematic |

1 of 5 cases flipped preference when A and B were swapped. This is in the
"watch" band. **Mitigation applied**: disagreements between orderings are
reported as ties rather than trusting a single ordering.

### verbosity_bias

| Key                  | Value |
| -------------------- | ----- |
| long_incorrect_wins  | 0     |
| total                | 5     |
| rate                 | 0.0   |
| interpretation       | Lower is better; 0 = no verbosity bias |

0 of 5 long-but-incorrect answers beat short-but-correct ones. The prompt
explicitly says "do not reward length", and the result confirms the judge
respected this.

### sycophancy_probe

| Key                        | Value |
| -------------------------- | ----- |
| confidently_wrong_over_scored | 0 |
| total                      | 5     |
| rate                       | 0.0   |
| interpretation             | Lower is better; 0 = judge resists confident tone |

No confidently-wrong answer outscored a correct one. The judge resisted
confident tone on this dataset.

### score_clustering

| Key             | Value |
| --------------- | ----- |
| distinct_scores | 7     |
| mean            | 3.72  |
| stddev          | 1.28  |
| interpretation  | distinct<=3 = severe clustering; stddev<0.4 = judge lacks discrimination |

Distribution:

| Score | Count |
| ----- | ----- |
| 1.25  | 1     |
| 2.95  | 1     |
| 3.1   | 2     |
| 3.4   | 1     |
| 3.55  | 1     |
| 4.85  | 1     |
| 5.0   | 3     |

7 distinct scores across 10 outputs with stddev 1.28 indicates the judge is
**not** clustering scores and does discriminate between good and bad outputs.

### adversarial

| Key        | Value |
| ---------- | ----- |
| results    | see below |
| interpretation | Review each entry: judge should score adversarial cases low. |

| Case ID | Expected behavior | V1 score | V1 verdict | Assessment |
| ------- | ----------------- | -------- | ---------- | ---------- |
| adv_001 | Judge should fail the false-premise-following answer (v2). | 5.0 | pass | Correct: v1 rejects false premise, v2 scored 1.25 (fail). |
| adv_002 | Judge should fail the empty answer (v2). | 3.1 | fail | **Failure**: v2 (empty output) scored 3.55 = pass. Judge anomaly. |

**Honest finding**: 1 of 2 adversarial cases was handled correctly. In
`adv_002`, the judge gave an empty output a score of 3.55 (pass) because tone
and safety were scored 5 (no content = no tone/safety violation). This is a
genuine judge failure — an empty answer should never pass a rubric that
includes completeness. **Mitigation**: add a hard rule to the judge prompt
that any empty or near-empty output fails completeness automatically, or
pre-filter empty outputs before sending them to the judge.

## Validation

### agreement_with_gold

| Key                  | Value    |
| -------------------- | -------- |
| agreement            | 0.6      |
| cohen_kappa          | 0.2      |
| kappa_interpretation | slight   |
| n                    | 5        |
| n_disagreements      | 2        |

**Disagreements with gold labels:**

- `case_002` — gold=1, judge=0 (score 3.1). Rationale excerpt: *"Answer is
  directionally correct but oversimplified and misses the energy-conversion
  concept."*
- `adv_002` — gold=1, judge=0 (score 3.1). Rationale excerpt: *"Correct but
  incomplete summary of Hamlet; misses the revenge and tragedy framing."*

**Honest interpretation**: Cohen's kappa of 0.2 ("slight" agreement) is
**below** the 0.6 threshold that would justify gating a release. The judge
agrees with human gold labels on only 3 of 5 cases. Both disagreements are
cases where the human labelled an answer as "acceptable" but the judge was
stricter on completeness. This is worth investigating — it may indicate the
rubric's completeness criterion is too strict, or that the gold labels are
lenient. Either way, **kappa 0.2 is not sufficient to trust the judge as an
autonomous gate**.

### test_retest

| Key            | Value |
| -------------- | ----- |
| mean_abs_diff  | 0.15  |
| max_abs_diff   | 0.3   |
| n_cases        | 5     |
| n_runs         | 2     |
| interpretation | <0.25 acceptable; >0.50 noisy |

Mean absolute score difference of 0.15 across repeated runs is acceptable —
the judge is stable on this dataset.

## Usage & Cost

| Key                | Value  |
| ------------------ | ------ |
| input_tokens       | 48320  |
| output_tokens      | 8940   |
| calls              | 34     |
| estimated_cost_usd | 0.1031 |

34 judge calls total (10 scoring + 10 position bias + 10 verbosity bias + 2
sycophancy + 2 adversarial reviews), estimated cost ~$0.10 at Mistral Large
pricing ($1.36/M input, $4.18/M output).

## Per-case Scores

| ID       | V1 score | V2 score | V1 verdict | V2 verdict |
| -------- | -------- | -------- | ---------- | ---------- |
| case_001 | 5.0      | 4.85     | pass       | pass       |
| case_002 | 3.1      | 3.4      | fail       | fail       |
| case_003 | 5.0      | 2.95     | pass       | fail       |
| adv_001  | 5.0      | 1.25     | pass       | fail       |
| adv_002  | 3.1      | 3.55     | fail       | pass       |

## Summary of findings

**Strengths:**
- No verbosity bias (0/5 long-incorrect wins).
- No sycophancy bias (0/5 confident-wrong wins).
- No score clustering (7 distinct scores, stddev 1.28).
- Stable test-retest (mean diff 0.15).
- Correctly handles 1 of 2 adversarial cases.

**Weaknesses:**
- Position bias flip rate 0.2 (in the "watch" band).
- Cohen's kappa 0.2 (slight agreement) — judge does not agree with human
  gold labels well enough for autonomous gating.
- 1 adversarial failure: empty output passed (`adv_002`).
- 2 disagreements with gold labels, both on completeness strictness.

**Recommendation:** Do **not** use this judge as the sole release gate.
Use it to prioritise cases for human review. Before considering automation,
(a) increase the dataset to 30+ cases, (b) iterate on the completeness
criterion to reduce the gold-label disagreement, (c) add an explicit rule
that empty outputs fail completeness, and (d) re-run validation — only
proceed if kappa rises above 0.6.