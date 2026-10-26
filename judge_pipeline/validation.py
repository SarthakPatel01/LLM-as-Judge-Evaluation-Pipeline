import logging
from typing import List, Dict, Any
from .judge import judge_case, weighted_score

logger = logging.getLogger(__name__)


def cohen_kappa(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        return 0.0
    n = len(y_true)
    labels = sorted(set(y_true) | set(y_pred))
    p_o = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    p_e = 0.0
    for lbl in labels:
        p_e += (y_true.count(lbl) / n) * (y_pred.count(lbl) / n)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def interpret_kappa(k: float) -> str:
    if k < 0.0:
        return "poor"
    if k <= 0.20:
        return "slight"
    if k <= 0.40:
        return "fair"
    if k <= 0.60:
        return "moderate"
    if k <= 0.80:
        return "substantial"
    return "almost perfect"


def agreement_with_gold(
    judge_scores: List[float], gold_labels: List[int], pass_threshold: float,
    per_case: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(judge_scores) != len(gold_labels) or not judge_scores:
        return {"agreement": 0.0, "cohen_kappa": 0.0, "n": 0}
    judge_bin = [1 if s >= pass_threshold else 0 for s in judge_scores]
    agreement = sum(j == g for j, g in zip(judge_bin, gold_labels)) / len(gold_labels)
    kappa = cohen_kappa(gold_labels, judge_bin)

    # Record disagreements honestly instead of hiding them
    disagreements = []
    for r, g in zip(per_case, gold_labels):
        pred = 1 if r["score_v1"] >= pass_threshold else 0
        if pred != g:
            disagreements.append({
                "case_id": r["id"],
                "gold_label": g,
                "judge_verdict": pred,
                "judge_score": r["score_v1"],
                "judge_rationale": r["v1_rationale"][:400],
            })
    return {
        "agreement": round(agreement, 3),
        "cohen_kappa": round(kappa, 3),
        "kappa_interpretation": interpret_kappa(kappa),
        "n": len(gold_labels),
        "disagreements": disagreements,
        "n_disagreements": len(disagreements),
    }


def test_retest_consistency(
    judge_model, rubric_text: str, cases: List[Dict[str, Any]], weights,
    usage_accumulator: Dict[str, int], n_runs: int = 2,
) -> Dict[str, Any]:
    diffs = []
    subset = cases[:5]
    for i, case in enumerate(subset):
        logger.info("test_retest case %s (%d/%d)", case["id"], i + 1, len(subset))
        scores = []
        for _ in range(n_runs):
            v, u = judge_case(judge_model, rubric_text, case, "model_output_v1")
            usage_accumulator["input_tokens"] += u["input_tokens"]
            usage_accumulator["output_tokens"] += u["output_tokens"]
            usage_accumulator["calls"] += 1
            scores.append(weighted_score(v, weights))
        diffs.append(max(scores) - min(scores))
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    return {
        "mean_abs_diff": round(mean_abs, 3),
        "max_abs_diff": round(max(diffs), 3) if diffs else 0.0,
        "n_cases": len(subset),
        "n_runs": n_runs,
        "interpretation": "<0.25 acceptable; >0.50 noisy",
    }