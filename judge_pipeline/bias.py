import logging
from collections import Counter
from typing import List, Dict, Any
from .judge import judge_case, judge_pairwise, weighted_score

logger = logging.getLogger(__name__)


def _pref_from_pairwise(verdict, a_label: str, b_label: str) -> str:
    w = (getattr(verdict, "winner", "") or "")
    w = str(w).strip().lower()
    if w in ("a", a_label.lower()):
        return a_label
    if w in ("b", b_label.lower()):
        return b_label
    return "tie"


def position_bias(
    judge_model, rubric_text: str, cases: List[Dict[str, Any]], weights,
    usage_accumulator: Dict[str, int], n: int = 10,
) -> Dict[str, Any]:
    flips = 0
    total = 0
    subset = cases[:n]
    for i, case in enumerate(subset):
        logger.info("position_bias case %s (%d/%d)", case["id"], i + 1, len(subset))
        p1, u1 = judge_pairwise(
            judge_model, rubric_text, case,
            case["model_output_v1"], case["model_output_v2"],
        )
        p2, u2 = judge_pairwise(
            judge_model, rubric_text, case,
            case["model_output_v2"], case["model_output_v1"],
        )
        for u in (u1, u2):
            usage_accumulator["input_tokens"] += u["input_tokens"]
            usage_accumulator["output_tokens"] += u["output_tokens"]
            usage_accumulator["calls"] += 1

        pref1 = _pref_from_pairwise(p1, "v1", "v2")
        pref2 = _pref_from_pairwise(p2, "v2", "v1")
        if pref1 != pref2:
            flips += 1
        total += 1
    rate = flips / total if total else 0.0
    return {
        "flip_rate": round(rate, 3),
        "flips": flips,
        "total": total,
        "interpretation": "<0.10 acceptable, 0.10-0.30 watch, >0.30 problematic",
    }


def verbosity_bias(
    judge_model, rubric_text: str, cases: List[Dict[str, Any]], weights,
    usage_accumulator: Dict[str, int],
) -> Dict[str, Any]:
    comparisons = []
    for i, case in enumerate(cases):
        long_incorrect = case.get("long_incorrect", "")
        if not long_incorrect:
            continue
        logger.info("verbosity_bias case %s (%d/%d)", case["id"], i + 1, len(cases))
        short_correct = case["model_output_v1"]

        v_short, u1 = judge_case(
            judge_model, rubric_text,
            {**case, "model_output_v1": short_correct}, "model_output_v1",
        )
        v_long, u2 = judge_case(
            judge_model, rubric_text,
            {**case, "model_output_v1": long_incorrect}, "model_output_v1",
        )
        for u in (u1, u2):
            usage_accumulator["input_tokens"] += u["input_tokens"]
            usage_accumulator["output_tokens"] += u["output_tokens"]
            usage_accumulator["calls"] += 1

        s_short = weighted_score(v_short, weights)
        s_long = weighted_score(v_long, weights)
        comparisons.append({
            "case_id": case["id"],
            "short_correct_score": s_short,
            "long_incorrect_score": s_long,
            "long_wins": s_long > s_short,
        })
    long_wins = sum(1 for c in comparisons if c["long_wins"])
    total = len(comparisons)
    return {
        "long_incorrect_wins": long_wins,
        "total": total,
        "rate": round(long_wins / total, 3) if total else 0.0,
        "details": comparisons,
        "interpretation": "Lower is better; 0 = no verbosity bias",
    }


def sycophancy_probe(
    judge_model, rubric_text: str, cases: List[Dict[str, Any]], weights,
    usage_accumulator: Dict[str, int],
) -> Dict[str, Any]:
    failures = []
    checked = 0
    for i, case in enumerate(cases):
        cw = case.get("confidently_wrong", "")
        if not cw:
            continue
        logger.info("sycophancy_probe case %s (%d/%d)", case["id"], i + 1, len(cases))
        checked += 1

        v_correct, u1 = judge_case(
            judge_model, rubric_text,
            {**case, "model_output_v1": case["model_output_v1"]}, "model_output_v1",
        )
        v_cw, u2 = judge_case(
            judge_model, rubric_text,
            {**case, "model_output_v1": cw}, "model_output_v1",
        )
        for u in (u1, u2):
            usage_accumulator["input_tokens"] += u["input_tokens"]
            usage_accumulator["output_tokens"] += u["output_tokens"]
            usage_accumulator["calls"] += 1

        s_correct = weighted_score(v_correct, weights)
        s_cw = weighted_score(v_cw, weights)
        if s_cw > s_correct:
            failures.append({
                "case_id": case["id"],
                "correct_score": s_correct,
                "confidently_wrong_score": s_cw,
            })
    return {
        "confidently_wrong_over_scored": len(failures),
        "total": checked,
        "rate": round(len(failures) / checked, 3) if checked else 0.0,
        "details": failures,
        "interpretation": "Lower is better; 0 = judge resists confident tone",
    }


def score_clustering(v1_scores: List[float], v2_scores: List[float]) -> Dict[str, Any]:
    """Measure whether the judge is clustering scores (low variance, few distinct values)."""
    all_scores = [round(s, 1) for s in v1_scores + v2_scores]
    if not all_scores:
        return {"distinct_scores": 0, "distribution": {}, "stddev": 0.0}
    counter = Counter(all_scores)
    distinct = len(counter)
    n = len(all_scores)
    mean = sum(all_scores) / n
    variance = sum((s - mean) ** 2 for s in all_scores) / n
    stddev = variance ** 0.5
    return {
        "distinct_scores": distinct,
        "distribution": dict(sorted(counter.items())),
        "mean": round(mean, 3),
        "stddev": round(stddev, 3),
        "interpretation": (
            "distinct<=3 = severe clustering; stddev<0.4 = judge lacks discrimination"
        ),
    }


def adversarial_results(
    per_case: List[Dict[str, Any]], cases: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Summarise scores on cases marked adversarial:true."""
    out = []
    by_id = {c["id"]: c for c in per_case}
    for case in cases:
        if not case.get("adversarial"):
            continue
        r = by_id.get(case["id"])
        if not r:
            continue
        out.append({
            "id": case["id"],
            "expected_behavior": case.get("expected_behavior", ""),
            "score_v1": r["score_v1"],
            "verdict_v1": r["verdict_v1"],
            "rationale_v1": r["v1_rationale"],
        })
    return out