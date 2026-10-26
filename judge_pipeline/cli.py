import argparse
import json
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .dataset import load_cases
from .judge import get_judge, judge_case, weighted_score
from .bias import (
    position_bias, verbosity_bias, sycophancy_probe,
    score_clustering, adversarial_results,
)
from .validation import agreement_with_gold, test_retest_consistency
from .report import build_report

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("judge")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge pipeline (Mistral + LangChain)")
    parser.add_argument("--data", default="data/test_cases.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "mistral-large-latest"))
    parser.add_argument("--skip-bias", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Only score cases")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    weights = config["weights"]
    threshold = config["pass_threshold"]
    rubric_text = json.dumps(config["criteria"], indent=2)

    cases = load_cases(args.data)
    judge_model = get_judge(args.judge_model)

    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    per_case = []
    raw_logs = []
    v1_scores, v2_scores = [], []

    for i, case in enumerate(cases):
        log.info("scoring case %s (%d/%d)", case["id"], i + 1, len(cases))
        v1, u1 = judge_case(judge_model, rubric_text, case, "model_output_v1")
        v2, u2 = judge_case(judge_model, rubric_text, case, "model_output_v2")
        for u in (u1, u2):
            usage["input_tokens"] += u["input_tokens"]
            usage["output_tokens"] += u["output_tokens"]
            usage["calls"] += 1

        s1 = weighted_score(v1, weights)
        s2 = weighted_score(v2, weights)
        v1_scores.append(s1)
        v2_scores.append(s2)
        per_case.append({
            "id": case["id"],
            "score_v1": s1,
            "score_v2": s2,
            "verdict_v1": "pass" if s1 >= threshold else "fail",
            "verdict_v2": "pass" if s2 >= threshold else "fail",
            "v1_rationale": v1.summary_rationale,
            "v2_rationale": v2.summary_rationale,
            "v1_criteria": [c.model_dump() for c in v1.criterion_scores],
            "v2_criteria": [c.model_dump() for c in v2.criterion_scores],
        })
        raw_logs.append({"case_id": case["id"], "v1": v1.model_dump(), "v2": v2.model_dump()})

    n = len(per_case)
    ab_summary = {
        "n_cases": n,
        "mean_v1": round(sum(v1_scores) / n, 3),
        "mean_v2": round(sum(v2_scores) / n, 3),
        "pass_rate_v1": round(sum(1 for s in v1_scores if s >= threshold) / n, 3),
        "pass_rate_v2": round(sum(1 for s in v2_scores if s >= threshold) / n, 3),
        "win_rate_v1": round(sum(1 for a, b in zip(v1_scores, v2_scores) if a > b) / n, 3),
        "win_rate_v2": round(sum(1 for a, b in zip(v1_scores, v2_scores) if b > a) / n, 3),
        "tie_rate": round(sum(1 for a, b in zip(v1_scores, v2_scores) if a == b) / n, 3),
        "winner": ("v1" if sum(v1_scores) > sum(v2_scores)
                   else "v2" if sum(v2_scores) > sum(v1_scores) else "tie"),
    }

    bias_results = {}
    if not args.skip_bias and not args.fast:
        log.info("running position_bias")
        bias_results["position_bias"] = position_bias(
            judge_model, rubric_text, cases, weights, usage)
        log.info("running verbosity_bias")
        bias_results["verbosity_bias"] = verbosity_bias(
            judge_model, rubric_text, cases, weights, usage)
        log.info("running sycophancy_probe")
        bias_results["sycophancy_probe"] = sycophancy_probe(
            judge_model, rubric_text, cases, weights, usage)
        log.info("computing score_clustering")
        bias_results["score_clustering"] = score_clustering(v1_scores, v2_scores)
        bias_results["adversarial"] = {
            "results": adversarial_results(per_case, cases),
            "interpretation": "Review each entry: judge should score adversarial cases low.",
        }

    validation_results = {}
    if not args.skip_validation and not args.fast:
        gold_cases = [c for c in cases if "gold_label" in c]
        if gold_cases:
            log.info("running agreement_with_gold")
            scores_for_gold = [s for c, s in zip(cases, v1_scores) if "gold_label" in c]
            gold = [c["gold_label"] for c in gold_cases]
            gold_per_case = [r for c, r in zip(cases, per_case) if "gold_label" in c]
            validation_results["agreement_with_gold"] = agreement_with_gold(
                scores_for_gold, gold, threshold, gold_per_case)
        log.info("running test_retest_consistency")
        validation_results["test_retest"] = test_retest_consistency(
            judge_model, rubric_text, cases, weights, usage, n_runs=2)

    build_report(config, per_case, ab_summary, bias_results,
                 validation_results, raw_logs, usage, Path(args.output_dir))
    print(f"\nReport written to {args.output_dir}/report.md, report.json, per_case.csv")


if __name__ == "__main__":
    main()