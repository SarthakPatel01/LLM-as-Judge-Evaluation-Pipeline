import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from tabulate import tabulate

# Mistral Large pricing (USD per 1M tokens) — update if pricing changes
PRICE_INPUT_PER_M = 1.36
PRICE_OUTPUT_PER_M = 4.18


def estimate_cost(usage: dict) -> dict:
    in_cost = usage["input_tokens"] / 1_000_000 * PRICE_INPUT_PER_M
    out_cost = usage["output_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_M
    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "calls": usage["calls"],
        "estimated_cost_usd": round(in_cost + out_cost, 4),
    }


def build_report(
    config: dict,
    per_case: list,
    ab_summary: dict,
    bias_results: dict,
    validation_results: dict,
    raw_logs: list,
    usage: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    cost = estimate_cost(usage)

    # ---- JSON ----
    (output_dir / "report.json").write_text(json.dumps({
        "timestamp": ts,
        "config": config,
        "per_case": per_case,
        "ab_comparison": ab_summary,
        "bias_measurements": bias_results,
        "validation": validation_results,
        "usage_and_cost": cost,
    }, indent=2))

    # ---- CSV ----
    with open(output_dir / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "score_v1", "score_v2", "verdict_v1", "verdict_v2"],
        )
        writer.writeheader()
        for r in per_case:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # ---- Raw logs ----
    (output_dir / "raw_judge_logs.json").write_text(json.dumps(raw_logs, indent=2))

    # ---- Markdown ----
    md = ["# LLM-as-Judge Report", "", f"**Generated:** {ts}", ""]

    md.append("## A/B Comparison\n")
    md.append(tabulate([[k, v] for k, v in ab_summary.items()],
                       headers=["Metric", "Value"], tablefmt="github"))

    md.append("\n## Bias Measurements\n")
    for probe, data in bias_results.items():
        md.append(f"### {probe}\n")
        flat = {k: v for k, v in data.items() if k != "details"}
        md.append(tabulate([[k, v] for k, v in flat.items()],
                           headers=["Key", "Value"], tablefmt="github"))
        if "details" in data and data["details"]:
            md.append("\n**Details:**\n")
            md.append(tabulate(
                [list(d.values()) for d in data["details"]],
                headers=list(data["details"][0].keys()),
                tablefmt="github",
            ))
        md.append("")

    md.append("\n## Validation\n")
    for check, data in validation_results.items():
        md.append(f"### {check}\n")
        flat = {k: v for k, v in data.items() if k not in ("disagreements",)}
        md.append(tabulate([[k, v] for k, v in flat.items()],
                           headers=["Key", "Value"], tablefmt="github"))
        if data.get("disagreements"):
            md.append("\n**Disagreements with gold labels:**\n")
            for d in data["disagreements"]:
                md.append(f"- `{d['case_id']}` — gold={d['gold_label']}, "
                          f"judge={d['judge_verdict']} (score {d['judge_score']}). "
                          f"Rationale excerpt: _{d['judge_rationale']}_")
        md.append("")

    md.append("\n## Usage & Cost\n")
    md.append(tabulate([[k, v] for k, v in cost.items()],
                       headers=["Metric", "Value"], tablefmt="github"))

    md.append("\n## Per-case Scores\n")
    md.append(tabulate(
        [[c["id"], c["score_v1"], c["score_v2"], c["verdict_v1"], c["verdict_v2"]]
         for c in per_case],
        headers=["ID", "V1 score", "V2 score", "V1 verdict", "V2 verdict"],
        tablefmt="github",
    ))

    (output_dir / "report.md").write_text("\n".join(md))