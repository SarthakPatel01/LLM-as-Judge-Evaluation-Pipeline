import json
from pathlib import Path
from datetime import datetime, timezone
from tabulate import tabulate


def build_report(
    config: dict,
    per_case: list,
    ab_summary: dict,
    bias_results: dict,
    validation_results: dict,
    raw_logs: list,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    (output_dir / "report.json").write_text(json.dumps({
        "timestamp": ts,
        "config": config,
        "per_case": per_case,
        "ab_comparison": ab_summary,
        "bias_measurements": bias_results,
        "validation": validation_results,
    }, indent=2))

    (output_dir / "raw_judge_logs.json").write_text(json.dumps(raw_logs, indent=2))

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
        md.append("")
    md.append("\n## Validation\n")
    for check, data in validation_results.items():
        md.append(f"### {check}\n")
        md.append(tabulate([[k, v] for k, v in data.items()],
                           headers=["Key", "Value"], tablefmt="github"))
        md.append("")
    md.append("\n## Per-case Scores\n")
    md.append(tabulate(
        [[c["id"], c["score_v1"], c["score_v2"], c["verdict_v1"], c["verdict_v2"]]
         for c in per_case],
        headers=["ID", "V1 score", "V2 score", "V1 verdict", "V2 verdict"],
        tablefmt="github",
    ))
    (output_dir / "report.md").write_text("\n".join(md))