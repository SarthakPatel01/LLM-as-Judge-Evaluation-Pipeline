# LLM‑as‑Judge Pipeline (Mistral + LangChain)

A command‑line pipeline that evaluates model outputs against a rubric, compares two configurations, probes for judge bias, and produces a report.

## Setup

pip install -r requirements.txt# LLM-as-Judge Pipeline (Mistral + LangChain)

This is my submission for the LLM-as-Judge project. It's a command-line pipeline
that scores model outputs against a rubric, compares two prompt configurations,
runs bias probes, validates the judge, and writes a report. I built it with
LangChain and Mistral because those were the tools suggested in the assignment,
and I wanted something I could actually run end-to-end on my laptop.

The whole thing is designed so you can reproduce every number in my final report
with one command (see below).

## What it does

1. **Reads test cases from JSON** (inputs, system prompts, two model outputs,
   optional expected answer, optional gold label, plus fields for the bias probes).
2. **Asks Mistral to score each output** on six rubric criteria — correctness,
   faithfulness, completeness, instruction-following, tone, and safety — with an
   evidence-based rationale per criterion.
3. **Validates the LLM's output with Pydantic** and retries on malformed JSON.
   There's a fallback from JSON-schema structured output to function calling in
   case the model/API doesn't support the former.
4. **Compares Prompt V1 vs V2** on the same cases (pass rate, mean score, win
   rate, tie rate, winner).
5. **Runs bias probes**: position flip rate, verbosity bias, sycophancy probe,
   score clustering, and adversarial case review.
6. **Validates the judge** against gold labels (agreement + Cohen's kappa) and
   via test-retest consistency.
7. **Writes a report** as Markdown, JSON, CSV, and a raw log of every judge call.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# put your Mistral key in .env