import json
import logging
from typing import Dict, Any, Tuple
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import ValidationError

from .schemas import JudgeLLMOutput, PairwiseLLMOutput

logger = logging.getLogger(__name__)

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict, impartial evaluator. Score every criterion on a 1-5 scale "
     "using the rubric. You MUST provide evidence-based rationales that quote or "
     "paraphrase the model output. Do not reward length or confident tone; reward "
     "factual accuracy and instruction-following. Return JSON only."),
    ("human",
     "## Rubric (criteria and anchors)\n{rubric}\n\n"
     "## Test case\n"
     "System prompt: {system_prompt}\n"
     "User input: {input}\n"
     "Expected answer (optional): {expected_answer}\n"
     "Model output: {model_output}\n\n"
     "## Task\n"
     "Return a JSON object with `criterion_scores` (one entry per rubric criterion: "
     "{{criterion, score, rationale}}) and `summary_rationale`."),
])

PAIRWISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict, impartial A/B evaluator. Compare Output A and Output B on "
     "the rubric. Prefer correctness over length or confidence. Return JSON only."),
    ("human",
     "## Rubric\n{rubric}\n\n"
     "## Test case\n"
     "System prompt: {system_prompt}\n"
     "User input: {input}\n"
     "Expected answer (optional): {expected_answer}\n\n"
     "## Output A\n{output_a}\n\n"
     "## Output B\n{output_b}\n\n"
     "Return `winner` ('A','B','tie'), `confidence` ('low','medium','high'), "
     "`rationale`, and `criterion_evidence` (list of {{criterion, score, rationale}})."),
])


def get_judge(model_name: str, temperature: float = 0.0) -> ChatMistralAI:
    return ChatMistralAI(
        model=model_name,
        temperature=temperature,
        max_tokens=1200,
        timeout=60,
        max_retries=2,
    )


def _structured(llm: ChatMistralAI, schema):
    """Try JSON-schema structured output, fall back to function calling."""
    try:
        return llm.with_structured_output(schema, method="json_schema")
    except Exception as e:
        logger.warning("json_schema mode unavailable (%s); falling back to function_calling", e)
        return llm.with_structured_output(schema, method="function_calling")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ValidationError, ValueError, json.JSONDecodeError)),
    reraise=True,
)
def judge_case(
    judge_model: ChatMistralAI,
    rubric_text: str,
    case: Dict[str, Any],
    output_field: str = "model_output_v1",
) -> Tuple[JudgeLLMOutput, Dict[str, int]]:
    """Return (verdict, usage). usage has input_tokens/output_tokens (0 if unknown)."""
    structured = _structured(judge_model, JudgeLLMOutput)
    chain = JUDGE_PROMPT | structured

    # Use include_raw=True so we can read the AIMessage for token usage
    chain_with_raw = JUDGE_PROMPT | judge_model.with_structured_output(
        JudgeLLMOutput,
        method="json_schema",
        include_raw=True,
    )
    try:
        result = chain_with_raw.invoke({
            "rubric": rubric_text,
            "system_prompt": case.get("system_prompt", ""),
            "input": case["input"],
            "expected_answer": case.get("expected_answer", "N/A"),
            "model_output": case[output_field],
        })
    except Exception:
        # Fallback path without usage
        verdict = chain.invoke({
            "rubric": rubric_text,
            "system_prompt": case.get("system_prompt", ""),
            "input": case["input"],
            "expected_answer": case.get("expected_answer", "N/A"),
            "model_output": case[output_field],
        })
        return verdict, {"input_tokens": 0, "output_tokens": 0}

    parsed = result.get("parsed")
    raw = result.get("raw")
    if parsed is None:
        # If parsing failed but raw exists, raise so tenacity retries
        raise ValueError(f"Structured parse failed: {result.get('parsing_error')}")

    usage = {"input_tokens": 0, "output_tokens": 0}
    if raw is not None and getattr(raw, "usage_metadata", None):
        usage["input_tokens"] = raw.usage_metadata.get("input_tokens", 0)
        usage["output_tokens"] = raw.usage_metadata.get("output_tokens", 0)
    return parsed, usage


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ValidationError, ValueError, json.JSONDecodeError)),
    reraise=True,
)
def judge_pairwise(
    judge_model: ChatMistralAI,
    rubric_text: str,
    case: Dict[str, Any],
    output_a: str,
    output_b: str,
) -> Tuple[PairwiseLLMOutput, Dict[str, int]]:
    structured = _structured(judge_model, PairwiseLLMOutput)
    chain = PAIRWISE_PROMPT | structured
    chain_with_raw = PAIRWISE_PROMPT | judge_model.with_structured_output(
        PairwiseLLMOutput,
        method="json_schema",
        include_raw=True,
    )
    try:
        result = chain_with_raw.invoke({
            "rubric": rubric_text,
            "system_prompt": case.get("system_prompt", ""),
            "input": case["input"],
            "expected_answer": case.get("expected_answer", "N/A"),
            "output_a": output_a,
            "output_b": output_b,
        })
    except Exception:
        verdict = chain.invoke({
            "rubric": rubric_text,
            "system_prompt": case.get("system_prompt", ""),
            "input": case["input"],
            "expected_answer": case.get("expected_answer", "N/A"),
            "output_a": output_a,
            "output_b": output_b,
        })
        return verdict, {"input_tokens": 0, "output_tokens": 0}

    parsed = result.get("parsed")
    raw = result.get("raw")
    if parsed is None:
        raise ValueError(f"Structured parse failed: {result.get('parsing_error')}")

    usage = {"input_tokens": 0, "output_tokens": 0}
    if raw is not None and getattr(raw, "usage_metadata", None):
        usage["input_tokens"] = raw.usage_metadata.get("input_tokens", 0)
        usage["output_tokens"] = raw.usage_metadata.get("output_tokens", 0)
    return parsed, usage


def weighted_score(verdict: JudgeLLMOutput, weights: Dict[str, float]) -> float:
    """Weighted mean over criteria present in the verdict; weights renormalised."""
    by_criterion = {c.criterion: c.score for c in verdict.criterion_scores}
    total_w = 0.0
    total_s = 0.0
    for name, w in weights.items():
        if name in by_criterion:
            total_w += w
            total_s += w * by_criterion[name]
    return round(total_s / total_w, 3) if total_w else 0.0