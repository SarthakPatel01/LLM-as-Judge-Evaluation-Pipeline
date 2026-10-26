from pydantic import BaseModel, Field
from typing import List

class CriterionScore(BaseModel):
    criterion: str = Field(..., description="Name of the criterion (must match rubric keys)")
    score: int = Field(..., ge=1, le=5, description="Score 1-5")
    rationale: str = Field(..., description="Evidence-based justification quoting the output")

class JudgeLLMOutput(BaseModel):
    """What the LLM returns. overall_score/verdict are computed in Python."""
    criterion_scores: List[CriterionScore]
    summary_rationale: str

class PairwiseLLMOutput(BaseModel):
    winner: str = Field(..., description="'A', 'B', or 'tie'")
    confidence: str = Field(..., description="'low', 'medium', or 'high'")
    rationale: str
    criterion_evidence: List[CriterionScore]