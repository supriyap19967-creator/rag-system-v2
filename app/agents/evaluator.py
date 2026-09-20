from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Dict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

# Load env variables
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationResult(BaseModel):
    is_faithful: bool = Field(
        description="True if the response is faithful to the context without hallucinations or altered facts."
    )
    score: float = Field(
        description="Faithfulness score between 0.0 (completely hallucinated) and 1.0 (fully faithful)."
    )
    reasoning: str = Field(
        description="Explanation assessing the quality of the response, citing facts, and mapping supporting/contradicting context."
    )

# Setup Evaluator Agent
model_name = os.getenv("EVALUATOR_MODEL_NAME", "groq:llama-3.3-70b-versatile")
evaluator_agent = Agent(
    model_name,
    output_type=EvaluationResult,
    retries=3,
    system_prompt=(
        "You are an expert Multimodal Context Evaluator assessing the quality of a RAG system.\n\n"
        "Evaluation Guidelines:\n"
        "1. Treat structured CSV rows and OCR chart extractions as ground truth context.\n"
        "2. Numeric calculations (averages, totals, growth rates) derived correctly from provided numeric contexts must be scored as Faithful (1.0).\n"
        "3. Do not penalize responses for reformatting raw tabular data into concise bullet points or data cards, provided the numerical facts remain accurate.\n"
        "4. Penalize hallucinated facts, altered numbers, or claims not supported by either the textual chunks or the visual/CSV metadata."
    )
)

def execute_evaluation(
    user_query: str,
    retrieved_context: str,
    agent_response: str
) -> Dict[str, Any]:
    """
    Entrypoint function to evaluate agent response faithfulness against retrieved context.
    """
    prompt = (
        f"Evaluate the faithfulness of the response against the retrieved context:\n\n"
        f"- User Query: {user_query}\n\n"
        f"- Ground Truth Context (CSV / Text / OCR):\n{retrieved_context}\n\n"
        f"- Agent Response to Assess:\n{agent_response}\n"
    )
    run_result = evaluator_agent.run_sync(prompt)
    output: EvaluationResult = run_result.output
    return {
        "is_faithful": output.is_faithful,
        "score": output.score,
        "reasoning": output.reasoning
    }
