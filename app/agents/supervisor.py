from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

# Load env values before initializing agent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TargetAgent(str, Enum):
    RESEARCH_AGENT = "RESEARCH_AGENT"
    VISION_AGENT = "VISION_AGENT"
    DATA_AGENT = "DATA_AGENT"
    VALIDATION_AGENT = "VALIDATION_AGENT"
    DIRECT_LLM = "DIRECT_LLM"

class TaskPlan(BaseModel):
    target_agent: TargetAgent = Field(
        description="The sub-agent best suited to handle the primary query context."
    )
    reasoning: str = Field(
        description="Explanation justifying why the selected target agent was chosen."
    )
    sub_queries: List[str] = Field(
        description="List of specific sub-queries or tasks formatted for the target agent."
    )
    requires_validation: bool = Field(
        default=True,
        description="Flag indicating if safety guardrails and validation layers must be executed."
    )

# Setup Supervisor Agent with Groq or fallback model
model_name = os.getenv("SUPERVISOR_MODEL_NAME", "groq:llama-3.3-70b-versatile")
supervisor_agent = Agent(
    model_name,
    output_type=TaskPlan,
    retries=3,
    system_prompt=(
        "You are the Supervisor Orchestrator Agent. Your task is to analyze user queries and "
        "delegate them to the correct specialized sub-agent.\n\n"
        "Routing Rules:\n"
        "1. Select DATA_AGENT if the query requires mathematical calculation, statistics, row sorting, "
        "filtering, or operations on tables/CSV files.\n"
        "2. Select VISION_AGENT if the query mentions an image, chart, graph, diagram, figure, or visual crop.\n"
        "3. Select RESEARCH_AGENT if the query asks for standard text lookups, semantic search, or textual explanations.\n"
        "4. Select DIRECT_LLM for general knowledge questions or queries not relating to the documents."
    )
)

def orchestrate_query(user_query: str) -> Dict[str, Any]:
    """
    Orchestrator entrypoint. Analyzes the query, generates a structured TaskPlan,
    and returns routing results for execution delegation.
    """
    logger.info(f"Orchestrator processing query: '{user_query}'")
    
    # Run the supervisor agent to get structured TaskPlan output
    run_result = supervisor_agent.run_sync(user_query)
    plan: TaskPlan = run_result.output
    
    logger.info(f"Routed target: {plan.target_agent} | Reasoning: {plan.reasoning}")
    
    return {
        "user_query": user_query,
        "routing_decision": plan.target_agent.value,
        "reasoning": plan.reasoning,
        "sub_queries": plan.sub_queries,
        "requires_validation": plan.requires_validation
    }
