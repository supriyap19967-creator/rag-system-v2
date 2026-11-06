from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict, Optional

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

class SubTask(BaseModel):
    agent: TargetAgent = Field(
        description="The target specialized sub-agent (DATA_AGENT, VISION_AGENT, RESEARCH_AGENT, or DIRECT_LLM)."
    )
    sub_query: str = Field(
        description="Self-contained, isolated query targeted specifically for this agent."
    )
    rationale: str = Field(
        default="",
        description="Why this specific agent is required for this sub-task."
    )

class TaskPlan(BaseModel):
    is_compound: bool = Field(
        default=False,
        description="True if the user query requires cross-modal collaboration across multiple distinct agents (e.g. CSV+Visual, CSV+Text, Text+Visual)."
    )
    target_agent: TargetAgent = Field(
        description="The primary sub-agent if single-modality, or primary coordinating agent if compound."
    )
    tasks: List[SubTask] = Field(
        default_factory=list,
        description="List of 1 to 3 isolated sub-tasks to execute across specialized agents. For single queries, contains 1 sub-task. For compound queries, contains 2 or more sub-tasks."
    )
    comparison_goal: Optional[str] = Field(
        default=None,
        description="For compound queries, the specific analytical correlation, comparison, or synthesis to achieve between sources."
    )
    reasoning: str = Field(
        description="Explanation justifying why the selected target agent(s) and sub-tasks were chosen."
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="List of specific sub-queries or tasks formatted for the target agent(s)."
    )
    requires_validation: bool = Field(
        default=True,
        description="Flag indicating if safety guardrails and validation layers must be executed."
    )

TaskPlan.model_rebuild()

from app.agents.model_factory import get_agent_model

# Setup Supervisor Agent with optimized model
supervisor_model = get_agent_model("SUPERVISOR_MODEL_NAME")
supervisor_agent = Agent(
    supervisor_model,
    output_type=TaskPlan,
    retries=3,
    system_prompt=(
        "You are the Supervisor Orchestrator Agent for an enterprise multimodal RAG system.\n"
        "Your task is to analyze user queries, determine if they are single-modality or compound cross-modal queries, "
        "and generate a structured TaskPlan with isolated sub-tasks.\n\n"
        "### SPECIALIZED WORKER CAPABILITIES:\n"
        "1. DATA_AGENT: Runs Python/Pandas code in a sandbox over CSV/tabular data (GDP, CO2, emissions, tables). "
        "Use for exact mathematical calculations, statistics, rankings, sums, and averages.\n"
        "2. VISION_AGENT: Analyzes images, charts, graphs, diagrams, figures, and visual crops via Gemini Vision. "
        "Use for visual trend extraction, axis values, curve interpretations, and chart OCR.\n"
        "3. RESEARCH_AGENT: Searches Qdrant vector store (dense BGE-M3 + sparse BM25) and cross-encoder reranker. "
        "Use for qualitative text lookups, report policies, definitions, and narrative explanations.\n"
        "4. DIRECT_LLM: General conversational questions not relating to document datasets.\n\n"
        "### COMPOUND CROSS-MODAL ROUTING RULES:\n"
        "When a query combines two different modalities, set `is_compound = True`, define a clear `comparison_goal`, "
        "and decompose the prompt into 2 separate sub-tasks in `tasks`:\n"
        "A. CSV + VISUAL:\n"
        "   - Example: 'Compare the GDP numbers in the table with the mortality trends in Figure 5.1'\n"
        "   - Task 1 -> DATA_AGENT: calculate/extract the relevant GDP numbers from the CSV table.\n"
        "   - Task 2 -> VISION_AGENT: extract the mortality trends and values from Figure 5.1.\n"
        "B. CSV + TEXT:\n"
        "   - Example: 'Compare the high-income GDP metrics in the table with the standard harmonization policy arguments in the text'\n"
        "   - Task 1 -> DATA_AGENT: calculate/extract high-income GDP metrics from CSV.\n"
        "   - Task 2 -> RESEARCH_AGENT: retrieve policy explanations and arguments from the report text.\n"
        "C. TEXT + VISUAL:\n"
        "   - Example: 'Explain how the environmental policy instruments described in the text relate to the emissions curves in Figure 6.1'\n"
        "   - Task 1 -> RESEARCH_AGENT: retrieve the text explanations of environmental policy instruments.\n"
        "   - Task 2 -> VISION_AGENT: extract and analyze the emissions curves depicted in Figure 6.1.\n\n"
        "### SINGLE-MODALITY ROUTING RULES:\n"
        "If the query only requires ONE modality, set `is_compound = False`, assign `target_agent`, "
        "and populate `tasks` with exactly 1 SubTask targeting that agent."
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
    
    is_compound = plan.is_compound or (len(plan.tasks) > 1)
    logger.info(f"Routed target: {plan.target_agent} | Compound: {is_compound} | Tasks: {len(plan.tasks)} | Reasoning: {plan.reasoning}")
    
    # Ensure at least one task exists
    tasks_payload = []
    if plan.tasks:
        for t in plan.tasks:
            tasks_payload.append({
                "agent": t.agent.value,
                "sub_query": t.sub_query,
                "rationale": t.rationale
            })
    else:
        tasks_payload.append({
            "agent": plan.target_agent.value,
            "sub_query": user_query,
            "rationale": plan.reasoning
        })
        
    sub_queries = plan.sub_queries if plan.sub_queries else [t["sub_query"] for t in tasks_payload]
    
    return {
        "user_query": user_query,
        "routing_decision": plan.target_agent.value,
        "is_compound": is_compound,
        "tasks": tasks_payload,
        "comparison_goal": plan.comparison_goal or plan.reasoning,
        "reasoning": plan.reasoning,
        "sub_queries": sub_queries,
        "requires_validation": plan.requires_validation
    }
