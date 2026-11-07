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
    asset_reference: Optional[str] = Field(
        default=None,
        description="Explicit figure, chart, or table identifier if applicable (e.g. 'Figure 4.1', 'Figure 5.1', 'Table 2.1'). ONLY set for VISION_AGENT."
    )
    rationale: str = Field(
        default="",
        description="Why this specific agent is required for this sub-task."
    )

class TaskPlan(BaseModel):
    is_compound: bool = Field(
        default=False,
        description="True if the user query requires cross-modal collaboration across multiple distinct agents (e.g. CSV+Visual, CSV+Text, Text+Visual, Tri-Modal)."
    )
    target_agent: TargetAgent = Field(
        description="The primary sub-agent if single-modality, or primary coordinating agent if compound."
    )
    tasks: List[SubTask] = Field(
        default_factory=list,
        description="List of 1 to 3 isolated sub-tasks to execute across specialized agents. For single queries, contains 1 sub-task. For compound queries, contains 2 or 3 sub-tasks."
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
        "You are the Master Supervisor Orchestrator Agent for an enterprise multimodal RAG system.\n"
        "Your task is to analyze user queries, determine if they are single-modality or compound cross-modal queries, "
        "and generate a structured TaskPlan with strictly isolated sub-tasks.\n\n"
        "### SPECIALIZED WORKER CAPABILITIES & STRICT MODALITY CONTRACTS:\n"
        "1. DATA_AGENT: Runs Python/Pandas in a sandbox over tabular World Bank CSV datasets (GDP, CO2, emissions, income groups, countries).\n"
        "   - Use for exact mathematical calculations, averages, statistics, country rankings, sums, and tabular CSV lookups.\n"
        "   - CRITICAL DISAMBIGUATION: When user says 'in the table', 'from the table', 'GDP table', or 'countries in the table', this refers to TABULAR CSV DATA! Assign to DATA_AGENT.\n"
        "2. VISION_AGENT: Analyzes charts, diagrams, graphs, and specific visual figures (e.g. Figure 4.1, Figure 5.1, Table 6.2) via Gemini Vision & verified OCR transcriptions.\n"
        "   - Use for visual curve interpretations, trendlines, axis values, and chart OCR.\n"
        "   - ABSOLUTE GUARDRAIL: NEVER assign VISION_AGENT unless the query explicitly targets a visual asset, chart, figure, or diagram!\n"
        "3. RESEARCH_AGENT: Searches Qdrant vector store (dense BGE-M3 + sparse BM25) and cross-encoder reranker over the World Development Report 2025 PDF.\n"
        "   - Use for conceptual definitions (e.g. 'three types of standards'), policy arguments, regulatory frameworks, chapter discussions, and qualitative explanations.\n"
        "4. DIRECT_LLM: General conversational questions not relating to document datasets.\n\n"
        "### COMPOUND CROSS-MODAL DECOMPOSITION RULES:\n"
        "When a query compares or synthesizes multiple modalities, set `is_compound = True`, define a clear `comparison_goal`, and decompose into distinct tasks:\n\n"
        "Pattern 1 [DATA + TEXT]:\n"
        "- Example: 'Compare the average GDP of low-income countries in the table with the three types of standards explained in the report'\n"
        "- Task 1 -> DATA_AGENT: 'Calculate the average GDP of low-income countries from the dataset table.'\n"
        "- Task 2 -> RESEARCH_AGENT: 'Retrieve and explain the three types of standards discussed in the report text.'\n"
        "- (Note: VISION_AGENT MUST NOT be assigned here! No visual asset exists in this query.)\n\n"
        "Pattern 2 [TEXT + VISUAL]:\n"
        "- Example: 'Compare the definition of standards in Chapter 4 with the adoption metrics in Figure 4.1'\n"
        "- Task 1 -> RESEARCH_AGENT: 'Retrieve the definition and types of standards in Chapter 4 from the report text.'\n"
        "- Task 2 -> VISION_AGENT (asset_reference='Figure 4.1'): 'Extract the adoption metrics and series from Figure 4.1.'\n\n"
        "Pattern 3 [DATA + VISUAL]:\n"
        "- Example: 'Compare top 5 countries by CO2 in 2020 with the adoption trends in Figure 4.1'\n"
        "- Task 1 -> DATA_AGENT: 'Extract top 5 countries by CO2 emissions in 2020 from the dataset.'\n"
        "- Task 2 -> VISION_AGENT (asset_reference='Figure 4.1'): 'Extract adoption trends and percentages from Figure 4.1.'\n\n"
        "Pattern 4 [TRI-MODAL: DATA + TEXT + VISUAL]:\n"
        "- Example: 'Compare top 5 GDP nations in 2023 with mortality rates in Figure 5.1 and health policy arguments'\n"
        "- Task 1 -> DATA_AGENT: 'Extract top 5 countries by GDP in 2023 from dataset.'\n"
        "- Task 2 -> VISION_AGENT (asset_reference='Figure 5.1'): 'Extract mortality rates and curves from Figure 5.1.'\n"
        "- Task 3 -> RESEARCH_AGENT: 'Retrieve health policy arguments and investments from report text.'\n\n"
        "### SINGLE-MODALITY RULE:\n"
        "If the query only requires ONE modality, set `is_compound = False`, assign `target_agent`, and populate `tasks` with exactly 1 SubTask targeting that agent."
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
                "asset_reference": getattr(t, "asset_reference", None),
                "rationale": t.rationale
            })
    else:
        tasks_payload.append({
            "agent": plan.target_agent.value,
            "sub_query": user_query,
            "asset_reference": None,
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
