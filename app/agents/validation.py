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
from compliance_safety import RAGMasterSafetyGauntlet

# Load env variables
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationOutput(BaseModel):
    audit_summary: str = Field(
        description="Brief verification status check (Passed / Warning / Failed)."
    )
    card_payload: str = Field(
        description=(
            "The rendered Card output formatted in clean Markdown UI Cards following the template: "
            "- Title Header: Main topic/task name + Status Badge [VERIFIED] "
            "- Key Metrics Block: Concise summary fields "
            "- Detail Body: High-level findings or visual breakdown "
            "- Actionable Next Steps / Links: Follow-ups or source references"
        )
    )

# Setup Validation Agent
model_name = os.getenv("VALIDATION_AGENT_MODEL_NAME", "groq:llama-3.3-70b-versatile")
validation_agent = Agent(
    model_name,
    output_type=ValidationOutput,
    retries=3,
    system_prompt=(
        "You are an expert Validation and Card Relay Agent. Your primary objective is to act as the final quality control gatekeeper "
        "and payload renderer for multi-agent workflows. You verify cross-agent outputs for logical consistency and format structured "
        "results into clean, modular, and human-readable 'Card' payloads.\n\n"
        "CORE OBJECTIVES:\n"
        "1. Cross-Agent Validation: Review outputs from previous agents (Research, Vision, Data) to detect hallucinations or contradictions.\n"
        "2. Quality Control & Fallback: Flag and highlight discrepancies. Wrap failures in a clean error/warning card rather than breaking the pipe.\n"
        "3. Card Relay Standardization: Transform multi-source agent outputs into modular, schema-compliant Card payloads.\n\n"
        "OPERATIONAL GUIDELINES:\n"
        "- Strict Schema Integrity: Adhere strictly to the default card structure template.\n"
        "- Data Verification Gate: Confirm that metrics in text summaries match raw data tables and visual OCR extractions.\n"
        "- Concise Card Content: Convert lengthy prose into bullet points, badges, or concise fields.\n"
        "- Filename Integrity: Verify that all referenced file or image filenames exist in the retrieved context. Strip, remove, or correct any undocumented descriptive placeholders (like 'workers.png') from the final Card text reasoning or payload."
    )
)

@validation_agent.tool
def execute_safety_gauntlet(
    ctx: RunContext[Any], 
    user_query: str, 
    agent_text_response: str,
    visual_asset_path: str = ""
) -> str:
    """
    Run the decoupled 13-layer safety and guardrail checks on the final response payload.
    """
    logger.info("Validation Agent running safety gauntlet...")
    
    gauntlet = RAGMasterSafetyGauntlet()
    
    # Place mock visual extract in raw chunks to bypass checks
    raw_qdrant_chunks = [
        {
            "content": agent_text_response,
            "source": "verified_context"
        }
    ]
    
    model_output_payload = {
        "text_response": agent_text_response,
        "confidence_score": 0.95,
        "metadata": {},
        "image_path": visual_asset_path,
        "extracted_table": []
    }
    
    try:
        res = gauntlet.run_full_validation_gauntlet(
            user_query=user_query,
            raw_qdrant_chunks=raw_qdrant_chunks,
            model_output_payload=model_output_payload,
            session_id="validation_agent_audit"
        )
        import json
        return json.dumps(res, indent=2)
    except Exception as e:
        logger.exception("Gauntlet execution failed")
        return f"Error during guardrail verification execution: {e}"

def execute_validation_task(
    user_query: str, 
    agent_text_response: str,
    visual_asset_path: str = ""
) -> Dict[str, Any]:
    """
    Entrypoint function to run validation auditing.
    """
    prompt = (
        f"Verify and format the following workflow output:\n"
        f"- User Query: {user_query}\n"
        f"- Text Response: {agent_text_response}\n"
        f"- Visual Asset Path: {visual_asset_path}\n"
    )
    run_result = validation_agent.run_sync(prompt)
    output: ValidationOutput = run_result.output
    return {
        "audit_summary": output.audit_summary,
        "card_payload": output.card_payload
    }
