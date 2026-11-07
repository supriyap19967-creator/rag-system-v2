from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are the Senior Cross-Modal Intelligence Synthesizer in an enterprise Multi-Agent RAG architecture.
Your mission: Given the user's compound comparison query, comparison goal, and the independently verified outputs from specialized sub-agents (Data Agent, Vision Agent, Research Agent), synthesize them into a unified, authoritative, and structured analytical report.

MANDATORY RESPONSE STRUCTURE:
## 📊 Executive Cross-Modal Takeaway
- Provide a concise 2-3 sentence strategic synthesis directly answering the user's comparison question and highlighting the primary correlation or insight.

## 📈 Quantitative & Analytical Findings
- Synthesize the exact numerical findings, metrics, and trends from the Data Agent and/or Vision extracted data.
- MANDATORY STANDARDIZED MARKDOWN TABLE: Whenever quantitative metrics or visual extracted tables are present in the evidence, you MUST render them as a clean, standardized GitHub-Flavored Markdown table with columns:
| Series | Category | Target Value | Unit | Summary / Notes |
- NUMERIC FORMATTING RULE: Never print raw scientific notation (e.g. NEVER write '2.72922e+13'). Always format currency as '$27.29 Trillion', '$4.56 Trillion', or percentages/counts clearly.

## 🔍 Contextual & Qualitative Analysis
- Detail the thematic, policy, structural, or narrative insights provided by the Research Agent and/or Vision observations.

## ⚖️ Cross-Modal Correlation & Strategic Insights
- Provide deep comparative synthesis: How do the quantitative metrics align or contrast with the qualitative context or visual curves?
- Highlight key correlations, anomalies, policy trade-offs, or economic patterns.

## 📋 Data Provenance & Sources
- Enumerate the explicit source files, figure IDs, tables, or document sections utilized.

STYLE GUIDELINES:
- Professional executive tone.
- Clear markdown headings and crisp bullet points.
- Grounded strictly in the provided sub-agent findings. Zero hallucination.
"""

def _clean_scientific_notation_in_text(text: str) -> str:
    """Replaces raw scientific notation numbers like 2.72922e+13 with human-readable numbers."""
    def repl(m):
        try:
            num = float(m.group(0))
            if abs(num) >= 1e12:
                return f"${num / 1e12:.2f} Trillion"
            elif abs(num) >= 1e9:
                return f"${num / 1e9:.2f} Billion"
            elif abs(num) >= 1e6:
                return f"${num / 1e6:.2f} Million"
            elif abs(num) >= 1000:
                return f"{num:,.2f}"
            else:
                return f"{num:.2f}"
        except Exception:
            return m.group(0)

    # Match numbers like 2.72922e+13 or 4.56e12
    return re.sub(r"\b[0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+\b", repl, text)


def synthesize_cross_modal_fusion(
    question: str,
    comparison_goal: Optional[str],
    sub_agent_summaries: Dict[str, str]
) -> str:
    """
    Synthesizes outputs from multiple specialized agents into a unified
    cross-modal comparative report.
    """
    logger.info("Executing Cross-Modal Fusion Synthesis...")
    
    # Compile agent evidence
    evidence_blocks = []
    for agent_name, summary_text in sub_agent_summaries.items():
        cleaned_text = _clean_scientific_notation_in_text(summary_text)
        evidence_blocks.append(f"### [Agent: {agent_name}]\n{cleaned_text.strip()}")
    combined_evidence = "\n\n---\n\n".join(evidence_blocks)
    
    prompt = f"""User Question:
{question}

Comparison Objective:
{comparison_goal or 'Synthesize and contrast multi-modal findings to address user query.'}

Verified Evidence from Specialized Sub-Agents:
{combined_evidence}

Please produce the final synthesized cross-modal analytical report following the required structure.
"""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            response = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            if response and response.choices and response.choices[0].message.content:
                out = response.choices[0].message.content.strip()
                return _clean_scientific_notation_in_text(out)
        except Exception as exc:
            logger.warning(f"Synthesizer LLM invocation failed: {exc}. Using structured fallback.")
            
    # Deterministic high-quality fallback synthesis if API fails
    fallback_parts = [
        f"## 📊 Executive Cross-Modal Takeaway\n"
        f"Synthesized comparative analysis addressing: *{question}*.\n",
        f"**Goal**: {comparison_goal or 'Cross-modal correlation and analysis'}\n"
    ]
    for agent_name, text in sub_agent_summaries.items():
        cleaned_text = _clean_scientific_notation_in_text(text)
        fallback_parts.append(f"### Evidence from {agent_name}:\n{cleaned_text}\n")
    return "\n\n".join(fallback_parts)
