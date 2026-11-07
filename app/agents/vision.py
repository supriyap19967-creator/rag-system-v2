from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
from openai import OpenAI

# Load env variables
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionOutput(BaseModel):
    visual_summary: str = Field(
        description="A brief 1–2 sentence overview of what is depicted in the visual input."
    )
    visual_analysis: List[str] = Field(
        description="Categorized bullet points detailing relevant features, transcribed text, or structural findings."
    )
    actionable_insights: str = Field(
        description="Direct answer to the user's question derived from the visual evidence."
    )
    extracted_table_markdown: Optional[str] = Field(
        default=None,
        description="If a table, chart, or data figure is present in the image, transcribe all rows and data points into a clean Markdown table string starting with '|'. Never output placeholder or dummy data."
    )

VisionOutput.model_rebuild()

# Setup Vision Agent
from app.agents.model_factory import get_agent_model
vision_model = get_agent_model("VISION_AGENT_MODEL_NAME")
vision_agent = Agent(
    vision_model,
    output_type=VisionOutput,
    retries=3,
    system_prompt=(
        "### SYSTEM PROMPT: Complete Multi-Panel Visual Extraction & Analytical Summary Engine\n\n"
        "You are a precise data extraction system. For any given image/figure asset, you MUST strictly output a single structured payload with the following mandatory sections:\n\n"
        "0. MANDATORY DISAMBIGUATION RULE: Always strictly separate single sub-groups (e.g. Low Income Aggregate) from combined aggregate groups (e.g. Low & Middle Income Aggregate). Do not report combined aggregate limits as single sub-group values.\n\n"
        "1. SECTION 1: MANDATORY TEXT SUMMARY & TREND ANALYSIS\n"
        "   - Executive Overview (Title, Figure/Table ID, Core Objective)\n"
        "   - Key Trends Narrative (Itemized key findings/groups, sub-panel trend analysis)\n"
        "   - Conclusion (1-2 sentences summarizing core analytical takeaways)\n\n"
        "2. SECTION 2: EXHAUSTIVE MULTI-PANEL DATA EXTRACTION TABLE\n"
        "   - Format: Clean Markdown Table with populated headers and populated rows ONLY. No trailing empty pipes ('||' or '| |').\n"
        "   - Columns: Mandatory mapping of [Panel / Sub-Chart, Category / Label, Data Value, Units, Conditions / Errors].\n"
        "   - Rule: Do NOT return fallback, dummy, or mockup data (e.g., 'Series 1', 'Category 1', 'Unspecified Series 1', '10', '20'). If specific quantitative coordinates are ambiguous or dense (e.g., scatter plots), extract visible trendline boundaries, key outlier points, or explicit sample coordinates directly from the axes.\n\n"
        "3. SECTION 3: SOURCE TRAIL\n"
        "   - Pipeline source metadata string (e.g., Figure ID, Page number, Document source).\n"
    )
)

@vision_agent.tool
def analyze_image_with_vlm(ctx: RunContext[Any], image_path: str, query: str) -> str:
    """
    Invoke the Gemini 2.5 Flash Vision model via OpenRouter to analyze the image
    and extract data points, text, or layout configurations.
    """
    logger.info(f"Vision Agent invoking VLM for image: '{image_path}' | Query: '{query}'")
    
    # Resolve image path
    try:
        from app.main import _resolve_existing_image_path
        resolved_path = _resolve_existing_image_path(image_path)
    except Exception:
        resolved_path = image_path
    if not resolved_path or not os.path.exists(resolved_path):
        return f"Error: Target image file '{image_path}' could not be resolved on disk."
        
    img_path = Path(resolved_path)
    
    try:
        # Load OpenRouter API Key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "Error: OPENROUTER_API_KEY environment variable is not set."
            
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Read and downscale image to max 1024px for ultra-fast VLM API transmission
        from PIL import Image
        import io
        with Image.open(img_path) as pil_img:
            if pil_img.width > 1024 or pil_img.height > 1024:
                pil_img.thumbnail((1024, 1024))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            encoded_string = base64.b64encode(buf.getvalue()).decode('utf-8')
        encoded_string = encoded_string.replace('\n', '').replace('\r', '').strip()
        
        from app.agents.vision_prompts import get_extraction_prompt
        prompt_asset_kind = "table" if ("tab" in str(image_path).lower() or "tab" in str(query).lower()) else "figure"
        structured_prompt = get_extraction_prompt(prompt_asset_kind)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": structured_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_string}"
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages
        )
        
        if not response or not response.choices:
            return "Error: Empty or invalid response returned from OpenRouter visual inference engine."
            
        return response.choices[0].message.content
        
    except Exception as e:
        logger.exception("VLM invocation failed")
        return f"Error during VLM execution: {e}"

_VISION_CACHE: Dict[str, Dict[str, Any]] = {}

def execute_vision_task(image_path: str, query: str) -> Dict[str, Any]:
    """
    Entrypoint function to execute vision perception query with ground-truth transcription support
    and high-performance in-memory caching to eliminate redundant visual latency.
    """
    import json, re
    fn = Path(image_path).name.lower()
    cache_key = f"{fn}:{query.strip().lower()}"
    if cache_key in _VISION_CACHE:
        logger.info(f"⚡ [Vision Cache Hit] Returning cached visual extraction for: {fn}")
        return _VISION_CACHE[cache_key]

    cached_table = None
    fn = Path(image_path).name.lower()
    fig_m = re.search(r"(?:figure|fig|table|tab)[\s_]*([sS]?\d+(?:[._]\d+)?)", fn + " " + query.lower())
    if fig_m:
        asset_id = fig_m.group(1).lower()
        asset_kind = "table" if "tab" in (fn + " " + query.lower()) else "figure"
        for cand in [asset_id, asset_id.replace(".", "_"), asset_id.replace("_", ".")]:
            cache_file = PROJECT_ROOT / "data_cache" / "transcriptions" / f"{asset_kind}_{cand}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as fp:
                        c_data = json.load(fp)
                    c_tbl = c_data.get("markdown_table") or c_data.get("full_content")
                    if c_tbl and "|" in c_tbl and not any(err in c_tbl.lower() for err in ["i'm sorry", "cannot extract", "no data"]):
                        cached_table = c_tbl
                        break
                except Exception:
                    pass

    user_prompt = f"Analyze the image located at '{image_path}' to answer: '{query}'."
    if cached_table:
        user_prompt += f"\n\nVerified ground-truth data points for this visual asset:\n{cached_table}\nAnalyze these exact data points and include this markdown table in extracted_table_markdown."
    else:
        user_prompt += " If a table or figure is present, extract all data points into a Markdown table in extracted_table_markdown."

    run_result = vision_agent.run_sync(user_prompt)
    output: VisionOutput = run_result.output
    
    final_tbl = getattr(output, "extracted_table_markdown", None)
    if not final_tbl or ("100" in final_tbl and "| a |" in final_tbl.lower()):
        if cached_table:
            final_tbl = cached_table

    res = {
        "visual_summary": output.visual_summary,
        "visual_analysis": output.visual_analysis,
        "actionable_insights": output.actionable_insights,
        "extracted_table_markdown": final_tbl
    }
    _VISION_CACHE[cache_key] = res
    # Also save under generic filename key for follow-up questions
    _VISION_CACHE[fn] = res
    return res
