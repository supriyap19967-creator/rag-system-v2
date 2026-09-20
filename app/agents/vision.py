from __future__ import annotations

import base64
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
from openai import OpenAI
from app.main import _resolve_existing_image_path

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
        description="If a table, chart, or data figure is present in the image, transcribe all rows and data points into a clean Markdown table string starting with '|' (e.g. '| Category | Value |\n| --- | --- |\n| A | 100 |')."
    )

# Setup Vision Agent
model_name = os.getenv("VISION_AGENT_MODEL_NAME", "groq:llama-3.3-70b-versatile")
vision_agent = Agent(
    model_name,
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
    resolved_path = _resolve_existing_image_path(image_path)
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
        
        structured_prompt = (
            "### SYSTEM PROMPT: Complete Multi-Panel Visual Extraction & Analytical Summary Engine\n\n"
            "You are a precise data extraction system. For any given image/figure asset, you MUST strictly output a single structured payload with the following mandatory sections:\n\n"
            f"User instructions: {query}\n\n"
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

def execute_vision_task(image_path: str, query: str) -> Dict[str, Any]:
    """
    Entrypoint function to execute vision perception query.
    """
    user_prompt = f"Analyze the image located at '{image_path}' to answer: '{query}'. If a table or figure is present, extract all data points into a Markdown table in extracted_table_markdown."
    run_result = vision_agent.run_sync(user_prompt)
    output: VisionOutput = run_result.output
    return {
        "visual_summary": output.visual_summary,
        "visual_analysis": output.visual_analysis,
        "actionable_insights": output.actionable_insights,
        "extracted_table_markdown": getattr(output, "extracted_table_markdown", None)
    }
