import os
import sys
import json
import logging
import base64
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_visual")

# Helper to encode local image to base64 for VLM input
def encode_image_to_base64(image_path: str) -> str:
    try:
        if not os.path.exists(image_path):
            # Attempt to resolve using main project helper
            from app.main import _resolve_existing_image_path
            resolved = _resolve_existing_image_path(image_path)
            if resolved and os.path.exists(resolved):
                image_path = resolved
            else:
                logger.error(f"Image file does not exist: {image_path}")
                return ""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return ""

# 1. Pydantic Schema Validation Score
def calculate_schema_validation_score(response_data: dict) -> float:
    """
    Checks if the response data dictionary contains all required structured output fields
    (e.g., text_reasoning/image_summary, and extracted_table/table_rows).
    """
    try:
        # Check standard visual extraction output schemas
        required_keys = ["text_response", "confidence_score", "metadata"]
        for key in required_keys:
            if key not in response_data:
                return 0.0
                
        # Also check for table extraction lists
        extracted_table = response_data.get("extracted_table")
        if extracted_table is not None and not isinstance(extracted_table, list):
            return 0.0
            
        return 1.0
    except Exception:
        return 0.0

# 2. Multimodal LLM Judge (Visual Grounding & Completeness)
def evaluate_multimodal_metrics(query: str, response: str, image_path: str) -> dict:
    """
    Calls a vision-capable evaluator VLM (GPT-4o-mini via OpenRouter) to check:
    - visual_faithfulness: Do numerical data points match the image?
    - visual_completeness: Does explanation capture key trends without hallucinating?
    """
    scores = {
        "visual_faithfulness": 0.0,
        "visual_completeness": 0.0
    }
    
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        logger.warning(f"Skipping VLM multimodal evaluation due to missing/invalid image at {image_path}")
        return scores
        
    try:
        from openai import OpenAI
        
        # Initialize client with OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # VLM prompt requesting granular continuous scoring scale
        prompt = (
            "You are a strict Multimodal AI Evaluation Judge. Analyze the generated text response and extracted table "
            "against the provided chart/diagram image.\n\n"
            f"User Query: {query}\n"
            f"Generated Response: {response}\n\n"
            "Evaluate along these two dimensions on a granular continuous scale from 0.00 to 1.00:\n\n"
            "1. visual_faithfulness (0.00 - 1.00):\n"
            "   - 1.00: All extracted numbers, data points, and labels match the image context with 100% precision.\n"
            "   - 0.70 - 0.95: Most values match, but minor non-critical values have slight numerical rounding or labeling discrepancies.\n"
            "   - 0.30 - 0.65: Moderate hallucinations or incorrect data extraction from the visual source.\n"
            "   - 0.00: Complete hallucination or completely wrong values.\n\n"
            "2. visual_completeness (0.00 - 1.00):\n"
            "   - 1.00: Extracted ALL relevant data points, trend lines, legends, and axis context requested by the user.\n"
            "   - 0.70 - 0.95: Extracted the key data points, but omitted minor visual details or supplementary context.\n"
            "   - 0.30 - 0.65: Missed more than half of the required visual data points/rows.\n"
            "   - 0.00: Extracted no usable visual information.\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            "  \"visual_faithfulness\": <float between 0.00 and 1.00>,\n"
            "  \"visual_completeness\": <float between 0.00 and 1.00>,\n"
            "  \"reasoning\": \"Detailed explanation justifying the exact numeric scores based on visual evidence\"\n"
            "}"
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        output_data = json.loads(response.choices[0].message.content)
        scores["visual_faithfulness"] = float(output_data.get("visual_faithfulness", 0.0))
        scores["visual_completeness"] = float(output_data.get("visual_completeness", 0.0))
        
        logger.info(f"VLM metrics evaluated successfully: {scores} | Reason: {output_data.get('reasoning')}")
        
    except Exception as e:
        logger.error(f"Failed during Multimodal VLM evaluation: {e}")
        
    return scores

# 3. Main Evaluation Runner
def evaluate_visual_query(trace_id: str, query: str, response_data: dict, image_path: str) -> dict:
    logger.info(f"Running Visual-Extraction evaluation metrics for trace: {trace_id}")
    
    response_text = response_data.get("text_response", "")
    
    # 1. Compute schema validation score
    schema_score = calculate_schema_validation_score(response_data)
    
    # 2. Compute visual faithfulness and completeness
    multimodal_scores = evaluate_multimodal_metrics(query, response_text, image_path)
    
    scores = {
        "schema_validation_score": schema_score,
        "visual_faithfulness": multimodal_scores["visual_faithfulness"],
        "visual_completeness": multimodal_scores["visual_completeness"]
    }
    
    logger.info(f"Evaluated Visual metrics for trace {trace_id}: {scores}")
    
    # 3. Log metrics to Langfuse
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        )
        
        for name, val in scores.items():
            langfuse_client.score(
                trace_id=trace_id,
                name=name,
                value=val,
                comment="Visual out-of-band evaluation metric"
            )
            logger.info(f"Logged score '{name}' = {val} to Langfuse trace {trace_id}")
    except Exception as lf_exc:
        logger.error(f"Failed to push Visual scores to Langfuse: {lf_exc}")
        
    # 4. Save results to eval_results_visual.json
    output_path = PROJECT_ROOT / "evaluation" / "eval_results_visual.json"
    results_data = []
    
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)
                if not isinstance(results_data, list):
                    results_data = []
        except Exception:
            results_data = []
            
    results_data.append({
        "trace_id": trace_id,
        "query": query,
        "response_data": response_data,
        "image_path": image_path,
        "scores": scores
    })
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Appended Visual metrics results to {output_path}")
    except Exception as file_exc:
        logger.error(f"Failed to write Visual metrics to JSON file: {file_exc}")
        
    return scores
