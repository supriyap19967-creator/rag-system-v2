import os
import sys
import json
import logging
import re
import math
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_csv")

# 1. Deterministic Python Checker: Exact Value Match
def calculate_exact_value_match(response: str, reference: str, contexts: list[str] | None = None) -> float:
    """
    Checks if key numerical values present in the reference answer exist
    with exact numerical precision inside the generated output.
    """
    try:
        # Pre-clean comma separators in numbers (e.g. 3,346,107 -> 3346107)
        clean_response = re.sub(r"(?<=\d),(?=\d)", "", response)
        clean_reference = re.sub(r"(?<=\d),(?=\d)", "", reference)

        if "unknown target value" in clean_reference.lower() and contexts:
            # Gather all numeric strings from contexts to use as reference values
            ref_numbers = []
            for ctx in contexts:
                clean_ctx = re.sub(r"(?<=\d),(?=\d)", "", ctx)
                nums = re.findall(r"\b\d+(?:\.\d+)?\b", clean_ctx)
                for num_str in nums:
                    try:
                        val = float(num_str)
                        # Keep larger values or specific years to prevent false negatives
                        if val > 10.0 or val in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
                            ref_numbers.append(num_str)
                    except ValueError:
                        continue
            ref_numbers = list(set(ref_numbers))
        else:
            # Extract all floating point/integer numbers from reference
            ref_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", clean_reference)

        if not ref_numbers:
            # If no numbers are in the reference, check key categorical words (casing ignored)
            ref_words = [w.strip().lower() for w in clean_reference.split() if len(w.strip()) > 3]
            if not ref_words:
                return 1.0
            matches = sum(1 for w in ref_words if w in clean_response.lower())
            return float(matches / len(ref_words))

        # Check if each number in reference exists in response
        matches = 0
        resp_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", clean_response)
        
        for num_str in ref_numbers:
            ref_val = float(num_str)
            found = False
            for r_num_str in resp_numbers:
                try:
                    if math.isclose(float(r_num_str), ref_val, rel_tol=1e-5):
                        found = True
                        break
                except ValueError:
                    continue
            if found:
                matches += 1
                
        return float(matches / len(ref_numbers))
    except Exception as e:
        logger.error(f"Error calculating exact value match: {e}")
        return 0.0

# 2. Deterministic check: Format Consistency
def calculate_format_consistency(response: str) -> float:
    """
    Validates if the output matches the required structured output schema:
    - Markdown table check: 1.0
    - Key-value JSON check: 1.0
    - Short numeric value: 1.0
    - Heavy boilerplate: 0.0 or 0.5
    """
    cleaned = response.strip()
    
    # 1. Check for markdown table structure
    if "|" in cleaned and "-" in cleaned:
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        table_lines = sum(1 for line in lines if line.startswith("|") and line.endswith("|"))
        if table_lines >= 2:
            return 1.0

    # 2. Check for clean JSON structure
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            json.loads(cleaned)
            return 1.0
        except ValueError:
            pass

    # 3. Check if response is a concise direct numeric string (max 15 characters)
    if len(cleaned) <= 15:
        # Check if it contains digits
        if any(c.isdigit() for c in cleaned):
            return 1.0

    # 4. If it has too much conversational boilerplate, penalize
    boilerplate_words = ["sure", "here is", "the answer", "according to", "following table", "please find"]
    boilerplate_count = sum(1 for word in boilerplate_words if word in cleaned.lower())
    if boilerplate_count >= 2:
        return 0.0
        
    return 0.5

# 3. DeepSeek Evaluator: Row/Column Grounding
def get_row_column_grounding(query: str, response: str, contexts: list[str]) -> float:
    """
    Verifies if the extracted data correctly maps to the requested row entity
    and column target present in contexts using DeepSeek V3.
    """
    try:
        from langchain_openai import ChatOpenAI
        
        # Initialize DeepSeek client
        evaluator_llm = ChatOpenAI(
            model="deepseek/deepseek-chat",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            timeout=30.0,
            max_retries=3,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        context_str = "\n".join(contexts)
        prompt = (
            "You are a strict data validation assistant. Analyze whether the generated RAG response "
            "correctly maps to the correct row entities (e.g. specific country, year) and column targets "
            "specified in the query and present in the source contexts.\n\n"
            f"Query: {query}\n"
            f"Source Contexts:\n{context_str}\n"
            f"Generated Response:\n{response}\n\n"
            "Respond ONLY with a JSON object containing:\n"
            "{\n"
            "  \"grounded\": true/false,\n"
            "  \"explanation\": \"Brief justification of the choice\"\n"
            "}"
        )
        
        messages = [{"role": "user", "content": prompt}]
        res = evaluator_llm.invoke(messages)
        
        output_data = json.loads(res.content)
        is_grounded = output_data.get("grounded", False)
        
        logger.info(f"DeepSeek Grounding check: {is_grounded} | Explanation: {output_data.get('explanation')}")
        return 1.0 if is_grounded else 0.0
        
    except Exception as e:
        logger.error(f"Error during DeepSeek Row/Column Grounding evaluation: {e}")
        return 0.0

# 4. Main Evaluation Runner
def evaluate_csv_query(trace_id: str, query: str, response: str, reference: str, contexts: list[str]) -> dict:
    logger.info(f"Running CSV-Only evaluation metrics for trace: {trace_id}")
    
    # Calculate metrics
    exact_match = calculate_exact_value_match(response, reference, contexts)
    grounding = get_row_column_grounding(query, response, contexts)
    format_consistency = calculate_format_consistency(response)
    
    scores = {
        "exact_value_match": exact_match,
        "row_column_grounding": grounding,
        "format_consistency": format_consistency
    }
    
    logger.info(f"Evaluated CSV metrics for trace {trace_id}: {scores}")
    
    # 1. Log metrics directly to Langfuse
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        )
        
        for name, val in scores.items():
            if hasattr(langfuse_client, "create_score"):
                langfuse_client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=val,
                    comment="CSV out-of-band evaluation metric"
                )
            elif hasattr(langfuse_client, "score"):
                langfuse_client.score(
                    trace_id=trace_id,
                    name=name,
                    value=val,
                    comment="CSV out-of-band evaluation metric"
                )
            logger.info(f"Logged score '{name}' = {val} to Langfuse trace {trace_id}")
    except Exception as lf_exc:
        logger.error(f"Failed to push CSV scores to Langfuse: {lf_exc}")
        
    # 2. Append results to eval_results_csv.json
    output_path = PROJECT_ROOT / "evaluation" / "eval_results_csv.json"
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
        "response": response,
        "reference": reference,
        "contexts": contexts,
        "scores": scores
    })
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Appended CSV metrics results to {output_path}")
    except Exception as file_exc:
        logger.error(f"Failed to write CSV metrics to JSON file: {file_exc}")
        
    return scores
