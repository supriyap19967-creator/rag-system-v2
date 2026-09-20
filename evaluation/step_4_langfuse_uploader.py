import os
import sys
import logging
from dotenv import load_dotenv
from langfuse import Langfuse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langfuse_uploader")

# 1. Setup & Connection
# Ensure credentials are present, falling back to environment values
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

langfuse_client = Langfuse(
    public_key=public_key,
    secret_key=secret_key,
    host=host
)

from evaluation.step_3_eval_runner import run_evaluation_pipeline

def upload_eval_results_to_langfuse():
    """
    Executes the Step 3 evaluation pipeline and logs the results to Langfuse.
    """
    logger.info("Starting offline evaluation pipeline...")
    results = run_evaluation_pipeline()
    
    logger.info(f"Evaluation finished. Uploading {len(results)} traces to Langfuse...")
    
    for item in results:
        modalities = item["modality"]
        if len(modalities) > 1:
            metric_name = "multimodal_hybrid_score"
        elif "visual" in modalities:
            metric_name = "visual_extraction_score"
        elif "csv" in modalities:
            metric_name = "csv_assertion_score"
        else:
            metric_name = "text_correctness_score"
            
        logger.info(f"Logging trace for sample {item['id']} under metric '{metric_name}'")
        
        # Log trace and score using standard SDK methods
        with langfuse_client.start_as_current_observation(
            name=f"Multimodal Eval - {item['id']}",
            input={
                "question": item["question"],
                "inputs": item["inputs"],
                "modality": item["modality"]
            },
            output=str(item["actual_output"]),
            metadata={
                "passed": item["passed"]
            }
        ):
            langfuse_client.score_current_trace(
                name=metric_name,
                value=float(item["score"]),
                comment=item["explanation"]
            )
        
    # 4. Reliable Delivery
    logger.info("Flushing trace upload queue to Langfuse...")
    try:
        langfuse_client.flush()
        logger.info("Langfuse upload complete.")
    except Exception as e:
        logger.error(f"Failed to flush Langfuse queues: {e}")

if __name__ == "__main__":
    upload_eval_results_to_langfuse()
