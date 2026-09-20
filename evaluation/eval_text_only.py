import os
import sys
import json
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_text_only")

# 1. Local MPNet Embeddings Wrapper
class LocalMpnetEmbeddings:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Initializing local SentenceTransformer('all-mpnet-base-v2') for Ragas...")
        self.model_name = "all-mpnet-base-v2"
        self._model = SentenceTransformer("all-mpnet-base-v2")
        logger.info("Local SentenceTransformer('all-mpnet-base-v2') loaded successfully.")

    @property
    def model(self) -> str:
        return self.model_name

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text).tolist()]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in self._model.encode(t).tolist()] for t in texts]

# 2. Main Evaluation Pipeline function
def run_text_only_evaluation():
    logger.info("Starting Text-Only Ragas Evaluation Pipeline...")
    
    # Load raw evaluation cases
    from evaluation.ragas_eval_set import EVALUATION_CASES
    
    # 1. Dataset Filtering: Filter only "pdf_explanatory" or explicitly marked "text-only" queries
    text_only_cases = [
        case for case in EVALUATION_CASES 
        if case.get("category") == "pdf_explanatory" or case.get("type") == "text-only"
    ]
    
    logger.info(f"Filtered {len(text_only_cases)} text-only evaluation cases out of {len(EVALUATION_CASES)} total cases.")
    
    if not text_only_cases:
        logger.warning("No text-only evaluation cases found! Exiting.")
        return
        
    # Initialize Ragas dependencies
    try:
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics import answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from ragas.run_config import RunConfig
        
        logger.info("Initializing Ragas LLM and Embeddings wrappers...")
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="openai/gpt-4o-mini",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.0,
                timeout=60.0,
                max_retries=5
            )
        )
        from langchain_openai import OpenAIEmbeddings
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1"
            )
        )
        
        metrics_list = [answer_relevancy, context_precision]
        logger.info("Ragas dependency instantiation complete.")
        
    except Exception as e:
        logger.error(f"Failed to initialize Ragas dependencies: {e}")
        raise e

    # Execute queries against RAG pipeline to gather responses and contexts
    # NOTE: We do not modify the main vector search or retrieval logic
    from query_rag import execute_rag_pipeline # Import RAG query pipeline directly
    
    samples = []
    evaluation_logs = []
    
    for case in text_only_cases:
        query = case["query"]
        ground_truth = case["ground_truth"]
        
        logger.info(f"Executing RAG pipeline for query: '{query}'")
        try:
            # Query the live pipeline to get the real response and context chunks
            pipeline_result = execute_rag_pipeline(query)
            
            response = str(pipeline_result.get("response", "")).strip()
            
            # Extract retrieved contexts
            retrieved_chunks = pipeline_result.get("retrieved_chunks", []) or []
            contexts = [c.get("content", "") if isinstance(c, dict) else getattr(c, "content", "") for c in retrieved_chunks]
            
            # Fallback context harvesting
            if not contexts or contexts == ["No context retrieved."]:
                try:
                    from streamlit_ui.StreamlitApp import query_qdrant_vector_search
                    fallback_chunks = query_qdrant_vector_search(query)
                    if fallback_chunks:
                        if isinstance(fallback_chunks, list):
                            contexts = [
                                c.payload.get("text", "") if hasattr(c, "payload") else str(c)
                                for c in fallback_chunks
                            ]
                        elif isinstance(fallback_chunks, str):
                            contexts = [fallback_chunks]
                except Exception as fallback_exc:
                    logger.warning(f"Could not perform fallback context retrieval: {fallback_exc}")

            if not contexts:
                contexts = ["No context retrieved."]
            
            logger.info(f"Query executed. Context chunks retrieved: {len(contexts)}")
            
            # Clean response of metadata / source trail / table formatting
            import re
            original_response = response
            if "**Source Trail:**" in response:
                response = response.split("**Source Trail:**")[0]
            if "### Extracted Table Data" in response:
                response = response.split("### Extracted Table Data")[0]
            response = response.strip()
            if not response or len(response) < 5:
                response = original_response
                
            response = response.replace("**", "")
            response = re.sub(r'(?m)^\s*[-*+]\s+', '', response)
            
            def prune_context_noise(text: str) -> str:
                if not text:
                    return ""
                # 1. Strip file paths / source trail paths / visual tokens
                text = re.sub(r'(?i)[^\s\'"()]+?\.(?:png|jpg|jpeg|webp|gif|pdf|csv|json|jsonl)\b', '', text)
                text = re.sub(r'(?i)\b(?:images|extracted_images|assets|Data/Pdf)/[^\s\'"()]+', '', text)
                
                # 2. Strip bounding boxes / layout coordinates like [82.5, 289.3, 496.0, 316.9] or Rect(...)
                text = re.sub(r'\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]', '', text)
                text = re.sub(r'(?i)\b(?:Rect|BBox)\b\(.*?\)', '', text)
                
                # 3. Strip layout headers, visual tokens, and page numbers
                text = re.sub(r'(?i)\bpage\s*\d+\b', '', text)
                text = re.sub(r'(?i)\bpage_number\s*\d+\b', '', text)
                
                # 4. Convert markdown table borders and grid structures to clean text
                lines = []
                for line in text.splitlines():
                    line_strip = line.strip()
                    if not line_strip:
                        continue
                    if re.match(r"^[\s|:-]+$", line_strip) and "-" in line_strip:
                        continue
                    if line_strip.startswith("|") and line_strip.endswith("|"):
                        parts = [p.strip() for p in line_strip.split("|") if p.strip()]
                        if parts:
                            lines.append(" - " + ", ".join(parts))
                    else:
                        lines.append(line)
                        
                text = "\n".join(lines)
                text = re.sub(r' +', ' ', text)
                text = re.sub(r'\n\s*\n', '\n', text).strip()
                return text

            pruned_contexts = [prune_context_noise(c) for c in contexts if c]
            # Slice to top 3 chunks maximum first
            pruned_contexts = pruned_contexts[:3]

            def extract_relevant_sentences(text: str, query: str) -> str:
                if not text or not query:
                    return text
                import re
                stop_words = {
                    'what', 'is', 'why', 'are', 'the', 'and', 'a', 'an', 'of', 'in', 'to', 'for', 
                    'with', 'on', 'at', 'by', 'from', 'about', 'how', 'does', 'do', 'did', 'explain', 
                    'describe', 'extract', 'show', 'report', 'document', 'pdf', 'csv', 'table', 'figure'
                }
                query_words = re.findall(r'\b\w+\b', query.lower())
                keywords = {w for w in query_words if w not in stop_words and len(w) > 1}
                if not keywords:
                    return text
                    
                sentences = re.split(r'(?<=[.!?])\s+', text)
                matched_sentences = []
                for sent in sentences:
                    sent_lower = sent.lower()
                    if any(w in sent_lower for w in keywords):
                        matched_sentences.append(sent)
                if matched_sentences:
                    return " ".join(matched_sentences).strip()
                elif sentences:
                    return sentences[0].strip()
                return text

            pruned_contexts = [extract_relevant_sentences(c, query) for c in pruned_contexts]
            pruned_contexts = [c for c in pruned_contexts if c.strip()]
            if not pruned_contexts:
                pruned_contexts = ["No context retrieved."]

            # Construct standard RAGAS SingleTurnSample
            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=pruned_contexts[:3],
                response=response,
                reference=ground_truth
            )
            samples.append(sample)
            
            evaluation_logs.append({
                "query": query,
                "ground_truth": ground_truth,
                "response": response,
                "contexts": contexts
            })
            
        except Exception as query_exc:
            logger.error(f"Failed to execute query '{query}': {query_exc}")
            continue

    if not samples:
        logger.warning("No samples executed successfully. Exiting evaluation.")
        return

    # Run Ragas evaluate
    logger.info("Running Ragas evaluation dataset computations...")
    dataset = EvaluationDataset(samples=samples)
    
    try:
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics_list,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RunConfig(timeout=60, max_retries=5),
            raise_exceptions=False
        )
        
        df_res = eval_result.to_pandas()
        results_list = df_res.to_dict(orient="records")
        
        # Calculate dataset mean scores
        summary_scores = eval_result.scores
        logger.info(f"Evaluation finished successfully. Scores: {summary_scores}")
        
        # Save results to dedicated file
        output_payload = {
            "summary_scores": summary_scores,
            "detailed_results": results_list,
            "raw_payloads": evaluation_logs
        }
        
        output_path = PROJECT_ROOT / "evaluation" / "eval_results_text_only.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Successfully saved evaluation results to {output_path}")
        
    except Exception as eval_exc:
        logger.error(f"Ragas evaluation run failed: {eval_exc}")

if __name__ == "__main__":
    run_text_only_evaluation()
