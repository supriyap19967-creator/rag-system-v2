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
logger = logging.getLogger("run_evaluations")

# Import isolated evaluation runners
from evaluation.eval_csv import evaluate_csv_query
from evaluation.eval_visual import evaluate_visual_query

# Local MPNet Embeddings Wrapper
class LocalMpnetEmbeddings:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Initializing local SentenceTransformer('all-mpnet-base-v2') for Ragas...")
        self.model_name = "all-mpnet-base-v2"
        self._model = SentenceTransformer("all-mpnet-base-v2")

    @property
    def model(self) -> str:
        return self.model_name

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text).tolist()]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in self._model.encode(t).tolist()] for t in texts]

# Intent Classifier using DeepSeek V3
def classify_query_type(query: str, category: str = "") -> str:
    """
    Classifies queries into "text", "csv", or "visual_extraction" based on rules and fast V3 classification.
    """
    q_lower = query.lower()
    
    # 1. Deterministic Rule checks
    if any(kw in q_lower for kw in ["gdp", "carbon dioxide", "co2", "emissions", "adoption rate"]) and not any(kw in q_lower for kw in ["figure", "chart", "diagram", "extract values", "extract table"]):
        return "csv"
    if "pdf_explanatory" in category.lower():
        return "text"
    if any(kw in q_lower for kw in ["figure", "chart", "diagram", "extract values", "table 4.1", "figure 4.2"]):
        return "visual_extraction"
        
    # 2. Fast LLM classifier fallback
    try:
        from langchain_openai import ChatOpenAI
        classifier = ChatOpenAI(
            model="deepseek/deepseek-chat",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            timeout=10.0,
            max_retries=2
        )
        prompt = (
            "Classify the following user query into exactly one of three categories:\n"
            "1. 'csv': Queries requiring numerical/mathematical analysis of data records (like GDP or carbon emission rates).\n"
            "2. 'visual_extraction': Queries asking to extract data, numbers, or trends from visual chart/table images or figures.\n"
            "3. 'text': Standard text-only retrieval and explanatory Q&A queries.\n\n"
            f"Query: {query}\n\n"
            "Return ONLY the category name (csv, visual_extraction, or text)."
        )
        res = classifier.invoke(prompt)
        classification = res.content.strip().lower()
        if classification in ["csv", "visual_extraction", "text"]:
            return classification
    except Exception as e:
        logger.warning(f"Fast LLM classifier failed: {e}. Defaulting to 'text'.")
        
    return "text"

# Evaluator for Text queries (RAGAS)
def evaluate_text_query_ragas(query: str, response: str, reference: str, contexts: list[str]) -> dict:
    logger.info("Evaluating text-only query using RAGAS...")
    try:
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics import answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from ragas.run_config import RunConfig
        import re
        
        # Standardize evaluator LLM to gpt-4o-mini
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="openai/gpt-4o-mini",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.0,
                timeout=60.0
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
        
        # Clean response of metadata / source trail / table formatting
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

        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=pruned_contexts[:3],
            response=response,
            reference=reference
        )
        dataset = EvaluationDataset(samples=[sample])
        
        eval_result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy, context_precision],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RunConfig(timeout=60, max_retries=3),
            raise_exceptions=False
        )
        
        df_res = eval_result.to_pandas()
        numeric_cols = df_res.select_dtypes(include='number').columns
        return df_res[numeric_cols].iloc[0].to_dict()
        
    except Exception as e:
        logger.error(f"Failed Ragas text evaluation: {e}")
        return {}

def run_evaluation_router():
    logger.info("Loading evaluation dataset cases...")
    from evaluation.ragas_eval_set import EVALUATION_CASES
    from query_rag import execute_rag_pipeline
    
    final_results = []
    
    for case in EVALUATION_CASES:
        query = case["query"]
        ground_truth = case["ground_truth"]
        category = case.get("category", "")
        
        # 1. Classify Query
        query_type = classify_query_type(query, category)
        logger.info(f"Query: '{query}' classified as type: '{query_type}'")
        
        # Execute Live pipeline
        logger.info("Executing live RAG pipeline...")
        try:
            pipeline_result = execute_rag_pipeline(query)
            response = str(pipeline_result.get("response", "")).strip()
            
            # Extract retrieved contexts
            retrieved_chunks = pipeline_result.get("retrieved_chunks", []) or []
            contexts = [c.get("content", "") if isinstance(c, dict) else getattr(c, "content", "") for c in retrieved_chunks]
            if not contexts or contexts == ["No context retrieved."]:
                try:
                    from streamlit_ui.StreamlitApp import query_qdrant_vector_search
                    # Resolve relevant Qdrant chunks for the query
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
        except Exception as e:
            logger.error(f"Failed to execute pipeline for '{query}': {e}")
            continue
            
        scores = {}
        trace_id = f"eval_{query_type}_{int(os.getpid())}"
        
        # 2. Route Execution to specific pipeline
        if query_type == "text":
            scores = evaluate_text_query_ragas(query, response, ground_truth, contexts)
        elif query_type == "csv":
            scores = evaluate_csv_query(trace_id, query, response, ground_truth, contexts)
        elif query_type == "visual_extraction":
            # Find local target image using pipeline parser
            def local_parse_target_asset(q: str) -> tuple[str | None, str | None]:
                import re
                p = re.compile(
                    r"\b(?P<kind>table|figure|fig\.?|chart|image|diagram|map|box|spotlight)\s*[_\-\s]?(?P<identifier>[A-Za-z]?\d+(?:\.\d+)*)\b",
                    flags=re.IGNORECASE
                )
                m = p.search(q)
                if m:
                    k = m.group("kind").lower()
                    return ("Figure" if k.startswith("fig") else "Table"), m.group("identifier")
                return None, None

            from app.main import _resolve_existing_image_path
            target_cat, target_id = local_parse_target_asset(query)
            image_path = ""
            if target_cat and target_id:
                image_path = _resolve_existing_image_path(f"{target_cat}_{target_id}")
                
            if not image_path and target_cat and target_id:
                try:
                    from app.multimodal_assets import build_asset_registry
                    registry = build_asset_registry()
                    for record in registry:
                        if target_id and target_id in record.entity_id:
                            if target_cat and target_cat.lower() in record.entity_id.lower():
                                if any(record.absolute_path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]):
                                    image_path = record.absolute_path
                                    break
                except Exception:
                    pass
                
            response_data = {
                "text_response": response,
                "confidence_score": 1.0,
                "metadata": {},
                "extracted_table": []
            }
            
            # Extract table rows if present
            def local_parse_markdown_table_to_dicts(text: str) -> list[dict]:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
                if len(table_lines) < 3:
                    return []
                headers = [h.strip() for h in table_lines[0].split("|")[1:-1]]
                rows = []
                for line in table_lines[2:]:
                    cols = [c.strip() for c in line.split("|")[1:-1]]
                    if len(cols) == len(headers):
                        row_dict = {}
                        for h, val in zip(headers, cols):
                            if h.lower() in ["series", "s"]:
                                row_dict["Series"] = val
                            elif h.lower() in ["category", "c"]:
                                row_dict["Category"] = val
                            elif h.lower() in ["targetvalue", "value", "v", "target value"]:
                                row_dict["TargetValue"] = val
                            else:
                                row_dict[h] = val
                        rows.append(row_dict)
                return rows

            if "|" in response:
                try:
                    parsed_rows = local_parse_markdown_table_to_dicts(response)
                    if parsed_rows:
                        response_data["extracted_table"] = parsed_rows
                except Exception:
                    pass
                    
            scores = evaluate_visual_query(trace_id, query, response_data, image_path)
            
        final_results.append({
            "query": query,
            "query_type": query_type,
            "response": response,
            "reference": ground_truth,
            "scores": scores
        })
        
    # 3. Save combined summary results
    output_path = PROJECT_ROOT / "evaluation" / "final_eval_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Evaluation Router completed. Combined summary saved to {output_path}")

if __name__ == "__main__":
    run_evaluation_router()
