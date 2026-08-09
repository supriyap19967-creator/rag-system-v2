from __future__ import annotations

import os
import sys
import io
import uuid
import logging
from typing import Any, List, Dict, Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent, ModelSettings, RunContext

# Add project root to path for dynamic imports
PROJECT_ROOT = Path("C:/Users/supri/recovered-rag-project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from PIL import Image
from qdrant_client import QdrantClient

ACTIVE_USER_QUERY = ""
VISION_ELEMENT_PROCESSED = False
VISION_TOOL_SUCCEEDED = False
VALIDATION_ATTEMPT_COUNT = 0

# ==========================================
# OPEN TELEMETRY & OBSERVABILITY INITIALIZATION
# ==========================================
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

try:
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    # Enable global auto-instrumentation for Pydantic AI Agents
    Agent.instrument_all()
except Exception as te_exc:
    logging.warning("Failed to initialize OpenTelemetry auto-instrumentation: %s", te_exc)

# ==========================================
# STEP 7: FRAMEWORK METRIC INSTRUMENTATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("backend_pipeline.log", mode="w", encoding="utf-8")]
)
# Explicitly redirect PydanticAI's internal framework tracing to this exact file
pydantic_ai_logger = logging.getLogger("pydantic_ai")
pydantic_ai_logger.setLevel(logging.DEBUG)
pydantic_ai_logger.addHandler(logging.FileHandler("backend_pipeline.log", mode="a", encoding="utf-8"))

logger = logging.getLogger("production_pipeline")


# ==========================================
# STEP 3: ENVIRONMENT INDEPENDENT DEPENDENCY CONTAINER
# ==========================================
class SystemPipelinesDeps:
    """
    Decoupled runtime dependency injection class. Houses live sessions, 
    dataframes, and cryptographic tracking signatures.
    """
    def __init__(
        self, 
        image_folder_path: str, 
        session_user: str = "default_user",
        pandas_df: pd.DataFrame | None = None,
        qdrant_client: QdrantClient | None = None,
        vision_runner: Any = None,
        user_query: str = "",
        gdp_df: pd.DataFrame | None = None,
        gdp_metadata_df: pd.DataFrame | None = None,
        co2_df: pd.DataFrame | None = None,
        co2_metadata_df: pd.DataFrame | None = None
    ):
        self.image_folder_path = image_folder_path
        self.session_signature = f"{session_user}_{uuid.uuid4().hex[:6].upper()}"
        self.pandas_df = pandas_df
        self.qdrant_client = qdrant_client
        self.vision_runner = vision_runner
        self.user_query = user_query
        self.gdp_df = gdp_df
        self.gdp_metadata_df = gdp_metadata_df
        self.co2_df = co2_df
        self.co2_metadata_df = co2_metadata_df
        self.vision_element_processed = False


def parse_markdown_table_to_dicts(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    import re
    lines = [line.strip() for line in text.splitlines() if "|" in line]
    if len(lines) < 3:
        return []
    separator_index = -1
    for idx, line in enumerate(lines):
        if re.match(r"^[\s|:-]+$", line) and "-" in line:
            separator_index = idx
            break
    if separator_index == -1 or separator_index == 0:
        return []
    header_line = lines[separator_index - 1]
    headers = [col.strip() for col in header_line.split("|") if col.strip()]
    
    data_rows = []
    for line in lines[separator_index + 1:]:
        if re.match(r"^[\s|:-]+$", line):
            continue
        cols = [col.strip() for col in line.split("|")]
        if line.startswith("|"):
            cols = cols[1:]
        if line.endswith("|"):
            cols = cols[:-1]
        cols = [c.strip() for c in cols]
        if len(cols) < 2:
            continue
        row_dict = {}
        matched_keys = {}
        for c_idx, h in enumerate(headers):
            h_lower = h.lower()
            if "series" in h_lower:
                matched_keys["Series"] = c_idx
            elif "category" in h_lower or "group" in h_lower:
                matched_keys["Category"] = c_idx
            elif "value" in h_lower or "target" in h_lower:
                matched_keys["TargetValue"] = c_idx
                
        assigned = set(matched_keys.values())
        for key in ["Series", "Category", "TargetValue"]:
            if key not in matched_keys:
                for idx_candidate in range(len(cols)):
                    if idx_candidate not in assigned:
                        matched_keys[key] = idx_candidate
                        assigned.add(idx_candidate)
                        break
                        
        row_dict["Series"] = cols[matched_keys.get("Series", 0)] if len(cols) > matched_keys.get("Series", 0) else ""
        row_dict["Category"] = cols[matched_keys.get("Category", 1)] if len(cols) > matched_keys.get("Category", 1) else ""
        
        raw_val = cols[matched_keys.get("TargetValue", 2)] if len(cols) > matched_keys.get("TargetValue", 2) else "0"
        try:
            val_clean = re.sub(r"[^\d.-]", "", raw_val)
            row_dict["TargetValue"] = float(val_clean) if "." in val_clean else int(val_clean)
        except ValueError:
            row_dict["TargetValue"] = raw_val
            
        data_rows.append(row_dict)
    return data_rows


# ==========================================
# STEP 2 & 5: STRUCTURAL METRIC COMPLIANCE SCHEMA
# ==========================================
class ChartTableRow(BaseModel):
    Series: str = Field(description="The name of the line, bar group, or data series (e.g. country name, variable name, or indicator).")
    Category: str = Field(description="The category label/X-axis label/dimension (e.g. year, age group, or class).")
    TargetValue: float | int | str = Field(description="The numerical value or raw value associated with this category/series.")

    @model_validator(mode='before')
    @classmethod
    def map_fields(cls, data: Any) -> Any:
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except Exception:
                pass
        if isinstance(data, dict):
            norm_data = {str(k).lower().strip(): v for k, v in data.items()}
            
            series_val = None
            for k, v in norm_data.items():
                if k in {"series", "line", "group", "label", "legend", "name", "country", "indicator"}:
                    series_val = v
                    break
            if series_val is None:
                for k, v in norm_data.items():
                    if "series" in k or "line" in k or "title" in k:
                        series_val = v
                        break
            if series_val is None and len(data) > 0:
                series_val = list(data.values())[0]

            category_val = None
            for k, v in norm_data.items():
                if k in {"category", "x-axis", "dimension", "year", "age", "class", "income group", "income"}:
                    category_val = v
                    break
            if category_val is None:
                for k, v in norm_data.items():
                    if "category" in k or "axis" in k or "dimension" in k:
                        category_val = v
                        break
            if category_val is None and len(data) > 1:
                category_val = list(data.values())[1]

            target_val = None
            for k, v in norm_data.items():
                if k in {"targetvalue", "value", "y-axis", "val", "amount", "number", "quantity", "count", "percentage", "rate"}:
                    target_val = v
                    break
            if target_val is None:
                for k, v in norm_data.items():
                    if "value" in k or "val" in k or "amount" in k or "y-axis" in k:
                        target_val = v
                        break
            if target_val is None and len(data) > 2:
                target_val = list(data.values())[2]
                
            if target_val is None:
                for v in data.values():
                    if isinstance(v, (int, float)):
                        target_val = v
                        break
                        
            res = {}
            res["Series"] = str(series_val) if series_val is not None else "N/A"
            res["Category"] = str(category_val) if category_val is not None else "N/A"
            
            if target_val is not None:
                if isinstance(target_val, (int, float)):
                    res["TargetValue"] = target_val
                else:
                    try:
                        val_str = re.sub(r"[^\d.-]", "", str(target_val))
                        res["TargetValue"] = float(val_str) if "." in val_str else int(val_str)
                    except ValueError:
                        res["TargetValue"] = target_val
            else:
                res["TargetValue"] = 0
                
            return res
        return data


class ChartTableData(BaseModel):
    source_routing_trail: str = Field(description="The source file and location metadata.")
    text_reasoning: str = Field(description="The step-by-step logical summary.")
    extracted_table: List[ChartTableRow] = Field(description="List of precise parsed table rows.")

    @model_validator(mode='after')
    def validate_table_integrity(self) -> 'ChartTableData':
        v = self.extracted_table
        global ACTIVE_USER_QUERY, VISION_ELEMENT_PROCESSED, VISION_TOOL_SUCCEEDED, VALIDATION_ATTEMPT_COUNT
        user_query = ACTIVE_USER_QUERY
        
        vision_processed = VISION_ELEMENT_PROCESSED
        vision_succeeded = VISION_TOOL_SUCCEEDED
        val_logger = logging.getLogger("pydantic_ai")
        val_logger.info(f"[VALIDATION DEBUG] vision_processed: {vision_processed}, vision_succeeded: {vision_succeeded}")

        def raise_validation_error(error_msg: str):
            global VALIDATION_ATTEMPT_COUNT
            VALIDATION_ATTEMPT_COUNT += 1
            from opentelemetry import trace
            tracer = trace.get_tracer("pydantic_ai")
            with tracer.start_as_current_span("validation_failure") as failure_span:
                failure_span.set_attribute("validation.non_compliant_output", self.model_dump_json())
                failure_span.set_attribute("validation.error_message", error_msg)
                failure_span.set_attribute("validation.feedback_prompt", f"ValueError: {error_msg}")
                failure_span.set_attribute("validation.attempt_count", VALIDATION_ATTEMPT_COUNT)
                failure_span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, error_msg))
            raise ValueError(error_msg)

        is_extraction = False
        if user_query:
            query_lower = user_query.lower()
            if any(k in query_lower for k in ["table", "figure", "chart", "graph", "extract", "values"]):
                is_extraction = True

        if not v:
            if is_extraction and vision_processed and vision_succeeded:
                # Step 1: Attempt to parse markdown table from text_reasoning
                parsed_rows = parse_markdown_table_to_dicts(self.text_reasoning)
                if parsed_rows:
                    val_logger.info(f"[VALIDATION SUCCESS] Automatically parsed {len(parsed_rows)} rows from Markdown text.")
                    self.extracted_table = parsed_rows
                    return self
                
                val_logger.warning("[VALIDATION FAILED] Empty table payload not allowed for data extraction query.")
                raise_validation_error(
                    "You are executing a visual data extraction query, but the 'extracted_table' field is empty. "
                    "You must call the 'process_vision_element' tool and populate 'extracted_table' with a list of "
                    "dictionaries containing the keys: 'Series', 'Category', 'TargetValue'."
                )
            else:
                val_logger.info("[VALIDATION PASS] Empty table payload allowed for conversational query.")
                return self

        val_logger.info(f"[VALIDATION START] Inspecting and healing {len(v)} visual/tabular extraction rows...")
        
        healed_v = []
        for index, row in enumerate(v):
            if not isinstance(row, dict):
                val_logger.warning(f"[VALIDATION FAILED] Row {index} is not a dictionary: {row}")
                raise_validation_error(
                    f"Row {index} must be a dictionary. Got: {type(row).__name__}. "
                    "Ensure the visual parser returns a list of dictionaries with matching keys."
                )
            
            # If standard keys exist, keep them
            if "Series" in row and "Category" in row and "TargetValue" in row:
                healed_v.append(row)
                continue
            
            # Otherwise, auto-map keys:
            mapped_row = {"Series": "", "Category": "", "TargetValue": 0}
            keys = list(row.keys())
            
            # Find numerical values
            val_found = False
            for k in keys:
                val = row[k]
                if isinstance(val, (int, float)) and not val_found:
                    mapped_row["TargetValue"] = val
                    val_found = True
                    
            # Map other keys to Series/Category
            string_keys = [k for k in keys if not isinstance(row[k], (int, float))]
            if len(string_keys) >= 2:
                mapped_row["Series"] = str(row[string_keys[0]])
                mapped_row["Category"] = str(row[string_keys[1]])
            elif len(string_keys) == 1:
                mapped_row["Series"] = str(row[string_keys[0]])
                mapped_row["Category"] = "N/A"
            else:
                # If all columns are numeric, map first as Series, etc.
                mapped_row["Series"] = str(keys[0]) if len(keys) > 0 else "N/A"
                mapped_row["Category"] = "N/A"
                if len(keys) > 1 and not val_found:
                    mapped_row["TargetValue"] = row[keys[1]]
            
            val_logger.info(f"[HEALED ROW {index}] Mapped keys {keys} -> Series, Category, TargetValue")
            healed_v.append(mapped_row)

        self.extracted_table = healed_v
        val_logger.info("[VALIDATION SUCCESS] All visual extraction rows validated and healed successfully.")
        return self




settings = ModelSettings(
    temperature=0.0
)

# ==========================================
# INITIALIZE THE AGENT ENGINE WITH RUNTIME RETRIES (STEP 8)
# ==========================================
multimodal_agent = Agent(
    'openrouter:google/gemini-2.5-flash',
    deps_type=SystemPipelinesDeps,
    output_type=ChartTableData,
    model_settings=settings,
    retries=3 
)


@multimodal_agent.system_prompt
def system_prompt(ctx: RunContext[SystemPipelinesDeps]) -> str:
    return (
        "You are a multimodal RAG system helper agent. Your task is to analyze user queries and extract data "
        "using your tools (query_pandas_dataframe, query_qdrant_vector_search, process_vision_element).\n\n"
        "GUIDELINES FOR VISUAL ELEMENTS:\n"
        "1. Identify if the element is a Chart/Diagram or a Document Table.\n"
        "2. IF THE ELEMENT IS A CHART, GRAPH, OR DIAGRAM: Extract every single data point, group, and entity present. Format them completely into a structured Markdown table using logical, generic column headers. You must then provide an exhaustive, point-by-point explanation of all extracted data inside the 'text_reasoning' field, ensuring no entity or metric is omitted.\n"
        "3. IF THE ELEMENT IS A DOCUMENT TABLE: Do not force it into a chart format. Provide a comprehensive, highly detailed text overview, row-by-row thematic breakdown, and thorough explanation of the topics covered directly inside the 'text_reasoning' field. The model is fully permitted to populate 'text_reasoning' with this comprehensive text overview while leaving 'extracted_table' empty without triggering any validation failures.\n"
        "4. GENERAL RULE FOR COMPLETENESS: Aim to cover as many distinct topics and data categories as possible. Prioritize explaining all the core themes and concepts visible in the asset thoroughly rather than demanding a rigid, word-for-word replication of every individual text cell.\n\n"
        "GENERAL GUIDELINES:\n"
        "- Prioritize answering the query precisely and step-by-step using tools.\n"
        "- For queries targeting CSV/DataFrame/tabular datasets (such as mathematical computations, statistical trends, row filtering, or aggregations on GDP/CO2 variables), you MUST call `query_pandas_dataframe` only and return the final answer inside the 'text_reasoning' field in a natural sentence (do NOT return table format) and ALWAYS leave 'extracted_table' as an empty list ([]). You MUST retrieve the exact unit or metric from the 'Indicator Name' column of the dataframe (e.g., 't CO2e/capita' or 'current US$') and include it in your sentence answer rather than hardcoding assumptions like 'kilotons' or 'dollars'. Do NOT call `query_qdrant_vector_search` or `process_vision_element` for queries that can be answered directly using the DataFrames.\n"
        "- For comparison, ranking, or statistical queries targeting multiple countries or years, you MUST append a brief 1-2 sentence analytical summary to the final output sentence, comparing the values (e.g., identifying which country/year has the highest or lowest GDP/emissions, and highlighting the difference or trend direction).\n"
        "- If any tool returns an error message or fails (such as vision runner quota exhaustion or execution failures), "
        "DO NOT retry calling the same tool or keep calling tools in a loop. Immediately summarize the failure inside "
        "your 'text_reasoning' field, leave 'extracted_table' as an empty list ([]), and complete the run.\n"
        "- Do not exceed 5 tool calls total."
    )



# =====================================================================
# CONTEXT-AWARE TOOLS WITH IDENTITY TRACING
# =====================================================================

@multimodal_agent.tool
def query_pandas_dataframe(ctx: RunContext[SystemPipelinesDeps], python_code: str, query_intent: str) -> str:
    """
    Call this tool when mathematical computations, matrix operations, statistical trends,
    data aggregation, or direct data row comparisons are requested on the loaded CSV layouts.
    
    IMPORTANT: The tabular datasets (gdp_df and co2_df) are structured in a WIDE format. 
    The columns are: ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', '1961', ..., '2015', '2016', ...]
    Do NOT query for columns like 'Year', 'year', 'Value', or 'value'. Instead, select the row by country name
    and retrieve the value using the specific year string (e.g. ['2015']) as the column index.
    
    Available DataFrames:
    - gdp_df: World Bank GDP data
    - gdp_metadata_df: metadata for GDP
    - co2_df: CO2 emissions data
    - co2_metadata_df: metadata for CO2
    """
    # PROVE IDENTITY & SANITARY BOUNDARY ISOLATION
    logger.info("═"*60)
    logger.info("🔍 ENTERING CONTEXT SECURITY BOUNDARY (Pandas Pipeline)")
    logger.info(f"   ↳ Active Request Signature: {ctx.deps.session_signature}")
    logger.info(f"   ↳ Isolated File Path Context: {ctx.deps.image_folder_path}")
    logger.info("═"*60)

    # Expose all dataframes to the local code execution environment
    locs = {
        "gdp_df": ctx.deps.gdp_df,
        "gdp_metadata_df": ctx.deps.gdp_metadata_df,
        "co2_df": ctx.deps.co2_df,
        "co2_metadata_df": ctx.deps.co2_metadata_df,
        "df": ctx.deps.pandas_df # fallback
    }
    stdout = io.StringIO()
    old_stdout = sys.stdout

    from opentelemetry import trace
    tracer = trace.get_tracer("pydantic_ai")
    with tracer.start_as_current_span("pandas_execution") as pandas_span:
        pandas_span.set_attribute("pandas.query_logic", python_code)
        
        # Capture dataframe metadata
        df_meta = {}
        for df_key in ["gdp_df", "gdp_metadata_df", "co2_df", "co2_metadata_df"]:
            df_obj = locs.get(df_key)
            if df_obj is not None:
                df_meta[df_key] = {
                    "shape": list(df_obj.shape),
                    "columns": list(df_obj.columns)[:15] # log first 15 columns for layout sanity
                }
        import json
        pandas_span.set_attribute("pandas.dataframe_metadata", json.dumps(df_meta))

        try:
            sys.stdout = stdout
            exec(python_code, {}, locs)
        except Exception as exc:
            pandas_span.record_exception(exc)
            pandas_span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, str(exc)))
            return f"Pandas execution failed with runtime error: {exc}"
        finally:
            sys.stdout = old_stdout

    output = stdout.getvalue().strip()
    if not output:
        output = str(locs.get("result", locs.get("ans", "Code executed successfully with no printed output.")))
    return output


@multimodal_agent.tool
def query_qdrant_vector_search(ctx: RunContext[SystemPipelinesDeps], semantic_query: str, target_collection: str) -> str:
    """
    Call this tool for natural language inquiries, contextual knowledge lookups,
    and text chunk extraction from the document collection.
    """
    logger.info("═"*60)
    logger.info("🔍 ENTERING CONTEXT SECURITY BOUNDARY (Qdrant Pipeline)")
    logger.info(f"   ↳ Active Request Signature: {ctx.deps.session_signature}")
    logger.info(f"   ↳ Isolated File Path Context: {ctx.deps.image_folder_path}")
    logger.info("═"*60)

    client = ctx.deps.qdrant_client
    if client is None or not isinstance(client, QdrantClient):
        return "Error: Injected qdrant_client dependency is not a valid QdrantClient instance."

    try:
        import re
        from app.embeddings import get_query_vector
        query_vector = get_query_vector(semantic_query)
        
        # Local helper to parse category and identifier
        def parse_target_asset(query: str) -> tuple[str | None, str | None]:
            pattern = re.compile(
                r"\b(?P<kind>table|tabel|tab|figure|fig|chart|diagram|graph)[\s_]*"
                r"(?P<identifier>[sS]?\d+(?:\.\d+)*)\b",
                flags=re.IGNORECASE
            )
            match = pattern.search(query)
            if match:
                kind = match.group("kind").lower()
                cat = "Table" if kind.startswith("tab") else "Figure"
                return cat, match.group("identifier")
            return None, None
            
        qdrant_filter = None
        target_cat, target_id = parse_target_asset(semantic_query)
        if target_cat and target_id:
            from qdrant_client import models
            asset_type = "table" if "table" in target_cat.lower() else "figure"
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(key="metadata.asset_type", match=models.MatchValue(value=asset_type)),
                    models.FieldCondition(key="metadata.asset_id", match=models.MatchValue(value=target_id))
                ]
            )
            
        from opentelemetry import trace
        tracer = trace.get_tracer("pydantic_ai")
        with tracer.start_as_current_span("retriever") as retriever_span:
            retriever_span.set_attribute("vector_search.query", semantic_query)
            retriever_span.set_attribute("vector_search.collection", target_collection)
            retriever_span.set_attribute("vector_search.limit", 5)
            
            # Log raw vector search parameters context (truncating dense vector array float output)
            retriever_span.set_attribute("vector_search.raw_parameters", f"dense_dims={len(query_vector)}, filter={str(qdrant_filter)}")

            results = client.search(
                collection_name=target_collection,
                query_vector=("dense", query_vector),
                query_filter=qdrant_filter,
                limit=5
            )
            
            if not results:
                retriever_span.set_attribute("vector_search.chunks_count", 0)
                return "No matching context fragments returned from Qdrant vector store."
                
            retriever_span.set_attribute("vector_search.chunks_count", len(results))
            retriever_span.set_attribute("vector_search.scores", [p.score for p in results])
            
            retrieved_texts = []
            formatted_chunks = []
            for index, point in enumerate(results, start=1):
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                text = payload.get("text") or payload.get("page_content") or ""
                source = payload.get("source") or metadata.get("source_file") or "unknown_source"
                page = metadata.get("page_number", "N/A")
                chapter = metadata.get("chapter_number", "N/A")
                
                retrieved_texts.append(text)
                chunk_str = (
                    f"[{index}] Source: {source} (Ch: {chapter}, Pg: {page}) | Score: {point.score:.4f}\n"
                    f"Content: {text.strip()}\n"
                )
                formatted_chunks.append(chunk_str)
                
            retriever_span.set_attribute("vector_search.retrieved_chunks", retrieved_texts)
            return "\n---\n".join(formatted_chunks)
    except Exception as exc:
        logger.exception("Qdrant vector search failed with exception")
        return f"Qdrant vector search failed with runtime error: {exc}"


@multimodal_agent.tool
def process_vision_element(ctx: RunContext[SystemPipelinesDeps], visual_asset_path: str, extraction_instructions: str) -> str:
    """
    Call this tool when the query refers to an image, graph, chart, diagram, or figure name.
    Instructs the Vision model to extract visual data points into raw text or structural data.
    """
    global VISION_ELEMENT_PROCESSED
    VISION_ELEMENT_PROCESSED = True
    ctx.deps.vision_element_processed = True
    # PROVE IDENTITY & SANITARY BOUNDARY ISOLATION
    logger.info("═"*60)
    logger.info("🔍 ENTERING CONTEXT SECURITY BOUNDARY (Vision Pipeline via OpenRouter)")
    logger.info(f"   ↳ Active Request Signature: {ctx.deps.session_signature}")
    logger.info(f"   ↳ Isolated File Path Context: {ctx.deps.image_folder_path}")
    logger.info("═"*60)

    visual_asset_path = os.path.normpath(visual_asset_path.replace("\\\\", "\\"))
    from app.main import _resolve_existing_image_path
    resolved_str = _resolve_existing_image_path(visual_asset_path)
    if resolved_str and os.path.exists(resolved_str):
        visual_asset_path = resolved_str
    img_path = Path(visual_asset_path)
    logger.info(f"Resolved visual asset path to: {img_path}")
    if not img_path.exists():
        try:
            from app.multimodal_assets import build_asset_registry, normalize_entity_id
            filename = os.path.basename(visual_asset_path)
            norm_id = normalize_entity_id(filename)
            logger.info(f"Normalizing filename '{filename}' (from path '{visual_asset_path}') to '{norm_id}' for registry lookup")
            
            registry = build_asset_registry()
            matching_record = None
            # Pass 1: Prioritize matching records that are image files
            for record in registry:
                if record.entity_id == norm_id:
                    path_suffix = Path(record.absolute_path).suffix.lower()
                    if path_suffix in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                        matching_record = record
                        break
            # Pass 2: Fallback to any matching record if no image was found
            if not matching_record:
                for record in registry:
                        if record.entity_id == norm_id:
                            matching_record = record
                            break
                        
                if matching_record:
                    resolved_path = Path(matching_record.absolute_path)
                    if resolved_path.suffix.lower() in [".csv", ".json"]:
                        resolved = False
                        page_match = re.search(r"page_?(\d+)", resolved_path.name, re.IGNORECASE)
                        if not page_match:
                            page_match = re.search(r"pdf-?(\d+)", resolved_path.name, re.IGNORECASE)
                        if not page_match:
                            page_match = re.search(r"-(\d+)(?:\.\d+)?\.[^.]+$", resolved_path.name)
                            
                        if page_match:
                            page_no = page_match.group(1)
                            for folder in ["assets/extracted_images", "extracted_images"]:
                                folder_path = Path("C:/Users/supri/recovered-rag-project") / folder
                                if folder_path.exists():
                                    for file in folder_path.glob("*"):
                                        if (file.name.lower().startswith(f"page{page_no}_") or file.name.lower().startswith(f"page_{page_no}_")) and file.suffix.lower() == ".png" and not file.name.lower().endswith(".raw.png"):
                                            img_path = file
                                            resolved = True
                                            logger.info(f"{resolved_path.suffix.upper()} resolved to image fallback: {img_path}")
                                            break
                                    if resolved:
                                        break
                        if not resolved:
                            img_path = resolved_path
                    else:
                        img_path = resolved_path
                    logger.info(f"Registry match found: {img_path}")
                else:
                    # Direct fallback to folder
                    fallback_path = Path(ctx.deps.image_folder_path) / img_path.name
                    if fallback_path.exists():
                        img_path = fallback_path
                    else:
                        # Try appending suffix if missing
                        resolved = False
                        for ext in [".png", ".jpg", ".jpeg"]:
                            temp_path = Path(ctx.deps.image_folder_path) / f"{img_path.name}{ext}"
                            if temp_path.exists():
                                img_path = temp_path
                                resolved = True
                                break
                        if not resolved:
                            # Try exact match with suffix inside directory
                            for file in Path(ctx.deps.image_folder_path).glob("*"):
                                if visual_asset_path.lower() in file.name.lower() or norm_id.lower() in file.name.lower():
                                    img_path = file
                                    resolved = True
                                    break
        except Exception as e:
            logger.warning(f"Registry lookup failed: {e}")

    # Prioritize raw (untrimmed) crop if it exists on disk
    if img_path.exists():
        raw_check = img_path.with_name(f"{img_path.stem}.raw{img_path.suffix}")
        if raw_check.exists():
            img_path = raw_check
            logger.info(f"Prioritizing untrimmed raw crop: {img_path}")

    if not img_path.exists():
        return f"Error: Target visual asset path '{visual_asset_path}' could not be resolved or does not exist on disk."

    try:
        import base64
        from openai import OpenAI
        
        # Load environment variables from .env file if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # Load OpenRouter API Key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "Error: OPENROUTER_API_KEY environment variable is not set."
            
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Read the image file in binary mode
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        encoded_string = encoded_string.replace('\n', '').replace('\r', '').strip()
        
        structured_prompt = (
            "You are a high-fidelity visual parser. Analyze the provided image and extract information based on the user's instructions.\n\n"
            f"User extraction instructions: {extraction_instructions}\n\n"
            "FORMATTING GUIDELINES:\n"
            "1. If the user instructions ask for a description or explanation of a document table, extract all data points conceptually and format them entirely as standard text paragraphs or clean Markdown sections. Ensure the text fully covers every topic and category dimension visible in the image.\n"
            "2. If the user instructions ask for data from a visual chart/graph, extract all raw data points across all entities and groups. Present them completely as a clean Markdown table using appropriate generic column names. You must capture and detail every single data point and entity present in the figure without shortcuts or omissions. Do not round approximations unnecessarily."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Extract information from this visual element:\n\n{structured_prompt}"
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
        
        # Extract image dimensions using PIL safely
        img_dims = [0, 0]
        try:
            from PIL import Image
            with Image.open(img_path) as pil_img:
                img_dims = list(pil_img.size)
        except Exception:
            pass

        from opentelemetry import trace
        tracer = trace.get_tracer("pydantic_ai")
        with tracer.start_as_current_span("visual_extraction") as vision_span:
            vision_span.set_attribute("visual.asset_path", str(img_path))
            vision_span.set_attribute("visual.image_width", img_dims[0])
            vision_span.set_attribute("visual.image_height", img_dims[1])
            vision_span.set_attribute("visual.base64_length", len(encoded_string))
            vision_span.set_attribute("visual.base64_prefix", encoded_string[:50])

            response = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=messages
            )
            
            if not response or not response.choices:
                vision_span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, "Empty visual response"))
                return "Error: Empty or invalid response returned from OpenRouter visual inference engine."
                
            raw_content = response.choices[0].message.content
            print(f"--- [RAW UNVALIDATED VISION RESPONSE] ---\n{raw_content}\n-----------------------------------------", flush=True)
            global VISION_TOOL_SUCCEEDED
            VISION_TOOL_SUCCEEDED = True
            return raw_content
        
    except Exception as exc:
        return f"Vision inference pipeline (OpenRouter) failed with runtime error: {exc}"

