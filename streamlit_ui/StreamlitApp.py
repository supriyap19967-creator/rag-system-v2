from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv(override=True)

import hashlib
import base64
import html
import json
import logging
import mimetypes
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Generator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# OPEN TELEMETRY & OBSERVABILITY INITIALIZATION
# ==========================================
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pydantic_ai import Agent

try:
    provider = TracerProvider()
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    
    if public_key and secret_key:
        import base64
        auth_token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        endpoint = f"{base_url.rstrip('/')}/api/public/otel/v1/traces"
        headers = {
            "Authorization": f"Basic {auth_token}",
            "x-langfuse-ingestion-version": "4"
        }
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        
        # Enable global auto-instrumentation for Pydantic AI Agents
        Agent.instrument_all()
    else:
        logging.warning("Langfuse credentials not found. OpenTelemetry traces not configured.")
except Exception as te_exc:
    logging.warning("Failed to initialize OpenTelemetry auto-instrumentation: %s", te_exc)

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from qdrant_client import QdrantClient, models
from vectordb.fastembed_runtime import SafeSparseEncoder
from vectordb.qdrant_client_manager import QdrantSettings, get_qdrant_client as build_managed_qdrant_client
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_sambanova import ChatSambaNova
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent, ModelSettings, RunContext

# ==========================================
# PYDANTIC & PYDANTIC_AI SELF-CORRECTING AGENT DEFINITION
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
        self.last_vision_raw_content = ""
        self.retrieved_chunks = []


def swap_and_clean_row(series: str, category: str, target_val: Any) -> tuple[str, str, Any]:
    s = str(series).strip()
    c = str(category).strip()
    v = target_val
    
    income_groups = {"low income", "lower middle income", "upper middle income", "high income"}
    
    # 1. Swap if income group is in Series
    if s.lower() in income_groups and c.lower() not in income_groups:
        s, c = c, s
        
    # 2. Swap if c is a line/series name and s is an income group
    if s.lower() in income_groups:
        s, c = c, s
        
    # If category matches income groups but Series is empty or dummy, make Series "Standard adopted"
    if c.lower() in income_groups and (s == "" or s.lower() in ("n/a", "data point")):
        s = "Standard adopted"
        
    # 3. If s is numeric, it shouldn't be the Series. If v is dummy/empty, move s to v.
    is_s_numeric = False
    try:
        float(s.replace(",", "").strip())
        is_s_numeric = True
    except ValueError:
        pass
        
    if is_s_numeric:
        try:
            val_clean = float(s.replace(",", "").strip())
            v = int(val_clean) if val_clean.is_integer() else val_clean
        except ValueError:
            v = s
        s = "Standard adopted" if c.lower() in income_groups else "Data Point"

    return s, c, v


def parse_markdown_table_to_dicts(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    import re
    lines = [line.strip() for line in text.splitlines() if "|" in line]
    if len(lines) < 2:
        return []
    separator_index = -1
    for idx, line in enumerate(lines):
        if re.match(r"^[\s|:-]+$", line) and "-" in line:
            separator_index = idx
            break
    
    if separator_index != -1 and separator_index > 0:
        header_line = lines[separator_index - 1]
        headers = [col.strip() for col in header_line.split("|") if col.strip()]
        data_start_idx = separator_index + 1
    else:
        header_line = lines[0]
        headers = [col.strip() for col in header_line.split("|") if col.strip()]
        data_start_idx = 1
    
    data_rows = []
    for line in lines[data_start_idx:]:
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
        
        if len(cols) == 2:
            row_dict["Series"] = cols[0] if cols[0] else "Data Point"
            row_dict["Category"] = cols[1] if cols[1] else "N/A"
            raw_val = cols[1]
            try:
                val_clean = re.sub(r"[^\d.-]", "", raw_val)
                row_dict["TargetValue"] = float(val_clean) if "." in val_clean else int(val_clean)
            except ValueError:
                row_dict["TargetValue"] = raw_val
        else:
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
                            
            row_dict["Series"] = cols[matched_keys.get("Series", 0)] if len(cols) > matched_keys.get("Series", 0) else "Data Point"
            row_dict["Category"] = cols[matched_keys.get("Category", 1)] if len(cols) > matched_keys.get("Category", 1) else "N/A"
            
            raw_val = cols[matched_keys.get("TargetValue", 2)] if len(cols) > matched_keys.get("TargetValue", 2) else ""
            try:
                val_clean = re.sub(r"[^\d.-]", "", raw_val)
                row_dict["TargetValue"] = float(val_clean) if "." in val_clean else int(val_clean)
            except ValueError:
                row_dict["TargetValue"] = raw_val if raw_val else "N/A"
        
        # Absolute Safeguard: Fill in any empty or None values to bypass validator failures
        s, c, v = swap_and_clean_row(row_dict.get("Series", ""), row_dict.get("Category", ""), row_dict.get("TargetValue", ""))
        row_dict["Series"] = s
        row_dict["Category"] = c
        row_dict["TargetValue"] = v

        if not row_dict.get("Series"):
            row_dict["Series"] = "Data Point"
        if not row_dict.get("Category"):
            row_dict["Category"] = "N/A"
        if row_dict.get("TargetValue") in ["", None]:
            row_dict["TargetValue"] = "N/A"
            
        data_rows.append(row_dict)
    return data_rows


class ChartTableRow(BaseModel):
    Series: str = Field(description="The name of the line, bar group, or data series (e.g. country name, variable name, or indicator).")
    Category: str = Field(description="The category label/X-axis label/dimension (e.g. year, age group, or class).")
    TargetValue: float | int | str = Field(description="The numerical value or raw value associated with this category/series.")


class ChartTableData(BaseModel):
    source_routing_trail: str = Field(description="The source file and location metadata.")
    text_reasoning: str = Field(description="The step-by-step logical summary.")
    extracted_table: list[ChartTableRow] = Field(description="List of precise parsed table rows.")
    chart_title: str | None = Field(default=None, description="The title or caption of the chart/figure, if identifiable.")
    x_axis_label: str | None = Field(default=None, description="The title or label of the X-axis (e.g., GDP per capita).")
    y_axis_label: str | None = Field(default=None, description="The title or label of the Y-axis (e.g., CO2 emissions).")
    chart_type: str | None = Field(default=None, description="The visual layout type (e.g., scatter_plot, bar_chart, line_graph, table).")
    units: str | None = Field(default=None, description="The units of measurement (e.g., USD, tonnes, %).")

    @model_validator(mode='after')
    def validate_table_integrity(self) -> 'ChartTableData':
        v = self.extracted_table
        global ACTIVE_USER_QUERY, VISION_ELEMENT_PROCESSED, VISION_TOOL_SUCCEEDED, VALIDATION_ATTEMPT_COUNT, LAST_VISION_RAW_CONTENT
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

        def is_table_invalid(table: list[Any]) -> bool:
            if not table:
                return True
            for row in table:
                if isinstance(row, ChartTableRow):
                    s = str(row.Series).strip()
                    c = str(row.Category).strip()
                    val_raw = row.TargetValue
                elif isinstance(row, dict):
                    s = str(row.get("Series", "")).strip()
                    c = str(row.get("Category", "")).strip()
                    val_raw = row.get("TargetValue", "")
                else:
                    return True
                val_str = str(val_raw).strip()
                is_dummy = (
                    (s == "" or s.lower() == "n/a") and
                    (c == "" or c.lower() == "n/a") and
                    (val_str == "" or val_str == "0" or val_str == "0.0" or val_str.lower() == "n/a" or val_raw is None)
                )
                if not is_dummy:
                    return False
            return True

        if is_extraction and vision_processed and vision_succeeded:
            if is_table_invalid(v):
                # Try to parse from text_reasoning
                if "|" in self.text_reasoning:
                    parsed_rows = parse_markdown_table_to_dicts(self.text_reasoning)
                    if parsed_rows and not is_table_invalid(parsed_rows):
                        val_logger.info(f"[VALIDATION SUCCESS] Automatically parsed {len(parsed_rows)} rows from Markdown text.")
                        self.extracted_table = [ChartTableRow(**r) for r in parsed_rows]
                        return self
                
                # Fallback: parse from the raw vision response stored globally
                if LAST_VISION_RAW_CONTENT and "|" in LAST_VISION_RAW_CONTENT:
                    parsed_rows = parse_markdown_table_to_dicts(LAST_VISION_RAW_CONTENT)
                    if parsed_rows and not is_table_invalid(parsed_rows):
                        val_logger.info(f"[VALIDATION SUCCESS] Automatically parsed {len(parsed_rows)} rows from global vision raw content.")
                        self.extracted_table = [ChartTableRow(**r) for r in parsed_rows]
                        return self
                
                val_logger.warning("[VALIDATION WARNING] Empty or dummy table payload detected. Skipping validation failure to allow post-agent interception.")
                return self
        elif not v:
            val_logger.info("[VALIDATION PASS] Empty table payload allowed.")
            return self

        v = self.extracted_table
        val_logger.info(f"[VALIDATION START] Inspecting and healing {len(v)} visual/tabular extraction rows...")
        healed_v = []
        for index, row in enumerate(v):
            if not isinstance(row, (dict, ChartTableRow)):
                val_logger.warning(f"[VALIDATION WARNING] Row {index} is not a dictionary or ChartTableRow: {row}. Skipping row.")
                continue
            
            # Convert row to dict representation for uniform parsing/healing
            row_dict = row.model_dump() if isinstance(row, ChartTableRow) else row
            
            # If standard keys exist, keep them
            if "Series" in row_dict and "Category" in row_dict and "TargetValue" in row_dict:
                healed_v.append(ChartTableRow(**row_dict))
                continue
            
            # Otherwise, auto-map keys:
            mapped_row = {"Series": "", "Category": "", "TargetValue": 0}
            keys = list(row_dict.keys())
            
            # Find numerical values
            val_found = False
            for k in keys:
                val = row_dict[k]
                if isinstance(val, (int, float)) and not val_found:
                    mapped_row["TargetValue"] = val
                    val_found = True
                    
            # Map other keys to Series/Category
            string_keys = [k for k in keys if not isinstance(row_dict[k], (int, float))]
            if len(string_keys) >= 2:
                mapped_row["Series"] = str(row_dict[string_keys[0]])
                mapped_row["Category"] = str(row_dict[string_keys[1]])
            elif len(string_keys) == 1:
                mapped_row["Series"] = str(row_dict[string_keys[0]])
                mapped_row["Category"] = "N/A"
            else:
                # If all columns are numeric, map first as Series, etc.
                mapped_row["Series"] = str(keys[0]) if len(keys) > 0 else "N/A"
                mapped_row["Category"] = "N/A"
                if len(keys) > 1 and not val_found:
                    mapped_row["TargetValue"] = row_dict[keys[1]]
            
            # Apply swap and clean
            s_clean, c_clean, v_clean = swap_and_clean_row(mapped_row.get("Series", ""), mapped_row.get("Category", ""), mapped_row.get("TargetValue", ""))
            mapped_row["Series"] = s_clean
            mapped_row["Category"] = c_clean
            mapped_row["TargetValue"] = v_clean
            
            val_logger.info(f"[HEALED ROW {index}] Mapped keys -> Series: {s_clean}, Category: {c_clean}, TargetValue: {v_clean}")
            healed_v.append(ChartTableRow(**mapped_row))

        self.extracted_table = healed_v
        val_logger.info("[VALIDATION SUCCESS] All visual extraction rows validated and healed successfully.")
        return self


ChartTableRow.model_rebuild()
ChartTableData.model_rebuild()

agent_settings = ModelSettings(
    temperature=0.0
)

multimodal_agent = Agent(
    'openrouter:google/gemini-2.5-flash',
    deps_type=SystemPipelinesDeps,
    output_type=ChartTableData,
    model_settings=agent_settings,
    retries=3 
)


@multimodal_agent.output_validator
def validate_result(ctx: RunContext[SystemPipelinesDeps], result: ChartTableData) -> ChartTableData:
    val_logger = logging.getLogger("pydantic_ai")
    val_logger.info("[RESULT VALIDATOR] Checking final structured table output and preserving chart metadata...")
    import re
    
    def is_table_invalid_res(table: list[ChartTableRow]) -> bool:
        if not table:
            return True
        for row in table:
            s = str(row.Series).strip()
            c = str(row.Category).strip()
            val_raw = row.TargetValue
            val_str = str(val_raw).strip()
            is_dummy = (
                (s == "" or s.lower() == "n/a") and
                (c == "" or c.lower() == "n/a") and
                (val_str == "" or val_str == "0" or val_str == "0.0" or val_str.lower() == "n/a" or val_raw is None)
            )
            if not is_dummy:
                return False
        return True

    # 1. Resolve raw markdown text source
    raw_markdown = ""
    if result.text_reasoning and "|" in result.text_reasoning:
        raw_markdown = result.text_reasoning
    elif ctx.deps.last_vision_raw_content and "|" in ctx.deps.last_vision_raw_content:
        raw_markdown = ctx.deps.last_vision_raw_content

    # 2. Extract and preserve high-level chart metadata if missing
    if raw_markdown:
        chart_title = result.chart_title
        x_axis = result.x_axis_label
        y_axis = result.y_axis_label
        chart_type = result.chart_type
        units = result.units

        # Parse Chart Type
        if not chart_type:
            lowered_md = raw_markdown.lower()
            if "scatter" in lowered_md:
                chart_type = "scatter_plot"
            elif "bar" in lowered_md:
                chart_type = "bar_chart"
            elif "line" in lowered_md:
                chart_type = "line_graph"
            elif "pie" in lowered_md:
                chart_type = "pie_chart"
            else:
                chart_type = "table"

        # Parse Chart Title
        if not chart_title:
            for line in raw_markdown.splitlines():
                line_clean = line.strip().lstrip("#").strip()
                if line_clean and any(k in line_clean.lower() for k in ["figure", "fig.", "table", "chart", "scatter"]):
                    chart_title = line_clean
                    break

        # Parse Axes Labels from Markdown headers
        lines_with_pipe = [line.strip() for line in raw_markdown.splitlines() if "|" in line]
        if len(lines_with_pipe) >= 2:
            separator_idx = -1
            for idx, line in enumerate(lines_with_pipe):
                if re.match(r"^[\s|:-]+$", line) and "-" in line:
                    separator_idx = idx
                    break
            header_line = None
            if separator_idx != -1 and separator_idx > 0:
                header_line = lines_with_pipe[separator_idx - 1]
            elif len(lines_with_pipe) > 0:
                header_line = lines_with_pipe[0]

            if header_line:
                headers = [col.strip() for col in header_line.split("|") if col.strip()]
                if len(headers) >= 2:
                    if not x_axis:
                        x_axis = headers[1]
                    if len(headers) >= 3 and not y_axis:
                        y_axis = headers[2]

        # Parse Units of measurement
        if not units:
            for term in ["%", "percent", "usd", "tonnes", "kg", "gdp", "emissions"]:
                if term in raw_markdown.lower():
                    units = term.upper() if term in ["usd", "kg"] else term
                    break

        # Save metadata back to the result model
        result = result.model_copy(update={
            "chart_title": chart_title,
            "x_axis_label": x_axis,
            "y_axis_label": y_axis,
            "chart_type": chart_type,
            "units": units
        })

    # 3. Handle visual table data fallback parsing
    if ctx.deps.vision_element_processed:
        if is_table_invalid_res(result.extracted_table):
            val_logger.info("⚠️ [RESULT VALIDATOR] extracted_table is empty/dummy. Performing programmatic fallback parsing from tool output...")
            if raw_markdown:
                parsed_rows = parse_markdown_table_to_dicts(raw_markdown)
                if parsed_rows:
                    val_logger.info(f"✅ [RESULT VALIDATOR] Successfully parsed {len(parsed_rows)} rows. Forcefully populating extracted_table.")
                    new_table = [ChartTableRow(**r) for r in parsed_rows]
                    result = result.model_copy(update={"extracted_table": new_table})
                else:
                    val_logger.warning("❌ [RESULT VALIDATOR] Parser found no valid Markdown table in raw_markdown.")
            else:
                val_logger.warning("❌ [RESULT VALIDATOR] No raw markdown content containing '|' found in text_reasoning or deps.")
                
    return result


@multimodal_agent.system_prompt
def system_prompt(ctx: RunContext[SystemPipelinesDeps]) -> str:
    return (
        "You are a multimodal RAG system helper agent. Your task is to analyze user queries and extract data "
        "using your tools (query_pandas_dataframe, query_qdrant_vector_search, process_vision_element).\n\n"
        "GUIDELINES FOR VISUAL ELEMENTS:\n"
        "1. Identify if the element is a Chart/Diagram or a Document Table.\n"
        "2. IF THE ELEMENT IS A CHART, GRAPH, OR DIAGRAM: Extract every single data point, group, and entity present. "
        "Format them completely into a structured Markdown table using logical, generic column headers inside the 'text_reasoning' field. "
        "Additionally, you MUST programmatically populate the 'extracted_table' field of the output schema with a list of dictionaries "
        "representing these extracted data points. Each dictionary in 'extracted_table' must have exactly these keys:\n"
        "  - 'Series': The name of the data series/line/group (e.g., 'Standard adopted', country name, or indicator). Line/series names must always go here. Numbers must never be mapped as the Series name.\n"
        "  - 'Category': The category/X-axis label/dimension. Income groups ('Low income', 'Lower middle income', 'Upper middle income', 'High income') must ALWAYS be extracted into the 'Category' column.\n"
        "  - 'TargetValue': The precise numerical value (must be formatted as a float, integer, or raw number).\n"
        "You must then provide an exhaustive, point-by-point explanation of all extracted data inside the 'text_reasoning' field, ensuring no entity or metric is omitted.\n"
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
    import io
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
def query_qdrant_vector_search(ctx: RunContext[SystemPipelinesDeps], semantic_query: str, target_collection: str = "conversational_rag") -> str:
    """
    Call this tool for natural language inquiries, contextual knowledge lookups,
    and text chunk extraction from the document collection.
    Use 'conversational_rag' as the target_collection name.
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
        from app.embeddings import get_query_vector
        query_vector = get_query_vector(semantic_query)
        
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

            response = client.query_points(
                collection_name=target_collection,
                query=query_vector,
                query_filter=qdrant_filter,
                using="dense",
                limit=5
            )
            results = response.points
            
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
                
            # Store raw chunks in dependencies context for guardrail evaluations
            if hasattr(ctx.deps, "retrieved_chunks") and isinstance(ctx.deps.retrieved_chunks, list):
                for point in results:
                    chunk = point.payload or {}
                    chunk["id"] = point.id
                    ctx.deps.retrieved_chunks.append(chunk)

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
    img_path = Path(visual_asset_path)
    if not img_path.exists():
        try:
            from app.multimodal_assets import build_asset_registry, normalize_entity_id
            norm_id = normalize_entity_id(visual_asset_path)
            logger.info(f"Normalizing '{visual_asset_path}' to '{norm_id}' for registry lookup")
            
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
                if resolved_path.suffix.lower() == ".csv":
                    resolved = False
                    page_match = re.search(r"page_(\d+)", resolved_path.name, re.IGNORECASE)
                    if page_match:
                        page_no = page_match.group(1)
                        for folder in ["assets/extracted_images", "extracted_images"]:
                            folder_path = Path("C:/Users/supri/recovered-rag-project") / folder
                            if folder_path.exists():
                                for file in folder_path.glob("*"):
                                    if (file.name.lower().startswith(f"page{page_no}_") or file.name.lower().startswith(f"page_{page_no}_")) and file.suffix.lower() == ".png" and not file.name.lower().endswith(".raw.png"):
                                        img_path = file
                                        resolved = True
                                        logger.info(f"CSV resolved to image fallback: {img_path}")
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

    if not img_path.exists():
        return f"Error: Target visual asset path '{visual_asset_path}' could not be resolved or does not exist on disk."

    try:
        import base64
        from openai import OpenAI
        
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
            "2. If the user instructions ask for data from a visual chart/graph, extract all raw data points across all entities and groups. Present them completely as a clean Markdown table using appropriate generic column names. You must capture and detail every single data point and entity present in the figure without shortcuts or omissions. Do not round approximations unnecessarily.\n"
            "STRICT COLUMN MAPPING RULES:\n"
            "- Income groups ('Low income', 'Lower middle income', 'Upper middle income', 'High income') must ALWAYS be mapped to the 'Category' column.\n"
            "- Line/Series names (e.g., 'Standard adopted') belong in the 'Series' column.\n"
            "- Numbers must never be mapped to the 'Series' column name."
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
            global VISION_TOOL_SUCCEEDED, LAST_VISION_RAW_CONTENT
            VISION_TOOL_SUCCEEDED = True
            LAST_VISION_RAW_CONTENT = raw_content
            ctx.deps.last_vision_raw_content = raw_content
            return raw_content
        
    except Exception as exc:
        return f"Vision inference pipeline (OpenRouter) failed with runtime error: {exc}"



from app.embeddings import get_query_vector
from app.reranker import TransformersReranker
from app.conversation_manager import MultimodalConversationManager
from app.multimodal_assets import (
    candidate_asset_paths,
    preview_csv,
    requested_asset_type as detect_requested_asset_type,
    resolve_best_asset,
    validate_asset_path,
)
from app.structured_query import (
    StructuredConstraint,
    StructuredQueryResult,
    extract_structured_constraints,
    get_structured_query_engine,
    looks_like_structured_query,
    should_use_structured_csv_query,
)
from gateway_guardrails import (
    GatewayGuardrailViolation,
    GatewayInfrastructure,
    InsufficientSemanticContent,
    PromptLengthExceeded,
    RateLimitExceeded,
    RetrievalCoverageExceeded,
    TokenBudgetExceeded,
)
from self_rag_utils import step_zero_extract_entities
from compliance_safety import RAGMasterSafetyGauntlet


load_dotenv()
ACTIVE_USER_QUERY = ""
VISION_ELEMENT_PROCESSED = False
VISION_TOOL_SUCCEEDED = False
LAST_VISION_RAW_CONTENT = ""
VALIDATION_ATTEMPT_COUNT = 0

logging.basicConfig(level=logging.INFO, force=True)
import sys
import re

class Base64LogFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str) and "data:image" in record.msg and "base64" in record.msg:
            record.msg = re.sub(
                r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s\r\n]{50,}", 
                "data:image/png;base64,[BASE64_IMAGE_DATA_TRUNCATED]", 
                record.msg
            )
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str) and "data:image" in arg and "base64" in arg:
                    arg = re.sub(
                        r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s\r\n]{50,}", 
                        "data:image/png;base64,[BASE64_IMAGE_DATA_TRUNCATED]", 
                        arg
                    )
                new_args.append(arg)
            record.args = tuple(new_args)
        return True

logging.getLogger().addFilter(Base64LogFilter())

pydantic_ai_logger = logging.getLogger("pydantic_ai")
pydantic_ai_logger.setLevel(logging.DEBUG)
pydantic_ai_logger.propagate = False
pydantic_ai_logger.addFilter(Base64LogFilter())

pydantic_file_handler = logging.FileHandler("streamlit_pydantic.log", mode="a", encoding="utf-8")
pydantic_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
pydantic_file_handler.addFilter(Base64LogFilter())

pydantic_console_handler = logging.StreamHandler(sys.stdout)
pydantic_console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
pydantic_console_handler.addFilter(Base64LogFilter())

if not pydantic_ai_logger.handlers:
    pydantic_ai_logger.addHandler(pydantic_file_handler)
    pydantic_ai_logger.addHandler(pydantic_console_handler)

for logger_name in ["httpx", "openai"]:
    l = logging.getLogger(logger_name)
    l.setLevel(logging.INFO)
    l.propagate = False
    l.addFilter(Base64LogFilter())
    if not l.handlers:
        l.addHandler(pydantic_file_handler)
        l.addHandler(pydantic_console_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APP_TITLE = "Enterprise Multi-Format RAG Assistant"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "conversational_rag"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
GEMINI_MODEL_NAME = "gemini-3.5-flash"
GROQ_LLAMA_70B_MODEL_NAME = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
LLAMA_70B_MAX_ATTEMPTS = 3
LLAMA_70B_INITIAL_BACKOFF_SECONDS = 3.0
GEMINI_MAX_ATTEMPTS = 4
GEMINI_RETRY_BACKOFF_SECONDS = 3.0
GEMINI_TRANSIENT_FAILURE_MESSAGE = (
    "Gemini is temporarily busy due to rate or capacity limits. "
    "Please pause for a moment and try your question again."
)
NVIDIA_LLAMA_MODEL_NAME = os.getenv("NVIDIA_LLAMA_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_FINAL_MODEL_NAME = os.getenv("NVIDIA_FINAL_MODEL", "meta/llama-3.3-70b-instruct")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
PREFETCH_LIMIT = 20
RRF_LIMIT = 15
RERANK_TOP_N = 10
PRIMARY_DENSE_TOP_K = 5
GLOBAL_ANALYTICS_LIMIT = 15
HYBRID_RESULT_LIMIT = PRIMARY_DENSE_TOP_K
ASSET_QUERY_INTERNAL_LIMIT = int(os.getenv("ASSET_QUERY_INTERNAL_LIMIT", "12"))
SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
BM25_MODEL_NAME = os.getenv("FASTEMBED_BM25_MODEL", "Qdrant/bm25")
RRF_K = 60
INDEXED_ENTITY_PAYLOAD_FIELDS = (
    "entity_id",
    "entity",
    "label",
    "name",
    "country",
    "country_name",
    "metadata.entity_id",
    "metadata.entity_ids",
    "metadata.entity",
    "metadata.entity_label",
    "metadata.label",
    "metadata.name",
    "metadata.country",
    "metadata.country_name",
    "metadata.figure_id",
    "metadata.cross_reference",
    "metadata.cross_references",
    "metadata.source_file",
    "metadata.title",
)
PRIMARY_ENTITY_PAYLOAD_FIELDS = (
    "entity_id",
    "entity",
    "label",
    "name",
    "country",
    "country_name",
    "metadata.entity_id",
    "metadata.entity_ids",
    "metadata.entity",
    "metadata.entity_label",
    "metadata.label",
    "metadata.name",
    "metadata.country",
    "metadata.country_name",
    "metadata.figure_id",
    "metadata.source_file",
    "metadata.title",
)

ASSET_PAYLOAD_FIELDS = (
    "entity_type",
    "metadata.entity_type",
    "csv_path",
    "csv_paths",
    "table_csv_path",
    "table_csv_paths",
    "image_path",
    "image_paths",
    "table_image_path",
    "table_image_paths",
    "figure_image_path",
    "figure_image_paths",
    "chart_image_path",
    "chart_image_paths",
    "metadata.csv_path",
    "metadata.csv_paths",
    "metadata.table_csv_path",
    "metadata.table_csv_paths",
    "metadata.image_path",
    "metadata.image_paths",
    "metadata.table_image_path",
    "metadata.table_image_paths",
    "metadata.figure_image_path",
    "metadata.figure_image_paths",
    "metadata.chart_image_path",
    "metadata.chart_image_paths",
)
CHAPTER_REFERENCE_PATTERN = re.compile(
    r"\bchapter\s+(?P<number>\d+|[ivxlcdm]+)\b",
    flags=re.IGNORECASE,
)
CHAPTER_PAYLOAD_FIELDS = (
    "chapter_number",
    "metadata.chapter_number",
)
TRANSCRIBE_API_URL = os.getenv("TRANSCRIBE_API_URL", "http://localhost:8080/api/transcribe")
COMPARATIVE_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "across charts",
    "both",
    "between",
)
HARD_ENTITY_PATTERN = re.compile(
    r"\b(?P<kind>fig(?:ure|ured)?|figure|figured|fig|tab(?:le|el)?|table|tabel|chart)[\s_]*"
    r"(?P<identifier>[Oo0]?\s*\.?\s*\d+(?:\s*\.\s*\d+)*)",
    flags=re.IGNORECASE,
)
STRUCTURAL_REFERENCE_PATTERN = re.compile(
    r"\b(?P<kind>fig(?:ure)?|table|chart)[\s_]*(?P<identifier>\d+(?:\.\d+)*)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
STRUCTURAL_IDENTIFIER_PATTERN = re.compile(r"\b(?:[Oo]\.)?\d+\.\d+\b", flags=re.IGNORECASE)
TABLE_ASSET_PATTERN = re.compile(r"\b(?:table|tabel)\b", flags=re.IGNORECASE)
FIGURE_TABLE_GUARDRAIL = (
    "Critical guardrail for figures and tables: Figures labeled with an 'O' such as Figure O.8 are "
    "Overview figures and are usually executive-summary duplicates or identical reprints of corresponding "
    "chapter figures such as Figure 8.4. If retrieved descriptions for an Overview figure and a Chapter "
    "figure have minor wording variations, do not assume the physical chart data points are different. "
    "Look for core semantic alignment: if both charts cover the same countries, years, and metrics, treat "
    "them as the same underlying graphic, state that they represent the same data, and synthesize details "
    "together. Flag a difference only if the context explicitly states that one chart modifies, updates, "
    "or expands upon the other."
)
EXECUTIVE_ANSWER_STYLE = (
    "Answer style: act as a sharp, executive-level research analyst. Start immediately with a direct "
    "1-2 sentence analytical thesis that answers the core question and outlines the overarching relationship "
    "or mechanism, with no setup phrases like 'Based on the retrieved context'. Break the explanation into "
    "logical thematic sections using bold headings, short paragraphs, or substantive bullets. Do not merely "
    "list figures; explain the causal chain, including how or why one factor influences another. Integrate "
    "chart and text references naturally as inline evidence anchors, such as 'which lowers export costs for "
    "local firms [Figure 3.10]'. Never make a figure number the grammatical subject of a sentence, and avoid "
    "phrases like 'Figure 3.2 shows'. Write with professional clarity and a natural, fluid voice; avoid rigid, "
    "formulaic bullet prefixes unless they genuinely serve the narrative flow."
)
GROUNDED_NO_DATA_RESPONSE = (
    "Request failed Layer 1 Retrieval validation because retrieved evidence was insufficient to support generation. "
    "The response was blocked before delivery."
)
NO_RELEVANT_EVIDENCE_RESPONSE = (
    "Request failed Layer 1 Retrieval validation because no matching document chunks were retrieved from Qdrant. "
    "Generation was intentionally blocked."
)
SAFE_REFUSAL_RESPONSE = "I cannot process that request because it attempts to bypass system controls or access internal instructions."
TOKEN_BUDGET_RESPONSE = (
    "Your message is too long to process safely. Please shorten it and try again."
)
INSUFFICIENT_SEMANTIC_CONTENT_RESPONSE = (
    "Request contains insufficient semantic content. Please submit a meaningful question."
)
RETRIEVAL_COVERAGE_RESPONSE = (
    "Request exceeds retrieval coverage limits. Please ask for a specific chapter, section, table, figure, or topic."
)
RATE_LIMIT_RESPONSE = "Too many requests were sent in a short time. Please wait a moment and try again."
SCHEMA_FAILURE_RESPONSE = (
    "The generated response for Request failed Layer 8 schema validation because it did not conform to the required "
    "response schema. The response was rejected before delivery."
)
PROMPT_LEAKAGE_RESPONSE = (
    "Protected system instructions were detected in generated output for Request during Layer 11 prompt leakage "
    "validation and were automatically removed."
)
GENERATION_FAILURE_RESPONSE = (
    "Request failed during grounded generation because the validated response could not be produced from the retrieved "
    "source data. The response was blocked before delivery."
)
USER_FACING_PERSONA_GUARDRAIL = (
    "Document-grounding guardrail: answer only from retrieved uploaded-document context. Never answer from world "
    "knowledge, training data, assumptions, or general background knowledge. Never answer questions unrelated to "
    "retrieved context. If validation fails, return a structured validation-layer failure message instead of a "
    "generic no-data response. "
    "User-facing persona guardrail: never mention internal database logistics, retrieval mechanics, context chunks, "
    "vector search, payloads, image-processing quality, backend failures, safety mechanisms, guardrails, internal rules, "
    "or database structure to the user. Absolutely do not use phrases "
    "such as 'The provided context does not contain sufficient information', 'According to Context Chunk X', "
    "'The image is too blurry/simplistic to extract data', or 'I cannot find this information in the database'. "
    "If the user asks a vague follow-up such as 'tell me more about this' or 'explain this further', use the immediate "
    "prior conversation turn to infer what 'this' refers to. If retrieved material contains internal engineering notes, "
    "image processing errors, OCR caveats, or phrases like 'blurry image', ignore those notes completely and do not echo "
    "them. If there is not enough clean, concrete evidence to answer, return a structured validation-layer failure "
    "message instead of a generic no-data response."
)
BANNED_USER_FACING_PHRASES = (
    "the provided context does not contain sufficient information",
    "according to context chunk",
    "context chunk",
    "the image is too blurry",
    "too blurry/simplistic",
    "i cannot find this information in the database",
)
RELEVANCE_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "and",
    "any",
    "are",
    "because",
    "before",
    "between",
    "both",
    "can",
    "contents",
    "could",
    "data",
    "database",
    "document",
    "documents",
    "does",
    "explain",
    "find",
    "for",
    "from",
    "give",
    "have",
    "how",
    "into",
    "more",
    "not",
    "pdf",
    "qdrant",
    "retrieved",
    "show",
    "source",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "there",
    "this",
    "uploaded",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def _debug_log_chunks(step_name: str, chunks: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 96}\n--- {step_name} ---\nTotal chunks: {len(chunks)}", file=sys.stderr, flush=True)
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Chunk {index} | id={chunk.get('id', 'unknown')} | source={chunk.get('source', 'unknown')} | "
            f"fusion_score={chunk.get('fusion_score')} | rrf_score={chunk.get('rrf_score')} | "
            f"rerank_score={chunk.get('rerank_score')} | dense_rank={chunk.get('dense_rank')} | "
            f"sparse_rank={chunk.get('sparse_rank')} | metadata={chunk.get('metadata', {})}\n"
            f"TEXT:\n{chunk.get('content', '')}",
            file=sys.stderr,
            flush=True,
        )
    print("=" * 96, file=sys.stderr, flush=True)
GLOBAL_ANALYTICS_PATTERN = re.compile(
    r"\b(highest|lowest|maximum|max|min(?:imum)?|largest|smallest|total|sum|aggregate|"
    r"across\s+(?:the\s+)?(?:entire\s+)?(?:dataset|file|table|csv)|entire\s+(?:dataset|file|table|csv))\b",
    flags=re.IGNORECASE,
)
GLOBAL_ANALYTICS_RETRIEVAL_SUFFIX = (
    "\nPrioritize complete dataset summaries, table headers, CSV rows, country records, regional rows, "
    "and records needed to calculate a dataset-wide aggregate or extremum."
)
GLOBAL_ANALYTICS_FORMATTER_GUARDRAIL = (
    "Global analytics guardrail: this is a dataset-wide aggregate or highest/lowest request. Calculate only "
    "from the visible records. If the retrieved material is a limited subset rather than a complete dataset-wide "
    "cross-section, explicitly qualify the answer with a concise phrase such as 'Based on the retrieved report "
    "chapters...' and do not claim a definitive global maximum, minimum, or total."
)
QUERY_CONDENSER_PROMPT = """You are an advanced Conversational Query Condenser designed for a production RAG pipeline. Your sole objective is to take a user's latest query along with the recent conversation history and output targeted standalone search queries optimized for a vector and keyword database.

Follow these strict operational rules:

1. RESOLVE CONTEXT DRIFT & PRONOUNS:
If the user's latest message relies on the context of the past conversation (using terms like "it", "they", "this", "by how much", "what about [Year]", "is it higher?"), reconstruct the question entirely. Infuse all necessary entity anchors (e.g., exact country names, specific metrics, indices, table references, and dates) from the history into the new query.

2. DETECT TOPIC SWITCHES (CRITICAL):
If the user's latest query introduces a completely new metric, schema, column name, or concept that was NOT present or related to the immediate history, DO NOT force the old context into the new query. Drop the history entirely and rewrite the query to focus 100% on the new target across the entire dataset. Do not trap the user in an old topic.

CRITICAL TOPIC-SWITCH RULE: Evaluate if the user's latest query is a sudden, complete departure from the previous chat history (e.g., switching from abstract standards back to country metrics like GDP). If a complete topic switch is detected, do NOT merge it with the history. Instead, completely ignore the history and pass the latest query through verbatim as a standalone search query.

3. STRIP ALL GRAPHICS AND LAYOUT META-COMMENTARY:
Never include phrases regarding chunk formatting, database structural complaints, or image quality (e.g., do NOT include "in the blurry image", "as seen in the context chunk"). Keep it strictly focused on the core data.

4. PRESERVE HARD IDENTIFIERS EXACTLY:
If the user's message contains an explicit identifier such as "Table X.X", "Figure X.X", or a specific number, preserve every such string literal exactly as typed in the standalone query. Never renumber, normalize, omit, paraphrase, or replace those literals.
If the user's input query mentions multiple structural entities, chart labels, figures, or table identifiers (e.g., "Figure 4.1", "Table 2.2", "3.7"), the generated standalone query MUST explicitly preserve and list ALL alphanumeric identifiers. Do not compress them into generic pronouns like "both figures" or "the previous chart".

5. SYSTEM CONTRACT - OUTPUT STRUCTURE:
- Analyze the user's input for ANY mentions of multiple data points, tables, figures, charts, chapters, or comparative concepts.
- If multiple entities or structural elements are detected, decompose the request into one targeted standalone search string per unique entity or structural element.
- Output ONLY a valid JSON array of search strings, even when there is only one query.
- Do NOT include markdown code blocks.
- Do NOT include conversational filler, introductory remarks, or explanations.
- If the user's query is already fully standalone, preserve its wording inside a single-item JSON array.

Example Input: "Compare Table 1.1 with Figure 4.2"
Example Output: ["Table 1.1 data and metrics", "Figure 4.2 chart data visualization"]

Example Input: "Summarize the metrics in Chapter 5 tables"
Example Output: ["Chapter 5 tables metrics", "Chapter 5 data infrastructure"]

EXAMPLES OF EXPECTED BEHAVIOR:

Example 1 (Fragmented Follow-up):
- History: [User: "What is India's GDP in 2024?", AI: "It is approximately $3.909 trillion."]
- Latest Query: "Is it higher or lower than China?"
- Output: Compare the 2024 GDP of India with the 2024 GDP of China

Example 2 (The "By How Much" Edge Case):
- History: [User: "Is India's GDP higher or lower than China?", AI: "India's GDP is lower than China's."]
- Latest Query: "By how much?"
- Output: What is the exact numerical difference in USD between the GDP of China and the GDP of India in 2024

Example 3 (Topic Switch Detection):
- History: [User: "What are the vehicle emission trends for China?", AI: "China progressed through stages 1-7 between 2008 and 2016."]
- Latest Query: "Which country has the highest GDP in the dataset?"
- Output: Which country or region has the maximum GDP value across the entire dataset"""
HYDE_SYSTEM_PROMPT = """You are an expert Data Simulator for an advanced HyDE (Hypothetical Document Embedding) RAG pipeline. Your job is to take a standalone user query and generate a fake, ideal document snippet that looks exactly like a high-quality chunk extracted from our underlying dataset (reports, CSV logs, or academic text).

Follow these strict structural rules:

1. SIMULATE THE RIGHT SCHEMA:
   - If the query asks for numerical comparisons, metrics, or data logs, output a simulated text block or markdown table snippet containing those data fields.
   - If the query is conceptual, output a dense, factual textbook or enterprise report paragraph.

2. THE PLACEHOLDER MANDATE (CRITICAL):
   - Never invent or guess specific numbers, metrics, or percentages if they are not explicitly implied by the query.
   - Use uppercase variables or bracketed placeholders (e.g., [X], [VALUE], [Y%], [DATE]) for all unknown data points.
   - Focus 100% on writing a grammatically perfect answer structure so the vector matching engine can map "answer semantics" to "answer semantics".

3. SYSTEM CONTRACT - OUTPUT STRUCTURE:
   - Output ONLY the simulated text or table chunk.
   - Do NOT include conversational preambles ("Here is the simulated document:").
   - Do NOT include markdown code blocks.

EXAMPLES OF EXPECTED HYDE BEHAVIOR:

Example 1 (Tabular Metric Intent):
- Input Query: "Compare the 2024 GDP of India with the 2024 GDP of China"
- Output: In the 2024 economic reporting period, China's Gross Domestic Product (GDP) reached [X] trillion USD, while India's GDP for the same fiscal year was logged at [Y] trillion USD, representing an absolute difference of [Z] trillion USD.

Example 2 (Global Ranking Analytics):
- Input Query: "Which country or region has the maximum GDP value across the entire dataset"
- Output: Region/Country: [COUNTRY_NAME] | Metric: Gross Domestic Product (GDP) | Year: [YEAR] | Value: [MAX_VALUE_USD] | Status: Highest global recorded value in dataset."""
INTENT_ROUTER_PROMPT = """You are a strict intent router for a production RAG assistant.

Classify the user's latest message into exactly one category:

DIRECT_RESPONSE
- Use only for greetings, compliments, pleasantries, thanks, farewells, or meta-questions about the AI assistant itself.

DATA_RETRIEVAL
- Use for any query requiring facts, metrics, comparisons, explanations of report content, figure or table details, document search, or data analysis.
- If uncertain, choose DATA_RETRIEVAL.

Output ONLY one raw token: DIRECT_RESPONSE or DATA_RETRIEVAL.
Do not include markdown, punctuation, explanations, or formatting."""
DIRECT_RESPONSE_PROMPT = """You are a concise, professional conversational assistant.
Respond naturally to the user's greeting, pleasantry, compliment, thanks, farewell, or meta-question about the assistant itself.
Do not claim to have searched documents or analyzed data.
Keep the answer brief and helpful."""
CONTEXT_EVALUATOR_PROMPT = """You are a highly precise, automated Context Relevance Gatekeeper. Your sole function is to analyze a user query against a block of retrieved document chunks and determine if the text contains the factual information required to answer the query. You must ignore fluff and look specifically for alphanumeric entities, table references, figure IDs, or matching concepts.

You must respond in strict JSON format with no markdown wrappers, no conversational filler, and no explanation. Your output must strictly match this structure:
{"is_relevant": "yes"} 
OR
{"is_relevant": "no"}"""
GROUNDED_QA_PROMPT = """You are an expert document analysis engine. Your goal is to answer the user's question accurately based on the provided text chunks.

Rules for Synthesis:
0. You are looking at a combined view of extracted text tables and visual figures. Analyze how the structural numbers in the table align with the trends plotted in the corresponding chart image/description. Provide comparative summaries, point out correlations, and explicitly reference both by their titles in your answer.
1. Be Semantically Flexible: If the user asks about a specific table or concept (e.g., "Table 2.1" or a definition) and the chunks contain highly relevant data under a slightly different label (e.g., "Table 3.1" or structural examples of the concept), explain the connection to the user rather than giving a blank rejection.
2. Synthesize Across Elements: Gather information from all retrieved chunks simultaneously to construct your response.
3. No Hallucinations: Keep your facts strictly tied to the provided text blocks. Never answer from world knowledge, training data, assumptions, or general background knowledge.
4. Fallback: If the chunks do not contain the answer, return a structured retrieval validation failure instead of a generic no-data response. """
SECURE_GENERATION_PROMPT = """Step 4: Secure Generation.
You are producing a held-back draft answer for a Self-RAG pipeline. This draft will be verified by a later hallucination gatekeeper before it is shown to the user.

Rules:
1. Use ONLY the verified context chunks provided in this request. These chunks have already passed relevance grading or exact fallback retrieval.
2. Ground the entire answer in the supplied chunks. Do not use outside knowledge, world knowledge, training data, assumptions, or the HyDE text as evidence.
3. Reference specific alphanumeric entities such as Table 3.1, Table 3.2, Figure 4.2, section identifiers, country names, years, and metric labels whenever they appear in the chunks.
4. If tables, matrices, or row data are present, render them as valid GitHub-Flavored Markdown tables before explaining them.
5. If figure or image metadata is present, reference the figure by its exact title or identifier and include verified image paths using Markdown image syntax only when a path is supplied in metadata.
6. For every metric, chart insight, table value, figure description, or diagram interpretation, explicitly name the specific Figure or Table identifier/title from the context that supports it.
7. Produce a structured analytical draft with a direct answer first, then concise supporting bullets or tables."""
HALLUCINATION_JUDGE_PROMPT = """You are an extremely strict, zero-tolerance Hallucination Judge. Your job is to verify if a Draft Answer is 100% textually grounded in the provided Context Chunks.

CRITICAL RULES:
1. If the Draft Answer uses superlative, subjective, or ranking language (e.g., 'most important', 'best', 'only', 'highest') but the Context Chunks merely list, classify, or present data without explicitly stating that exact opinion or ranking, you MUST mark it as a hallucination.
2. The Draft Answer must not assume, infer, or extrapolate beyond the raw text. 
3. If there is ANY minor mismatch or unverified opinion inserted by the generator, the answer is NOT grounded.

Respond ONLY in this strict JSON format with no markdown wrappers or backticks:
{"is_grounded": "no"}
OR
{"is_grounded": "yes"}"""
SELF_CORRECTED_REWRITE_PROMPT = """Self-corrected rewrite instruction:
The previous draft may have included unsupported claims. Rewrite the answer using ONLY facts explicitly visible in the retrieved chunks.
Delete any claim, number, metric, comparison, table row, figure interpretation, or inference that is not directly supported by the chunks.
If the chunks do not support a specific requested detail, return a structured validation-layer failure instead of a generic no-data response.
Keep the answer concise, structured, and grounded."""
EXECUTIVE_FORMATTER_PROMPT = """You are an elite corporate research analyst providing executive briefs to leadership. Answer the user's query utilizing ONLY the facts, metrics, and tables present in the provided retrieved context.

Follow these strict professional formatting and behavior guardrails:

1. Bottom-Line Up Front (BLUF): Answer the core question immediately in the very first sentence. Use bold text for key metrics, numbers, and dates.
2. Absolute Math Determinism: If the user is asking for a comparison, a percentage change, or a numerical difference (e.g., "by how much?"), look at the retrieved text/tables, calculate the exact mathematical difference, and present the calculation clearly. Never let the model guess or gloss over numerical comparisons.
3. No Robotic/System Filler Text: NEVER include engineering notes, system meta-commentary, or lazy academic boilerplate headers such as "Conclusion:", "Key Findings:", "Data Source:", "Introduction:", or "According to Context Chunk 2...".
4. The Invisible Database: Seamlessly integrate statistics into your sentences naturally. Do not refer to "the provided dataset", "the database", "evidence items", or "retrieved chunks". Speak as though you possess the data organically (e.g., "World Development Report metrics demonstrate that...").
5. Concise Density: Use clean bullet points for supporting context. Keep paragraphs strictly to a maximum of two sentences.

THE GROUNDING MANDATE:
- You will be provided with three components: a User Query, a Hypothetical Answer (HyDE), and Real Retrieved Chunks from Qdrant.
- CRITICAL: The Hypothetical Answer contains FAKE placeholder data used solely for database routing. Completely IGNORE, DESTROY, and DISREGARD any numbers, percentages, dates, or metrics found inside the Hypothetical Answer.
- Ground the final response 100% strictly in the data found within the Real Retrieved Chunks from Qdrant. If a number is not in the Qdrant chunks, it does not exist.

STRUCTURAL ADAPTATION:
- You may use the structural layout suggested by the user's intent or the HyDE document, such as a markdown table comparison, bulleted list, or financial report style.
- Populate that layout using ONLY real Qdrant chunk data.
- When presenting extracted table data, format it as a clean Markdown table using pipe-delimited rows such as `| Column | Value |`.
- When a relevant visual figure or diagram has a verified local image path in its retrieved metadata, include it using standard Markdown image syntax: `![Chart Description](path_to_extracted_image.png)`. Never invent an image path or base64 value.
- Whenever the retrieved context contains raw table data or a matrix such as Table 3.1 or Table 3.2, you MUST explicitly format it as a GitHub-flavored Markdown table using `|` dividers. Do not just summarize it in prose; output the structural table first, followed by your description.
- If a figure or image pathway such as Figure 3.1 is present in the retrieved context metadata, output it using standard Markdown image syntax: `![Figure Description](image_path_or_base64_string)`.
- Whenever you include a table or comparative matrix in your response, you MUST format it as a valid GitHub-Flavored Markdown table using pipe characters `|` for columns and a structural alignment row such as `|---|---|`. Never output a table as a plain text list or a standard block of text. If you reference a visual chart or diagram file path, always insert it using the explicit Markdown image syntax: `![Caption](path_to_image)`.

UNCERTAINTY HANDLING:
- If the Real Retrieved Chunks from Qdrant do not contain the requested answer, return a structured validation-layer failure instead of a generic no-data response.
- Never use pre-trained knowledge or the hypothetical document as factual evidence."""


@st.cache_resource
def load_reranker_model() -> TransformersReranker:
    return TransformersReranker(RERANK_MODEL_NAME)


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    settings = QdrantSettings(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name=COLLECTION_NAME,
    )
    logger.info("Connecting Streamlit app to Qdrant at %s", settings.url or f"{settings.host}:{settings.port}")
    return build_managed_qdrant_client(settings)


@st.cache_resource
def get_sparse_encoder() -> SafeSparseEncoder:
    return SafeSparseEncoder(BM25_MODEL_NAME)


@st.cache_resource
def get_groq_client(api_key: str) -> Groq:
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY before running conversational queries.")
    return Groq(api_key=api_key)


class NvidiaLlamaModel:
    """Small NVIDIA NIM text-generation wrapper used by pre-retrieval stages."""

    def __init__(self, api_key: str, model_name: str = NVIDIA_LLAMA_MODEL_NAME) -> None:
        if not api_key:
            raise RuntimeError("Set NVIDIA_API_KEY before running NVIDIA LLaMA stages.")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL, timeout=60.0)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            timeout=60.0,
        )
        return str(response.choices[0].message.content or "").strip()


class GroqModel:
    """Explicit Groq Llama wrapper used by the query generation, HyDE, and generation stages."""

    def __init__(self, api_key: str) -> None:
        self.client = get_groq_client(api_key)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        primary_model = "llama-3.3-70b-versatile"
        fallback_model = "llama-3.1-8b-instant"

        def is_503_or_429(exc: Exception) -> bool:
            s = str(exc).lower()
            return any(k in s for k in ["503", "429", "unavailable", "resource_exhausted", "rate limit", "quota", "overloaded", "apierror", "service"])

        try:
            response = self.client.chat.completions.create(
                model=primary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
            )
            return str(response.choices[0].message.content or "").strip()
        except Exception as exc:
            if is_503_or_429(exc):
                print(f"⚠️ Warning: Groq {primary_model} busy or rate-limited ({exc}), switching to {fallback_model} fallback...", file=sys.stderr, flush=True)
                logger.warning("Groq primary busy or rate-limited, switching to fallback... Error: %s", exc)
                try:
                    response = self.client.chat.completions.create(
                        model=fallback_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature,
                    )
                    return str(response.choices[0].message.content or "").strip()
                except Exception as fallback_exc:
                    if is_503_or_429(fallback_exc):
                        print(f"⚠️ Warning: Groq fallback also rate-limited ({fallback_exc}), pausing for 5 seconds to cool down...", file=sys.stderr, flush=True)
                        logger.warning("Groq fallback also rate-limited, pausing for 5 seconds to cool down... Error: %s", fallback_exc)
                        time.sleep(5)
                        print(f"🔄 Retrying final request to {primary_model}...", file=sys.stderr, flush=True)
                        logger.info("Retrying final request to Groq primary after cooldown.")
                        try:
                            response = self.client.chat.completions.create(
                                model=primary_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=temperature,
                            )
                            return str(response.choices[0].message.content or "").strip()
                        except Exception as final_exc:
                            logger.error("Final primary retry failed: %s", final_exc)
                            raise final_exc
                    else:
                        logger.error("Groq fallback failed with non-quota error: %s", fallback_exc)
                        raise fallback_exc
            else:
                raise exc


@st.cache_resource
def get_nvidia_llama_model(api_key: str) -> NvidiaLlamaModel:
    return NvidiaLlamaModel(api_key)


@st.cache_resource
def get_nvidia_final_model(api_key: str) -> NvidiaLlamaModel:
    return NvidiaLlamaModel(api_key, model_name=NVIDIA_FINAL_MODEL_NAME)


@st.cache_resource
def get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable before running OpenRouter queries.")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


@st.cache_resource
def get_groq_model(api_key: str) -> GroqModel:
    return GroqModel(api_key)


@st.cache_resource
def get_gateway() -> GatewayInfrastructure:
    return GatewayInfrastructure()


def mask_pii_text(text: Any) -> str:
    return get_gateway().mask_pii(str(text or ""))


def gateway_user_message(exc: GatewayGuardrailViolation) -> str:
    if isinstance(exc, (PromptLengthExceeded, TokenBudgetExceeded)):
        return (
            f"Request failed Layer 3 Rate Limiting and Token Budget validation: {exc}. "
            "The request was blocked before retrieval."
        )
    if isinstance(exc, InsufficientSemanticContent):
        return (
            f"Request failed Layer 3 semantic-content validation: {exc}. "
            "The request was blocked before retrieval."
        )
    if isinstance(exc, RetrievalCoverageExceeded):
        return (
            f"Request failed retrieval coverage validation: {exc}. "
            "The request was blocked before vector-store retrieval."
        )
    if isinstance(exc, RateLimitExceeded):
        return (
            f"Request failed Layer 3 Rate Limiting and Token Budget validation: {exc}. "
            "The request was blocked before retrieval."
        )
    return f"Request failed gateway validation: {exc}. The request was blocked before retrieval."


def layer3_user_message(reason: str) -> str:
    if reason in {"Prompt length exceeded", "Token budget exceeded"}:
        return f"Request failed Layer 3 Rate Limiting and Token Budget validation: {reason}. The request was blocked before retrieval."
    if reason == "Rate limit reached":
        return f"Request failed Layer 3 Rate Limiting and Token Budget validation: {reason}. The request was blocked before retrieval."
    return f"Request failed gateway validation: {reason}. The request was blocked before retrieval."


def requested_entity_name(query: str, locked_entities: list[str] | None = None) -> str:
    for entity in locked_entities or []:
        value = str(entity or "").strip()
        if value:
            return value
    hard_entities = extract_hard_entities(query)
    if hard_entities:
        return hard_entities[0]["label"]
    chapter_refs = extract_chapter_references(query)
    if chapter_refs:
        return f"Chapter {chapter_refs[0]}"
    return (str(query or "Request").strip()[:120] or "Request")


def _structured_constraint_label(constraint: StructuredConstraint) -> str:
    indicator_label = constraint.indicator.upper() if constraint.indicator else "value"
    return f"{constraint.country_name} {indicator_label} {constraint.year}".strip()


def _structured_csv_chunk(document: Any, index: int) -> dict[str, Any]:
    metadata = dict(getattr(document, "metadata", {}) or {})
    metadata.setdefault("document_type", "csv")
    metadata.setdefault("source_type", "csv")
    metadata.setdefault("contains_csv", True)
    metadata.setdefault("retrieval_mode", "structured_csv_exact")
    metadata.setdefault("retrieval_source", metadata.get("retrieval_source") or "pandas_structured")
    source = str(metadata.get("source") or metadata.get("source_files") or "Data/csv")
    return {
        "id": f"structured_csv::{metadata.get('source_files') or Path(source).name}::{metadata.get('country_iso3') or 'row'}::{metadata.get('year') or index}",
        "content": str(getattr(document, "page_content", "") or ""),
        "source": source,
        "fusion_score": 1.0,
        "rerank_score": 1.0,
        "matched_sub_queries": [],
        "metadata": metadata,
    }


def _structured_csv_answer(result: StructuredQueryResult) -> tuple[str, list[dict[str, Any]]]:
    chunks = [_structured_csv_chunk(document, index) for index, document in enumerate(result.answer_documents, start=1)]
    answer = "\n\n".join(chunk["content"] for chunk in chunks if str(chunk.get("content") or "").strip())
    return answer, chunks


def _run_structured_csv_query(user_query: str) -> tuple[str, list[dict[str, Any]], bool]:
    constraints = extract_structured_constraints(user_query)
    if not constraints or not looks_like_structured_query(user_query):
        return "", [], False
    if not should_use_structured_csv_query(user_query):
        return "", [], False

    result = get_structured_query_engine().answer(user_query)
    if result.has_complete_answer:
        answer, chunks = _structured_csv_answer(result)
        return answer, chunks, True

    missing = result.missing_constraints or constraints
    missing_label = _structured_constraint_label(missing[0])
    return retrieval_failure_message(missing_label), [], True


def retrieval_failure_message(entity_name: str, knowledge_base: str = "uploaded knowledge base") -> str:
    return (
        f"{entity_name} was not found in the {knowledge_base}. Layer 1 retrieval validation failed because no "
        "matching document chunks were retrieved from Qdrant, so generation was intentionally blocked."
    )


def asset_path_failure_message(asset_name: str) -> str:
    return (
        f"A reference to {asset_name} was detected, but Layer 4 asset path validation failed because the corresponding "
        "asset path could not be verified on disk. The request was blocked to prevent hallucinated visual content."
    )


def file_not_found_failure_message(filename: str) -> str:
    return (
        f"The file '{filename}' failed file access validation because it does not exist in the approved document corpus. "
        "Access was denied."
    )


def path_traversal_failure_message(path: str) -> str:
    return (
        f"The requested path '{path}' failed path traversal validation because it is outside the approved asset "
        "directory. Access was denied for security reasons."
    )


def layout_validation_failure_message(entity_name: str) -> str:
    return (
        f"Visual metadata for {entity_name} failed Layer 5 layout validation because the bounding box format was "
        "invalid. The visual response was rejected before delivery."
    )


def entity_cross_check_failure_message(value: str) -> str:
    return (
        f"The generated value '{value}' failed Layer 6 entity cross-check validation because it could not be verified "
        "in the retrieved source data. The response was blocked to prevent unsupported claims."
    )


def quote_anchor_failure_message(entity_name: str = "Request") -> str:
    return (
        f"The quoted text for {entity_name} failed Layer 7 quote-anchor validation because it could not be located "
        "in the retrieved document context. The unsupported quote was removed."
    )


def schema_failure_message(entity_name: str = "Request") -> str:
    return (
        f"The generated response for {entity_name} failed Layer 8 schema validation because it did not conform to "
        "the required response schema. The response was rejected before delivery."
    )


def null_asset_failure_message(asset_name: str = "visual asset") -> str:
    return (
        f"The response referenced {asset_name}, but Layer 9 null asset validation failed because the asset path was "
        "empty or null. Rendering was blocked."
    )


def prompt_leakage_failure_message(entity_name: str = "Request") -> str:
    return (
        f"Protected system instructions were detected in generated output for {entity_name} during Layer 11 prompt "
        "leakage validation and were automatically removed."
    )


def dlp_failure_message(entity_name: str = "Request") -> str:
    return (
        f"Potentially sensitive infrastructure information was detected in the response for {entity_name} during "
        "Layer 12 DLP validation and was removed from the response."
    )


def format_masked_history(history: list[dict[str, Any]], max_turns: int = 3) -> str:
    recent_history = history[-max(max_turns, 0) * 2 :] if max_turns else []
    history_text = "\n".join(
        f"{turn.get('role', '')}: {mask_pii_text(turn.get('content', ''))}"
        for turn in recent_history
    )
    return mask_pii_text(history_text)


def resolve_groq_api_key(explicit_key: str = "") -> str:
    if explicit_key.strip():
        return explicit_key.strip()
    return os.getenv("GROQ_API_KEY", "").strip()


def resolve_nvidia_api_key(explicit_key: str = "") -> str:
    return explicit_key.strip() or os.getenv("NVIDIA_API_KEY", "").strip()





def is_resource_exhausted_error(exc: Exception) -> bool:
    message = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return (
        str(status) == "429"
        or "429" in message
        or "resource_exhausted" in message
        or "quota" in message
        or "rate_limit" in message
        or "rate limit" in message
    )


def is_transient_provider_error(exc: Exception) -> bool:
    message = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return (
        is_resource_exhausted_error(exc)
        or str(status) == "503"
        or "503" in message
        or "service unavailable" in message
        or "overloaded" in message
    )


def call_with_llama_retry(callable_fn: Any, *, description: str = "Llama 3.3 70B call") -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, LLAMA_70B_MAX_ATTEMPTS + 1):
        try:
            return callable_fn()
        except Exception as exc:
            last_exc = exc
            if is_transient_provider_error(exc) and attempt < LLAMA_70B_MAX_ATTEMPTS:
                delay = LLAMA_70B_INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "%s hit transient provider error on attempt %s/%s; backing off %.1fs. Error: %s",
                    description,
                    attempt,
                    LLAMA_70B_MAX_ATTEMPTS,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{description} failed without a captured exception.")


def openrouter_invoke(
    user_query: str | None = None,
    document_context: str = "",
    pre_extracted_vision_text: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system_instruction: str | None = None,
    prompt: str | None = None,
) -> str:
    client = get_openrouter_client()
    messages = []
    
    if system_instruction is not None or prompt is not None:
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if prompt:
            messages.append({"role": "user", "content": prompt})
    else:
        sys_inst = (
            "You are an advanced RAG synthesis engine. Your task is to provide a comprehensive, "
            "grounded answer using the provided Document Context and Pre-Extracted Image Data. "
            "If the data is conflicting, prioritize the extracted image data for visual queries."
        )
        usr_prompt = (
            f"[DOCUMENT CONTEXT]\n{document_context}\n\n"
            f"[PRE-EXTRACTED IMAGE DATA]\n{pre_extracted_vision_text}\n\n"
            f"[USER QUERY]\n{user_query or ''}"
        )
        messages = [
            {"role": "system", "content": sys_inst},
            {"role": "user", "content": usr_prompt}
        ]

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("OpenRouter primary model call failed: %s. Falling back to free model meta-llama/llama-3.3-70b-instruct:free", exc)
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return str(response.choices[0].message.content or "").strip()
        except Exception as fallback_exc:
            logger.error("OpenRouter fallback model call failed: %s", fallback_exc)
            raise fallback_exc


def openrouter_invoke_stream(
    user_query: str | None = None,
    document_context: str = "",
    pre_extracted_vision_text: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system_instruction: str | None = None,
    prompt: str | None = None,
) -> Generator[str, None, None]:
    client = get_openrouter_client()
    messages = []
    
    if system_instruction is not None or prompt is not None:
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if prompt:
            messages.append({"role": "user", "content": prompt})
    else:
        sys_inst = (
            "You are an advanced RAG synthesis engine. Your task is to provide a comprehensive, "
            "grounded answer using the provided Document Context and Pre-Extracted Image Data. "
            "If the data is conflicting, prioritize the extracted image data for visual queries."
        )
        usr_prompt = (
            f"[DOCUMENT CONTEXT]\n{document_context}\n\n"
            f"[PRE-EXTRACTED IMAGE DATA]\n{pre_extracted_vision_text}\n\n"
            f"[USER QUERY]\n{user_query or ''}"
        )
        messages = [
            {"role": "system", "content": sys_inst},
            {"role": "user", "content": usr_prompt}
        ]

    try:
        response_stream = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response_stream:
            text = str(chunk.choices[0].delta.content or "")
            if text:
                yield text
    except Exception as exc:
        logger.warning("OpenRouter primary streaming model call failed: %s. Falling back to free model meta-llama/llama-3.3-70b-instruct:free", exc)
        try:
            response_stream = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response_stream:
                text = str(chunk.choices[0].delta.content or "")
                if text:
                    yield text
        except Exception as fallback_exc:
            logger.error("OpenRouter fallback streaming model call failed: %s", fallback_exc)
            raise fallback_exc


def _parse_gemini_json_payload(response_text: str) -> dict[str, Any]:
    cleaned = str(response_text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def gemini_invoke_text(messages: list[BaseMessage], *, description: str = "OpenRouter call") -> str:
    system_instruction, user_contents = _gemini_messages_to_prompt(messages)
    return openrouter_invoke(
        system_instruction=system_instruction,
        prompt=user_contents,
        temperature=0,
    )


class RAGMemoryManager:
    """Independent in-memory conversation store with bounded prompt history."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def get_optimized_history(self, session_id: str, max_turns: int = 3) -> list:
        with self._lock:
            history = self._sessions.get(session_id, [])
            return list(history[-max(max_turns, 0) * 2 :]) if max_turns else []

    def get_full_history(self, session_id: str) -> list:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def update_history(self, session_id: str, user_query: str, ai_response: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": ai_response})

    def attach_sources(self, session_id: str, sources: list[dict[str, Any]]) -> None:
        with self._lock:
            history = self._sessions.get(session_id, [])
            if history and history[-1].get("role") == "assistant":
                history[-1]["sources"] = sources

    def clear_history(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


@st.cache_resource
def get_memory_manager() -> MultimodalConversationManager:
    return MultimodalConversationManager()


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        session_id = str(st.query_params.get("session_id", "") or "").strip()
        if not session_id:
            session_id = uuid.uuid4().hex
            st.query_params["session_id"] = session_id
        st.session_state.session_id = session_id
    if "query_text" not in st.session_state:
        st.session_state.query_text = ""
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "submitted_query" not in st.session_state:
        st.session_state.submitted_query = ""
    if "clear_query_after_run" not in st.session_state:
        st.session_state.clear_query_after_run = False
    if "pending_voice_query_text" not in st.session_state:
        st.session_state.pending_voice_query_text = ""
    if "query_input_nonce" not in st.session_state:
        st.session_state.query_input_nonce = 0
    if "last_voice_audio_hash" not in st.session_state:
        st.session_state.last_voice_audio_hash = ""
    if "last_voice_transcript" not in st.session_state:
        st.session_state.last_voice_transcript = ""
    if "current_image_path" not in st.session_state:
        st.session_state.current_image_path = None
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_images" not in st.session_state:
        st.session_state.current_images = []
    if "messages" not in st.session_state:
        st.session_state.messages = get_memory_manager().get_full_history(st.session_state.session_id)



def _consume_voice_query_params() -> None:
    voice_query = str(st.query_params.get("voice_query", "") or "").strip()
    voice_error = str(st.query_params.get("voice_error", "") or "").strip()
    if voice_query:
        st.session_state.pending_voice_query_text = voice_query
    if voice_error:
        st.warning(f"Voice input failed: {voice_error}")
    if voice_query or voice_error:
        session_id = str(st.session_state.get("session_id", "") or "")
        st.query_params.clear()
        if session_id:
            st.query_params["session_id"] = session_id


def _render_microphone_button() -> None:
    transcribe_url = json.dumps(TRANSCRIBE_API_URL)
    components.html(
        f"""
        <button id="mic-button" title="Record voice query" style="
            width: 100%;
            min-height: 42px;
            border: 1px solid #d0d5dd;
            border-radius: 8px;
            background: #ffffff;
            color: #344054;
            font-size: 18px;
            cursor: pointer;
        ">🎙</button>
        <div id="mic-status" style="
            margin-top: 4px;
            color: #667085;
            font-family: sans-serif;
            font-size: 11px;
            text-align: center;
        ">Voice</div>
        <script>
        const transcribeUrl = {transcribe_url};
        const button = document.getElementById("mic-button");
        const statusEl = document.getElementById("mic-status");
        let recorder = null;
        let chunks = [];

        function pushResultParam(key, value) {{
            const target = new URL(window.parent.location.href);
            target.searchParams.set(key, value);
            window.parent.location.href = target.toString();
        }}

        function setIdle() {{
            button.textContent = "🎙";
            button.style.background = "#ffffff";
            button.style.color = "#344054";
            button.disabled = false;
            statusEl.textContent = "Voice";
        }}

        function setRecording() {{
            button.textContent = "■";
            button.style.background = "#d92d20";
            button.style.color = "#ffffff";
            button.disabled = false;
            statusEl.textContent = "Recording...";
        }}

        function setBusy() {{
            button.textContent = "…";
            button.style.background = "#f2f4f7";
            button.style.color = "#475467";
            button.disabled = true;
            statusEl.textContent = "Transcribing...";
        }}

        async function uploadAudio(blob) {{
            const formData = new FormData();
            const extension = blob.type.includes("mp4") ? "mp4" : "webm";
            formData.append("audio", blob, `voice-query.${{extension}}`);
            const response = await fetch(transcribeUrl, {{
                method: "POST",
                body: formData,
            }});
            if (!response.ok) {{
                const errorText = await response.text();
                throw new Error(errorText || `HTTP ${{response.status}}`);
            }}
            const data = await response.json();
            const text = (data.text || "").trim();
            if (!text) {{
                throw new Error("Transcription returned empty text.");
            }}
            pushResultParam("voice_query", text);
        }}

        async function startRecording() {{
            if (!navigator.mediaDevices || !window.MediaRecorder) {{
                throw new Error("This browser does not support microphone recording.");
            }}
            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            chunks = [];
            recorder = new MediaRecorder(stream);
            recorder.ondataavailable = (event) => {{
                if (event.data && event.data.size > 0) chunks.push(event.data);
            }};
            recorder.onstop = async () => {{
                try {{
                    setBusy();
                    stream.getTracks().forEach((track) => track.stop());
                    const blob = new Blob(chunks, {{ type: recorder.mimeType || "audio/webm" }});
                    await uploadAudio(blob);
                }} catch (error) {{
                    pushResultParam("voice_error", error.message || String(error));
                }} finally {{
                    setIdle();
                    recorder = null;
                    chunks = [];
                }}
            }};
            recorder.start();
            setRecording();
        }}

        button.addEventListener("click", async () => {{
            try {{
                if (recorder && recorder.state === "recording") {{
                    recorder.stop();
                    return;
                }}
                await startRecording();
            }} catch (error) {{
                pushResultParam("voice_error", error.message || String(error));
            }}
        }});
        </script>
        """,
        height=76,
    )


def _render_voice_recorder() -> None:
    status_slot = st.empty()
    audio_file = st.audio_input("Voice query", label_visibility="collapsed")
    if audio_file is None:
        return

    audio_bytes = audio_file.getvalue()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if st.session_state.last_voice_audio_hash == audio_hash:
        return

    try:
        status_slot.markdown(
            """
            <style>
            .voice-mini-spinner {
                width: 16px;
                height: 16px;
                margin: 4px auto 0 auto;
                border: 2px solid rgba(148, 163, 184, 0.35);
                border-top-color: #f97316;
                border-radius: 999px;
                animation: voice-spin 0.8s linear infinite;
            }
            @keyframes voice-spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            </style>
            <div class="voice-mini-spinner" aria-label="Transcribing audio"></div>
            """,
            unsafe_allow_html=True,
        )
        response = requests.post(
            TRANSCRIBE_API_URL,
            files={
                "audio": (
                    audio_file.name or "voice-query.wav",
                    audio_bytes,
                    audio_file.type or "audio/wav",
                )
            },
            timeout=90,
        )
        response.raise_for_status()
        transcribed_text = str(response.json().get("text") or "").strip()
        if not transcribed_text:
            raise RuntimeError("Transcription returned empty text.")
        st.session_state.pending_voice_query_text = transcribed_text
        st.session_state.last_voice_transcript = transcribed_text
        st.session_state.last_voice_audio_hash = audio_hash
        status_slot.empty()
        st.rerun()
    except Exception as exc:
        status_slot.empty()
        st.error(f"Voice transcription failed: {exc}")


def _submit_current_query() -> None:
    st.session_state.submitted_query = str(st.session_state.get("query_input", "") or "").strip()


def sanitize_user_answer(answer: str) -> str:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return schema_failure_message()
    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in BANNED_USER_FACING_PHRASES):
        return prompt_leakage_failure_message()
    return cleaned


def extract_llm_response_text(response: Any) -> str:
    """Normalize streamed or batched LLM payloads into plain markdown text."""

    if response is None:
        return ""

    if isinstance(response, str):
        return response

    if hasattr(response, "content") and not isinstance(response, dict):
        return extract_llm_response_text(getattr(response, "content"))

    if isinstance(response, list):
        parts = [extract_llm_response_text(item) for item in response]
        return "".join(part for part in parts if part)

    if isinstance(response, dict):
        text_value = response.get("text")
        if text_value is not None and str(text_value).strip():
            return str(text_value)
        for nested_key in ("content", "message", "delta"):
            if nested_key in response:
                nested_text = extract_llm_response_text(response[nested_key])
                if nested_text:
                    return nested_text
        return ""

    return str(response).strip()


def _normalize_entity_identifier(raw_identifier: str) -> str:
    identifier = re.sub(r"\s+", "", raw_identifier or "").upper().replace("0.", "O.")
    if re.fullmatch(r"[O0]\d+", identifier):
        identifier = f"O.{identifier[1:]}"
    return identifier


def structural_reference_variants(query: str) -> list[str]:
    """Return all stable payload-key spellings for figure/table/chart references."""

    variants: list[str] = []
    seen: set[str] = set()
    for match in STRUCTURAL_REFERENCE_PATTERN.finditer(query or ""):
        kind_raw = match.group("kind").lower()
        prefix = "Table" if kind_raw == "table" else "Chart" if kind_raw == "chart" else "Figure"
        number = match.group("identifier")
        candidates = (
            f"{prefix}_{number}",
            f"{prefix.lower()}_{number}",
            f"{prefix} {number}",
            f"{prefix.lower()} {number}",
            number,
        )
        for candidate in candidates:
            key = candidate.lower()
            if candidate and key not in seen:
                seen.add(key)
                variants.append(candidate)
    return variants


def hard_entity_label_variants(entity: dict[str, str]) -> list[str]:
    identifier = entity["identifier"]
    kind = entity.get("kind", "figure")
    prefix = "Table" if kind == "table" else "Chart" if kind == "chart" else "Figure"
    variants = {
        entity["label"],
        f"{prefix}_{identifier}",
        f"{prefix.lower()}_{identifier}",
        f"{prefix} {identifier}",
        f"{prefix.lower()} {identifier}",
        identifier,
        identifier.lower(),
        identifier.upper(),
    }
    if identifier.upper().startswith("O."):
        zero_identifier = f"0.{identifier.split('.', 1)[1]}"
        variants.update(
            {
                f"{prefix}_{zero_identifier}",
                f"{prefix.lower()}_{zero_identifier}",
                f"{prefix} {zero_identifier}",
                f"{prefix.lower()} {zero_identifier}",
                zero_identifier,
            }
        )
    variants.update(structural_reference_variants(entity["label"]))
    return [variant for variant in variants if variant]


def hard_entity_strict_label_variants(entity: dict[str, str]) -> list[str]:
    """Return kind-qualified variants so Table 4.1 does not match Figure 4.1."""

    identifier = entity["identifier"]
    kind = entity.get("kind", "figure")
    prefix = "Table" if kind == "table" else "Chart" if kind == "chart" else "Figure"
    variants = {
        entity["label"],
        f"{prefix}_{identifier}",
        f"{prefix.lower()}_{identifier}",
        f"{prefix} {identifier}",
        f"{prefix.lower()} {identifier}",
    }
    if identifier.upper().startswith("O."):
        zero_identifier = f"0.{identifier.split('.', 1)[1]}"
        variants.update(
            {
                f"{prefix}_{zero_identifier}",
                f"{prefix.lower()}_{zero_identifier}",
                f"{prefix} {zero_identifier}",
                f"{prefix.lower()} {zero_identifier}",
            }
        )
    return [variant for variant in variants if variant]


def extract_hard_entities(user_query: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in HARD_ENTITY_PATTERN.finditer(user_query or ""):
        kind_raw = match.group("kind").lower()
        kind = "table" if kind_raw.startswith(("tab", "table")) else "chart" if kind_raw == "chart" else "figure"
        identifier = _normalize_entity_identifier(match.group("identifier"))
        if not identifier:
            continue
        key = (kind, identifier)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            {
                "kind": kind,
                "identifier": identifier,
                "label": f"{'Table' if kind == 'table' else 'Chart' if kind == 'chart' else 'Figure'} {identifier}",
            }
        )
    return entities


def has_explicit_identifier_or_number(query: str) -> bool:
    return bool(extract_hard_entities(query) or EXPLICIT_NUMBER_PATTERN.search(query or ""))


IMAGE_ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
IMAGE_ASSET_DIRS = [
    *([Path(os.getenv("NVIDIA_VISION_ASSETS_DIR", "")).expanduser()] if os.getenv("NVIDIA_VISION_ASSETS_DIR", "").strip() else []),
    Path("extracted_images"),
    Path("extracted_charts"),
    Path("Data/extracted_visuals_smoke"),
]
IMAGE_FILENAME_PATTERN = re.compile(
    r"(?P<filename>[^\\/\r\n:*?\"<>|]*(?:figure|chart|diagram|image)[^\\/\r\n:*?\"<>|]*\.(?:png|jpg|jpeg|webp|gif))",
    flags=re.IGNORECASE,
)


def display_image_robustly(img_path: str):
    resolved_path = img_path
    if resolved_path:
        resolved_path = resolved_path.replace("\\", "/")
        if "C:/" in resolved_path or (len(resolved_path) > 1 and resolved_path[1] == ":") or resolved_path.startswith("/") or resolved_path.startswith("/mount") or resolved_path.startswith("/home"):
            filename = os.path.basename(resolved_path)
            for candidate_dir in ["assets/extracted_images", "extracted_images", "assets/extracted_charts", "extracted_charts"]:
                local_dir_path = os.path.join(os.getcwd(), candidate_dir)
                candidate_path = os.path.join(local_dir_path, filename)
                if os.path.exists(candidate_path):
                    resolved_path = candidate_path
                    break
                    
    st.write(f"DEBUG Image Path: {resolved_path}, Exists: {os.path.exists(resolved_path)}")
    
    if not resolved_path or not os.path.exists(resolved_path):
        st.warning(f"Image file not found: {resolved_path}")
        return
        
    try:
        st.image(resolved_path, use_container_width=True)
    except Exception as e:
        try:
            with open(resolved_path, "rb") as f:
                img_bytes = f.read()
            st.image(img_bytes, use_container_width=True)
        except Exception as e2:
            st.warning(f"Could not render image file: {resolved_path}. Error: {e} | {e2}")

def _resolve_existing_image_path(value: object) -> str:
    raw_path = str(value or "").strip()
    if not raw_path:
        return ""
    raw_path = raw_path.strip(" '\"`").replace("\\", "/")
    
    # Check if absolute path, extract basename and resolve using os.path.join(os.getcwd(), ...)
    if "C:/" in raw_path or (len(raw_path) > 1 and raw_path[1] == ":") or raw_path.startswith("/") or raw_path.startswith("/mount") or raw_path.startswith("/home"):
        filename = os.path.basename(raw_path)
        for candidate_dir in ["assets/extracted_images", "extracted_images", "assets/extracted_charts", "extracted_charts", "Data/extracted_visuals_smoke"]:
            candidate_path = os.path.join(os.getcwd(), candidate_dir, filename)
            if os.path.exists(candidate_path):
                return candidate_path

    cleaned_path = raw_path
    marker = "recovered-rag-project/"
    if marker in cleaned_path:
        rel_portion = cleaned_path.split(marker, 1)[1]
    else:
        parts = cleaned_path.split("/")
        if "assets" in parts:
            rel_portion = "/".join(parts[parts.index("assets"):])
        elif "extracted_images" in parts:
            rel_portion = "assets/" + "/".join(parts[parts.index("extracted_images"):])
        elif "extracted_charts" in parts:
            rel_portion = "assets/" + "/".join(parts[parts.index("extracted_charts"):])
        else:
            rel_portion = cleaned_path

    try_path = Path(os.path.join(os.getcwd(), rel_portion)).resolve()
    if try_path.is_file():
        return str(try_path)

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(os.path.join(os.getcwd(), str(path))).resolve()
    if path.is_file():
        return str(path)
        
    filename = Path(raw_path).name
    if not filename:
        return ""
        
    for asset_dir in IMAGE_ASSET_DIRS:
        if not str(asset_dir):
            continue
        resolved_dir = asset_dir
        if not resolved_dir.is_absolute():
            resolved_dir = Path(os.path.join(os.getcwd(), str(resolved_dir))).resolve()
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            continue
            
        direct_path = (resolved_dir / filename).resolve()
        if direct_path.is_file():
            return str(direct_path)
            
        for candidate in resolved_dir.rglob(filename):
            if candidate.is_file():
                return str(candidate.resolve())
    return ""


def _extract_image_filename_from_text(value: object) -> str:
    text = str(value or "")
    for line in text.splitlines():
        match = IMAGE_FILENAME_PATTERN.search(line)
        if match:
            return match.group("filename").strip(" '\"`.,;)")
    match = IMAGE_FILENAME_PATTERN.search(text)
    return match.group("filename").strip(" '\"`.,;)") if match else ""


def _extract_image_reference_from_metadata(metadata: dict[str, Any]) -> str:
    for key in ("image_path", "image_local_path", "image_name", "filename", "file_name", "path"):
        value = metadata.get(key)
        resolved = _resolve_existing_image_path(value)
        if resolved:
            return resolved
        filename = _extract_image_filename_from_text(value)
        resolved = _resolve_existing_image_path(filename)
        if resolved:
            return resolved
    for value in metadata.values():
        if isinstance(value, dict):
            resolved = _extract_image_reference_from_metadata(value)
            if resolved:
                return resolved
        elif isinstance(value, (str, int, float)):
            filename = _extract_image_filename_from_text(value)
            resolved = _resolve_existing_image_path(filename)
            if resolved:
                return resolved
    return ""


def _locked_entity_asset_tokens(locked_entities: list[str]) -> list[str]:
    tokens: list[str] = []
    for entity in locked_entities or []:
        for match in re.findall(r"\b(?:[A-Za-z]+\s*)?([Oo0]?\s*\.?\s*\d+(?:\s*\.\s*\d+)*)\b", str(entity)):
            normalized = _normalize_entity_identifier(match)
            variants = {normalized, normalized.replace("O.", "0."), normalized.replace("0.", "O.")}
            for variant in variants:
                token = re.sub(r"[^a-z0-9]+", "_", variant.lower()).strip("_")
                if token and token not in tokens:
                    tokens.append(token)
    return tokens


def _locked_entity_kinds(locked_entities: list[str]) -> set[str]:
    kinds: set[str] = set()
    for entity in locked_entities or []:
        for hard_entity in extract_hard_entities(str(entity)):
            kinds.add(hard_entity["kind"])
    return kinds


def _locked_entity_exact_asset_tokens(locked_entities: list[str]) -> list[str]:
    tokens: list[str] = []
    for entity in locked_entities or []:
        for hard_entity in extract_hard_entities(str(entity)):
            for variant in hard_entity_label_variants(hard_entity):
                if variant == hard_entity["identifier"]:
                    continue
                token = _normalized_identifier_blob(variant)
                if token and token not in tokens:
                    tokens.append(token)
    return tokens


def _image_asset_sort_key(image_path: str, locked_entities: list[str]) -> tuple[int, int, str]:
    path = Path(str(image_path or ""))
    name_blob = _normalized_identifier_blob(path.stem)
    exact_tokens = _locked_entity_exact_asset_tokens(locked_entities)
    matches_exact_entity = any(token and token in name_blob for token in exact_tokens)
    is_full_page_fallback = "full_page" in name_blob or "fallback" in name_blob
    return (0 if matches_exact_entity else 1, 1 if is_full_page_fallback else 0, str(path).lower())


def _find_matching_image_asset(locked_entities: list[str]) -> str:
    kinds = _locked_entity_kinds(locked_entities)
    tokens = _locked_entity_asset_tokens(locked_entities)
    if not tokens:
        return ""
    matches: list[str] = []
    for asset_dir in IMAGE_ASSET_DIRS:
        if "assets/extracted_images" in str(asset_dir).replace("\\", "/"):
            continue
        if not str(asset_dir) or not asset_dir.exists() or not asset_dir.is_dir():
            continue
        for path in asset_dir.rglob("*"):
            if path.suffix.lower() not in IMAGE_ASSET_EXTENSIONS or not path.is_file():
                continue
            normalized_name = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
            if any(token in normalized_name for token in tokens):
                matches.append(str(path.resolve()))
    return sorted(matches, key=lambda candidate: _image_asset_sort_key(candidate, locked_entities))[0] if matches else ""


def _locked_entity_search_terms(locked_entities: list[str]) -> list[str]:
    terms: list[str] = []
    for entity in locked_entities or []:
        text = str(entity or "").strip()
        hard_entities = extract_hard_entities(text)
        if hard_entities:
            for hard_entity in hard_entities:
                for variant in hard_entity_strict_label_variants(hard_entity):
                    if variant and variant.lower() not in [term.lower() for term in terms]:
                        terms.append(variant)
            continue
        if text and text.lower() not in [term.lower() for term in terms]:
            terms.append(text)
        for token in _locked_entity_asset_tokens([text]):
            dotted = token.replace("_", ".")
            spaced = token.replace("_", " ")
            for variant in (token, dotted, spaced):
                if variant and variant.lower() not in [term.lower() for term in terms]:
                    terms.append(variant)
    return terms


def _chunk_search_blob(chunk: dict[str, Any]) -> str:
    metadata = dict(chunk.get("metadata") or {})
    return f"{chunk.get('content', '')} {chunk.get('source', '')} {metadata}".lower()


def _normalized_chunk_entities(chunk: dict[str, Any]) -> set[str]:
    metadata = dict(chunk.get("metadata") or {})
    metadata_entity_ids = metadata.get("entity_ids") or []
    chunk_entity_ids = chunk.get("entity_ids") or []
    if isinstance(metadata_entity_ids, str):
        metadata_entity_ids = [metadata_entity_ids]
    if isinstance(chunk_entity_ids, str):
        chunk_entity_ids = [chunk_entity_ids]
    values = [
        chunk.get("entity_id"),
        metadata.get("entity_id"),
        metadata.get("figure_id"),
        *metadata_entity_ids,
        *chunk_entity_ids,
    ]
    entities: set[str] = set()
    for value in values:
        for entity in extract_hard_entities(str(value or "")):
            entities.add(entity["label"].lower())
    return entities


def _chunk_primary_entity_label(chunk: dict[str, Any]) -> str:
    metadata = dict(chunk.get("metadata") or {})
    primary_candidates = [
        metadata.get("entity_id"),
        chunk.get("entity_id"),
        metadata.get("figure_id"),
    ]
    for value in primary_candidates:
        entities = extract_hard_entities(str(value or ""))
        if entities:
            return entities[0]["label"].lower()
    labels = sorted(_normalized_chunk_entities(chunk))
    return labels[0] if labels else ""


def _locked_entity_labels(locked_entities: list[str]) -> set[str]:
    labels: set[str] = set()
    for entity in locked_entities or []:
        for hard_entity in extract_hard_entities(str(entity or "")):
            labels.add(hard_entity["label"].lower())
    return labels


def _extract_target_entity_tuple(query: str) -> tuple[str, str]:
    hard_entities = extract_hard_entities(query or "")
    if not hard_entities:
        return "", ""
    primary = hard_entities[0]
    target_type = "table" if primary["kind"] == "table" else "figure" if primary["kind"] in {"figure", "chart"} else ""
    target_number = str(primary.get("identifier") or "").strip().lower()
    return target_type, target_number


def _extract_chunk_entity_tuple(chunk: dict[str, Any]) -> tuple[str, str]:
    metadata = dict(chunk.get("metadata") or {})
    candidates = [
        metadata.get("entity_id"),
        chunk.get("entity_id"),
        metadata.get("figure_id"),
    ]
    metadata_entity_ids = metadata.get("entity_ids") or []
    chunk_entity_ids = chunk.get("entity_ids") or []
    if isinstance(metadata_entity_ids, str):
        metadata_entity_ids = [metadata_entity_ids]
    if isinstance(chunk_entity_ids, str):
        chunk_entity_ids = [chunk_entity_ids]
    candidates.extend(metadata_entity_ids)
    candidates.extend(chunk_entity_ids)
    for value in candidates:
        entities = extract_hard_entities(str(value or ""))
        if not entities:
            continue
        primary = entities[0]
        entity_type = "table" if primary["kind"] == "table" else "figure" if primary["kind"] in {"figure", "chart"} else ""
        entity_number = str(primary.get("identifier") or "").strip().lower()
        if entity_type and entity_number:
            return entity_type, entity_number
    return "", ""


def _chunk_matches_target_tuple(chunk: dict[str, Any], target_type: str, target_number: str) -> bool:
    if not target_type or not target_number:
        return True
    entity_type, entity_number = _extract_chunk_entity_tuple(chunk)
    if not entity_type or not entity_number:
        return False
    if entity_type == "table" and target_type == "figure":
        return False
    if entity_type == "figure" and target_type == "table":
        return False
    return entity_type == target_type and entity_number == target_number


def _has_conflicting_visual_asset(metadata: dict[str, Any], requested_kinds: set[str]) -> bool:
    table_only = "table" in requested_kinds and not ({"figure", "chart"} & requested_kinds)
    figure_only = ({"figure", "chart"} & requested_kinds) and "table" not in requested_kinds
    if table_only:
        return bool(
            metadata.get("figure_image_path")
            or metadata.get("figure_image_paths")
            or metadata.get("chart_image_path")
            or metadata.get("chart_image_paths")
        )
    if figure_only:
        return bool(
            metadata.get("table_csv_path")
            or metadata.get("table_csv_paths")
            or metadata.get("table_image_path")
            or metadata.get("table_image_paths")
            or metadata.get("csv_path")
            or metadata.get("csv_paths")
        )
    return False


def _chunk_matches_locked_entity(chunk: dict[str, Any], locked_entities: list[str]) -> bool:
    target_type, target_number = _extract_target_entity_tuple(" ".join(locked_entities or []))
    if target_type and target_number:
        return _chunk_matches_target_tuple(chunk, target_type, target_number)
    locked_labels = _locked_entity_labels(locked_entities)
    primary_label = _chunk_primary_entity_label(chunk)
    if primary_label and locked_labels:
        return primary_label in locked_labels
    return False


def _is_vision_chunk(chunk: dict[str, Any]) -> bool:
    metadata = dict(chunk.get("metadata") or {})
    blob = f"{chunk.get('source', '')} {metadata.get('source', '')} {metadata.get('source_type', '')} "\
        f"{metadata.get('content_type', '')} {metadata.get('visual_type', '')} {metadata.get('caption_source', '')}".lower()
    return any(marker in blob for marker in ("vision", "visual", "chart", "diagram", "figure", "image", "map", "qwen", "gemini"))


def _is_table_chunk(chunk: dict[str, Any]) -> bool:
    metadata = dict(chunk.get("metadata") or {})
    entity_type = str(
        metadata.get("entity_type")
        or chunk.get("entity_type")
        or ""
    ).lower()
    has_table_asset = bool(
        metadata.get("table_csv_path")
        or metadata.get("table_csv_paths")
        or metadata.get("table_image_path")
        or metadata.get("table_image_paths")
        or metadata.get("csv_path")
        or metadata.get("csv_paths")
    )
    has_conflicting_visual = bool(
        metadata.get("figure_image_path")
        or metadata.get("figure_image_paths")
        or metadata.get("chart_image_path")
        or metadata.get("chart_image_paths")
    )
    if entity_type == "table" and has_table_asset and not has_conflicting_visual:
        return True
    return bool(has_table_asset and not has_conflicting_visual)


def _image_path_matches_requested_kinds(image_path: str, kinds: set[str]) -> bool:
    if not image_path or not kinds:
        return bool(image_path)
    name_blob = _normalized_identifier_blob(Path(str(image_path)).stem)
    table_only = "table" in kinds and not ({"figure", "chart"} & kinds)
    figure_only = ({"figure", "chart"} & kinds) and "table" not in kinds
    if table_only:
        return "table" in name_blob
    if figure_only:
        return "table" not in name_blob
    return True


def _chunk_matches_requested_asset_kinds(chunk: dict[str, Any], kinds: set[str]) -> bool:
    if not kinds:
        return True
    metadata = dict(chunk.get("metadata") or {})
    table_only = "table" in kinds and not ({"figure", "chart"} & kinds)
    figure_only = ({"figure", "chart"} & kinds) and "table" not in kinds
    if table_only:
        return _is_table_chunk(chunk) and not _has_conflicting_visual_asset(metadata, kinds)
    if figure_only:
        return _is_vision_chunk(chunk) and not _is_table_chunk(chunk) and not _has_conflicting_visual_asset(metadata, kinds)
    return True


def promote_locked_entity_candidates(
    candidates: list[dict[str, Any]],
    locked_entities: list[str],
) -> list[dict[str, Any]]:
    if not locked_entities or not candidates:
        return candidates
    promoted: list[dict[str, Any]] = []
    regular: list[dict[str, Any]] = []
    requested_kinds = _locked_entity_kinds(locked_entities)
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        metadata = dict(item.get("metadata") or {})
        matches_locked = _chunk_matches_locked_entity(item, locked_entities)
        kind_match = _chunk_matches_requested_asset_kinds(item, requested_kinds)
        vision_match = _is_vision_chunk(item) and matches_locked and kind_match
        if matches_locked and kind_match and not _has_conflicting_visual_asset(metadata, requested_kinds):
            item["rerank_score"] = max(float(item.get("rerank_score", 0.0)), 1_000_000.0 - index)
            item["fusion_score"] = max(float(item.get("fusion_score", 0.0)), 1_000_000.0 - index)
            item["locked_entity_boost"] = True
            promoted.append(item)
        elif vision_match:
            item["rerank_score"] = max(float(item.get("rerank_score", 0.0)), 1_000_000.0 - index)
            item["fusion_score"] = max(float(item.get("fusion_score", 0.0)), 1_000_000.0 - index)
            item["locked_entity_boost"] = True
            promoted.append(item)
        else:
            regular.append(item)
    if promoted:
        print(
            f"DEBUG [Reranker Interception]: Promoted {len(promoted)} chunks for locked entities {locked_entities}",
            file=sys.stderr,
            flush=True,
        )
    return [*promoted, *regular]


def _chunk_image_path(chunk: dict[str, Any]) -> str:
    metadata = dict(chunk.get("metadata") or {})
    return (
        _extract_image_reference_from_metadata(metadata)
        or _resolve_existing_image_path(metadata.get("image_path"))
        or _resolve_existing_image_path(metadata.get("figure_image_path"))
        or _resolve_existing_image_path(metadata.get("chart_image_path"))
        or _resolve_existing_image_path(metadata.get("table_image_path"))
        or _resolve_existing_image_path(metadata.get("image_local_path"))
        or _resolve_existing_image_path(metadata.get("visual_path"))
        or _resolve_existing_image_path(_extract_image_filename_from_text(chunk.get("content", "")))
    )


def _requested_kind_image_path(chunk: dict[str, Any], kinds: set[str]) -> str:
    image_path = _chunk_image_path(chunk)
    if (
        image_path
        and _chunk_matches_requested_asset_kinds(chunk, kinds)
        and _image_path_matches_requested_kinds(image_path, kinds)
    ):
        return image_path
    return ""


def bind_image_paths_to_chunks(
    retrieved_chunks: list[dict[str, Any]],
    locked_entities: list[str],
    source_pool: list[dict[str, Any]] | None = None,
) -> str:
    if not locked_entities:
        for index, chunk in enumerate(retrieved_chunks):
            retrieved_chunks[index] = strip_visual_metadata([chunk])[0]
        return ""

    kinds = _locked_entity_kinds(locked_entities)
    source_pool = source_pool or retrieved_chunks
    fallback_image_path = _find_matching_image_asset(locked_entities)
    if fallback_image_path and not _image_path_matches_requested_kinds(fallback_image_path, kinds):
        fallback_image_path = ""
    locked_match_image_paths: list[str] = []
    for chunk in source_pool:
        if _chunk_matches_locked_entity(chunk, locked_entities):
            locked_match_image_path = _requested_kind_image_path(chunk, kinds)
            if locked_match_image_path:
                locked_match_image_paths.append(locked_match_image_path)
            filename = _extract_image_filename_from_text(chunk.get("content", ""))
            locked_match_image_path = _resolve_existing_image_path(filename)
            if locked_match_image_path and _image_path_matches_requested_kinds(locked_match_image_path, kinds):
                locked_match_image_paths.append(locked_match_image_path)
    locked_match_image_path = (
        sorted(set(locked_match_image_paths), key=lambda candidate: _image_asset_sort_key(candidate, locked_entities))[0]
        if locked_match_image_paths
        else ""
    )
    pool_image_paths: list[str] = []
    for chunk in source_pool:
        filename = _extract_image_filename_from_text(chunk.get("content", ""))
        pool_image_path = _resolve_existing_image_path(filename)
        if (
            pool_image_path
            and _chunk_matches_requested_asset_kinds(chunk, kinds)
            and _image_path_matches_requested_kinds(pool_image_path, kinds)
        ):
            pool_image_paths.append(pool_image_path)
    pool_image_path = (
        sorted(set(pool_image_paths), key=lambda candidate: _image_asset_sort_key(candidate, locked_entities))[0]
        if pool_image_paths
        else ""
    )
    vision_image_paths: list[str] = []
    for chunk in source_pool:
        if (
            _is_vision_chunk(chunk)
            and _chunk_matches_locked_entity(chunk, locked_entities)
            and _chunk_matches_requested_asset_kinds(chunk, kinds)
        ):
            vision_image_path = _requested_kind_image_path(chunk, kinds)
            if vision_image_path:
                vision_image_paths.append(vision_image_path)
    vision_image_path = (
        sorted(set(vision_image_paths), key=lambda candidate: _image_asset_sort_key(candidate, locked_entities))[0]
        if vision_image_paths
        else ""
    )
    selected_image_path = ""
    for chunk in sorted(
        retrieved_chunks,
        key=lambda item: _image_asset_sort_key(_requested_kind_image_path(item, kinds), locked_entities),
    ):
        metadata = dict(chunk.get("metadata") or {})
        image_path = _requested_kind_image_path(chunk, kinds)
        if not image_path and _chunk_matches_locked_entity(chunk, locked_entities):
            image_path = locked_match_image_path or vision_image_path or pool_image_path or fallback_image_path
            if image_path:
                metadata["image_path"] = image_path
        elif image_path:
            metadata["image_path"] = image_path
        else:
            for key in ("image_path", "image_local_path", "image_name"):
                metadata.pop(key, None)
        chunk["metadata"] = metadata
        if image_path and not selected_image_path:
            selected_image_path = image_path
    return selected_image_path


def extract_structural_identifier_queries(query: str) -> list[str]:
    """Return stable one-identifier queries for deterministic multi-entity retrieval."""

    queries: list[str] = []
    seen: set[str] = set()
    for entity in extract_hard_entities(query):
        key = entity["label"].lower()
        if key not in seen:
            seen.add(key)
            queries.append(entity["label"])
    for match in STRUCTURAL_IDENTIFIER_PATTERN.finditer(query or ""):
        identifier = match.group(0)
        if any(identifier.lower() in existing.lower() for existing in queries):
            continue
        key = identifier.lower()
        if key not in seen:
            seen.add(key)
            queries.append(identifier)
    return queries


def preserve_explicit_literals(original_query: str, rewritten_query: str) -> str:
    literals = [match.group(0) for match in HARD_ENTITY_PATTERN.finditer(original_query or "")]
    literals.extend(EXPLICIT_NUMBER_PATTERN.findall(original_query or ""))
    missing = [literal for literal in dict.fromkeys(literals) if literal not in rewritten_query]
    return f"{rewritten_query}\nExact literals: {', '.join(missing)}" if missing else rewritten_query


def parse_condensed_queries(raw_response: str, original_query: str) -> list[str]:
    try:
        parsed = json.loads(str(raw_response or "").strip())
    except json.JSONDecodeError:
        parsed = [str(raw_response or "").strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    queries = [str(item).strip() for item in parsed if str(item).strip()]
    structural_queries = extract_structural_identifier_queries(original_query)
    if structural_queries:
        queries = [*structural_queries, *queries]
    elif len(queries) == 1:
        queries = [preserve_explicit_literals(original_query, queries[0])]
    return list(dict.fromkeys(queries)) or [original_query]


def enforce_locked_entities(queries: list[str], locked_entities: list[str]) -> list[str]:
    locked_entities = [str(entity).strip() for entity in (locked_entities or []) if str(entity).strip()]
    if not locked_entities:
        return queries
    output = list(queries)
    combined = "\n".join(output)
    for entity in locked_entities:
        if entity not in combined:
            output.append(entity)
    return list(dict.fromkeys(output))


def query_condenser_prompt_with_locks(locked_entities: list[str]) -> str:
    if not locked_entities:
        return QUERY_CONDENSER_PROMPT
    return (
        f"{QUERY_CONDENSER_PROMPT}\n\n"
        "CRITICAL PERIMETER GUARDRAIL: "
        f"The user has explicitly locked down these specific document identifiers: {locked_entities}. "
        "When you reformulate the conversational history into a standalone query, you MUST explicitly "
        "preserve and append these exact text strings to the end of your output query. Do not alter, "
        "delete, or summarize them."
    )


def hard_entity_query_suffix(entities: list[dict[str, str]]) -> str:
    if not entities:
        return ""
    labels = ", ".join(entity["label"] for entity in entities)
    return f"\nHard entity labels that must be retrieved exactly: {labels}"


def _hard_entity_asset_path_tokens(entity: dict[str, str]) -> list[str]:
    identifier = str(entity.get("identifier") or "").strip()
    kind = str(entity.get("kind") or "").strip().lower()
    if not identifier or not kind:
        return []
    prefix = "table" if kind == "table" else "chart" if kind == "chart" else "figure"
    variants = {
        f"{prefix}_{identifier}",
        f"{prefix} {identifier}",
        f"{prefix}_{identifier.replace('.', '_')}",
        f"{prefix}{identifier.replace('.', '_')}",
    }
    return [variant for variant in variants if variant]


def _asset_path_field_names() -> tuple[str, ...]:
    return tuple(field for field in ASSET_PAYLOAD_FIELDS if "path" in field)


def _entity_matches_payload_or_asset_path(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    text: str,
    entity: dict[str, str],
) -> bool:
    searchable = f"{text} {payload} {metadata}".lower()
    label_variants = hard_entity_strict_label_variants(entity)
    if any(_structural_variant_matches_blob([variant], searchable) for variant in label_variants):
        return True

    path_tokens = _hard_entity_asset_path_tokens(entity)
    if not path_tokens:
        return False

    path_values: list[str] = []
    for field_name in _asset_path_field_names():
        container = metadata if field_name.startswith("metadata.") else payload
        key = field_name.split(".", 1)[1] if field_name.startswith("metadata.") else field_name
        raw_value = container.get(key)
        if isinstance(raw_value, str):
            path_values.append(raw_value)
        elif isinstance(raw_value, (list, tuple, set)):
            path_values.extend(str(item) for item in raw_value if str(item or "").strip())

    for path_value in path_values:
        normalized_path = _normalized_identifier_blob(Path(str(path_value)).name or str(path_value))
        if any(_normalized_identifier_blob(token) in normalized_path for token in path_tokens):
            return True
    return False


def build_hard_entity_filter(
    entities: list[dict[str, str]],
    *,
    include_cross_references: bool = False,
) -> models.Filter | None:
    if not entities:
        return None
    conditions = []
    fields = INDEXED_ENTITY_PAYLOAD_FIELDS if include_cross_references else PRIMARY_ENTITY_PAYLOAD_FIELDS
    match_text_cls = getattr(models, "MatchText", None)
    for entity in entities:
        variants = hard_entity_label_variants(entity) if include_cross_references else hard_entity_strict_label_variants(entity)
        for key in fields:
            conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=variants)))
        if match_text_cls is not None:
            for token in _hard_entity_asset_path_tokens(entity):
                for key in _asset_path_field_names():
                    conditions.append(models.FieldCondition(key=key, match=match_text_cls(text=token)))
    return models.Filter(should=conditions)


def extract_chapter_references(query: str) -> list[str]:
    references: list[str] = []
    roman_values = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "viii": 8,
        "ix": 9,
        "x": 10,
    }
    for match in CHAPTER_REFERENCE_PATTERN.finditer(query or ""):
        value = match.group("number").lower()
        normalized = str(roman_values.get(value, value))
        if normalized not in references:
            references.append(normalized)
    return references


def build_chapter_filter(chapter_numbers: list[str]) -> models.Filter | None:
    numbers = [str(number).strip() for number in chapter_numbers if str(number).strip()]
    if not numbers:
        return None
    conditions = [
        models.FieldCondition(key=field, match=models.MatchAny(any=numbers))
        for field in CHAPTER_PAYLOAD_FIELDS
    ]
    return models.Filter(should=conditions)


def combine_filters(*filters: models.Filter | None) -> models.Filter | None:
    active = [item for item in filters if item is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return models.Filter(should=active)



def _normalized_identifier_blob(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _structural_variant_matches_blob(variants: list[str], searchable: str) -> bool:
    normalized_blob = _normalized_identifier_blob(searchable)
    for variant in variants:
        value = str(variant or "").strip()
        if not value:
            continue
        if value.lower() in searchable:
            return True
        normalized_value = _normalized_identifier_blob(value)
        if normalized_value and normalized_value in normalized_blob:
            return True
    return False


def _exact_asset_priority(item: dict[str, Any]) -> tuple[int, int, int]:
    metadata = dict(item.get("metadata") or {})
    image_path = _extract_image_reference_from_metadata(metadata)
    contains_chart = bool(metadata.get("contains_chart") or item.get("contains_chart"))
    name_blob = _normalized_identifier_blob(Path(image_path).stem) if image_path else ""
    is_full_page_fallback = "full_page" in name_blob or "fallback" in name_blob
    return (0 if image_path else 1, 0 if contains_chart else 1, 1 if is_full_page_fallback else 0)


def ensure_entity_payload_indexes(client: QdrantClient) -> None:
    for field_name in (*INDEXED_ENTITY_PAYLOAD_FIELDS, *ASSET_PAYLOAD_FIELDS):
        try:
            client.create_payload_index(
                COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            logger.debug("Payload index %s already exists or could not be created: %s", field_name, exc)
    for field_name in CHAPTER_PAYLOAD_FIELDS:
        try:
            client.create_payload_index(
                COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            logger.debug("Payload index %s already exists or could not be created: %s", field_name, exc)


def _candidate_entity_ids(candidate: dict[str, Any]) -> list[str]:
    metadata = dict(candidate.get("metadata") or {})
    entity_ids = metadata.get("entity_ids") or []
    cross_references = metadata.get("cross_references") or []
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]
    if isinstance(cross_references, str):
        cross_references = [cross_references]
    values = [
        metadata.get("entity_id"),
        *entity_ids,
        metadata.get("figure_id"),
        metadata.get("cross_reference"),
        *cross_references,
    ]
    entities: list[str] = []
    for value in values:
        if value:
            entities.extend(entity["label"] for entity in extract_hard_entities(str(value)))
    return list(dict.fromkeys(entities))


def co_retrieve_cross_references(candidates: list[dict[str, Any]], query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Pull table/figure companions into the same context window before reranking."""

    client = get_qdrant_client()
    ensure_entity_payload_indexes(client)
    labels = [entity["label"] for entity in extract_hard_entities(query)]
    for candidate in candidates:
        labels.extend(_candidate_entity_ids(candidate))
    labels = list(dict.fromkeys(label for label in labels if label))
    if not labels:
        return candidates

    conditions = []
    for label in labels:
        entity = extract_hard_entities(label)
        if not entity:
            continue
        variants = hard_entity_label_variants(entity[0])
        for key in INDEXED_ENTITY_PAYLOAD_FIELDS:
            conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=variants)))
    if not conditions:
        return candidates

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(should=conditions),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    expanded = list(candidates)
    existing = {str(candidate.get("id")) for candidate in candidates}
    for point in points:
        if str(point.id) in existing:
            continue
        payload = dict(point.payload or {})
        text = extract_payload_text(payload)
        if not text:
            continue
        expanded.append(
            {
                "id": str(point.id),
                "content": text,
                "source": str(payload.get("source") or "unknown"),
                "fusion_score": 1.0,
                "metadata": dict(payload.get("metadata") or {}),
                "cross_reference_match": True,
            }
        )
    return expanded


def contextualize_query(
    query: str,
    history: list[dict[str, str]],
    nvidia_llama_model: NvidiaLlamaModel,
    locked_entities: list[str] | None = None,
) -> list[str]:
    locked_entities = locked_entities or []
    history_text = format_masked_history(history)
    rewritten = nvidia_llama_model.generate(
        query_condenser_prompt_with_locks(locked_entities),
        f"Conversation history:\n{history_text}\n\nLatest user message:\n{mask_pii_text(query)}",
        temperature=0.0,
    )
    queries = enforce_locked_entities(parse_condensed_queries(rewritten, mask_pii_text(query)), locked_entities)
    return [
        get_memory_manager().redact_condensed_payload(item, query, history)
        for item in queries
    ]


def generate_hypothetical_document(condensed_query: str, groq_model: GroqModel) -> str:
    try:
        return groq_model.generate(HYDE_SYSTEM_PROMPT, condensed_query, temperature=0.3) or condensed_query
    except Exception as exc:
        if is_resource_exhausted_error(exc):
            print(
                f"--- STEP 3: HYDE GROQ 429 FALLBACK ---\n"
                f"Groq quota was exhausted. Using the NVIDIA LLaMA-condensed query for Qdrant search.\n"
                f"Error: {exc}",
                file=sys.stderr,
                flush=True,
            )
        else:
            logger.warning("HyDE generation failed; using condensed query fallback: %s", exc)
        return condensed_query


def is_comparative_query(query: str) -> bool:
    normalized = f" {query.lower()} "
    if any(marker in normalized for marker in COMPARATIVE_MARKERS):
        return True
    figure_mentions = re.findall(r"\bfig(?:ure)?\.?\s+[A-Za-z0-9.:-]+", query, flags=re.IGNORECASE)
    return len(set(mention.lower() for mention in figure_mentions)) >= 2


def _parse_json_query_list(raw_text: str) -> list[str]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def generate_retrieval_queries(
    query: str,
    history: list[dict[str, str]],
    nvidia_llama_model: NvidiaLlamaModel,
) -> list[str]:
    if not is_comparative_query(query):
        return [query]

    history_text = format_masked_history(history)
    raw_response = nvidia_llama_model.generate(
        (
            "Split comparative or synthesis search questions into distinct retrieval queries. "
            "Return ONLY a JSON array of strings. Each string must focus on one figure, chart, table, "
            "metric, entity, or comparison target. Preserve exact figure and table identifiers. "
            "For non-comparative queries, return a single-item JSON array."
        ),
        f"Chat history:\n{history_text}\n\nStandalone query:\n{mask_pii_text(query)}",
        temperature=0.0,
    )
    sub_queries = _parse_json_query_list(raw_response)
    return [
        get_memory_manager().redact_condensed_payload(item, query, history)
        for item in (sub_queries or [query])
    ]


def encode_query(query: str) -> list[float]:
    return [float(value) for value in get_query_vector(query)]


def encode_sparse_query(query: str) -> models.SparseVector:
    return get_sparse_encoder().encode_query(query)


def _collection_sparse_vector_names(client: QdrantClient) -> set[str]:
    try:
        sparse_vectors = client.get_collection(COLLECTION_NAME).config.params.sparse_vectors
        return set(sparse_vectors) if isinstance(sparse_vectors, dict) else set()
    except Exception as exc:
        logger.warning("Could not inspect Qdrant sparse vectors: %s", exc)
        return set()


def is_global_analytics_query(query: str) -> bool:
    return bool(GLOBAL_ANALYTICS_PATTERN.search(query))


def global_analytics_search_query(query: str) -> str:
    return f"{query}{GLOBAL_ANALYTICS_RETRIEVAL_SUFFIX}" if is_global_analytics_query(query) else query


def _summary_header_boost(candidate: dict[str, Any]) -> float:
    searchable = f"{candidate.get('content', '')} {candidate.get('metadata', {})}".lower()
    return 0.05 if any(term in searchable for term in ("dataset summary", "table header", "csv", "summary")) else 0.0


def _asset_query_boost(query: str, candidate: dict[str, Any]) -> float:
    requested = detect_requested_asset_type(query)
    if not requested:
        return 0.0
    metadata = dict(candidate.get("metadata") or {})
    if requested == "table" and (
        metadata.get("contains_table")
        or metadata.get("entity_type") == "table"
        or metadata.get("table_csv_path")
        or metadata.get("csv_path")
        or metadata.get("table_image_path")
    ):
        return 25.0
    if requested == "image" and (
        metadata.get("contains_figure")
        or metadata.get("contains_chart")
        or metadata.get("contains_image")
        or metadata.get("image_path")
        or metadata.get("figure_image_path")
        or metadata.get("chart_image_path")
    ):
        return 25.0
    if requested == "csv" and (metadata.get("contains_csv") or metadata.get("document_type") == "csv"):
        return 25.0
    return 0.0


def _chunk_entity_labels(chunk: dict[str, Any]) -> list[str]:
    metadata = dict(chunk.get("metadata") or {})
    labels: list[str] = []
    values = [
        chunk.get("entity_id"),
        metadata.get("entity_id"),
        metadata.get("figure_id"),
        *(metadata.get("entity_ids") or [] if not isinstance(metadata.get("entity_ids"), str) else [metadata.get("entity_ids")]),
    ]
    for value in values:
        for entity in extract_hard_entities(str(value or "")):
            label = entity["label"]
            if label not in labels:
                labels.append(label)
    return labels


def _normalized_entity_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _chunk_debug_id(chunk: dict[str, Any]) -> str:
    metadata = dict(chunk.get("metadata") or {})
    return str(
        chunk.get("id")
        or metadata.get("chunk_id")
        or metadata.get("parent_id")
        or f"{chunk.get('source', 'unknown')}::{hash(chunk.get('content', ''))}"
    )


def _append_strict_gate_issue(metadata: dict[str, Any], reason: str) -> None:
    issues = metadata.get("_strict_gate_issues") or []
    if isinstance(issues, str):
        issues = [issues]
    if reason not in issues:
        issues.append(reason)
    metadata["_strict_gate_issues"] = issues


def _path_matches_entity_label(path_value: object, entity_label: str) -> bool:
    raw_path = str(path_value or "").strip()
    if not raw_path or not entity_label:
        return False
    filename = Path(raw_path).stem
    entity_matches = extract_hard_entities(entity_label)
    if not entity_matches:
        return False
    entity = entity_matches[0]
    prefix = "table" if entity["kind"] == "table" else "chart" if entity["kind"] == "chart" else "figure"
    expected = _normalized_entity_token(f"{prefix}_{entity['identifier']}")
    actual = _normalized_entity_token(filename)
    return expected == actual


def sanitize_chunk_metadata_bindings(chunk: dict[str, Any]) -> dict[str, Any]:
    item = dict(chunk)
    metadata = dict(item.get("metadata") or {})
    if metadata.get("document_type") == "csv" or str(item.get("source") or "").lower().endswith(".csv"):
        return item
    entity_labels = _chunk_entity_labels(item)
    primary_entity = entity_labels[0] if entity_labels else ""
    primary_kind = ""
    primary_entities = extract_hard_entities(primary_entity)
    if primary_entities:
        primary_kind = primary_entities[0]["kind"]
    path_specs = (
        ("table_csv_path", "table_csv_paths", "table"),
        ("csv_path", "csv_paths", "table"),
        ("table_image_path", "table_image_paths", "table"),
        ("figure_image_path", "figure_image_paths", "figure"),
        ("chart_image_path", "chart_image_paths", "chart"),
        ("image_path", "image_paths", ""),
    )
    for singular_key, plural_key, required_kind in path_specs:
        values = []
        raw_values = metadata.get(plural_key) or metadata.get(singular_key)
        if isinstance(raw_values, str):
            values = [raw_values]
        elif isinstance(raw_values, (list, tuple, set)):
            values = [str(value) for value in raw_values if str(value or "").strip()]
        elif raw_values:
            values = [str(raw_values)]
        kept: list[str] = []
        if required_kind and primary_kind and primary_kind != required_kind:
            if values:
                _append_strict_gate_issue(metadata, "entity/path mismatch")
            metadata.pop(singular_key, None)
            metadata.pop(plural_key, None)
            continue
        for value in values:
            if not primary_entity:
                continue
            if _path_matches_entity_label(value, primary_entity):
                kept.append(value)
            else:
                _append_strict_gate_issue(metadata, "entity/path mismatch")
                logger.warning(
                    "Stripping mismatched asset binding: entity=%s key=%s path=%s",
                    primary_entity,
                    singular_key,
                    value,
                )
        if kept:
            metadata[singular_key] = kept[0]
            metadata[plural_key] = kept
        else:
            metadata.pop(singular_key, None)
            metadata.pop(plural_key, None)
    item["metadata"] = metadata
    return item


def _chunk_has_structural_table_payload(chunk: dict[str, Any]) -> bool:
    metadata = dict(chunk.get("metadata") or {})
    content = str(chunk.get("content") or "")
    strict_extracted = str(metadata.get("strict_extracted_text") or "")
    description = str(metadata.get("description") or "")
    if metadata.get("document_type") == "csv" or str(chunk.get("source") or "").lower().endswith(".csv"):
        return True
    if metadata.get("table_csv_path") or metadata.get("csv_path"):
        signals = (
            "data sheet metric lookup",
            "row id:",
            "context/trend summary:",
            "|",
            ",",
        )
        haystack = f"{content}\n{strict_extracted}\n{description}".lower()
        return any(signal in haystack for signal in signals)
    return False


def _is_narrative_table_proxy(chunk: dict[str, Any]) -> bool:
    metadata = dict(chunk.get("metadata") or {})
    content = str(chunk.get("content") or "").strip()
    if not content:
        return True
    if not _is_table_chunk(chunk):
        return False
    if metadata.get("document_type") == "csv" or str(chunk.get("source") or "").lower().endswith(".csv"):
        return False
    if metadata.get("table_csv_path") and not _chunk_has_structural_table_payload(chunk):
        return True
    return False


def _strip_chunk_asset_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    item = dict(chunk)
    metadata = dict(item.get("metadata") or {})
    for key in (
        "image_path",
        "image_paths",
        "image_local_path",
        "image_name",
        "csv_path",
        "csv_paths",
        "table_csv_path",
        "table_csv_paths",
        "figure_image_path",
        "figure_image_paths",
        "chart_image_path",
        "chart_image_paths",
        "table_image_path",
        "table_image_paths",
    ):
        metadata.pop(key, None)
    item["metadata"] = metadata
    return item


def _strict_match_and_enforce(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bypass_layer_1 = os.getenv("BYPASS_GATEWAY", "true").lower() != "false" or os.getenv("DISABLE_GATEWAY", "true").lower() != "false"
    if bypass_layer_1:
        return chunks
        
    requested_type = detect_requested_asset_type(query)
    target_type, target_number = _extract_target_entity_tuple(query)
    locked_labels = {entity["label"].lower() for entity in extract_hard_entities(query)}

    isolate_assets = bool(target_type in {"table", "figure"} and target_number)
    sanitized = [sanitize_chunk_metadata_bindings(chunk) for chunk in chunks]
    filtered: list[dict[str, Any]] = []
    strict_table_rows: list[dict[str, Any]] = []
    nearby_companions: dict[str, dict[str, Any]] = {}
    visual_primaries: dict[str, dict[str, Any]] = {}
    kept_ids: list[str] = []
    dropped: list[tuple[str, str]] = []
    for chunk in sanitized:
        metadata = dict(chunk.get("metadata") or {})
        chunk_id = _chunk_debug_id(chunk)
        
        # Check if the chunk is a CSV chunk
        is_csv = (metadata.get("document_type") == "csv" or 
                  str(chunk.get("source") or "").lower().endswith(".csv") or
                  "rows" in metadata or "columns" in metadata or "table" in metadata)
        
        # Check if the chunk is a visual chunk
        is_visual = (metadata.get("document_type") in ("image", "visual") or
                     any(key in metadata for key in ("image_path", "figure_image_path", "chart_image_path", "table_image_path")))
        
        # Heterogeneous payload handling: if it's a pure text chunk (lacks CSV/visual keys), let it pass through as raw text
        if not is_csv and not is_visual:
            filtered.append(chunk)
            kept_ids.append(chunk_id)
            continue
            
        if is_csv:
            filtered.append(chunk)
            kept_ids.append(chunk_id)
            continue
        chunk_labels = {label.lower() for label in _chunk_entity_labels(chunk)}
        primary_label = _chunk_primary_entity_label(chunk)
        tuple_match = _chunk_matches_target_tuple(chunk, target_type, target_number)
        locked_overlap = locked_labels & chunk_labels if locked_labels and chunk_labels else set()
        has_matching_asset_path = False
        if isolate_assets:
            requested_asset_key = "table" if target_type == "table" else "image"
            target_tokens = {
                _normalized_identifier_blob(f"{target_type}_{target_number}"),
                _normalized_identifier_blob(f"{target_type} {target_number}"),
                _normalized_identifier_blob(f"{target_type}_{target_number.replace('.', '_')}"),
            }
            for asset_path, asset_type in candidate_asset_paths(chunk, requested_asset_key):
                validation_asset_type = "table" if requested_asset_key == "table" else "image"
                validation = validate_asset_path(asset_path, validation_asset_type)
                if not validation.ok:
                    continue
                normalized_path = _normalized_identifier_blob(Path(validation.path).name)
                if any(token and token in normalized_path for token in target_tokens):
                    has_matching_asset_path = True
                    break
        if target_type and target_number and not tuple_match:
            if isolate_assets and has_matching_asset_path:
                tuple_match = True
            else:
                entity_type, entity_number = _extract_chunk_entity_tuple(chunk)
                if entity_type and entity_number:
                    dropped.append((chunk_id, f"failed exact tuple match ({entity_type}, {entity_number})"))
                else:
                    dropped.append((chunk_id, "missing exact tuple metadata"))
                continue
        if isolate_assets and not has_matching_asset_path:
            logger.warning("Chunk %s is missing verified matching asset path, but letting it pass as text-only context.", chunk_id)
            metadata.pop("image_path", None)
            metadata.pop("figure_image_path", None)
            metadata.pop("chart_image_path", None)
            metadata.pop("table_image_path", None)
            chunk["metadata"] = metadata
            chunk.pop("image_path", None)
        if locked_labels and primary_label and primary_label not in locked_labels:
            dropped.append((chunk_id, "failed strict primary entity match"))
            continue
        if locked_labels and chunk_labels and not locked_overlap:
            dropped.append((chunk_id, "failed strict entity overlap"))
            continue
        if requested_type == "table":
            overlap_label = primary_label if primary_label in locked_labels else (sorted(locked_overlap)[0] if locked_overlap else "")
            if _has_conflicting_visual_asset(metadata, {"table"}):
                if not isolate_assets and overlap_label and str(chunk.get("source") or "").lower().endswith(".pdf"):
                    companion = _strip_chunk_asset_metadata(chunk)
                    companion_metadata = dict(companion.get("metadata") or {})
                    companion_metadata["_strict_gate_role"] = "table_nearby_companion"
                    companion["metadata"] = companion_metadata
                    nearby_companions.setdefault(overlap_label, companion)
                    kept_ids.append(chunk_id)
                else:
                    dropped.append((chunk_id, "table query rejected conflicting figure asset"))
                continue
            if not _is_table_chunk(chunk):
                if not isolate_assets and overlap_label and str(chunk.get("source") or "").lower().endswith(".pdf"):
                    companion = _strip_chunk_asset_metadata(chunk)
                    companion_metadata = dict(companion.get("metadata") or {})
                    companion_metadata["_strict_gate_role"] = "table_nearby_companion"
                    companion["metadata"] = companion_metadata
                    nearby_companions.setdefault(overlap_label, companion)
                    kept_ids.append(chunk_id)
                else:
                    dropped.append((chunk_id, "table query rejected non-table chunk"))
                continue
            if _is_narrative_table_proxy(chunk):
                dropped.append((chunk_id, "narrative table proxy rejected"))
                continue
            if _chunk_has_structural_table_payload(chunk):
                strict_table_rows.append(chunk)
            filtered.append(chunk)
            kept_ids.append(chunk_id)
            continue
        if requested_type == "image":
            overlap_label = primary_label if primary_label in locked_labels else (sorted(locked_overlap)[0] if locked_overlap else "")
            if _has_conflicting_visual_asset(metadata, {"figure"}):
                dropped.append((chunk_id, "figure query rejected conflicting table asset"))
                continue
            if _is_vision_chunk(chunk) and not _is_table_chunk(chunk):
                if overlap_label:
                    visual_primaries.setdefault(overlap_label, chunk)
                filtered.append(chunk)
                kept_ids.append(chunk_id)
                continue
            if not isolate_assets and overlap_label and str(chunk.get("source") or "").lower().endswith(".pdf"):
                companion = _strip_chunk_asset_metadata(chunk)
                companion_metadata = dict(companion.get("metadata") or {})
                companion_metadata["_strict_gate_role"] = "figure_nearby_companion"
                companion["metadata"] = companion_metadata
                nearby_companions.setdefault(overlap_label, companion)
                kept_ids.append(chunk_id)
                continue
            dropped.append((chunk_id, "figure query rejected non-visual chunk"))
            continue
        filtered.append(chunk)
        kept_ids.append(chunk_id)
    if requested_type == "table" and strict_table_rows:
        non_table_chunks = [c for c in filtered if not _is_table_chunk(c)]
        deduped: dict[str, dict[str, Any]] = {}
        for chunk in strict_table_rows:
            key = str(chunk.get("id") or "") or f"{chunk.get('source')}::{hash(chunk.get('content', ''))}"
            deduped[key] = chunk
        filtered = list(deduped.values()) + list(nearby_companions.values()) + non_table_chunks
        kept_ids = [_chunk_debug_id(chunk) for chunk in filtered]
    elif requested_type == "image" and visual_primaries:
        non_visual_chunks = [c for c in filtered if not (_is_vision_chunk(c) and not _is_table_chunk(c))]
        deduped_visuals: dict[str, dict[str, Any]] = {}
        for chunk in visual_primaries.values():
            key = str(chunk.get("id") or "") or f"{chunk.get('source')}::{hash(chunk.get('content', ''))}"
            deduped_visuals[key] = chunk
        filtered = list(deduped_visuals.values()) + list(nearby_companions.values()) + non_visual_chunks
        kept_ids = [_chunk_debug_id(chunk) for chunk in filtered]
    mismatch_drops: list[tuple[str, str]] = []
    for chunk in sanitized:
        metadata = dict(chunk.get("metadata") or {})
        for issue in metadata.get("_strict_gate_issues") or []:
            mismatch_drops.append((_chunk_debug_id(chunk), str(issue)))
    if sanitized:
        print(
            f"DEBUG [Strict Gate]: query={query!r} kept_chunk_ids={kept_ids} "
            f"dropped={ [{'chunk_id': chunk_id, 'reason': reason} for chunk_id, reason in [*dropped, *mismatch_drops]] }",
            file=sys.stderr,
            flush=True,
        )
    return filtered


def extract_payload_text(payload: dict[str, Any]) -> str:
    nested_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    return str(
        payload.get("text")
        or payload.get("page_content")
        or payload.get("content")
        or nested_payload.get("text")
        or ""
    ).strip()


def _exact_identifier_payload_matches(
    client: QdrantClient,
    hard_entities: list[dict[str, str]],
    limit: int,
) -> list[dict[str, Any]]:
    if not hard_entities:
        return []
    ensure_entity_payload_indexes(client)
    entity_filter = build_hard_entity_filter(hard_entities)
    if entity_filter is None:
        return []
    entity_variants = [hard_entity_strict_label_variants(entity) for entity in hard_entities]
    matches: list[dict[str, Any]] = []
    offset = None
    scan_limit = max(limit * 8, 32)
    while len(matches) < scan_limit:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=entity_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            metadata = dict(payload.get("metadata") or {})
            text = extract_payload_text(payload)
            if any(
                _entity_matches_payload_or_asset_path(payload, metadata, text, hard_entity)
                for hard_entity in hard_entities
            ):
                matches.append(
                    {
                        "id": str(point.id),
                        "content": text,
                        "source": str(payload.get("source") or "unknown"),
                        "fusion_score": 1.0,
                        "metadata": metadata,
                        "sparse_rank": len(matches) + 1,
                    }
                )
                if len(matches) >= scan_limit:
                    break
        if offset is None:
            break
    return sorted(matches, key=_exact_asset_priority)[:limit]


def step_three_exact_entity_fallback(
    locked_entities: list[str],
    limit_per_entity: int = PRIMARY_DENSE_TOP_K,
) -> list[dict[str, Any]]:
    """Bypass vector search and fetch literal payload-text matches for locked entities."""

    entities = [str(entity).strip() for entity in locked_entities or [] if str(entity).strip()]
    if not entities:
        return []
    print(
        f"[WARN] Relevance check failed. Step 3 fallback triggered for entities: {entities}",
        file=sys.stderr,
        flush=True,
    )
    client = get_qdrant_client()
    ensure_entity_payload_indexes(client)
    fallback: list[dict[str, Any]] = []
    seen: set[str] = set()

    for entity in entities:
        hard_entities = extract_hard_entities(entity)
        entity_filter = build_hard_entity_filter(hard_entities)
        if entity_filter is None:
            logger.warning("Step 3 exact fallback skipped unindexed entity text scan for %s", entity)
            continue
        entity_matches = 0
        offset = None
        entity_scan_limit = max(limit_per_entity * 8, 32)
        while entity_matches < entity_scan_limit:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=entity_filter,
                limit=limit_per_entity,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                metadata = dict(payload.get("metadata") or {})
                text = extract_payload_text(payload)
                if not any(
                    _entity_matches_payload_or_asset_path(payload, metadata, text, hard_entity)
                    for hard_entity in hard_entities
                ):
                    continue
                parent_id = str(metadata.get("parent_id") or point.id)
                dedupe_key = parent_id or str(point.id)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                content = str(metadata.get("parent_text") or text).strip()
                fallback.append(
                    {
                        "id": str(point.id),
                        "content": content,
                        "source": str(metadata.get("source") or payload.get("source") or "unknown"),
                        "fusion_score": 1.0,
                        "rerank_score": 1.0,
                        "metadata": metadata,
                        "parent_id": parent_id,
                        "step3_fallback_entity": entity,
                    }
                )
                entity_matches += 1
                if entity_matches >= entity_scan_limit:
                    break
            if offset is None:
                break
    return sorted(fallback, key=_exact_asset_priority)[: limit_per_entity * len(entities)]


def _point_response_to_candidates(response: Any) -> list[dict[str, Any]]:
    points = response.points or []
    if not points:
        return []

    candidates: list[dict[str, Any]] = []
    empty_payloads = 0
    for point in points:
        payload = dict(point.payload or {})
        text = extract_payload_text(payload)
        if not text:
            empty_payloads += 1
            continue
        candidates.append(
            {
                "id": str(point.id),
                "content": text,
                "source": str(payload.get("source") or "unknown"),
                "fusion_score": float(point.score),
                "metadata": dict(payload.get("metadata") or {}),
            }
        )

    if not candidates:
        raise RuntimeError(
            "Qdrant returned search matches, but none contained payload['text']. "
            f"Empty payload matches skipped: {empty_payloads}. Re-run ingestion with ingest_data.py."
        )
    return candidates


def _rrf_merge(
    dense_candidates: list[dict[str, Any]],
    sparse_candidates: list[dict[str, Any]],
    limit: int = HYBRID_RESULT_LIMIT,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path, candidates in (("dense", dense_candidates), ("sparse", sparse_candidates)):
        for rank, candidate in enumerate(candidates, start=1):
            point_id = str(candidate.get("id") or "")
            dedupe_key = point_id or f"{candidate.get('source')}::{hash(candidate.get('content', ''))}"
            item = merged.setdefault(dedupe_key, dict(candidate))
            item["rrf_score"] = float(item.get("rrf_score", 0.0)) + (1.0 / (RRF_K + rank))
            item["fusion_score"] = item["rrf_score"]
            item[f"{path}_rank"] = rank
    return sorted(merged.values(), key=lambda item: float(item["rrf_score"]), reverse=True)[:limit]


def hybrid_search(
    condensed_query: str,
    hypothetical_doc: str,
    candidate_limit: int = PRIMARY_DENSE_TOP_K,
    result_limit: int = HYBRID_RESULT_LIMIT,
    sparse_only: bool = False,
    structural_intent: str = "CONCEPTUAL_TEXTUAL",
    qdrant_duration_accum: list[float] | None = None,
) -> list[dict[str, Any]]:
    print(
        f"\n{'=' * 96}\n--- STEP 1: RETRIEVAL INPUT ---\nCondensed query:\n{condensed_query}\n\n"
        f"HyDE dense-search document:\n{hypothetical_doc}\n{'=' * 96}",
        file=sys.stderr,
        flush=True,
    )
    hard_entities = extract_hard_entities(condensed_query)
    chapter_numbers = extract_chapter_references(condensed_query)
    sparse_query_text = f"{global_analytics_search_query(condensed_query)}{hard_entity_query_suffix(hard_entities)}"
    dense = None if sparse_only else encode_query(hypothetical_doc or condensed_query)
    hard_filter = build_hard_entity_filter(hard_entities)
    
    # Detect strict numerical metrics / timelines query
    is_numeric_query = (structural_intent == "TABULAR_NUMERIC")
    
    csv_filter = None
    # Skip csv_filter to allow unified parallel search across text, CSV, and visual collections simultaneously
    # if is_numeric_query:
    #     csv_filter = models.Filter(
    #         must=[
    #             models.FieldCondition(
    #                 key="metadata.document_type",
    #                 match=models.MatchValue(value="csv")
    #             )
    #         ]
    #     )
        
    client = get_qdrant_client()
    ensure_entity_payload_indexes(client)
    chapter_filter = build_chapter_filter(chapter_numbers)
    scoped_filter = combine_filters(hard_filter, chapter_filter, csv_filter)
    
    # To prevent CSV chunks from being strictly filtered out by hard/chapter filters,
    # we allow any chunk that matches the scoped_filter OR is a CSV chunk
    if scoped_filter is not None:
        scoped_filter = models.Filter(
            should=[
                scoped_filter,
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_type",
                            match=models.MatchValue(value="csv")
                        )
                    ]
                )
            ]
        )

    dense_limit = max(int(candidate_limit), 1) if hard_entities else min(max(int(candidate_limit), 1), PRIMARY_DENSE_TOP_K)

    def _search(active_filter: models.Filter | None) -> list[dict[str, Any]]:
        qdrant_start = time.perf_counter()
        dense_candidates = []
        if not sparse_only:
            dense_response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense,
                using="dense",
                query_filter=active_filter,
                limit=dense_limit,
                with_payload=True,
            )
            dense_candidates = _point_response_to_candidates(dense_response)
            _debug_log_chunks("STEP 1A: RAW DENSE QDRANT MATCHES", dense_candidates)
        if SPARSE_VECTOR_NAME not in _collection_sparse_vector_names(client):
            if sparse_only:
                logger.warning("Sparse vector slot is unavailable; using exact identifier payload scan.")
                result = _exact_identifier_payload_matches(client, hard_entities, result_limit)
            else:
                logger.warning("Sparse vector slot is unavailable; returning dense retrieval results.")
                result = dense_candidates[:result_limit]
            if qdrant_duration_accum is not None:
                qdrant_duration_accum.append(time.perf_counter() - qdrant_start)
            return result
        sparse_response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=encode_sparse_query(sparse_query_text),
            using=SPARSE_VECTOR_NAME,
            query_filter=active_filter,
            limit=dense_limit,
            with_payload=True,
        )
        sparse_candidates = _point_response_to_candidates(sparse_response)
        _debug_log_chunks("STEP 1B: RAW SPARSE QDRANT MATCHES", sparse_candidates)
        if sparse_only:
            logger.info("Explicit identifier detected; returning sparse-only keyword matches.")
            result = sparse_candidates[:result_limit] or _exact_identifier_payload_matches(client, hard_entities, result_limit)
        else:
            fused_candidates = _rrf_merge(dense_candidates, sparse_candidates, result_limit)
            _debug_log_chunks("STEP 1C: RRF-FUSED QDRANT MATCHES", fused_candidates)
            result = fused_candidates
        if qdrant_duration_accum is not None:
            qdrant_duration_accum.append(time.perf_counter() - qdrant_start)
        return result

    if scoped_filter:
        logger.info(
            "Applying metadata filter before retrieval: hard_entities=%s chapters=%s",
            ", ".join(entity["label"] for entity in hard_entities) or "(none)",
            ", ".join(chapter_numbers) or "(none)",
        )
        filtered_candidates = _search(scoped_filter)
        # Automatic Semantic Fallback: if search returns 0 chunks, strip filters and rerun
        if len(filtered_candidates) >= 1:
            return filtered_candidates
        logger.warning("Filtered search returned 0 chunks. Instantly stripping away all metadata filters and falling back to pure unfiltered semantic search.")
        
    return _search(None)



def merge_and_dedupe_candidates(candidate_groups: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for sub_query, candidates in candidate_groups:
        for candidate in candidates:
            point_id = str(candidate.get("id") or "")
            dedupe_key = point_id or f"{candidate.get('source')}::{hash(candidate.get('content', ''))}"
            if dedupe_key not in merged:
                item = dict(candidate)
                item["matched_sub_queries"] = [sub_query]
                merged[dedupe_key] = item
                continue
            existing = merged[dedupe_key]
            existing["fusion_score"] = max(
                float(existing.get("fusion_score", 0.0)),
                float(candidate.get("fusion_score", 0.0)),
            )
            existing.setdefault("matched_sub_queries", [])
            if sub_query not in existing["matched_sub_queries"]:
                existing["matched_sub_queries"].append(sub_query)
    return sorted(merged.values(), key=lambda item: float(item.get("fusion_score", 0.0)), reverse=True)


def rerank_context(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int = RERANK_TOP_N,
    locked_entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    query = mask_pii_text(query)
    pairs = [(query, candidate["content"]) for candidate in candidates]
    scores = load_reranker_model().score_pairs(pairs)

    reranked: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        boost = 10000.0 if check_entity_match(query, item) else 0.0
        item["rerank_score"] = float(score) + _summary_header_boost(item) + _asset_query_boost(query, item) + boost
        reranked.append(item)
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    reranked = promote_locked_entity_candidates(reranked, locked_entities or [])
    top_chunks = reranked[:top_n]
    print(f"\n{'=' * 96}\n--- STEP 2: RERANKING INPUT QUERY ---\n{query}", file=sys.stderr, flush=True)
    _debug_log_chunks("STEP 2: TOP CHUNKS AFTER CROSS-ENCODER RERANKING", top_chunks)
    return top_chunks


def rerank_balanced_context(
    query: str,
    candidate_groups: list[tuple[str, list[dict[str, Any]]]],
    per_bucket: int = 3,
    locked_entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    reranked_groups = [
        (sub_query, rerank_context(query, candidates, top_n=per_bucket, locked_entities=locked_entities))
        for sub_query, candidates in candidate_groups
    ]
    merged = merge_and_dedupe_candidates(reranked_groups)
    return promote_locked_entity_candidates(merged, locked_entities or [])


def expand_reranked_children_to_parents(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace selected child text with one full parent context per parent ID."""

    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        item = dict(candidate)
        metadata = dict(item.get("metadata") or {})
        parent_id = str(metadata.get("parent_id") or item.get("id") or "")
        if parent_id in seen:
            continue
        seen.add(parent_id)
        if metadata.get("preserve_child_text"):
            parent_text = str(metadata.get("parent_text") or item.get("content") or "").strip()
            item["supporting_parent_text"] = parent_text
        else:
            parent_text = str(metadata.get("parent_text") or item.get("content") or "").strip()
            item["content"] = parent_text
        item["parent_id"] = parent_id
        item["retrieved_child_id"] = str(item.get("id") or "")
        expanded.append(item)
    return expanded


def parse_context_relevance_response(response: str) -> bool:
    """Parse strict JSON relevance output from the NVIDIA gatekeeper."""

    try:
        parsed = json.loads(str(response or "").strip())
        return str(parsed.get("is_relevant", "")).strip().lower() == "yes"
    except Exception:
        normalized = str(response or "").strip().lower()
        return '"is_relevant"' in normalized and '"yes"' in normalized


def meaningful_query_terms(query: str) -> set[str]:
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(query or ""))
        if token.lower() not in RELEVANCE_STOPWORDS
    }
    return terms


def has_minimum_relevance_signal(query: str, chunks: list[dict[str, Any]], structural_intent: str = "CONCEPTUAL_TEXTUAL") -> bool:
    if not chunks:
        return False

    context = "\n".join(str(chunk.get("content") or "") for chunk in chunks).lower()

    if structural_intent == "TABULAR_NUMERIC":
        query_lower = query.lower()
        has_digit = any(c.isdigit() for c in query_lower)
        context_has_digit = any(c.isdigit() for c in context)
        query_terms = set(re.findall(r"[a-z0-9_-]+", query_lower))
        context_terms = set(re.findall(r"[a-z0-9_-]+", context))
        indicators = {"gdp", "emission", "emissions", "co2", "revenue", "metric", "indicator", "table", "timeline", "statistics", "stats", "percent", "percentage", "income", "group"}
        countries = {"india", "ind", "sri lanka", "lka", "timor-leste", "tls", "nauru", "nru", "bangladesh", "nepal", "bhutan", "maldives"}
        relevant_query_terms = query_terms & (indicators | countries)
        if not relevant_query_terms:
            stopwords = {"the", "and", "for", "what", "is", "of", "in", "to", "are", "with", "by", "at"}
            relevant_query_terms = {t for t in query_terms if t not in stopwords and len(t) > 2}
        term_overlap = bool(relevant_query_terms & context_terms)
        return term_overlap or (has_digit and context_has_digit)

    elif structural_intent == "ASSET_VISUAL":
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            for key in ("image_path", "figure_image_path", "chart_image_path", "table_image_path", "image_local_path", "visual_path"):
                if key in meta and meta[key]:
                    path_val = str(meta[key])
                    if os.path.exists(path_val):
                        return True
        return False

    else:
        requested_entities = [entity["label"].lower() for entity in extract_hard_entities(query)]
        if requested_entities:
            return any(entity in context for entity in requested_entities)

        query_terms = meaningful_query_terms(query)
        if not query_terms:
            return False
        return bool(query_terms & set(re.findall(r"[a-z][a-z0-9_-]{2,}", context)))


def strip_visual_metadata(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        metadata = dict(item.get("metadata") or {})
        for key in ("image_path", "image_local_path", "image_name", "asset_path", "visual_path"):
            metadata.pop(key, None)
        item["metadata"] = metadata
        stripped.append(item)
    return stripped


_TRACKING_BRACKET_PATTERN = re.compile(
    r"\[(?:TRACKING|META|LAYOUT|COORD|BBOX)[^\]]*\]",
    re.IGNORECASE,
)
_NESTED_TRACKING_PATTERN = re.compile(r"\[\[.*?\]\]", re.DOTALL)
_LAYOUT_LINE_PATTERN = re.compile(
    r"^(?:page[_\s-]*\d*|layout|bbox|bounding[_\s-]*box|coordinates?|font[_\s-]*(?:size|name)?|"
    r"margin|reading[_\s-]*order|block[_\s-]*type|element[_\s-]*id)\s*[:=].*$",
    re.IGNORECASE | re.MULTILINE,
)
_TRAILING_LAYOUT_BLOCK_PATTERN = re.compile(
    r"\n(?:---+|\*{3,})\s*(?:layout|metadata|tracking|coordinates?).*$",
    re.IGNORECASE | re.DOTALL,
)
_LAYOUT_METADATA_KEYS = frozenset({
    "bbox",
    "bounding_box",
    "coordinates",
    "font_size",
    "font_name",
    "page_width",
    "page_height",
    "layout",
    "tracking_id",
    "block_type",
    "element_id",
    "reading_order",
    "x0",
    "y0",
    "x1",
    "y1",
})


def prune_chunk_text(text: Any) -> str:
    cleaned = str(text or "")
    cleaned = _TRACKING_BRACKET_PATTERN.sub("", cleaned)
    cleaned = _NESTED_TRACKING_PATTERN.sub("", cleaned)
    cleaned = _LAYOUT_LINE_PATTERN.sub("", cleaned)
    cleaned = _TRAILING_LAYOUT_BLOCK_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def prune_context_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip retrieval padding and layout noise while preserving text, tables, and diagram summaries."""

    pruned: list[dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        content = prune_chunk_text(item.get("content") or item.get("text") or item.get("page_content") or "")
        if not content:
            continue
        item["content"] = content
        item.pop("text", None)
        item.pop("page_content", None)

        metadata = dict(item.get("metadata") or {})
        for key in list(metadata.keys()):
            lowered = str(key).lower()
            if key in _LAYOUT_METADATA_KEYS or lowered.startswith("layout_") or lowered.endswith("_bbox"):
                metadata.pop(key, None)
        item["metadata"] = metadata
        pruned.append(item)
    return pruned


def parse_hallucination_response(response: str) -> bool:
    """Return True only when the judge explicitly marks the draft as grounded."""

    try:
        parsed = json.loads(str(response or "").strip())
        return str(parsed.get("is_grounded", "")).strip().lower() == "yes"
    except Exception:
        normalized = str(response or "").strip().lower()
        return '"is_grounded"' in normalized and '"yes"' in normalized


def _format_chunk_for_generation(chunk: dict[str, Any], index: int) -> str:
    metadata = dict(chunk.get("metadata") or {})
    role = str(metadata.get("_strict_gate_role") or "")
    if role == "table_nearby_companion":
        evidence_type = "Nearby narrative companion"
    elif role == "figure_nearby_companion":
        evidence_type = "Nearby narrative companion"
    elif metadata.get("document_type") == "csv" or str(chunk.get("source") or "").lower().endswith(".csv"):
        evidence_type = "Structured extracted table rows"
    elif _is_vision_chunk(chunk):
        evidence_type = "Extracted figure/chart evidence"
    else:
        evidence_type = "Retrieved context"
    return (
        f"Evidence Item {index}\n"
        f"Evidence Type: {evidence_type}\n"
        f"Source: [{chunk['source']}]\n"
        f"Metadata: {chunk.get('metadata', {})}\n"
        f"Text: {chunk['content']}"
    )


def generate_final_answer(
    query: str,
    condensed_query: str,
    hypothetical_doc: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
    nvidia_final_model: NvidiaLlamaModel,
    global_analytics: bool = False,
    generation_payload: dict[str, Any] | None = None,
) -> str:
    if generation_payload:
        history_text = mask_pii_text(generation_payload.get("chat_history_transcript") or "")
        context = str(generation_payload.get("compressed_context_text") or "")
        active_asset_paths = list(generation_payload.get("active_asset_paths") or [])
    else:
        history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-6:])
        history_text = mask_pii_text(history_text)
        context = "\n\n".join(
            _format_chunk_for_generation(chunk, index)
            for index, chunk in enumerate(chunks, start=1)
        )
        active_asset_paths = []
    system_prompt = (
        f"{SECURE_GENERATION_PROMPT}\n\n"
        f"{EXECUTIVE_FORMATTER_PROMPT}\n\n"
        f"{GROUNDED_QA_PROMPT}\n\n"
        "For table questions, load verified extracted table rows first and answer from those rows first. Add nearby "
        "narrative companion context only after the table facts are stated. Never invent a new table, never paste raw "
        "rows verbatim into the prose answer, and never restate the same extracted rows as a second duplicate table in "
        "description text. Convert row evidence into clear informative sentences and paragraphs.\n\n"
        "For figure or chart questions, describe the extracted visual evidence first, then add only the most relevant "
        "nearby narrative companion context afterward. Do not let nearby prose override extracted figure evidence.\n\n"
        f"{FIGURE_TABLE_GUARDRAIL}\n\n"
        f"{USER_FACING_PERSONA_GUARDRAIL}\n\n"
        f"{GLOBAL_ANALYTICS_FORMATTER_GUARDRAIL if global_analytics else ''}"
    )
    asset_block = "\n".join(f"- {path}" for path in active_asset_paths) or "(none)"
    llm_payload = (
        f"Recent conversation history:\n{history_text or '(none)'}\n\n"
        f"Original user query:\n{query}\n\n"
        f"Standalone retrieval query:\n{condensed_query}\n\n"
        f"Hypothetical Answer (HyDE; routing structure only, never evidence):\n{hypothetical_doc}\n\n"
        f"Active visual/data file paths for this turn:\n{asset_block}\n\n"
        f"Real Retrieved Chunks from Qdrant:\n{context}"
    )
    print(
        f"\n{'=' * 96}\n--- STEP 3: FINAL LLM PROMPT ASSEMBLY ---\nSYSTEM PROMPT:\n{system_prompt}\n\n"
        f"USER PAYLOAD:\n{llm_payload}\n{'=' * 96}",
        file=sys.stderr,
        flush=True,
    )
    draft_answer = nvidia_final_model.generate(system_prompt, llm_payload, temperature=0.1)
    print(
        f"\n{'=' * 96}\n--- STEP 4: SECURE GENERATION DRAFT ANSWER ---\n{draft_answer}\n{'=' * 96}",
        file=sys.stderr,
        flush=True,
    )
    judge_payload = (
        f"[DRAFT ANSWER]\n{draft_answer}\n[END OF DRAFT ANSWER]\n\n"
        f"[RETRIEVED CONTEXT CHUNKS]\n{context}\n[END OF RETRIEVED CONTEXT CHUNKS]\n\n"
        'Evaluate whether the draft answer is fully grounded. Respond only with {"is_grounded": "yes"} or {"is_grounded": "no"}.'
    )
    judge_response = nvidia_final_model.generate(HALLUCINATION_JUDGE_PROMPT, judge_payload, temperature=0.0)
    judge_response_text = str(getattr(judge_response, "text", judge_response) or "")
    judge_response_text = judge_response_text.strip()
    is_grounded = parse_hallucination_response(judge_response_text)
    print(
        f"DEBUG [Step 5 Hallucination]: Raw -> {judge_response_text} | Parsed -> {is_grounded}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"\n{'=' * 96}\n--- STEP 5: HALLUCINATION JUDGE RESPONSE ---\n{judge_response}\n{'=' * 96}",
        file=sys.stderr,
        flush=True,
    )
    if is_grounded:
        return sanitize_user_answer(draft_answer)

    logger.warning("Step 5 hallucination judge returned not grounded; invoking self-corrected rewrite.")
    print(
        "[WARN] Hallucination detected. Triggering self-corrected rewrite...",
        file=sys.stderr,
        flush=True,
    )
    corrected_draft = nvidia_final_model.generate(
        f"{system_prompt}\n\n{SELF_CORRECTED_REWRITE_PROMPT}",
        llm_payload,
        temperature=0.0,
    )
    print(
        f"\n{'=' * 96}\n--- STEP 5: SELF-CORRECTED REWRITE ---\n{corrected_draft}\n{'=' * 96}",
        file=sys.stderr,
        flush=True,
    )
    return sanitize_user_answer(corrected_draft)


def clean_context_metadata(text: str) -> str:
    """Strip out unnecessary coordinate maps, JSON syntax brackets, or double spacing to minimize API token payload."""
    if not text:
        return ""
    # Strip coordinate maps like [x1, y1, x2, y2]
    text = re.sub(r'\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]', '', text)
    # Strip other numeric arrays in brackets
    text = re.sub(r'\[\s*\d+(?:\.\d+)?\s*(?:,\s*\d+(?:\.\d+)?\s*)*\]', '', text)
    # Strip empty JSON/list brackets
    text = re.sub(r'\{\s*\}', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    # Normalize spacing
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def parse_target_asset(query: str) -> tuple[str | None, str | None]:
    """Parse user's query to catch the targeted category and identifier (e.g. Table 6.1)."""
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


def check_entity_match(query: str, candidate: dict[str, Any]) -> bool:
    """Check if the chunk's metadata or content matches the user's requested entity (e.g. Figure_4.2)."""
    target_cat, target_id = parse_target_asset(query)
    if not (target_cat and target_id):
        return False
        
    patterns = [
        f"{target_cat}_{target_id}".lower(),
        f"{target_cat} {target_id}".lower(),
    ]
    
    metadata = dict(candidate.get("metadata") or {})
    
    entity_ids = metadata.get("entity_ids") or []
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]
    elif not isinstance(entity_ids, list):
        entity_ids = [str(entity_ids)]
        
    for eid in entity_ids:
        eid_str = str(eid).lower()
        if any(pat in eid_str for pat in patterns):
            return True
            
    for key in ("entity_id", "figure_id", "linked_entity_id", "visual_title", "caption_text"):
        val_str = str(metadata.get(key) or "").lower()
        if any(pat in val_str for pat in patterns):
            return True
            
    content_str = str(candidate.get("content") or candidate.get("text") or "").lower()
    if any(pat in content_str for pat in patterns):
        return True
        
    return False


def extract_type_and_id_for_chunk(vis: dict[str, Any]) -> tuple[str, str]:
    metadata = dict(vis.get("metadata") or {})
    entity_type, entity_id = _extract_chunk_entity_tuple(vis)
    t = entity_type or vis.get("type") or metadata.get("entity_type") or metadata.get("asset_type") or ""
    i = entity_id or vis.get("id") or metadata.get("asset_id") or metadata.get("entity_id") or ""
    t_str = str(t).lower()
    if t_str in ("chart", "diagram", "graph"):
        t_str = "figure"
    return t_str, str(i).strip().lower()


def isolate_asset_text(text: str, target_cat: str, target_id: str) -> str:
    """Isolate text around the target asset and strip competing asset categories."""
    if not text:
        return ""
    
    escaped_id = re.escape(target_id)
    target_pattern = re.compile(rf"\b{target_cat}[\s_]*{escaped_id}\b", flags=re.IGNORECASE)
    
    all_cats = ["table", "figure", "chart", "diagram", "graph", "box"]
    target_cat_clean = target_cat.lower()
    if target_cat_clean in ("figure", "fig", "chart", "diagram", "graph"):
        target_group = {"figure", "fig", "chart", "diagram", "graph"}
    elif target_cat_clean in ("table", "tabel", "tab"):
        target_group = {"table", "tabel", "tab"}
    else:
        target_group = {target_cat_clean}
    
    competing_cats = [c for c in all_cats if c not in target_group]
    
    has_competing = False
    for comp_cat in competing_cats:
        comp_pattern = re.compile(rf"\b{comp_cat}[\s_]*{escaped_id}\b", flags=re.IGNORECASE)
        if comp_pattern.search(text):
            has_competing = True
            break
            
    if not has_competing:
        return text
        
    matches = list(target_pattern.finditer(text))
    if not matches:
        return text
        
    window_half = 500
    intervals = []
    for match in matches:
        start_idx = match.start()
        end_idx = match.end()
        w_start = max(0, start_idx - window_half)
        w_end = min(len(text), end_idx + window_half)
        intervals.append((w_start, w_end))
        
    intervals.sort()
    merged_intervals = []
    for current in intervals:
        if not merged_intervals:
            merged_intervals.append(current)
        else:
            prev = merged_intervals[-1]
            if current[0] <= prev[1]:
                merged_intervals[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged_intervals.append(current)
                
    parts = []
    for start, end in merged_intervals:
        prefix = "" if start == 0 else "... "
        suffix = "" if end == len(text) else " ..."
        parts.append(f"{prefix}{text[start:end]}{suffix}")
        
    return "\n\n".join(parts)


def extract_target_keyword(query: str) -> str | None:
    """Extract target keyword (like '4.2') from the user's query."""
    cat, id_val = parse_target_asset(query)
    if id_val:
        return id_val
    match = re.search(r'\b\d+\.\d+\b', query)
    if match:
        return match.group(0)
    return None


def split_and_filter_payload(content: str, target_keyword: str) -> str:
    """Split the payload and filter down to sub-objects/lines containing the keyword."""
    if not content:
        return ""
        
    parts = re.split(r'(?=\b(?:Figure|Table|Chart|Box|Diagram|Graph|Spotlight)[\s_]*(?:[sS]?\d+(?:\.\d+)*)\b)', content, flags=re.IGNORECASE)
    if len(parts) <= 1:
        parts = re.split(r'\n+', content)
        
    filtered_parts = []
    has_split_bomb = False
    
    for part in parts:
        part_str = part.strip()
        if not part_str:
            continue
        if target_keyword.lower() in part_str.lower():
            filtered_parts.append(part_str)
        else:
            has_split_bomb = True
            
    if has_split_bomb and len(filtered_parts) < len(parts):
        print(f"✂️ Discovered unified payload bomb. Split and filtered down to ONLY chunks containing our target keyword.")
        
    return "\n\n".join(filtered_parts)


def strip_geographic_noise(text: str) -> str:
    """Remove raw lists of uppercase/titlecase location headers without punctuation (more than 3 consecutive lines)."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned_lines = []
    consecutive_headers = []
    
    def is_location_header(line: str) -> bool:
        line_clean = line.strip()
        if not line_clean:
            return False
        if any(char in line_clean for char in [".", ",", ";", ":", "?", "!"]):
            return False
        words = line_clean.split()
        if not words:
            return False
        is_cap = all(w[0].isupper() if w else True for w in words)
        return is_cap and len(line_clean) < 35

    for line in lines:
        if is_location_header(line):
            consecutive_headers.append(line)
        else:
            if len(consecutive_headers) > 3:
                pass
            else:
                cleaned_lines.extend(consecutive_headers)
            consecutive_headers = []
            cleaned_lines.append(line)
            
    if len(consecutive_headers) > 3:
        pass
    else:
        cleaned_lines.extend(consecutive_headers)
        
    return "\n".join(cleaned_lines)


def mask_competing_sentences(text: str, target_cat: str, target_id: str) -> str:
    """Mask sentence structures that contain competing targets to protect token payload."""
    if not text:
        return ""
        
    all_cats = ["table", "figure", "chart", "diagram", "graph", "box", "spotlight"]
    target_cat_clean = target_cat.lower()
    if target_cat_clean in ("figure", "fig", "chart", "diagram", "graph"):
        target_group = {"figure", "fig", "chart", "diagram", "graph"}
    elif target_cat_clean in ("table", "tabel", "tab"):
        target_group = {"table", "tabel", "tab"}
    else:
        target_group = {target_cat_clean}
    
    competing_cats = [c for c in all_cats if c not in target_group]
    
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    sanitized_sentences = []
    
    for sentence in sentences:
        has_competing = False
        for comp_cat in competing_cats:
            escaped_id = re.escape(target_id)
            comp_pattern = re.compile(rf"\b{comp_cat}[\s_]*{escaped_id}\b", flags=re.IGNORECASE)
            if comp_pattern.search(sentence):
                has_competing = True
                break
                
        if has_competing:
            continue
        else:
            sanitized_sentences.append(sentence)
            
    return " ".join(sanitized_sentences)


def format_pruned_chunks_for_context(
    chunks: list[dict[str, Any]],
    target_cat: str | None = None,
    target_id: str | None = None
) -> str:
    context_parts: list[str] = []
    for index, chunk in enumerate(prune_context_chunks(list(chunks or [])), start=1):
        content = chunk.get("content") or ""
        source = chunk.get("source") or "Unknown Source"
        meta = chunk.get("metadata") or {}
        
        if not is_csv_chunk(content, meta):
            content = strip_geographic_noise(content)
            if target_cat and target_id:
                content = mask_competing_sentences(content, target_cat, target_id)
                content = isolate_asset_text(content, target_cat, target_id)
            
        if is_csv_chunk(content, meta):
            formatted_content = parse_csv_to_markdown(content)
            title = f"### Tabular CSV Context {index} (Source: {source})"
        else:
            formatted_content = content
            title = f"### Text Chunk Context {index} (Source: {source})"
        context_parts.append(f"{title}\n{formatted_content}")
    return "\n\n".join(context_parts)


def build_grounded_generation_messages(
    user_query: str,
    retrieved_chunks: list,
    chat_history: list | None = None,
    metrics_out: dict[str, float] | None = None,
) -> list[BaseMessage]:
    prompt_start = time.perf_counter()
    target_cat, target_id = parse_target_asset(user_query)
    context_block = format_pruned_chunks_for_context(list(retrieved_chunks or []), target_cat, target_id)
    context_block = clean_context_metadata(context_block)

    history_str = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"

    prompt = f"""You are an advanced, helpful, and accurate Conversational RAG assistant.
You are provided with a set of multi-source evidence chunks (retrieved from PDF text, CSV tables, and visual metadata collections) and the ongoing chat history.
Your goal is to generate a comprehensive, contextually accurate, and well-grounded response to the user's latest question.

GROUND RULES:
1. Base your answer STRICTLY on the facts provided in the multi-source evidence below. Do not assume or extrapolate.
2. If the evidence contains tables or CSV metrics, represent the numbers and data accurately in your response.
3. If the evidence contains visual captions or details, describe the visual elements accurately as they appear in the source.
4. Integrate information from text, CSV, and visual sources to answer mixed-data queries seamlessly.
5. If the evidence is insufficient to answer the question, state that clearly.
6. Write in a professional, engaging, and clear conversational tone.
7. NUMERICAL ACCURACY: Be extremely careful with numbers. If there are values in a figure or table, those values must be present in the answer; do not avoid or omit them, and ensure all values are accurate. If the text data looks slightly messy or conflicting, prioritize the exact numbers written in the main sentences/captions. Do not guess or change digits (like writing 17.1 instead of 17.3).

MULTI-SOURCE EVIDENCE:
{context_block}

CHAT HISTORY:
{history_str or "(No prior conversation history)"}

USER'S QUESTION:
{user_query}

Generate your comprehensive grounded response:"""

    prompt_duration = time.perf_counter() - prompt_start
    print(f"⏱️ PROMPT CONSTRUCTED IN: {prompt_duration:.2f} seconds", flush=True)
    print(f"📊 TOTAL CHARACTER LENGTH OF CONTEXT: {len(context_block)}", flush=True)
    if metrics_out is not None:
        metrics_out["prompt_construction"] = prompt_duration
        metrics_out["context_character_length"] = float(len(context_block))

    print(
        f"\n{'=' * 96}\n--- STEP 3: GEMINI FINAL ANSWER GENERATION ---\nPROMPT:\n{prompt}\n{'=' * 96}",
        file=sys.stderr,
        flush=True,
    )
    system_prompt = (
        "The context provided below contains high-density information including structured Markdown tables and isolated text passages. "
        "Analyze the alignment of rows and columns carefully to extract exact metrics, numbers, and tabular data to formulate your final answer. "
        "If the answer is in a table, reference it accurately."
    )
    if target_cat and target_id:
        system_prompt += (
            f"\n\nCRITICAL ENGINE RULE: The user is explicitly querying about a specific asset target: {target_cat} {target_id} (e.g., Table 6.1). "
            f"The provided vector context chunks may contain adjacent or overlapping references to other structural elements with the same identifier, "
            f"such as Figure {target_id} or Chart {target_id}. You are strictly forbidden from summarizing, processing, extracting, or mentioning "
            f"any data originating from a competing asset category. If the query asks for a Table, do not return descriptive details about a Figure, "
            f"and vice versa. Focus entirely on the text, properties, and values belonging explicitly to the requested target asset category."
        )
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]


def build_combined_synthesis_messages(
    user_query: str,
    qdrant_chunks: str,
    pandas_output: str,
    target_cat: str | None = None,
    target_id: str | None = None,
) -> list[BaseMessage]:
    blended_prompt = f"""Context from Database (Qdrant Document Vector Search):
{prune_chunk_text(qdrant_chunks) or "(No matching text or visual document context found in vector database)"}

Context from Table (Pandas DataFrame Data Extraction):
{prune_chunk_text(pandas_output)}"""

    synthesis_prompt = f"""You are an advanced, helpful, and accurate Conversational RAG assistant.
You are provided with context from two different sources: document text passages/figures from the Qdrant database, and tabular data/calculations extracted from our spreadsheets by a LangChain Pandas DataFrame agent.

Your goal is to write a single comprehensive, well-structured final answer that seamlessly integrates and cross-references both sources of information to answer the user's question.

CONTEXTS:
{blended_prompt}

USER'S QUESTION:
{user_query}

Write your comprehensive, integrated final answer:"""

    system_prompt = (
        "You are a synthesis engine. Read both the document context and the tabular data extraction, "
        "and write a single comprehensive, well-structured final answer cross-referencing the data. "
        "Maintain professional tone and accuracy. Do not make up facts."
    )
    if target_cat and target_id:
        system_prompt += (
            f"\n\nCRITICAL ENGINE RULE: The user is explicitly querying about a specific asset target: {target_cat} {target_id} (e.g., Table 6.1). "
            f"The provided vector context chunks may contain adjacent or overlapping references to other structural elements with the same identifier, "
            f"such as Figure {target_id} or Chart {target_id}. You are strictly forbidden from summarizing, processing, extracting, or mentioning "
            f"any data originating from a competing asset category. If the query asks for a Table, do not return descriptive details about a Figure, "
            f"and vice versa. Focus entirely on the text, properties, and values belonging explicitly to the requested target asset category."
        )
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=synthesis_prompt),
    ]


def _render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.header("Configuration")
        st.text_input("Qdrant Mode", value=f"server: {QDRANT_HOST}:{QDRANT_PORT}", disabled=True)
        st.text_input("Collection", value=COLLECTION_NAME, disabled=True)
        st.text_input("Embedding Model", value=EMBEDDING_MODEL_NAME, disabled=True)
        st.text_input("Reranker", value=RERANK_MODEL_NAME, disabled=True)
        st.text_input("Final Answer LLM", value=NVIDIA_FINAL_MODEL_NAME, disabled=True)
        api_key = st.text_input(
            "Groq API Key",
            value=resolve_groq_api_key(),
            type="password",
        )
        nvidia_api_key = st.text_input(
            "NVIDIA NIM API Key",
            value=resolve_nvidia_api_key(),
            type="password",
        )
        if st.button("Clear Chat", width="stretch"):
            get_memory_manager().clear_history(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()

    return api_key.strip(), nvidia_api_key.strip()


def _render_sources(chunks: list[dict[str, Any]]) -> None:
    with st.expander("View Retrieved Sources"):
        seen_image_paths = set()
        for index, chunk in enumerate(chunks, start=1):
            img_path = chunk.get("metadata", {}).get("image_path")
            if img_path:
                if img_path in seen_image_paths:
                    continue  # Skip processing this chunk completely if the image was already drawn
                seen_image_paths.add(img_path)

            st.markdown(f"**{index}. {chunk['source']}**")
            st.write(
                {
                    "point_id": chunk.get("id"),
                    "fusion_score": chunk.get("fusion_score"),
                    "rerank_score": chunk.get("rerank_score"),
                    "matched_sub_queries": chunk.get("matched_sub_queries", []),
                    "metadata": chunk.get("metadata", {}),
                }
            )
            # Safe extraction fallback
            display_text = chunk.get("content") or chunk.get("text") or chunk.get("metadata", {}).get("anchor_text") or "No explicit text content found."
            st.code(display_text[:2500], language="text")


def _markdown_table_html(lines: list[str]) -> str:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) < 2:
        return ""
    separator = rows[1]
    if not separator or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        return ""
    header = "".join(f"<th>{html.escape(cell)}</th>" for cell in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows[2:]
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _extract_markdown_tables(content: str) -> list[str]:
    tables: list[str] = []
    current: list[str] = []
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append(stripped)
            continue
        if current:
            table_html = _markdown_table_html(current)
            if table_html:
                tables.append(table_html)
            current = []
    if current:
        table_html = _markdown_table_html(current)
        if table_html:
            tables.append(table_html)
    return tables


def _image_data_url(image_path: str) -> str:
    validation = validate_asset_path(image_path, "image")
    if not validation.ok:
        return ""
    path = Path(validation.path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _csv_table_html(csv_path: str) -> str:
    validation = validate_asset_path(csv_path, "table")
    if not validation.ok:
        return ""
    rows = preview_csv(validation.path, max_rows=25)
    if not rows:
        return ""
    header = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows[1:]
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _render_multimodal_assets(chunks: list[dict[str, Any]], include_images: bool = True, target_cat: str | None = None) -> None:
    cards: list[str] = []
    seen: set[str] = set()
    rendered_table_entities: set[str] = set()
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        content = str(chunk.get("content") or "")
        entity_title = str(metadata.get("entity_id") or metadata.get("figure_id") or "Extracted table")
        is_strict_table = _is_table_chunk(chunk)
        is_strict_visual = _is_vision_chunk(chunk) and not is_strict_table
        for path, asset_type in candidate_asset_paths(chunk, "table"):
            if not is_strict_table:
                continue
            if asset_type != "table":
                continue
            suffix = Path(str(path)).suffix.lower()
            validation = validate_asset_path(path, "table" if suffix in {".csv", ".xlsx"} else "table_image")
            logger.info(
                "Multimodal asset validation: chunk_id=%s asset_type=%s path=%s ok=%s reason=%s",
                metadata.get("chunk_id") or chunk.get("id"),
                asset_type,
                path,
                validation.ok,
                validation.reason,
            )
            if not validation.ok:
                continue
            key = f"table_asset::{validation.path}"
            if key in seen:
                continue
            seen.add(key)
            title = html.escape(str(metadata.get("entity_id") or Path(validation.path).stem))
            if suffix == ".csv":
                if target_cat == "table" and (metadata.get("image_path") or chunk.get("image_path") or metadata.get("document_type") == "pdf_visual"):
                    continue
                table_html = _csv_table_html(validation.path)
                if table_html:
                    rendered_table_entities.add(entity_title.lower())
                    cards.append(f'<section class="rag-asset"><h4>{title}</h4>{table_html}</section>')
            elif include_images:
                if entity_title.lower() in rendered_table_entities:
                    continue
                image_url = _image_data_url(validation.path)
                if image_url:
                    rendered_table_entities.add(entity_title.lower())
                    cards.append(
                        f'<section class="rag-asset"><h4>{title}</h4>'
                        f'<img src="{image_url}" alt="{title}" loading="lazy"></section>'
                    )
        for table_html in _extract_markdown_tables(content):
            if not is_strict_table:
                continue
            if entity_title.lower() in rendered_table_entities:
                continue
            if target_cat == "table" and (metadata.get("image_path") or chunk.get("image_path") or metadata.get("document_type") == "pdf_visual"):
                continue
            key = f"table::{table_html}"
            if key not in seen:
                seen.add(key)
                rendered_table_entities.add(entity_title.lower())
                title = html.escape(str(metadata.get("entity_id") or "Extracted table"))
                cards.append(f'<section class="rag-asset"><h4>{title}</h4>{table_html}</section>')

        if include_images:
            if not is_strict_visual:
                continue
            for image_path, asset_type in candidate_asset_paths(chunk, "image"):
                if asset_type != "image":
                    continue
                validation = validate_asset_path(image_path, "image")
                logger.info(
                    "Multimodal asset validation: chunk_id=%s asset_type=%s path=%s ok=%s reason=%s",
                    metadata.get("chunk_id") or chunk.get("id"),
                    asset_type,
                    image_path,
                    validation.ok,
                    validation.reason,
                )
                if not validation.ok:
                    continue
                image_url = _image_data_url(validation.path)
                if not image_url:
                    continue
                key = f"image::{validation.path}"
                if key not in seen:
                    seen.add(key)
                    title = html.escape(str(metadata.get("entity_id") or metadata.get("figure_id") or "Extracted visual"))
                    cards.append(
                        f'<section class="rag-asset"><h4>{title}</h4>'
                        f'<img src="{image_url}" alt="{title}" loading="lazy"></section>'
                    )
    if not cards:
        return
    st.markdown(
        """
        <style>
        .rag-assets { display: flex; flex-direction: row; flex-wrap: wrap; gap: 16px;
          justify-content: space-between; align-items: stretch; margin: 12px 0; }
        .rag-asset { flex: 1 1 calc(50% - 16px); min-width: 300px; border: 1px solid #343a46;
          padding: 12px; border-radius: 6px; overflow-x: auto; }
        .rag-asset h4 { margin: 0 0 10px; font-size: 0.95rem; }
        .rag-asset img { width: 100%; height: auto; object-fit: contain; display: block; }
        .rag-asset table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
        .rag-asset th, .rag-asset td { border: 1px solid #48505e; padding: 6px 8px; text-align: left; }
        </style>
        <div class="rag-assets">""" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def format_nearby_context(text: str) -> str:
    if not text:
        return ""
        
    lines = [line.strip() for line in text.split('\n')]
    paragraphs = []
    current_para = []
    
    for line in lines:
        if not line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            continue
            
        if line.startswith(('-', '*', '•', '1.', '2.', '3.')):
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            paragraphs.append(line)
        else:
            current_para.append(line)
            
    if current_para:
        paragraphs.append(" ".join(current_para))
        
    cleaned_paras = []
    for p in paragraphs:
        p_clean = re.sub(r'\s+', ' ', p).strip()
        if len(p_clean) < 15 and (p_clean.isdigit() or p_clean.lower() in ("references", "wdr 2025", "world development report")):
            continue
        cleaned_paras.append(p_clean)
        
    if not cleaned_paras:
        return ""
        
    return "### Nearby Document Context\n\n" + "\n\n".join(cleaned_paras)


def _render_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = get_memory_manager().get_full_history(st.session_state.session_id)
    seen_image_paths = set()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = extract_llm_response_text(message.get("content"))
            st.markdown(content, unsafe_allow_html=True)
            if message["role"] == "assistant":
                # Re-render old visuals dynamically from history
                if "images" in message and message["images"]:
                    for img_path in message["images"]:
                        if img_path in seen_image_paths:
                            continue
                        seen_image_paths.add(img_path)
                        if os.path.exists(img_path):
                            label = ""
                            sources = message.get("sources", [])
                            for chunk in sources:
                                metadata = chunk.get("metadata", {})
                                if metadata.get("image_path") == img_path:
                                    for key in ("entity_id", "linked_entity_id", "visual_title", "caption_text"):
                                        value = str(metadata.get(key) or "").strip()
                                        if value:
                                            if len(value) > 100:
                                                label = f"Reference: {value[:97]}..."
                                            else:
                                                label = f"Reference: {value}"
                                            break
                                    if label:
                                        break
                            if not label:
                                label = "Reference Figure"
                            
                            st.subheader(label)
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.write(f"### DEBUG: Target Asset Asked: (History Mode) | Path sent to st.image: {img_path}")
                                display_image_robustly(img_path)
                
                if "nearby_context" in message and message["nearby_context"]:
                    formatted_context = format_nearby_context(message["nearby_context"])
                    if formatted_context:
                        st.markdown(formatted_context)

                msg_target_cat = None
                for prev_msg in st.session_state.messages:
                    if prev_msg == message:
                        break
                    if prev_msg["role"] == "user":
                        msg_target_cat, _ = parse_target_asset(prev_msg.get("content", ""))
                _render_multimodal_assets(message.get("sources", []), include_images=False, target_cat=msg_target_cat)
                _render_sources(message.get("sources", []))



def _render_raw_retrieval_debug(chunks: list) -> None:
    with st.expander("SYSTEM DEBUG: Raw Retrieved Chunks", expanded=True):
        st.write(f"Total Chunks Retrieved: {len(chunks)}")
        if not chunks:
            st.error("ALERT: Qdrant returned 0 chunks. The retrieval function is coming up completely empty.")
            return
        for index, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {index} Text Content:**")
            content = getattr(chunk, "page_content", None)
            if content is None and isinstance(chunk, dict):
                content = chunk.get("content") or chunk.get("text")
            st.code(str(content if content is not None else chunk)[:500], language="text")


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate or near-duplicate chunks based on content or point IDs."""
    seen_ids = set()
    seen_contents = []
    deduped = []
    for chunk in chunks:
        point_id = str(chunk.get("id") or "")
        content = str(chunk.get("content") or chunk.get("text") or "").strip()
        if not content:
            continue
        if point_id and point_id in seen_ids:
            continue
        
        # Normalize content whitespace and check similarity
        normalized_content = " ".join(content.lower().split())
        is_duplicate = False
        for seen in seen_contents:
            if normalized_content == seen or normalized_content in seen or seen in normalized_content:
                is_duplicate = True
                break
        if is_duplicate:
            continue
            
        if point_id:
            seen_ids.add(point_id)
        seen_contents.append(normalized_content)
        deduped.append(chunk)
    return deduped


def is_csv_chunk(content: str, metadata: dict[str, Any]) -> bool:
    """Detect if a chunk is a CSV table or structured tabular text."""
    # If the chunk is already in a key-value or historical list format,
    # we do not want to parse it as raw CSV since it is already highly readable.
    if "Country:" in content and "Indicator:" in content and "Historical Data:" in content:
        return False
        
    doc_type = str(metadata.get("document_type") or "").lower()
    if doc_type == "csv":
        return True
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if len(lines) >= 2:
        comma_counts = [line.count(',') for line in lines[:3]]
        if all(c >= 2 for c in comma_counts) and len(set(comma_counts)) == 1:
            return True
    return False


def parse_csv_to_markdown(csv_text: str) -> str:
    """Parse raw CSV text into a structured, clean Markdown table."""
    import csv
    import io
    csv_text = csv_text.strip()
    if not csv_text:
        return ""
    try:
        sample = csv_text[:1024]
        dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';'])
    except Exception:
        class DefaultDialect(csv.Dialect):
            delimiter = ','
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = '\n'
            quoting = csv.QUOTE_MINIMAL
        dialect = DefaultDialect
    
    f = io.StringIO(csv_text)
    reader = csv.reader(f, dialect)
    rows = []
    try:
        for r in reader:
            if r:
                rows.append(r)
    except Exception:
        rows = [line.split(',') for line in csv_text.split('\n') if line.strip()]
    
    if not rows:
        return csv_text
        
    md_lines = []
    headers = [str(col).strip().replace('|', '\\|') for col in rows[0]]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows[1:]:
        cols = [str(col).strip().replace('|', '\\|') for col in row]
        if len(cols) < len(headers):
            cols += [""] * (len(headers) - len(cols))
        elif len(cols) > len(headers):
            cols = cols[:len(headers)]
        md_lines.append("| " + " | ".join(cols) + " |")
    return "\n".join(md_lines)


class RAGModules:
    """Stateless RAG stages used by the Streamlit orchestration layer."""

    @staticmethod
    def classify_structural_intent(query: str, model: NvidiaLlamaModel) -> str:
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["figure", "fig ", "fig.", "chart", "diagram", "image", "visual", "picture", "illustration"]):
            return "ASSET_VISUAL"
        if any(kw in query_lower for kw in ["gdp", "emission", "co2", "revenue", "metric", "indicator", "table", "timeline", "statistics", "stats", "percent", "percentage", "income group"]):
            return "TABULAR_NUMERIC"
            
        try:
            prompt = """Analyze the user query and classify its structural intent into exactly one category:
- TABULAR_NUMERIC: Query is seeking table numbers, numeric data rows, statistics, or timelines.
- ASSET_VISUAL: Query specifically requests chart/figure images, visuals, drawings, or coordinate bindings.
- CONCEPTUAL_TEXTUAL: Query is asking for narrative descriptions, definitions, procedures, or text concepts.

Output ONLY the category name: TABULAR_NUMERIC, ASSET_VISUAL, or CONCEPTUAL_TEXTUAL. Do not write anything else."""
            intent = model.generate(prompt, query, temperature=0.0).strip().upper()
            if intent in {"TABULAR_NUMERIC", "ASSET_VISUAL", "CONCEPTUAL_TEXTUAL"}:
                return intent
        except Exception as exc:
            logger.warning("LLM structural intent router failed: %s", exc)
        return "CONCEPTUAL_TEXTUAL"

    @staticmethod
    def format_tabular_key_value_query(query: str, model: NvidiaLlamaModel) -> str:
        return query

    @staticmethod
    def module_route_intent(user_query: str, nvidia_llama_model: NvidiaLlamaModel) -> str:
        try:
            intent = nvidia_llama_model.generate(INTENT_ROUTER_PROMPT, user_query, temperature=0.0).upper()
            return intent if intent in {"DIRECT_RESPONSE", "DATA_RETRIEVAL"} else "DATA_RETRIEVAL"
        except Exception as exc:
            logger.warning("Intent router failed; defaulting to data retrieval: %s", exc)
            return "DATA_RETRIEVAL"

    @staticmethod
    def module_direct_response(user_query: str, chat_history: list, nvidia_llama_model: NvidiaLlamaModel) -> str:
        try:
            history_text = format_masked_history(chat_history)
            return nvidia_llama_model.generate(
                DIRECT_RESPONSE_PROMPT,
                f"Recent conversation:\n{history_text or '(none)'}\n\nLatest user message:\n{mask_pii_text(user_query)}",
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("Direct response generation failed: %s", exc)
            return "Hello. How can I help with your report analysis?"

    @staticmethod
    def module_condense_query(
        latest_query: str,
        chat_history: list,
        nvidia_llama_model: NvidiaLlamaModel,
        locked_entities: list[str] | None = None,
    ) -> list[str]:
        try:
            return contextualize_query(latest_query, chat_history, nvidia_llama_model, locked_entities)
        except Exception as exc:
            logger.warning("Query condenser failed; using raw user query: %s", exc)
            queries = enforce_locked_entities([mask_pii_text(latest_query)], locked_entities or [])
            return [mask_pii_text(query) for query in queries]

    @staticmethod
    def module_generate_hyde(condensed_query: str, groq_model: GroqModel) -> str:
        try:
            return generate_hypothetical_document(condensed_query, groq_model)
        except Exception as exc:
            logger.warning("HyDE module failed; using condensed query: %s", exc)
            return condensed_query

    @staticmethod
    def module_retrieve_hybrid(
        condensed_query: str | list[str],
        hyde_doc: str,
        top_k: int = HYBRID_RESULT_LIMIT,
        candidate_limit: int = RRF_LIMIT,
        chat_history: list | None = None,
        nvidia_llama_model: NvidiaLlamaModel | None = None,
        sparse_only: bool = False,
        locked_entities: list[str] | None = None,
        structural_intent: str = "CONCEPTUAL_TEXTUAL",
        qdrant_duration_accum: list[float] | None = None,
    ) -> list:
        try:
            retrieval_queries = [
                get_memory_manager().redact_condensed_payload(query, chat_history or [])
                for query in (condensed_query if isinstance(condensed_query, list) else [condensed_query])
            ]
            
            # 1. TABULAR_NUMERIC Query Re-formatting
            if structural_intent == "TABULAR_NUMERIC" and nvidia_llama_model is not None:
                retrieval_queries = [RAGModules.format_tabular_key_value_query(q, nvidia_llama_model) for q in retrieval_queries]
                hyde_doc = RAGModules.format_tabular_key_value_query(hyde_doc, nvidia_llama_model)
                logger.info("Tabular/Numeric query reformatted to key-value structure: %s", retrieval_queries)

            combined_query = get_memory_manager().redact_condensed_payload(
                "\n".join(retrieval_queries),
                chat_history or [],
                condensed_query,
                hyde_doc,
            )
            final_limit = min(max(int(top_k), 1), PRIMARY_DENSE_TOP_K)
            locked_entities = locked_entities or []
            is_asset_query = False if structural_intent == "TABULAR_NUMERIC" else bool(detect_requested_asset_type(combined_query) or locked_entities)
            internal_window = max(ASSET_QUERY_INTERNAL_LIMIT, final_limit) if is_asset_query else final_limit
            pre_truncation_limit = max(
                final_limit,
                min(max(int(candidate_limit), internal_window), max(RRF_LIMIT, internal_window)),
            ) if is_asset_query else final_limit
            logger.info("Running bucketed retrieval plan: %s", retrieval_queries)
            candidate_groups = [
                (
                    sub_query,
                    hybrid_search(
                        condensed_query=sub_query,
                        hypothetical_doc=sub_query if has_explicit_identifier_or_number(sub_query) else hyde_doc,
                        candidate_limit=pre_truncation_limit,
                        result_limit=pre_truncation_limit,
                        sparse_only=sparse_only and has_explicit_identifier_or_number(sub_query),
                        structural_intent=structural_intent,
                        qdrant_duration_accum=qdrant_duration_accum,
                    ),
                )
                for sub_query in retrieval_queries
            ]
            retrieved_pool = [candidate for _sub_query, candidates in candidate_groups for candidate in candidates]
            
            # Strict Pre-Rerank Metadata Type & ID Filter
            target_cat, target_id_val = parse_target_asset(combined_query)
            if target_cat and target_id_val:
                target_type = "figure" if target_cat.lower() in ("figure", "fig", "chart", "diagram", "graph") else target_cat.lower()
                target_id = target_id_val.lower()
                
                filtered_candidates = []
                for chunk in retrieved_pool:
                    is_visual_chunk = bool(
                        chunk.get("image_path") or
                        chunk.get("metadata", {}).get("image_path") or
                        chunk.get("metadata", {}).get("document_type") == "pdf_visual"
                    )
                    if is_visual_chunk:
                        vis_type, vis_id = extract_type_and_id_for_chunk(chunk)
                        if vis_type == target_type and vis_id == target_id:
                            filtered_candidates.append(chunk)
                    else:
                        filtered_candidates.append(chunk)
                        
                if filtered_candidates:
                    retrieved_pool = filtered_candidates
                    candidate_groups = [
                        (sub_query, [c for c in candidates if c in filtered_candidates])
                        for sub_query, candidates in candidate_groups
                    ]
            
            # Enforce strict path checks for ASSET_VISUAL queries
            if structural_intent == "ASSET_VISUAL":
                for chunk in retrieved_pool:
                    meta = chunk.get("metadata") or {}
                    for key in ("image_path", "figure_image_path", "chart_image_path", "table_image_path"):
                        if key in meta:
                            path_val = str(meta[key])
                            if path_val and not os.path.exists(path_val):
                                meta.pop(key, None)
                                logger.warning("Enforcing strict visual path check: removed missing path %s", path_val)

            balanced = rerank_balanced_context(
                combined_query,
                candidate_groups,
                per_bucket=final_limit,
                locked_entities=locked_entities,
            )
            target_cat, target_id_val = parse_target_asset(combined_query)
            if target_cat and target_id_val:
                cross_referenced = _strict_match_and_enforce(combined_query, balanced)
            else:
                expanded = expand_reranked_children_to_parents(balanced)
                expanded = _strict_match_and_enforce(combined_query, expanded)
                cross_referenced = expand_reranked_children_to_parents(
                    co_retrieve_cross_references(expanded, combined_query, limit=final_limit)
                )
                cross_referenced = _strict_match_and_enforce(combined_query, cross_referenced)
            
            if structural_intent == "TABULAR_NUMERIC":
                pass
            else:
                cross_referenced = promote_locked_entity_candidates(cross_referenced, locked_entities)
                
            bind_image_paths_to_chunks(cross_referenced, locked_entities, source_pool=retrieved_pool)
            
            # Deduplicate chunks to prevent context bloat
            cross_referenced = deduplicate_chunks(cross_referenced)
            
            if target_cat and target_id_val:
                target_type = target_cat.lower()
                target_id = target_id_val.lower()
                for vis in cross_referenced:
                    vis_type, vis_id = extract_type_and_id_for_chunk(vis)
                    vis["type"] = vis_type
                    vis["id"] = vis_id
                
                matched_visuals = [
                    vis for vis in cross_referenced 
                    if vis.get("type").lower() == target_type and vis.get("id") == target_id
                ]
                cross_referenced = matched_visuals
            
            # Apply in-memory payload splitter and target keyword filtering
            target_keyword = extract_target_keyword(combined_query)
            if target_keyword:
                for chunk in cross_referenced:
                    for key in ("content", "text", "page_content"):
                        if key in chunk and chunk[key]:
                            chunk[key] = split_and_filter_payload(chunk[key], target_keyword)

            return cross_referenced[:final_limit]
        except Exception:
            logger.exception("Hybrid retrieval module failed")
            raise

    @staticmethod
    def module_evaluate_context(
        condensed_query: str,
        retrieved_chunks: list,
        nvidia_llama_model: NvidiaLlamaModel,
        structural_intent: str = "CONCEPTUAL_TEXTUAL",
    ) -> bool:
        if not has_minimum_relevance_signal(condensed_query, retrieved_chunks, structural_intent=structural_intent):
            return False
        try:
            context = "\n\n".join(str(chunk.get("content") or "") for chunk in retrieved_chunks)
            if structural_intent == "TABULAR_NUMERIC":
                prompt_instruction = (
                    "Evaluate the context chunks above. Does the context contain the relevant structured CSV metrics, tabular data, "
                    "or historical year-by-year numbers needed to answer the user query? Respond with "
                    '{"is_relevant": "yes"} or {"is_relevant": "no"}.'
                )
            elif structural_intent == "ASSET_VISUAL":
                prompt_instruction = (
                    "Evaluate the context chunks above. Does the context contain visual figures, chart details, "
                    "image coordinates, or page visual extractions related to the user's visual asset query? Respond with "
                    '{"is_relevant": "yes"} or {"is_relevant": "no"}.'
                )
            else:
                prompt_instruction = (
                    "Evaluate the context chunks above. Does the context contain the factual metrics, tables, or data required to answer the user query? "
                    'Respond with {"is_relevant": "yes"} or {"is_relevant": "no"}.'
                )
            response = nvidia_llama_model.generate(
                CONTEXT_EVALUATOR_PROMPT,
                f"[CONTEXT CHUNKS FOR EVALUATION]\n{context}\n[END OF CONTEXT CHUNKS]\n\n"
                f"[USER QUERY]\n{condensed_query}\n[END OF USER QUERY]\n\n"
                f"{prompt_instruction}",
                temperature=0.0,
            )
            response_text = str(getattr(response, "text", response) or "")
            response_text = response_text.strip()
            is_relevant = parse_context_relevance_response(response_text)
            print(
                f"DEBUG [Step 2 Relevance]: Raw -> {response_text} | Parsed -> {is_relevant} | Intent -> {structural_intent}",
                file=sys.stderr,
                flush=True,
            )
            return is_relevant
        except Exception as exc:
            logger.warning("Context evaluator failed; blocking retrieved chunks: %s", exc)
            return False

    @staticmethod
    def module_grounded_generation(
        user_query: str,
        retrieved_chunks: list,
        condensed_query: str = "",
        hyde_doc: str = "",
        chat_history: list | None = None,
        nvidia_final_model: NvidiaLlamaModel | None = None,
        global_analytics: bool = False,
        generation_payload: dict[str, Any] | None = None,
        groq_model: GroqModel | None = None,
    ) -> str:
        try:
            if groq_model is not None:
                # 2. Aggregating all retrieved dense and sparse vector matches into a single, comprehensive context block
                context_parts = []
                for index, chunk in enumerate(retrieved_chunks, start=1):
                    content = chunk.get("content") or chunk.get("text") or ""
                    source = chunk.get("source") or "Unknown Source"
                    meta = chunk.get("metadata") or {}
                    
                    if is_csv_chunk(content, meta):
                        formatted_content = parse_csv_to_markdown(content)
                        title = f"### Tabular CSV Context {index} (Source: {source})"
                    else:
                        formatted_content = content
                        title = f"### Text Chunk Context {index} (Source: {source})"
                        
                    context_parts.append(f"{title}\n{formatted_content}")
                context_block = "\n\n".join(context_parts)
                
                # Format recent chat history for the conversation context
                history_str = ""
                if chat_history:
                    for msg in chat_history[-6:]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        content = msg.get("content", "")
                        history_str += f"{role}: {content}\n"
                        
                prompt = f"""You are an advanced, helpful, and accurate Conversational RAG assistant.
You are provided with a set of multi-source evidence chunks (retrieved from PDF text, CSV tables, and visual metadata collections) and the ongoing chat history.
Your goal is to generate a comprehensive, contextually accurate, and well-grounded response to the user's latest question.

GROUND RULES:
1. Base your answer STRICTLY on the facts provided in the multi-source evidence below. Do not assume or extrapolate.
2. If the evidence contains tables or CSV metrics, represent the numbers and data accurately in your response.
3. If the evidence contains visual captions or details, describe the visual elements accurately as they appear in the source.
4. Integrate information from text, CSV, and visual sources to answer mixed-data queries seamlessly.
5. If the evidence is insufficient to answer the question, state that clearly.
6. Write in a professional, engaging, and clear conversational tone.
7. NUMERICAL ACCURACY: Be extremely careful with numbers. If there are values in a figure or table, those values must be present in the answer; do not avoid or omit them, and ensure all values are accurate. If the text data looks slightly messy or conflicting, prioritize the exact numbers written in the main sentences/captions. Do not guess or change digits (like writing 17.1 instead of 17.3).

MULTI-SOURCE EVIDENCE:
{context_block}

CHAT HISTORY:
{history_str or "(No prior conversation history)"}

USER'S QUESTION:
{user_query}

Generate your comprehensive grounded response:"""

                print(
                    f"\n{'=' * 96}\n--- STEP 3: GROQ FINAL ANSWER GENERATION ---\nPROMPT:\n{prompt}\n{'=' * 96}",
                    file=sys.stderr,
                    flush=True,
                )
                system_prompt = (
                    "The context provided below contains high-density information including structured Markdown tables and isolated text passages. "
                    "Analyze the alignment of rows and columns carefully to extract exact metrics, numbers, and tabular data to formulate your final answer. "
                    "If the answer is in a table, reference it accurately."
                )
                answer = groq_model.generate(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    temperature=0.2
                ).strip()
                return answer

            if nvidia_final_model is None:
                raise RuntimeError("NVIDIA LLaMA 70B model is required for grounded generation.")
            return generate_final_answer(
                user_query,
                condensed_query or user_query,
                hyde_doc or condensed_query or user_query,
                retrieved_chunks,
                chat_history or [],
                nvidia_final_model,
                global_analytics=global_analytics,
                generation_payload=generation_payload,
            )
        except Exception as exc:
            logger.warning("Grounded generation module failed: %s", exc)
            return GENERATION_FAILURE_RESPONSE



def analyze_query_context(chat_history: list[dict[str, str]], latest_input: str, groq_model: GroqModel) -> dict[str, Any]:
    history_str = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("content", "")
        history_str += f"{role}: {content}\n"

    prompt = f"""You are an expert query analyzer and standalone query generator for a multi-turn Conversational RAG system.
Given the following chat history and the user's latest input, evaluate the context and classify the intent.

CLASSIFICATION RULES:
1. Determine if the user's latest input is a FOLLOW-UP query to the previous conversation (shares the same topic, refers to previous entities/years/charts) OR if it introduces a COMPLETELY DIFFERENT TOPIC.
2. If it is a FOLLOW-UP, set "intent" to "follow_up" and generate a single, optimized, history-aware "standalone_query" that resolves all pronouns (e.g. "it", "that", "those charts", "them") and implicit context from the history.
3. If it introduces a COMPLETELY DIFFERENT TOPIC, set "intent" to "new_topic" and generate a clean "standalone_query" based solely on the new input, ignoring the history entirely.

CHAT HISTORY:
{history_str or "(No prior conversation history)"}

USER'S LATEST INPUT:
{latest_input}

Respond ONLY with a valid JSON object. Do not include any markdown formatting (like ```json), explanations, or conversational text. The JSON must contain exactly these two keys:
{{
  "intent": "follow_up" or "new_topic",
  "standalone_query": "optimized standalone search query text"
}}"""

    try:
        response_text = gemini_invoke_text(
            [
                SystemMessage(content="You are a JSON query routing assistant."),
                HumanMessage(content=prompt),
            ],
            description="Gemini conversational context router",
        )
        data = _parse_gemini_json_payload(response_text)
        if "intent" in data and "standalone_query" in data:
            return data
    except Exception as exc:
        logger.warning("Gemini query context analysis failed; falling back to Groq. Error: %s", exc)
        try:
            response_text = groq_model.generate(
                system_prompt="You are a JSON query routing assistant.",
                user_prompt=prompt,
                temperature=0.0,
            ).strip()
            data = _parse_gemini_json_payload(response_text)
            if "intent" in data and "standalone_query" in data:
                return data
        except Exception as groq_exc:
            logger.warning("Groq query context analysis fallback failed: %s", groq_exc)

    return {
        "intent": "new_topic" if not chat_history else "follow_up",
        "standalone_query": latest_input,
    }


def gemini_orchestrate_routing(query: str) -> dict[str, Any]:
    """Gemini dual-engine orchestrator: classify route and split mixed queries."""
    target_cat, target_id = parse_target_asset(query)
    if target_cat and target_id:
        return {
            "route": "TEXT/VISUAL-ONLY",
            "csv_query": None,
            "text_visual_query": query
        }

    prompt = f"""You are the main orchestrator for a dual-engine RAG application.

Analyze the user query and return JSON with these rules:

ROUTES:
1. CSV-ONLY — exclusively tabular/spreadsheet/statistical questions about GDP, CO2, country indicators, regions, or income groups.
2. TEXT/VISUAL-ONLY — document text, chapter summaries, figures, charts, diagrams, or image/visual extraction tasks.
3. COMBINED — the query explicitly requires BOTH CSV/tabular analysis AND text/visual extraction in one submission.

SPLITTING:
- If route is COMBINED, populate both "csv_query" and "text_visual_query" as self-contained sub-tasks.
- If route is CSV-ONLY, set "csv_query" to the full query and "text_visual_query" to null.
- If route is TEXT/VISUAL-ONLY, set "text_visual_query" to the full query and "csv_query" to null.

Respond ONLY with valid JSON:
{{
  "route": "CSV-ONLY" | "TEXT/VISUAL-ONLY" | "COMBINED",
  "csv_query": "string or null",
  "text_visual_query": "string or null"
}}

Query:
{query}"""

    try:
        response_text = gemini_invoke_text(
            [
                SystemMessage(content="You are a dual-engine query orchestrator. Respond only with valid JSON."),
                HumanMessage(content=prompt),
            ],
            description="Gemini dual-engine orchestrator",
        )
        data = _parse_gemini_json_payload(response_text)
        route = str(data.get("route", "")).strip().upper().replace("_", "-")
        if "CSV" in route and "ONLY" in route:
            normalized_route = "CSV-ONLY"
        elif "COMBINED" in route:
            normalized_route = "COMBINED"
        else:
            normalized_route = "TEXT/VISUAL-ONLY"
        csv_query = str(data.get("csv_query") or "").strip() or None
        text_visual_query = str(data.get("text_visual_query") or "").strip() or None
        if normalized_route == "CSV-ONLY":
            csv_query = csv_query or query
            text_visual_query = None
        elif normalized_route == "TEXT/VISUAL-ONLY":
            text_visual_query = text_visual_query or query
            csv_query = None
        elif not csv_query or not text_visual_query:
            csv_query = csv_query or isolate_tabular_query_heuristic(query)
            text_visual_query = text_visual_query or isolate_text_visual_query_heuristic(query)
        return {
            "route": normalized_route,
            "csv_query": csv_query,
            "text_visual_query": text_visual_query,
        }
    except Exception as exc:
        logger.warning("Gemini orchestrator failed; using heuristic routing. Error: %s", exc)
        return _heuristic_orchestrator_routing(query)


def isolate_tabular_query_heuristic(query: str) -> str:
    lowered = query.lower()
    if any(marker in lowered for marker in ("gdp", "co2", "emission", "spreadsheet", "table", "csv", "indicator", "statistics")):
        return query
    return query.split(" and ")[-1].strip() if " and " in query else query


def isolate_text_visual_query_heuristic(query: str) -> str:
    lowered = query.lower()
    if any(marker in lowered for marker in ("chapter", "figure", "chart", "diagram", "visual", "summarize", "section")):
        return query
    return query.split(" and ")[0].strip() if " and " in query else query


def _heuristic_orchestrator_routing(query: str) -> dict[str, Any]:
    lowered = query.lower()
    csv_markers = ("gdp", "co2", "emission", "spreadsheet", "indicator", "statistics", "tabular", "csv")
    visual_markers = ("chapter", "figure", "chart", "diagram", "visual", "summarize", "section", "image")
    has_csv = any(marker in lowered for marker in csv_markers)
    has_visual = any(marker in lowered for marker in visual_markers)
    if has_csv and has_visual:
        return {
            "route": "COMBINED",
            "csv_query": isolate_tabular_query_heuristic(query),
            "text_visual_query": isolate_text_visual_query_heuristic(query),
        }
    if has_csv:
        return {"route": "CSV-ONLY", "csv_query": query, "text_visual_query": None}
    return {"route": "TEXT/VISUAL-ONLY", "csv_query": None, "text_visual_query": query}


@st.cache_resource
def _load_tabular_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gdp_df = pd.read_csv("C:/Users/supri/recovered-rag-project/Data/csv/GDP1.csv", skiprows=4, encoding="utf-8-sig")
    gdp_metadata_df = pd.read_csv("C:/Users/supri/recovered-rag-project/Data/csv/GDP2.csv", encoding="utf-8-sig")
    co2_df = pd.read_csv("C:/Users/supri/recovered-rag-project/Data/csv/CO21.csv", skiprows=4, encoding="utf-8-sig")
    co2_metadata_df = pd.read_csv("C:/Users/supri/recovered-rag-project/Data/csv/CO22.csv", encoding="utf-8-sig")
    return gdp_df, gdp_metadata_df, co2_df, co2_metadata_df


def _build_pandas_agent(llm: Any):
    dfs = list(_load_tabular_dataframes())
    return create_pandas_dataframe_agent(
        llm,
        dfs,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="zero-shot-react-description",
        include_df_in_prompt=False,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


@st.cache_resource
def get_pandas_agent_sambanova() -> Any:
    llm = ChatSambaNova(
        model="Meta-Llama-3.3-70B-Instruct",
        max_tokens=2048,
        temperature=0,
        sambanova_api_key=os.getenv("SAMBANOVA_API_KEY", "2cae239b-8569-4c93-aec0-2846a3085490"),
        max_retries=3,
    )
    return _build_pandas_agent(llm)





@st.cache_resource
def get_pandas_agent_groq(api_key: str) -> Any:
    llm = ChatGroq(
        model=GROQ_LLAMA_70B_MODEL_NAME,
        groq_api_key=api_key,
        temperature=0,
        max_tokens=2048,
    )
    return _build_pandas_agent(llm)


def invoke_pandas_agent_with_retry(agent: Any, enhanced_query: str) -> str:
    def _invoke() -> str:
        response = agent.invoke({"input": enhanced_query})
        return str(response.get("output", ""))

    return call_with_llama_retry(_invoke, description="Llama 3.3 70B pandas agent")


def run_pandas_agent_with_fallback(query: str, api_key: str) -> str:
    enhanced_query = f"{query}\n\n{_tabular_df_description()}"

    try:
        agent = get_pandas_agent_sambanova()
        return invoke_pandas_agent_with_retry(agent, enhanced_query)
    except Exception as primary_exc:
        logger.warning("SambaNova Llama 3.3 70B pandas agent failed after retries: %s", primary_exc)
        groq_key = resolve_groq_api_key(api_key)
        if not groq_key:
            return f"Error executing tabular query: {primary_exc}"
        try:
            logger.info("Falling back to Groq %s for tabular pandas execution.", GROQ_LLAMA_70B_MODEL_NAME)
            groq_agent = get_pandas_agent_groq(groq_key)
            return invoke_pandas_agent_with_retry(groq_agent, enhanced_query)
        except Exception as fallback_exc:
            logger.error("Groq Llama 3.3 70B pandas fallback failed: %s", fallback_exc)
            return f"Error executing tabular query: {fallback_exc}"


def _tabular_df_description() -> str:
    return (
        "Available DataFrames:\n"
        "- df1 (GDP Data): GDP (current US$) for 266 countries from 1960 to 2025.\n"
        "  Columns: 'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', and year columns ('1960', '1961', ..., '2025').\n"
        "- df2 (GDP Metadata): Metadata for GDP countries.\n"
        "  Columns: 'Country Code', 'Region', 'IncomeGroup', 'SpecialNotes', 'TableName'.\n"
        "- df3 (CO2 Data): Carbon dioxide (CO2) emissions per capita (t CO2e/capita) for 266 countries from 1960 to 2025.\n"
        "  Columns: 'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', and year columns ('1960', '1961', ..., '2025').\n"
        "- df4 (CO2 Metadata): Metadata for CO2 countries.\n"
        "  Columns: 'Country Code', 'Region', 'IncomeGroup', 'SpecialNotes', 'TableName'.\n\n"
        "Instructions:\n"
        "1. Do NOT redefine df1, df2, df3, or df4 in your code! Use these pre-loaded variables directly.\n"
        "2. The country name column is 'Country Name' (not 'Country'), and the country code column is 'Country Code'.\n"
        "3. Use standard pandas syntax to filter and extract the required values. For example, to get the 2020 GDP of United States: "
        "df1.loc[df1['Country Name'] == 'United States', '2020'].values[0]\n"
        "4. Make sure to perform the correct calculations and comparison in python, and output the final result clearly."
    )


def _figure_reference_label(retrieval_results: list[dict[str, Any]]) -> str:
    for chunk in retrieval_results:
        metadata = dict(chunk.get("metadata") or {})
        for key in ("entity_id", "linked_entity_id", "visual_title", "caption_text"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return f"Reference: {value}"
    return "Reference Figure"


def _image_path_from_retrieval_metadata(retrieval_results: list[dict[str, Any]], query: str = "") -> str:
    """Resolve the best on-disk image path from Qdrant chunk payloads."""
    target_cat, target_id = parse_target_asset(query)
    
    filtered_chunks = []
    if target_cat and target_id:
        norm_cat = "table" if "table" in target_cat.lower() else "figure"
        for chunk in retrieval_results:
            meta = dict(chunk.get("metadata") or {})
            meta_asset_type = str(meta.get("asset_type") or "").lower()
            meta_asset_id = str(meta.get("asset_id") or "").lower()
            if meta_asset_type == norm_cat and meta_asset_id == target_id.lower():
                filtered_chunks.append(chunk)
    else:
        filtered_chunks = retrieval_results

    for chunk in filtered_chunks:
        resolved = _chunk_image_path(chunk)
        if resolved:
            return resolved
    for chunk in filtered_chunks:
        metadata = dict(chunk.get("metadata") or {})
        for key in (
            "image_path",
            "figure_image_path",
            "chart_image_path",
            "table_image_path",
            "diagram_image_path",
            "image_local_path",
            "visual_path",
        ):
            raw_path = str(metadata.get(key) or chunk.get(key) or "").strip()
            if not raw_path:
                continue
            resolved = _resolve_existing_image_path(raw_path)
            if resolved:
                return resolved
            return raw_path
    return ""


def _all_image_paths_from_retrieval_metadata(retrieval_results: list[dict[str, Any]], query: str = "") -> list[tuple[str, str, str]]:
    """Resolve all unique on-disk image paths, labels, and nearby contexts from Qdrant chunk payloads."""
    target_cat, target_id = parse_target_asset(query)
    results = []
    seen = set()
    for chunk in retrieval_results:
        metadata = dict(chunk.get("metadata") or {})
        
        if target_cat and target_id:
            norm_cat = "table" if "table" in target_cat.lower() else "figure"
            meta_asset_type = str(metadata.get("asset_type") or "").lower()
            meta_asset_id = str(metadata.get("asset_id") or "").lower()
            if meta_asset_type != norm_cat or meta_asset_id != target_id.lower():
                continue

        resolved = _chunk_image_path(chunk)
        if not resolved:
            metadata = dict(chunk.get("metadata") or {})
            for key in (
                "image_path",
                "figure_image_path",
                "chart_image_path",
                "table_image_path",
                "diagram_image_path",
                "image_local_path",
                "visual_path",
            ):
                raw_path = str(metadata.get(key) or chunk.get(key) or "").strip()
                if not raw_path:
                    continue
                resolved = _resolve_existing_image_path(raw_path) or raw_path
                if resolved:
                    break
        if resolved:
            display_path = _resolve_existing_image_path(resolved) or resolved
            if os.path.exists(display_path) and display_path not in seen:
                seen.add(display_path)
                metadata = dict(chunk.get("metadata") or {})
                label = ""
                for key in ("entity_id", "linked_entity_id", "visual_title", "caption_text"):
                    value = str(metadata.get(key) or "").strip()
                    if value:
                        if len(value) > 100:
                            label = f"Reference: {value[:97]}..."
                        else:
                            label = f"Reference: {value}"
                        break
                if not label:
                    label = "Reference Figure"
                nearby_context = metadata.get("nearby_context", "")
                results.append((display_path, label, nearby_context))
    return results


def execute_gemini_extraction(
    query: str,
    retrieval_results: list[dict[str, Any]],
    timings: dict[str, float] | None = None,
) -> Generator[str, None, None]:
    """Fast grounded synthesis over retrieved chunks using OpenRouter replacing Gemini."""

    target_cat, target_id = parse_target_asset(query)

    if target_cat and target_id:
        logger.info("Executing visual verification and generation queue for %s %s...", target_cat, target_id)
        
        for idx, chunk in enumerate(retrieval_results):
            meta = chunk.get("metadata") or {}
            
            anchor_text = meta.get("anchor_text") or ""
            nearby_paragraph = meta.get("context_before", "")
            if meta.get("context_after"):
                nearby_paragraph += "\n" + meta.get("context_after")
            if not nearby_paragraph.strip():
                nearby_paragraph = chunk.get("text", "")

            # Extract visual data transcription from the retrieved chunk content
            raw_content = chunk.get("content") or chunk.get("text") or ""
            visual_data = ""
            match_vis = re.search(r"\[VISUAL DATA\]:\s*(.*?)(?=\[CONTEXT AFTER\]|$)", raw_content, re.DOTALL)
            if match_vis:
                visual_data = match_vis.group(1).strip()
            else:
                visual_data = raw_content

            # Identify if this point has already matched via metadata
            meta_asset_type = str(meta.get("asset_type") or "").lower()
            meta_asset_id = str(meta.get("asset_id") or "").lower()
            
            is_verified = False
            if target_cat and target_id:
                norm_cat = "table" if "table" in target_cat.lower() else "figure"
                if meta_asset_type == norm_cat and meta_asset_id == target_id.lower():
                    is_verified = True

            # DATA AUDITOR PROMPT GUARDRAIL
            if is_verified:
                system_instruction = (
                    "You are a precise document analyst and data auditor.\n"
                    "This content has been verified to represent the requested asset. Answer the user's question directly and clearly.\n"
                    "Do not summarize, crop, condense or omit any details. You must extract and output every piece of narrative data, "
                    "every category value, every country metric, and all numerical information contained within the retrieved visual evidence completely and exhaustively.\n"
                    "Do not reply with DATA_MISMATCH.\n\n"
                    "VISUAL DATA FORMATTING MANDATE: Do not output dense text walls or raw, endless bulleted lists for chart details. You must present visual data points using clean, professional Markdown structures:\n"
                    "- Use clear ### Subheadings to separate sections (e.g., Metrics by Category, Color-Coded Groupings).\n"
                    "- Use Markdown Tables to organize categorical values or country rankings (e.g., Column 1: Country/Category, Column 2: Value/Status) so the data is readable at a glance.\n"
                    "- Bold key metrics, axis thresholds, and critical insights to ensure excellent visual hierarchy.\n\n"
                    "- TABLE-SPECIFIC MANDATE: If the user's question is about a TABLE, the table image is already displayed below, so do not recreate or print the table in text/Markdown format. Instead, provide a descriptive summary of the table. Explain what the table shows, the key findings, important trends, comparisons, and the main insights in natural language. Keep the image, but replace the generated table with a textual explanation."
                )
            else:
                system_instruction = (
                    "You are a precise document analyst and data auditor.\n"
                    "Verify if the retrieved anchor text, visual data, and text content represent the user's requested asset type and ID. \n"
                    "If they do not match, or contain messy layout noise, reply exactly with: DATA_MISMATCH.\n"
                    "Otherwise, answer the user's question directly and clearly, using ONLY the facts from the visual data and text content.\n"
                    "Do not summarize, crop, condense or omit any details. You must extract and output every piece of narrative data, "
                    "every category value, every country metric, and all numerical information contained within the retrieved visual evidence completely and exhaustively.\n\n"
                    "VISUAL DATA FORMATTING MANDATE: Do not output dense text walls or raw, endless bulleted lists for chart details. You must present visual data points using clean, professional Markdown structures:\n"
                    "- Use clear ### Subheadings to separate sections (e.g., Metrics by Category, Color-Coded Groupings).\n"
                    "- Use Markdown Tables to organize categorical values or country rankings (e.g., Column 1: Country/Category, Column 2: Value/Status) so the data is readable at a glance.\n"
                    "- Bold key metrics, axis thresholds, and critical insights to ensure excellent visual hierarchy.\n\n"
                    "- TABLE-SPECIFIC MANDATE: If the user's question is about a TABLE, the table image is already displayed below, so do not recreate or print the table in text/Markdown format. Instead, provide a descriptive summary of the table. Explain what the table shows, the key findings, important trends, comparisons, and the main insights in natural language. Keep the image, but replace the generated table with a textual explanation."
                )
            
            prompt = (
                f"REQUESTED ASSET TYPE: {target_cat}\n"
                f"REQUESTED ASSET ID: {target_id}\n\n"
                f"RETRIEVED ANCHOR TEXT:\n{anchor_text}\n\n"
                f"VISUAL DATA (Extracted Text/Content from Image):\n{visual_data}\n\n"
                f"NEARBY TEXT CONTEXT:\n{nearby_paragraph}\n\n"
                f"USER QUESTION:\n{query}\n\n"
                "Extracted answer:"
            )
            
            # Call OpenRouter
            logger.info("Auditing chunk index %d/%d...", idx+1, len(retrieval_results))
            try:
                start_time = time.perf_counter()
                response_text = openrouter_invoke(
                    system_instruction=system_instruction,
                    prompt=prompt,
                    temperature=0,
                )
                if timings is not None:
                    timings["gemini_extraction"] = time.perf_counter() - start_time
                
                # Check for DATA_MISMATCH
                if "DATA_MISMATCH" in response_text:
                    logger.warning("DATA_MISMATCH detected for chunk %d! Skipping and dropping down to the next candidate in the queue.", idx+1)
                    continue
                    
                # Graceful match found, yield and finish
                yield response_text
                return
            except Exception as exc:
                logger.error("Failed to generate content for chunk %d: %s", idx+1, exc)
                continue
                
        # If all candidates fail or mismatch
        yield "Data mismatch: requested asset could not be verified or matched from the retrieved candidates."
        return

    # Fallback to standard textual generation if not targeting a visual element
    context = format_pruned_chunks_for_context(retrieval_results, target_cat, target_id)
    context = clean_context_metadata(context)

    system_instruction = (
        "You are a precise document analyst. Using ONLY the retrieved evidence below, answer the user's question "
        "with the most important extracted facts. Be concise, direct, and well-structured. "
        "Do not invent information. If the evidence is insufficient, say so clearly."
    )
    if target_cat and target_id:
        system_instruction += (
            f"\n\nCRITICAL ENGINE RULE: The user is explicitly querying about a specific asset target: {target_cat} {target_id} (e.g., Table 6.1). "
            f"The provided vector context chunks may contain adjacent or overlapping references to other structural elements with the same identifier, "
            f"such as Figure {target_id} or Chart {target_id}. You are strictly forbidden from summarizing, processing, extracting, or mentioning "
            f"any data originating from a competing asset category. If the query asks for a Table, do not return descriptive details about a Figure, "
            f"and vice versa. Focus entirely on the text, properties, and values belonging explicitly to the requested target asset category."
        )

    prompt = (
        f"RETRIEVED EVIDENCE:\n{context or '(no retrieved evidence)'}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        "Extracted answer:"
    )
    try:
        start = time.perf_counter()
        first_chunk_received = False
        for chunk_text in openrouter_invoke_stream(
            system_instruction=system_instruction,
            prompt=prompt,
            temperature=0,
        ):
            if not first_chunk_received:
                if timings is not None:
                    timings["gemini_extraction"] = time.perf_counter() - start
                first_chunk_received = True
            yield chunk_text
        return
    except Exception as exc:
        logger.error("OpenRouter extraction failed: %s", exc)
        yield "Extraction error: Unable to synthesize response."


def render_retrieved_figure(retrieval_results: list[dict[str, Any]], query: str = "") -> str | None:
    """Render all figures and visual chunks sequentially after the text answer."""
    target_cat, target_id = parse_target_asset(query)
    seen_images = set()
    last_display_path = None

    for chunk in retrieval_results:
        metadata = chunk.get("metadata", {})
        
        if target_cat and target_id:
            norm_cat = "table" if "table" in target_cat.lower() else "figure"
            meta_asset_type = str(metadata.get("asset_type") or "").lower()
            meta_asset_id = str(metadata.get("asset_id") or "").lower()
            if meta_asset_type != norm_cat or meta_asset_id != target_id.lower():
                continue

        # 2. Extract visual information safely
        img_path = _chunk_image_path(chunk)
        
        # Resolve path
        if img_path:
            display_path = _resolve_existing_image_path(img_path) or img_path
            
            # 3. ONLY display the image and its background paragraph if we haven't seen this file yet
            if os.path.exists(display_path) and display_path not in seen_images:
                # Add label or title
                label = ""
                for key in ("entity_id", "linked_entity_id", "visual_title", "caption_text"):
                    value = str(metadata.get(key) or "").strip()
                    if value:
                        if len(value) > 100:
                            label = f"Reference: {value[:97]}..."
                        else:
                            label = f"Reference: {value}"
                        break
                if not label:
                    label = "Reference Figure"
                
                st.subheader(label)
                
                # Wrap image in columns so it renders beautifully and slightly smaller
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.write(f"### DEBUG: Target Asset Asked: {query} | Path sent to st.image: {display_path}")
                    display_image_robustly(display_path)
                
                # Display the nearby paragraphs cleanly underneath
                nearby_text = metadata.get("nearby_context")
                if not nearby_text:
                    before = metadata.get("context_before", "")
                    after = metadata.get("context_after", "")
                    raw_text = chunk.get("text", "")
                    if before or after:
                        nearby_text = f"{before}\n\n{after}".strip()
                    else:
                        nearby_text = raw_text

                if nearby_text:
                    formatted_context = format_nearby_context(nearby_text)
                    if formatted_context:
                        st.markdown(formatted_context)
                    
                seen_images.add(display_path)
                last_display_path = display_path
                
    if last_display_path:
        st.session_state.current_image = last_display_path
        return last_display_path
    return None


def _gemini_messages_to_prompt(stream_messages: list[BaseMessage]) -> tuple[str | None, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in stream_messages:
        content = prune_chunk_text(extract_llm_response_text(getattr(message, "content", "")))
        if isinstance(message, SystemMessage):
            system_parts.append(content)
        else:
            user_parts.append(content)
    system_instruction = "\n\n".join(part for part in system_parts if part).strip() or None
    user_contents = "\n\n".join(part for part in user_parts if part).strip()
    return system_instruction, user_contents


def stream_gemini_messages(
    stream_messages: list[BaseMessage],
    timings: dict[str, float] | None = None,
) -> Generator[str, None, None]:
    system_instruction, user_contents = _gemini_messages_to_prompt(stream_messages)
    try:
        start = time.perf_counter()
        first_token_logged = False
        for chunk_text in openrouter_invoke_stream(
            system_instruction=system_instruction,
            prompt=user_contents,
            temperature=0,
        ):
            if not first_token_logged:
                llm_connection_duration = time.perf_counter() - start
                print(
                    f"⏱️ LLM STREAM FIRST TOKEN RESPONDED IN: {llm_connection_duration:.2f} seconds",
                    flush=True,
                )
                if timings is not None:
                    timings["llm_first_token"] = llm_connection_duration
                first_token_logged = True
            yield chunk_text
        return
    except Exception as exc:
        logger.error("OpenRouter streaming failed: %s", exc)
        yield "Error: OpenRouter stream encountered an issue."


def _execute_text_visual_pipeline(
    user_query: str,
    retrieval_query: str,
    history: list[dict[str, str]],
    groq_model: GroqModel,
    nvidia_llama_model: NvidiaLlamaModel,
    timings: dict[str, float],
) -> tuple[str | None, list[dict[str, Any]], str | None, list[BaseMessage] | None]:
    start = time.time()
    intent = RAGModules.module_route_intent(retrieval_query, nvidia_llama_model)
    timings["intent_router"] = time.time() - start
    if intent == "DIRECT_RESPONSE":
        history_text = format_masked_history(history)
        stream_messages = [
            SystemMessage(content=DIRECT_RESPONSE_PROMPT),
            HumanMessage(
                content=(
                    f"Recent conversation:\n{history_text or '(none)'}\n\n"
                    f"Latest user message:\n{mask_pii_text(retrieval_query)}"
                )
            ),
        ]
        timings["direct_response_generation"] = 0.0
        return None, [], None, stream_messages

    start = time.time()
    locked_entities = step_zero_extract_entities(retrieval_query)
    condensed_queries = [retrieval_query]
    condensed_query = retrieval_query
    timings["query_condenser"] = time.time() - start
    global_analytics = is_global_analytics_query(condensed_query)
    retrieval_limit = GLOBAL_ANALYTICS_LIMIT if global_analytics else RRF_LIMIT

    sparse_only = has_explicit_identifier_or_number(retrieval_query)
    start = time.time()
    hypothetical_doc = (
        retrieval_query
        if sparse_only
        else RAGModules.module_generate_hyde(retrieval_query, groq_model)
    )
    timings["hyde_generation"] = time.time() - start

    start = time.time()
    structural_intent = RAGModules.classify_structural_intent(retrieval_query, nvidia_llama_model)
    logger.info("Classified structural intent: %s", structural_intent)

    qdrant_durations: list[float] = []
    target_cat, target_id = parse_target_asset(retrieval_query)
    top_chunks = None
    if target_cat and target_id:
        target_cat_lower = target_cat.lower()
        if "table" in target_cat_lower:
            asset_type = "table"
        elif "figure" in target_cat_lower or "chart" in target_cat_lower or "diagram" in target_cat_lower or "graph" in target_cat_lower:
            asset_type = "figure"
        else:
            asset_type = None

        if asset_type:
            logger.info("Executing Direct Payload Retrieval for %s %s...", asset_type, target_id)
            start_qdrant = time.time()
            try:
                qdrant_filter = models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.asset_type", match=models.MatchValue(value=asset_type)),
                        models.FieldCondition(key="metadata.asset_id", match=models.MatchValue(value=target_id))
                    ]
                )
                temp_client = get_qdrant_client()
                scroll_res, _ = temp_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                    limit=10,
                    with_payload=True,
                    with_vectors=False
                )
                top_chunks = []
                for point in scroll_res:
                    chunk = point.payload or {}
                    chunk["id"] = point.id
                    top_chunks.append(chunk)
                qdrant_duration = time.time() - start_qdrant
                qdrant_durations.append(qdrant_duration)
                logger.info("Direct Payload Retrieval completed. Found %d chunks in %.4f seconds.", len(top_chunks), qdrant_duration)
            except Exception as e:
                logger.error("Direct Payload Retrieval failed: %s. Falling back to hybrid.", e)
                top_chunks = None

    if top_chunks is None:
        top_chunks = RAGModules.module_retrieve_hybrid(
            condensed_queries,
            hypothetical_doc,
            top_k=HYBRID_RESULT_LIMIT,
            candidate_limit=retrieval_limit,
            chat_history=history,
            nvidia_llama_model=nvidia_llama_model,
            sparse_only=sparse_only,
            locked_entities=locked_entities,
            structural_intent=structural_intent,
            qdrant_duration_accum=qdrant_durations,
        )
    timings["hybrid_retrieval_and_rerank"] = time.time() - start
    qdrant_duration = sum(qdrant_durations)
    timings["qdrant_retrieval"] = qdrant_duration
    print(f"⏱️ QDRANT RETRIEVED IN: {qdrant_duration:.2f} seconds", flush=True)

    bypass_layer_1 = os.getenv("BYPASS_GATEWAY", "true").lower() != "false" or os.getenv("DISABLE_GATEWAY", "true").lower() != "false"
    if bypass_layer_1:
        is_relevant = True
    else:
        start = time.time()
        is_relevant = RAGModules.module_evaluate_context(
            condensed_query, top_chunks, nvidia_llama_model, structural_intent=structural_intent
        )
        timings["nvidia_context_gate"] = time.time() - start
    if not is_relevant:
        start = time.time()
        fallback_chunks = step_three_exact_entity_fallback(locked_entities)
        if fallback_chunks:
            top_chunks = fallback_chunks
        else:
            logger.warning("Layer 1 retrieval validation failed; blocking generation and suppressing sources/assets.")
            timings["step3_exact_payload_fallback"] = time.time() - start
            entity_name = requested_entity_name(condensed_query or user_query, locked_entities)
            return retrieval_failure_message(entity_name), [], None, None
        timings["step3_exact_payload_fallback"] = time.time() - start
        if not has_minimum_relevance_signal(condensed_query, top_chunks, structural_intent=structural_intent):
            logger.warning("Exact fallback chunks failed relevance signal; blocking generation.")
            entity_name = requested_entity_name(condensed_query or user_query, locked_entities)
            return retrieval_failure_message(entity_name), [], None, None

    image_path = bind_image_paths_to_chunks(top_chunks, locked_entities)
    asset_resolution = resolve_best_asset(retrieval_query, top_chunks)
    if asset_resolution.ok:
        image_path = asset_resolution.path if asset_resolution.renderer == "image" else image_path

    stream_messages = build_grounded_generation_messages(
        user_query,
        top_chunks,
        chat_history=history,
        metrics_out=timings,
    )
    timings["gemini_flash_generation"] = 0.0
    return None, top_chunks, image_path or None, stream_messages


from langfuse.decorators import observe, langfuse_context

@observe()
def run_pipeline(
    user_query: str,
    groq_api_key: str,
    nvidia_api_key: str,
) -> tuple[str | None, list[dict[str, Any]], dict[str, float], str | None, list[BaseMessage] | None]:
    timings: dict[str, float] = {}
    start_time = time.time()
    
    # Update Langfuse trace metadata with session, user, and deployment tags context
    try:
        active_session_id = st.session_state.get("session_id", "anonymous-session")
        active_user_id = st.session_state.get("user_id", "default-user")
        tags_list = os.environ.get("DEPLOYMENT_TAGS", "production,v2-rag").split(",")
        langfuse_context.update_current_trace(
            session_id=active_session_id,
            user_id=active_user_id,
            tags=tags_list
        )
    except Exception as lf_exc:
        logger.warning("Failed to update Langfuse trace metadata context: %s", lf_exc)
    
    # 1. Initialize dependencies for the Pydantic AI agent
    try:
        gdp_df, gdp_metadata_df, co2_df, co2_metadata_df = _load_tabular_dataframes()
    except Exception as exc:
        logger.warning("Failed to load tabular dataframes: %s", exc)
        gdp_df = None
        gdp_metadata_df = None
        co2_df = None
        co2_metadata_df = None

    client = get_qdrant_client()
    
    vision_runner = None
    try:
        from google import genai
        vision_runner = genai.Client()
    except Exception as e:
        logger.warning("google-genai Client failed to initialize: %s", e)

    deps = SystemPipelinesDeps(
        image_folder_path="C:/Users/supri/recovered-rag-project/extracted_images",
        pandas_df=gdp_df,
        qdrant_client=client,
        vision_runner=vision_runner,
        user_query=user_query,
        gdp_df=gdp_df,
        gdp_metadata_df=gdp_metadata_df,
        co2_df=co2_df,
        co2_metadata_df=co2_metadata_df
    )

    # 2. Run the Pydantic AI agent
    try:
        global ACTIVE_USER_QUERY, VISION_ELEMENT_PROCESSED, VISION_TOOL_SUCCEEDED, VALIDATION_ATTEMPT_COUNT, LAST_VISION_RAW_CONTENT
        ACTIVE_USER_QUERY = user_query
        VISION_ELEMENT_PROCESSED = False
        VISION_TOOL_SUCCEEDED = False
        LAST_VISION_RAW_CONTENT = ""
        VALIDATION_ATTEMPT_COUNT = 0
        
        from pydantic_ai.usage import UsageLimits
        from opentelemetry import trace
        
        tracer = trace.get_tracer("pydantic_ai")
        with tracer.start_as_current_span("Self_Correction_Orchestrator") as orchestrator_span:
            with tracer.start_as_current_span("Query_Orchestrator_Router") as router_span:
                # Evaluate routing intent pathways
                has_csv = looks_like_structured_query(user_query) and should_use_structured_csv_query(user_query)
                pathway = "pandas_dataframe" if has_csv else "hybrid_retriever_agent"
                router_span.set_attribute("routing.selected_pathway", pathway)
                router_span.set_attribute("routing.confidence_score", 1.0)
                router_span.set_attribute("routing.user_query", user_query)
                
                result = multimodal_agent.run_sync(
                    user_query,
                    deps=deps,
                    message_history=[],
                    usage_limits=UsageLimits(request_limit=100)
                )
                
                # Post-Execution Interceptor for Visual Extraction
                if result and hasattr(result, "output") and result.output:
                    if VISION_ELEMENT_PROCESSED and VISION_TOOL_SUCCEEDED:
                        extracted = result.output.extracted_table
                        
                        def is_table_invalid_post(table: list[Any]) -> bool:
                            if not table:
                                return True
                            for row in table:
                                if isinstance(row, ChartTableRow):
                                    s = str(row.Series).strip()
                                    c = str(row.Category).strip()
                                    val_raw = row.TargetValue
                                elif isinstance(row, dict):
                                    s = str(row.get("Series", "")).strip()
                                    c = str(row.get("Category", "")).strip()
                                    val_raw = row.get("TargetValue", "")
                                else:
                                    return True
                                val_str = str(val_raw).strip()
                                is_dummy = (
                                    (s == "" or s.lower() == "n/a") and
                                    (c == "" or c.lower() == "n/a") and
                                    (val_str == "" or val_str == "0" or val_str == "0.0" or val_str.lower() == "n/a" or val_raw is None)
                                )
                                if not is_dummy:
                                    return False
                            return True

                        if is_table_invalid_post(extracted):
                            logger.info("⚠️ [Post-Agent Interceptor] extracted_table is empty or dummy. Intercepting raw vision content to parse programmatically...")
                            
                            raw_markdown = ""
                            if result.output.text_reasoning and "|" in result.output.text_reasoning:
                                raw_markdown = result.output.text_reasoning
                            elif deps.last_vision_raw_content and "|" in deps.last_vision_raw_content:
                                raw_markdown = deps.last_vision_raw_content
                            
                            if raw_markdown:
                                parsed_rows = parse_markdown_table_to_dicts(raw_markdown)
                                if parsed_rows:
                                    logger.info(f"✅ [Post-Agent Interceptor] Successfully parsed {len(parsed_rows)} rows. Forcefully populating extracted_table.")
                                    result.output.extracted_table = [ChartTableRow(**r) for r in parsed_rows]
            
            # Log final successfully parsed Pydantic object
            if result and hasattr(result, "data") and result.data:
                orchestrator_span.set_attribute("validation.final_parsed_object", result.data.model_dump_json())
        
        # Log tool calls and token usage metrics
        for msg in result.new_messages():
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "tool_name") and hasattr(part, "args"):
                        logger.info(f"⚙️ [PydanticAI Tool Call] Invoked: {part.tool_name} with Args: {part.args}")
        usage = result.usage
        logger.info(f"📊 [PydanticAI Token Usage] Input: {usage.input_tokens or 0} | Output: {usage.output_tokens or 0} | Total: {usage.total_tokens or 0}")
        sources = []
        image_path = None
        try:
            target_cat, target_id = parse_target_asset(user_query)
            if target_cat and target_id:
                asset_type = "table" if "table" in target_cat.lower() else "figure"
                qdrant_filter = models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.asset_type", match=models.MatchValue(value=asset_type)),
                        models.FieldCondition(key="metadata.asset_id", match=models.MatchValue(value=target_id))
                    ]
                )
                scroll_res, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                    limit=10,
                    with_payload=True,
                    with_vectors=False
                )
                for point in scroll_res:
                    chunk = point.payload or {}
                    chunk["id"] = point.id
                    sources.append(chunk)
                
                # Resolve using build_asset_registry
                from app.multimodal_assets import build_asset_registry, normalize_entity_id
                norm_id = normalize_entity_id(f"{target_cat}_{target_id}")
                registry = build_asset_registry()
                for record in registry:
                    if record.entity_id == norm_id:
                        image_path = record.absolute_path
                        break
        except Exception as pre_exc:
            logger.warning("Failed to pre-resolve visual sources for rendering: %s", pre_exc)

        # 2. Format payload and execute the 14-layer compliance gauntlet
        answer_text = result.output.text_reasoning
        try:
            gauntlet = RAGMasterSafetyGauntlet()
            payload_dict = {
                "text_response": result.output.text_reasoning,
                "source_routing_trail": result.output.source_routing_trail or "",
                "extracted_table": [r.model_dump() for r in result.output.extracted_table] if result.output.extracted_table else [],
                "chart_title": result.output.chart_title or "",
                "x_axis_label": result.output.x_axis_label or "",
                "y_axis_label": result.output.y_axis_label or "",
                "chart_type": result.output.chart_type or "",
                "units": result.output.units or ""
            }
            chunks_to_vet = list(sources) if sources else getattr(deps, "retrieved_chunks", [])
            vetted_res = gauntlet.run_full_validation_gauntlet(
                user_query=user_query,
                raw_qdrant_chunks=chunks_to_vet,
                model_output_payload=payload_dict,
                session_id=active_session_id,
                agent_steps=VALIDATION_ATTEMPT_COUNT
            )
            
            # If Layer 14 triggered fallback due to guardrail breaches
            if vetted_res.get("metadata", {}).get("safe_fallback"):
                logger.warning("⚠️ [Layer 14 Fallback Router] Intercepted safety/invariant failure: %s. Returning fallback payload.", vetted_res["metadata"].get("failure_type"))
                answer_text = vetted_res["text_response"]
                result.output.extracted_table = []
                result.output.source_routing_trail = ""
            else:
                answer_text = vetted_res.get("text_response", result.output.text_reasoning)
                if "extracted_table" in vetted_res:
                    result.output.extracted_table = [ChartTableRow(**r) for r in vetted_res["extracted_table"]]
        except Exception as gauntlet_exc:
            logger.exception("Error running RAGMasterSafetyGauntlet: %s", gauntlet_exc)

        # 3. Append source trail if present and not blocked
        if result.output.source_routing_trail and not (result.output.extracted_table == [] and answer_text == RAGMasterSafetyGauntlet.SAFE_FALLBACK_TEXT):
            answer_text += f"\n\n**Source Trail:** {result.output.source_routing_trail}"
            
        # 4. Append extracted table if present and not blocked
        if result.output.extracted_table:
            answer_text += "\n\n### Extracted Table Data\n"
            df_temp = pd.DataFrame([row.model_dump() for row in result.output.extracted_table])
            df_temp.rename(columns={"TargetValue": "Target Value"}, inplace=True, errors="ignore")
            answer_text += df_temp.to_markdown(index=False)

        timings["agent_execution_seconds"] = time.time() - start_time
        return answer_text, sources, timings, image_path, None
        
    except Exception as exc:
        logger.exception("Agent execution inside Streamlit run_pipeline failed")
        return f"Agent execution failed: {exc}", [], timings, None, None



def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_session_state()
    _consume_voice_query_params()
    groq_api_key, nvidia_api_key = _render_sidebar()

    st.title(APP_TITLE)
    st.markdown(
        """
        <style>
        /* Force chat messages and tables to use the same standard font family as the UI */
        .stChatMessage, .stChatMessage p, .stChatMessage li, .stChatMessage div, .stChatMessage td, .stChatMessage th {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.caption("Hybrid dense+sparse retrieval over PDFs, CSV rows, tables, charts, and enriched text.")
    _render_history()



    if st.session_state.clear_query_after_run:
        st.session_state.query_input = ""
        st.session_state.clear_query_after_run = False

    if st.session_state.pending_voice_query_text:
        st.session_state.query_input = st.session_state.pending_voice_query_text
        st.session_state.pending_voice_query_text = ""

    input_col, mic_col, send_col = st.columns([12, 1, 1.4], vertical_alignment="bottom")
    with input_col:
        st.text_input(
            "Ask a question about your data",
            key="query_input",
            placeholder="Ask a question about your data...",
            label_visibility="collapsed",
        )
    with mic_col:
        _render_voice_recorder()
    with send_col:
        st.button("Send", width="stretch", on_click=_submit_current_query)

    if st.session_state.last_voice_transcript:
        st.caption(f"Voice transcript: {st.session_state.last_voice_transcript}")

    user_query = str(st.session_state.submitted_query or "").strip()
    st.session_state.submitted_query = ""
    if not user_query:
        return
    st.session_state.current_image_path = None
    st.session_state.current_image = None
    st.session_state.current_images = []

    with st.chat_message("user"):
        st.markdown(user_query)

    answer: str | None = None
    sources: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    image_path: str | None = None
    stream_messages: list[BaseMessage] | None = None

    with st.chat_message("assistant"):
        with st.spinner("Searching database, validating context, and generating answer..."):
            try:
                answer, sources, timings, image_path, stream_messages = run_pipeline(
                    user_query, groq_api_key, nvidia_api_key
                )
            except Exception as exc:
                answer = f"Unable to complete the request: {exc}"
                sources = []
                timings = {}
                image_path = None
                stream_messages = None

        if stream_messages is not None and sources:
            answer = st.write_stream(execute_gemini_extraction(user_query, sources, timings=timings))
            answer = sanitize_user_answer(extract_llm_response_text(answer))
            render_retrieved_figure(sources, user_query)
        elif stream_messages is not None:
            answer = st.write_stream(stream_gemini_messages(stream_messages, timings=timings))
            answer = sanitize_user_answer(extract_llm_response_text(answer))
        elif answer is not None:
            answer = sanitize_user_answer(extract_llm_response_text(answer))
            st.markdown(answer, unsafe_allow_html=True)
            if sources:
                render_retrieved_figure(sources, user_query)

        # Collect all image paths already stored in previous assistant messages
        global_seen_images = set()
        for msg in st.session_state.get("messages", []):
            if "images" in msg and msg["images"]:
                for img_p in msg["images"]:
                    global_seen_images.add(img_p)

        unique_images_raw = _all_image_paths_from_retrieval_metadata(sources, user_query) if sources else []
        unique_images = []
        seen_img = set()
        for item in unique_images_raw:
            path_val = item[0]
            if path_val not in seen_img and path_val not in global_seen_images:
                seen_img.add(path_val)
                unique_images.append(item)

        st.session_state.current_images = unique_images
        resolved_figure_path = _image_path_from_retrieval_metadata(sources, user_query) if sources else ""
        st.session_state.current_image_path = _resolve_existing_image_path(resolved_figure_path or image_path) or None
        if st.session_state.current_image_path:
            st.session_state.current_image = st.session_state.current_image_path
            st.markdown("### Extracted Visual Asset")
            display_image_robustly(st.session_state.current_image_path)
        active_target_cat, _ = parse_target_asset(user_query)
        _render_multimodal_assets(sources, include_images=False, target_cat=active_target_cat)
        if timings:
            st.caption(" | ".join(f"{name}: {value:.2f}s" for name, value in timings.items()))
        _render_sources(sources)

    # Append the user's new question and the assistant answer to st.session_state.messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": user_query})
    images_list = [item[0] for item in unique_images] if 'unique_images' in locals() else []
    nearby_contexts = [item[2] for item in unique_images if item[2]] if 'unique_images' in locals() else []
    retrieved_nearby_context_text = "\n\n".join(nearby_contexts) if nearby_contexts else ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer or "",
        "sources": sources,
        "images": images_list,
        "nearby_context": retrieved_nearby_context_text
    })

    memory_manager = get_memory_manager()
    memory_manager.update_history(st.session_state.session_id, user_query, answer or "")
    memory_manager.attach_sources(st.session_state.session_id, sources)
    st.session_state.clear_query_after_run = True
    st.session_state.last_voice_audio_hash = ""
    st.rerun()



if __name__ == "__main__":
    main()
