from __future__ import annotations

import io
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
import pandas as pd

# Load env variables
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataOutput(BaseModel):
    executive_metrics: str = Field(
        description="Summary table or bullet points of primary metrics and KPIs."
    )
    analytical_breakdown: str = Field(
        description="In-depth interpretation of patterns, outliers, and trend lines."
    )
    code_reference: str = Field(
        description="Clear execution logic (Pandas/SQL) used to achieve the results."
    )

# Setup Data Agent
model_name = os.getenv("DATA_AGENT_MODEL_NAME", "groq:llama-3.3-70b-versatile")
data_agent = Agent(
    model_name,
    output_type=DataOutput,
    retries=10,
    system_prompt=(
        "You are a Senior Data Analyst Agent specializing in analytical, statistical, and comparative Pandas queries.\n\n"
        "CORE OBJECTIVES:\n"
        "1. Data Wrangling & Cleaning: Process raw tabular, CSV, or JSON data into clean, structured formats.\n"
        "2. Exploratory & Quantitative Analysis: Calculate descriptive statistics, correlations, and trends.\n"
        "3. Data Visualization & Reporting: Translate statistical output into plain-language summaries.\n\n"
        "DYNAMIC FILE RETRIEVAL & MERGING RULES:\n"
        "1. MULTI-ENTITY / MULTI-FILE LOOKUP:\n"
        "   - If the request asks for multiple metrics stored in separate files (e.g. GDP and CO2), write code to load BOTH datasets, filter each independently by the requested entities/years, and MERGE them on common key columns (e.g. ['Country Name', 'Year'] or similar standard keys).\n"
        "   - If the request asks for multiple entities from the same file, filter using vector conditions (e.g. df['Country Name'].isin(['India', 'China'])).\n"
        "2. COMBINED OUTPUT REQUIREMENT:\n"
        "   - ALWAYS combine final results into a single consolidated Pandas DataFrame or structured JSON response before printing/summarizing.\n"
        "   - Do NOT output separate disconnected tables for each entity or metric.\n"
        "3. DATA HYGIENE:\n"
        "   - Standardize key column names (strip spaces, match casing for country names).\n"
        "   - Handle missing values gracefully (use .fillna('N/A') or state missing entries explicitly).\n\n"
        "EXECUTION & LOGIC RULES:\n"
        "1. REGIONAL AGGREGATE FILTERING (Mandatory):\n"
        "   - Exclude non-country aggregate rows from the country column before running any math or sorting.\n"
        "   - The most robust way to filter out aggregates is by merging the data DataFrame (e.g. GDP1.csv or CO21.csv) with its metadata DataFrame (e.g. GDP2.csv or CO22.csv) on 'Country Code' and retaining only rows where 'Region' is NOT null/empty (e.g., `df = df[df['Region'].notna()]`). Regions/Income groups are aggregates and have null Region fields.\n"
        "   - Alternatively, filter out values matching: ['World', 'European Union', 'Euro area', 'High income', 'Low income', 'Middle income', 'East Asia & Pacific', 'Latin America & Caribbean', 'OECD members', 'Sub-Saharan Africa', 'North America', 'Upper middle income', 'Lower middle income', 'Arab World', 'Euro area', 'IDA & IBRD countries', 'developed countries', 'developing countries', 'transition economies', 'least developed countries'].\n"
        "2. NUMERIC CONVERSION & CALCULATIONS:\n"
        "   - Coerce target metric columns to numeric types: `df[col] = pd.to_numeric(df[col], errors='coerce')`.\n"
        "   - Perform `.groupby('Country Name', as_index=False)[col].mean()` or aggregations.\n"
        "   - Flatten MultiIndex headers resulting from aggregations (e.g. `df.columns = ['_'.join(c).strip() for c in df.columns]`) and preserve categorical columns.\n"
        "3. NUMERIC SORTING (Must happen BEFORE string formatting):\n"
        "   - Call `.sort_values(by=target_metric, ascending=False)` on the RAW NUMERIC COLUMN.\n"
        "   - Slice top records using `.head(N)` immediately after sorting.\n"
        "4. STRING FORMATTING (Must happen LAST):\n"
        "   - Only AFTER sorting and slicing, convert numbers into formatted display strings (e.g. '$2.84 Trillion' or '1.84 t CO2e').\n"
        "   - NEVER sort after formatting numbers with '$', 'Trillion', or other string suffixes.\n"
        "5. MULTI-FILE / MULTI-METRIC COMPARISON:\n"
        "   - If comparing metrics across different files (e.g., GDP vs CO2 emissions), load both files, execute necessary filters/aggregations on each, and perform an inner/outer `.merge()` on shared key columns.\n\n"
        "OPERATIONAL & FORMATTING RULES:\n"
        "1. NO MARKDOWN TABLES: Do not use Markdown table syntax (|---|).\n"
        "2. KEY-VALUE CARDS: Use bold text category headers and bullet points for all metrics.\n"
        "3. METRIC HUMANIZATION: Convert massive numbers into readable formats (e.g. '$2.84 Trillion') AND you MUST ALWAYS append the exact raw average/numerical value in parentheses next to it (e.g., '$8.22 Trillion ($8,219,305,102,400.00)'). Round floats to 2 decimal places.\n"
        "4. COMBINED ENTITY LAYOUT: Group metrics cleanly by Entity or Year (e.g. **📊 India (2019 Snapshot)**).\n"
        "5. ANALYTICAL SUMMARY: Always end with a concise 1-2 sentence key takeaway explaining the result (e.g. identifying the top performer or core correlation).\n"
        "- Metric Integrity: Specify the methods used (mean, median, SD, etc.).\n"
        "- Code-Driven Accuracy: Derive metrics directly from execution outputs. Do NOT hallucinate numbers.\n"
        "- Data Quality Transparency: Explicitly flag edge cases, outliers, or missing data points."
    )
)

@data_agent.tool
def execute_pandas_query(ctx: RunContext[Any], python_code: str) -> str:
    """
    Executes Python/Pandas code on the loaded extracted CSV tables on disk.
    
    CRITICAL: All CSV files in 'assets/extracted_tables/' are PRELOADED into the local execution scope as DataFrame variables.
    The dots (.) and dashes (-) in the filenames are replaced by underscores (_).
    For example:
    - 'page_206_Table_4.1.csv' is preloaded as the variable `page_206_Table_4_1`
    - 'page_208_Table_4.2.csv' is preloaded as the variable `page_208_Table_4_2`
    You must use these preloaded variables directly (e.g. `print(page_206_Table_4_1.head())`) and do NOT call `pd.read_csv` for them.
    
    Make sure to PRINT the final output/statistics using print() so it is captured in stdout.
    """
    logger.info("Executing Pandas code on local tables...")
    
    # Expose preloaded dataframes if they exist on disk, else load from Supabase CDN
    from app.multimodal_assets import get_supabase_asset_url
    locs = {}
    tables_dir = PROJECT_ROOT / "assets" / "extracted_tables"
    if tables_dir.exists():
        for csv_file in tables_dir.glob("*.csv"):
            var_name = csv_file.stem.replace(".", "_").replace("-", "_")
            try:
                locs[var_name] = pd.read_csv(csv_file)
            except Exception:
                pass
                
    # Also expose standard names (from local disk or Supabase Cloud CDN)
    for name, rel_path in [("gdp_df", "Data/gdp_data.csv"), ("co2_df", "Data/co2_data.csv"), ("all_tables_df", "Data/csv/World_Development_Report_2025_All_Tables.csv")]:
        local_p = PROJECT_ROOT / rel_path
        try:
            if local_p.exists() and local_p.is_file():
                locs[name] = pd.read_csv(local_p)
            else:
                supa_url = get_supabase_asset_url(rel_path)
                logger.info(f"🌐 Fetching CSV dataset '{name}' from Supabase CDN: {supa_url}")
                locs[name] = pd.read_csv(supa_url)
        except Exception as csv_err:
            logger.warning(f"Could not load CSV dataset '{name}': {csv_err}")
        
    stdout = io.StringIO()
    old_stdout = sys.stdout
    
    try:
        sys.stdout = stdout
        exec(python_code, {}, locs)
        sys.stdout = old_stdout
        return stdout.getvalue()
    except Exception as exc:
        sys.stdout = old_stdout
        return f"Error executing Python code: {exc}"

def execute_data_task(query: str) -> Dict[str, Any]:
    """
    Entrypoint function to execute data analytics query.
    """
    run_result = data_agent.run_sync(query)
    output: DataOutput = run_result.output
    return {
        "executive_metrics": output.executive_metrics,
        "analytical_breakdown": output.analytical_breakdown,
        "code_reference": output.code_reference
    }
