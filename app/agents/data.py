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

DataOutput.model_rebuild()

# Setup Data Agent
from app.agents.model_factory import get_agent_model
data_model = get_agent_model("DATA_AGENT_MODEL_NAME")
data_agent = Agent(
    data_model,
    output_type=DataOutput,
    retries=3,
    system_prompt=(
        "You are a Senior Data Analyst Agent specializing in analytical, statistical, and comparative Pandas queries.\n\n"
        "PRELOADED DATASETS IN EXECUTION ENVIRONMENT:\n"
        "- `gdp_clean_df`: PRE-CLEANED World Bank GDP dataset with metadata ALREADY merged and all non-country aggregates filtered out! Contains only valid sovereign countries with columns: 'Country Name', 'Country Code', 'Region', 'IncomeGroup', and yearly columns '1960' through '2023'. USE THIS DIRECTLY for country rankings, top/bottom lists, and country metrics!\n"
        "- `co2_clean_df`: PRE-CLEANED CO2 emissions dataset with metadata merged and aggregates filtered out.\n"
        "- `gdp_df`: World Bank GDP dataset. Columns include 'Country Name', 'Country Code', 'Indicator Name' ('GDP (current US$)'), 'Indicator Code' ('NY.GDP.MKTP.CD'), and yearly columns '1960' through '2023'.\n"
        "- `gdp_meta_df`: Metadata dataset with columns 'Country Code', 'Region', 'IncomeGroup'.\n"
        "- `co2_df`: World Bank CO2 emissions dataset.\n"
        "- `co2_meta_df`: CO2 metadata with 'Country Code', 'Region', 'IncomeGroup'.\n"
        "- Extracted report tables from `assets/extracted_tables/` (e.g. `page_111_Table_2_1`).\n"
        "NOTE: Yearly columns are strings (e.g. '2020', '2021', '2022'). Use `gdp_df['2022']` rather than `df['Year'] == 2022`.\n\n"
        "CRITICAL NUMERIC RULES:\n"
        "- When calculating mean, median, sum, or statistics across years or across rows, ALWAYS pass numeric_only=True (e.g. df.mean(numeric_only=True)) or explicitly select year columns (e.g. year_cols = [c for c in df.columns if c.isdigit()]), because metadata columns like 'Region', 'IncomeGroup', 'Country Name', 'Country Code' are strings and will raise a TypeError if included in string math!\n"
        "- For queries filtering by IncomeGroup (e.g. 'low-income' or 'high-income'):\n"
        "  * Use gdp_clean_df[gdp_clean_df['IncomeGroup'].str.lower() == 'low income'] or high income.\n"
        "  * To compute average GDP: avg_val = low_df.mean(numeric_only=True).mean() or specify the year.\n\n"
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
    logger.info(f"Executing Pandas code on local tables:\n{python_code}")
    
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
    standard_files = [
        ("gdp_df", "Data/csv/GDP1.csv", 4),
        ("gdp1_df", "Data/csv/GDP1.csv", 4),
        ("gdp_meta_df", "Data/csv/GDP2.csv", 0),
        ("gdp2_df", "Data/csv/GDP2.csv", 0),
        ("co2_df", "Data/csv/CO21.csv", 4),
        ("co21_df", "Data/csv/CO21.csv", 4),
        ("co2_meta_df", "Data/csv/CO22.csv", 0),
        ("co22_df", "Data/csv/CO22.csv", 0),
    ]
    for name, rel_path, skip in standard_files:
        local_p = PROJECT_ROOT / rel_path
        try:
            if local_p.exists() and local_p.is_file():
                if skip > 0:
                    locs[name] = pd.read_csv(local_p, skiprows=skip)
                else:
                    locs[name] = pd.read_csv(local_p)
            else:
                supa_url = get_supabase_asset_url(rel_path)
                logger.info(f"🌐 Fetching CSV dataset '{name}' from Supabase CDN: {supa_url}")
                if skip > 0:
                    locs[name] = pd.read_csv(supa_url, skiprows=skip)
                else:
                    locs[name] = pd.read_csv(supa_url)
        except Exception as csv_err:
            logger.warning(f"Could not load CSV dataset '{name}': {csv_err}")
        
    # Common DataFrame aliases and pre-cleaned country datasets
    if "gdp_df" in locs and "gdp_meta_df" in locs:
        m_gdp = locs["gdp_df"].merge(locs["gdp_meta_df"], on="Country Code", how="left")
        clean_gdp = m_gdp[m_gdp["Region"].notna()].copy()
        clean_gdp = clean_gdp.drop(columns=[c for c in ["SpecialNotes", "TableName"] if c in clean_gdp.columns])
        year_cols = [c for c in clean_gdp.columns if c.isdigit()]
        clean_gdp[year_cols] = clean_gdp[year_cols].apply(pd.to_numeric, errors='coerce')
        locs["gdp_clean_df"] = clean_gdp
        locs["gdp_countries_df"] = clean_gdp
    if "co2_df" in locs and "co2_meta_df" in locs:
        m_co2 = locs["co2_df"].merge(locs["co2_meta_df"], on="Country Code", how="left")
        clean_co2 = m_co2[m_co2["Region"].notna()].copy()
        clean_co2 = clean_co2.drop(columns=[c for c in ["SpecialNotes", "TableName"] if c in clean_co2.columns])
        year_cols = [c for c in clean_co2.columns if c.isdigit()]
        clean_co2[year_cols] = clean_co2[year_cols].apply(pd.to_numeric, errors='coerce')
        locs["co2_clean_df"] = clean_co2
        locs["co2_countries_df"] = clean_co2

    if "gdp_df" in locs:
        locs["GDP"] = locs["gdp_df"]
        locs["gdp"] = locs["gdp_df"]
        locs["df"] = locs.get("gdp_clean_df", locs["gdp_df"])
    if "co2_df" in locs:
        locs["CO2"] = locs["co2_df"]
        locs["co2"] = locs["co2_df"]

    # Provide smart pd wrapper so if code calls pd.read_csv('GDP.csv') or similar, it succeeds
    import types
    smart_pd = types.ModuleType("pandas_wrapper")
    for attr in dir(pd):
        setattr(smart_pd, attr, getattr(pd, attr))

    def smart_read_csv(filepath_or_buffer, *args, **kwargs):
        p_str = str(filepath_or_buffer).replace("\\", "/").lower()
        if "gdp" in p_str and "2" not in p_str:
            target = PROJECT_ROOT / "Data" / "csv" / "GDP1.csv"
            if target.exists():
                kwargs.setdefault("skiprows", 4)
                return pd.read_csv(target, *args, **kwargs)
        elif "gdp" in p_str and "2" in p_str:
            target = PROJECT_ROOT / "Data" / "csv" / "GDP2.csv"
            if target.exists():
                return pd.read_csv(target, *args, **kwargs)
        elif "co2" in p_str and "2" not in p_str:
            target = PROJECT_ROOT / "Data" / "csv" / "CO21.csv"
            if target.exists():
                kwargs.setdefault("skiprows", 4)
                return pd.read_csv(target, *args, **kwargs)
        elif "co2" in p_str and "2" in p_str:
            target = PROJECT_ROOT / "Data" / "csv" / "CO22.csv"
            if target.exists():
                return pd.read_csv(target, *args, **kwargs)
        try:
            return pd.read_csv(filepath_or_buffer, *args, **kwargs)
        except Exception:
            # Fallback with skiprows=4
            return pd.read_csv(filepath_or_buffer, *args, skiprows=4, **kwargs)

    smart_pd.read_csv = smart_read_csv
    locs["pd"] = smart_pd
    locs["pandas"] = smart_pd

    stdout = io.StringIO()
    old_stdout = sys.stdout
    
    try:
        sys.stdout = stdout
        exec(python_code, locs, locs)
        sys.stdout = old_stdout
        val = stdout.getvalue()
        logger.info(f"Pandas execution produced stdout ({len(val)} chars):\n{val[:300]}")
        return val if val.strip() else "Code executed successfully with no stdout. Ensure you use print(...) to output values."
    except Exception as exc:
        sys.stdout = old_stdout
        logger.warning(f"Error executing Python code: {exc}")
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
