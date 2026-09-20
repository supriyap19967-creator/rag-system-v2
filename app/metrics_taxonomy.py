"""
Canonical Metric Taxonomy Registry & Cross-Modality Compatibility Engine
Pillar 2 & 3 Implementation for Multimodal RAG Precision
"""

from typing import Any, Dict, List, Optional, Tuple

# Canonical Metric Definitions & Scope Rules
METRIC_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "NY.GDP.MKTP.CD": {
        "canonical_name": "Total GDP (Current US$)",
        "unit": "Current US$",
        "is_per_capita": False,
        "indicator_code": "NY.GDP.MKTP.CD",
        "description": "Gross Domestic Product total market value in current US dollars",
        "distinct_from": ["GDP per capita", "GDP growth rate", "Purchasing Power Parity (PPP)"]
    },
    "NY.GDP.PCAP.CD": {
        "canonical_name": "GDP per capita (Current US$)",
        "unit": "Current US$ per person",
        "is_per_capita": True,
        "indicator_code": "NY.GDP.PCAP.CD",
        "description": "Gross Domestic Product divided by midyear population",
        "distinct_from": ["Total GDP", "Gross National Income (GNI)"]
    },
    "EN.ATM.CO2E.PC": {
        "canonical_name": "CO2 emissions per capita (metric tons)",
        "unit": "metric tons per capita",
        "is_per_capita": True,
        "scope": "Excluding LULUCF (Production-based)",
        "indicator_code": "EN.ATM.CO2E.PC",
        "description": "Per-capita production-based carbon dioxide emissions excluding land-use change",
        "distinct_from": ["Consumption-based CO2 emissions per capita", "Total greenhouse gas emissions"]
    },
    "IncomeGroup": {
        "canonical_name": "World Bank Income Group Classification",
        "is_classification_only": True,
        "categories": ["Low income", "Lower middle income", "Upper middle income", "High income"],
        "forbidden_claims": [
            "compliance capacity",
            "compliance rate",
            "enforcement rate",
            "regulatory compliance score",
            "standard adoption capability"
        ],
        "description": "Analytical grouping of economies based on GNI per capita; does not measure compliance or regulatory enforcement directly."
    }
}

# Dummy & Placeholder Value Blacklist (Pillar 4)
DUMMY_VALUE_BLACKLIST = {
    "1234", "5678", "50000", "20000", "10000", "12345", "9999", "99999",
    "1234.0", "5678.0", "50000.0", "20000.0", "10000.0",
    "100", "120", "50", "60", "1000", "5000", "10.5", "20.3", "30.7", "10.0", "20.0", "30.0",
    "100.0", "120.0", "50.0", "60.0", "1000.0", "5000.0"
}

PLACEHOLDER_ENTITIES = {
    "country a", "country b", "country c", "country d",
    "income group 1", "income group 2", "income group 3", "income group 4",
    "series 1", "series 2", "category a", "category b", "data point", "entity a", "entity b"
}


def validate_metric_compatibility(
    csv_metric: str,
    pdf_visual_metric: str
) -> Tuple[bool, str]:
    """
    Checks if a CSV metric and a PDF/Visual metric are semantically compatible for direct merging/comparison.
    Returns (is_compatible, explanation_message).
    """
    csv_m = csv_metric.lower().strip()
    pdf_m = pdf_visual_metric.lower().strip()

    # 1. Total GDP vs GDP per Capita Check
    if ("total gdp" in csv_m or "ny.gdp.mktp.cd" in csv_m) and ("per capita" in pdf_m or "gdp per capita" in pdf_m):
        return (
            False,
            "CSV dataset contains Total GDP (Current US$), whereas PDF/Figure measures GDP per capita. "
            "These represent distinct economic scales and cannot be combined or compared directly without population normalization."
        )

    # 2. Production CO2 vs Consumption CO2 Check
    if ("excluding lulucf" in csv_m or "en.atm.co2e.pc" in csv_m) and ("consumption-based" in pdf_m or "trade-adjusted" in pdf_m):
        return (
            False,
            "CSV dataset contains Production-based CO2 per capita (excluding LULUCF), whereas PDF/Figure measures Consumption-based CO2. "
            "Consumption-based figures account for embodied carbon in international trade and differ from production totals."
        )

    return True, "Metrics are compatible."


def check_income_group_claims(text: str) -> Tuple[bool, Optional[str]]:
    """
    Scans text response for unsupported claims asserting IncomeGroup directly measures compliance capacity/rates.
    Returns (has_violation, warning_message).
    """
    lower_text = text.lower()
    if "income group" in lower_text or "income-group" in lower_text or "high income" in lower_text or "low income" in lower_text:
        forbidden = [
            "measures compliance capacity",
            "proves compliance rate",
            "equals compliance capacity",
            "direct measure of compliance",
            "compliance rate of income group",
            "enforcement rate of income group",
            "causes higher compliance",
            "causes lower compliance"
        ]
        for phrase in forbidden:
            if phrase in lower_text:
                return (
                    True,
                    f"Unsupported claim detected: World Bank Income Group is a GNI classification, not a direct measure of compliance capacity or enforcement rate. Reframe claim."
                )
    return False, None


def check_causation_vs_correlation_claims(text: str) -> Tuple[bool, Optional[str]]:
    """
    Scans text response for overstated causal claims when PDF text describes associations or correlations.
    Returns (has_violation, warning_message).
    """
    lower_text = text.lower()
    causal_overstatements = [
        "causes economic growth",
        "directly causes higher gdp",
        "causes co2 reduction",
        "proves that income group causes"
    ]
    for phrase in causal_overstatements:
        if phrase in lower_text:
            return (
                True,
                "Overstated causal claim detected: PDF document findings describe correlations/associations rather than direct controlled causation. Reframe as association."
            )
    return False, None


def contains_synthetic_or_dummy_data(obj: Any) -> Tuple[bool, Optional[str]]:
    """
    Checks if a text string or list of table rows contains synthetic placeholder tables or dummy fallbacks.
    Returns (has_dummy, explanation).
    """
    import re
    if isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict):
                series = str(row.get("Series", "")).strip().lower()
                category = str(row.get("Category", "")).strip().lower()
                val = str(row.get("TargetValue", "")).strip().lower()
                
                # Check for explicit placeholder series/categories in structured table rows
                if series in PLACEHOLDER_ENTITIES or category in PLACEHOLDER_ENTITIES:
                    return True, f"Extracted table contains synthetic placeholder entity ('{series or category}')."
                
                # Check for explicit dummy template numbers in table values
                if val in {"1234", "5678", "50000", "20000", "12345", "9999", "99999"}:
                    return True, f"Extracted table contains dummy fallback value '{val}'."
    elif isinstance(obj, str):
        text_lower = obj.lower()
        if "unable to calculate requested result from available csv data" in text_lower:
            return False, None
            
        # Check for explicit markdown table rows with placeholder entities (e.g. | Country A | 100 |)
        placeholder_table_pattern = re.compile(
            r"\|\s*(?:country\s+[a-d]|income\s+group\s+[1-4]|series\s+[1-2]|category\s+[a-b])\s*\|",
            re.IGNORECASE
        )
        if placeholder_table_pattern.search(obj):
            return True, "Response contains synthetic placeholder table syntax."
            
        # Check for explicit key-value dummy patterns like "Country A: 100" or "Country B: 120"
        kv_pattern = re.compile(
            r"\b(?:country\s+[a-d]|income\s+group\s+[1-4])\s*:\s*\d+",
            re.IGNORECASE
        )
        if kv_pattern.search(obj):
            return True, "Response contains synthetic placeholder key-value fallback syntax."

    return False, None


def format_calculation_provenance(
    dataset_name: str,
    columns_used: List[str],
    filtering_applied: str,
    missing_value_handling: str,
    aggregation_method: str,
    valid_row_count: int
) -> str:
    """
    Generates a structured calculation provenance metadata block for Pandas analytical queries.
    """
    cols_str = ", ".join([f"`{c}`" for c in columns_used])
    return (
        f"\n\n---\n**📊 Calculation Provenance**:\n"
        f"- **Dataset**: `{dataset_name}`\n"
        f"- **Columns Evaluated**: {cols_str}\n"
        f"- **Filtering Applied**: {filtering_applied}\n"
        f"- **Missing Value Handling**: {missing_value_handling} ({valid_row_count} valid records processed)\n"
        f"- **Aggregation Method**: `{aggregation_method}`\n"
    )


def is_blacklisted_dummy_value(val: Any) -> bool:
    """Checks if a string or numeric value is in the dummy/placeholder blacklist."""
    val_str = str(val).strip()
    return val_str in DUMMY_VALUE_BLACKLIST or val_str.lower() in PLACEHOLDER_ENTITIES
