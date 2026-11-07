"""
Deterministic Query Intent Router & Modality Isolation Pipeline
Pillar 1 Implementation for Multimodal RAG
"""

import re
from enum import Enum
from typing import Dict, List, Set, Tuple


class QueryIntent(str, Enum):
    CSV_ONLY = "CSV_ONLY"
    TEXT_ONLY = "TEXT_ONLY"
    VISUAL_SPECIFIC = "VISUAL_SPECIFIC"
    HYBRID_MULTIMODAL = "HYBRID_MULTIMODAL"
    CONVERSATIONAL = "CONVERSATIONAL"


# Country / Entity indicators that target tabular CSV datasets
COUNTRY_KEYWORDS = {
    "india", "usa", "us", "united states", "russia", "china", "japan", "germany",
    "uk", "united kingdom", "brazil", "france", "canada", "italy", "south africa",
    "australia", "indonesia", "mexico", "nigeria", "egypt", "vietnam"
}

# Metric & Aggregation indicators that target tabular CSV datasets
CSV_SPECIFIC_KEYWORDS = {
    "gdp", "co2", "emission", "emissions", "income group", "income-group", "high income", "high-income",
    "low income", "low-income", "middle income", "top 5", "top 10", "lowest", "highest",
    "average co2", "average gdp", "per capita co2", "total gdp", "world bank data", "world bank",
    "csv", "dataframe", "dataset", "indicator"
}

METRIC_KEYWORDS = CSV_SPECIFIC_KEYWORDS | {
    "country", "correlation", "groupby", "median", "aggregate", "aggregate value",
    "total value", "value of", "values of", "data of", "sum of", "statistics", "metric", "metrics"
}

CSV_KEYWORDS = COUNTRY_KEYWORDS | METRIC_KEYWORDS

VISUAL_KEYWORDS = {
    "figure", "fig", "chart", "diagram", "table 2.", "table 3.", "table 6.",
    "table 7.", "panel a", "panel b", "scatter plot", "bar chart", "line graph",
    "visual asset", "image", "crop", "this figure", "this chart", "the figure",
    "the chart", "above figure", "above chart", "from this figure", "from this chart",
    "from the table", "from the figure", "from the chart", "extracted table",
    "in this figure", "in this chart", "previous figure", "previous chart"
}

TEXT_KEYWORDS = {
    "what are the three types of standards", "definition of", "standards discussed",
    "policy", "world development report 2025", "wdr 2025", "wdr", "compliance capacity definition",
    "explain", "overview", "chapter", "framework", "recommendations", "standards",
    "pdf", "report", "text", "document", "what does", "how does", "why does",
    "harmonization", "standard harmonization", "harmonization policy", "arguments"
}


def classify_query_intent(query: str) -> Tuple[QueryIntent, Dict[str, bool]]:
    """
    Classifies user query intent and determines allowed tools with strict modality isolation.
    Returns (intent, allowed_tools_dict).
    """
    q_clean = query.lower().strip()

    # 1. Visual Modality Indicators (Explicit Figure/Table IDs or Chart Keywords)
    explicit_asset = bool(re.search(r"\b(?:fig|figure|table)[\s_]*[0-9]+(?:\.[0-9]+)*\b", q_clean))
    visual_terms = {
        "figure", "fig", "chart", "diagram", "panel a", "panel b", "scatter plot",
        "bar chart", "line graph", "visual asset", "crop", "curve", "curves", "plot"
    }
    has_visual = explicit_asset or any(re.search(r"\b" + re.escape(vt) + r"\b", q_clean) for vt in visual_terms)
    
    # Disambiguation: A mention of 'table' without a number when CSV indicators are present refers to CSV, NOT visual!
    if not explicit_asset and not any(vt in q_clean for vt in ["chart", "figure", "fig", "diagram", "graph", "curve", "curves", "plot", "image"]):
        has_visual = False

    # 2. Tabular / CSV Modality Indicators
    has_specific_csv = any(kw in q_clean for kw in CSV_SPECIFIC_KEYWORDS) or bool(re.search(r"\b(19|20)\d\d\b", q_clean))
    has_generic_metric = any(kw in q_clean for kw in METRIC_KEYWORDS)
    has_country = any(re.search(r"\b" + re.escape(c) + r"\b", q_clean) for c in COUNTRY_KEYWORDS)
    if explicit_asset:
        # If an explicit visual asset is targeted, only consider CSV active if specific CSV metrics or countries are named
        has_csv = bool(has_specific_csv or has_country)
    else:
        has_csv = bool(has_specific_csv or has_generic_metric or has_country)

    # 3. Document / Textual Modality Indicators
    text_specific = any(kw in q_clean for kw in TEXT_KEYWORDS)
    text_broad = any(kw in q_clean for kw in [
        "policy", "standard", "standards", "regulation", "regulations", "framework", 
        "chapter", "wdr", "recommendation", "recommendations", "definition", "definitions",
        "explain", "overview", "harmonization", "governance", "trade"
    ])
    has_text = bool(text_specific or text_broad)

    # 4. Comparative / Compound Markers
    comparison_markers = [
        "compare", "comparing", "comparison", "contrast", "contrasting",
        "correlate", "correlation", "relationship between", "versus", "vs",
        "against", "along with", "as well as", "combine", "cross-reference",
        "both", "difference between"
    ]
    is_comparison = any(re.search(r"\b" + re.escape(m) + r"\b", q_clean) for m in comparison_markers)
    explicit_hybrid = any(phrase in q_clean for phrase in [
        "compare csv", "combine csv", "with csv", "csv data", "dataframe", 
        "cross-reference csv", "against gdp dataset", "against co2 dataset"
    ])

    modalities_count = int(has_csv) + int(has_visual) + int(has_text)

    # Compound Cross-Modal Detection (HYBRID_MULTIMODAL)
    is_compound = (
        explicit_hybrid
        or (modalities_count >= 2 and (is_comparison or any(conn in q_clean for conn in [" and ", " with ", " as well as ", " versus ", " vs "])))
        or (is_comparison and (has_csv or has_visual))
    )

    if is_compound:
        return QueryIntent.HYBRID_MULTIMODAL, {
            "allow_pandas": bool(has_csv),
            "allow_qdrant": bool(has_text or (not has_visual and not has_csv)),
            "allow_vision": bool(has_visual)
        }

    # Single Modality 1: Visual Specific
    if has_visual:
        return QueryIntent.VISUAL_SPECIFIC, {
            "allow_pandas": False,
            "allow_qdrant": True,
            "allow_vision": True
        }

    # Single Modality 2: CSV Data Only
    if has_csv and not has_visual and not ("definition of" in q_clean or "three types of standards" in q_clean):
        return QueryIntent.CSV_ONLY, {
            "allow_pandas": True,
            "allow_qdrant": False,
            "allow_vision": False
        }

    # Single Modality 3: Text Only
    return QueryIntent.TEXT_ONLY, {
        "allow_pandas": False,
        "allow_qdrant": True,
        "allow_vision": False
    }
