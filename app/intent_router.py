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


# Country / Entity indicators that target tabular CSV datasets
COUNTRY_KEYWORDS = {
    "india", "usa", "us", "united states", "russia", "china", "japan", "germany",
    "uk", "united kingdom", "brazil", "france", "canada", "italy", "south africa",
    "australia", "indonesia", "mexico", "nigeria", "egypt", "vietnam"
}

# Metric & Aggregation indicators that target tabular CSV datasets
METRIC_KEYWORDS = {
    "gdp", "co2", "emissions", "income group", "income-group", "country",
    "top 5", "top 10", "lowest", "highest", "average co2", "average gdp",
    "correlation", "groupby", "median", "per capita co2", "total gdp",
    "world bank data", "csv", "dataframe", "dataset", "indicator",
    "aggregate", "aggregate value", "total value", "value of", "values of",
    "data of", "sum of", "statistics", "metric"
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
    "pdf", "report", "text", "document", "what does", "how does", "why does"
}


def classify_query_intent(query: str) -> Tuple[QueryIntent, Dict[str, bool]]:
    """
    Classifies user query intent and determines allowed tools.
    Returns (intent, allowed_tools_dict).
    """
    q_clean = query.lower().strip()

    has_csv = any(kw in q_clean for kw in METRIC_KEYWORDS) or bool(re.search(r"\b(19|20)\d\d\b", q_clean))
    has_visual = any(re.search(r"\b" + re.escape(kw) + r"\b", q_clean) for kw in VISUAL_KEYWORDS) or bool(re.search(r"\b(?:fig|figure|table)[\s_]*[0-9]+(?:\.[0-9]+)*\b", q_clean))
    has_text = any(kw in q_clean for kw in TEXT_KEYWORDS)
    has_country = any(re.search(r"\b" + re.escape(c) + r"\b", q_clean) for c in COUNTRY_KEYWORDS)

    has_compound = any(conn in q_clean for conn in [" and ", " as well as ", " along with ", " plus ", " also "])
    explicit_hybrid = any(phrase in q_clean for phrase in ["compare csv", "combine csv", "with csv", "csv data", "dataframe", "cross-reference csv", "against gdp dataset", "against co2 dataset"])

    # 1. Visual Specific: Target figure/chart/table explicitly named
    if has_visual:
        return QueryIntent.VISUAL_SPECIFIC, {
            "allow_pandas": False,
            "allow_qdrant": True,
            "allow_vision": True
        }

    # 3. CSV Only: Data calculation / ranking / aggregation query
    if (has_csv or has_country) and not has_visual and not ("definition of" in q_clean or "three types of standards" in q_clean):
        return QueryIntent.CSV_ONLY, {
            "allow_pandas": True,
            "allow_qdrant": False,
            "allow_vision": False
        }

    # 4. Text Only Default: Conceptual, policy, standard definition inquiries
    return QueryIntent.TEXT_ONLY, {
        "allow_pandas": False,
        "allow_qdrant": True,
        "allow_vision": False
    }
