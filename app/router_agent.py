import logging
import re
from dataclasses import dataclass
from typing import List

from app.structured_query import StructuredConstraint, extract_structured_constraints, looks_like_structured_query
from app.utils import log_event


logger = logging.getLogger(__name__)

VISUAL_TERMS = {
    "chart",
    "charts",
    "graph",
    "graphs",
    "figure",
    "figures",
    "table",
    "tables",
    "trend",
    "visual",
    "visuals",
    "diagram",
}

EXPLANATORY_TERMS = {
    "explain",
    "describe",
    "why",
    "how",
    "what does",
    "report",
    "regulation",
    "regulations",
    "standard",
    "standards",
    "growth",
    "impact",
    "effect",
    "effects",
    "support",
}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    use_structured: bool
    use_pdf_retrieval: bool
    use_visual_retrieval: bool
    constraints: List[StructuredConstraint]
    reasoning: str


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _has_visual_intent(normalized: str) -> bool:
    tokens = set(normalized.split())
    return bool(tokens & VISUAL_TERMS)


def _has_explanatory_intent(normalized: str) -> bool:
    return any(term in normalized for term in EXPLANATORY_TERMS)


def route_query(question: str) -> RouteDecision:
    normalized = _normalize(question)
    constraints = extract_structured_constraints(question)
    structured_shape = looks_like_structured_query(question)
    has_visual = _has_visual_intent(normalized)
    has_explanation = _has_explanatory_intent(normalized)

    use_structured = bool(constraints) or (structured_shape and not has_explanation and not has_visual)
    use_visual = has_visual
    use_pdf = has_explanation or has_visual or not structured_shape

    if structured_shape and not constraints and not has_explanation and not has_visual:
        route = "structured"
        reasoning = "query appears numeric but is missing country, year, or metric constraints"
    elif use_structured and (use_pdf or use_visual):
        route = "hybrid"
        reasoning = "query contains exact structured data constraints plus explanatory or visual intent"
    elif use_structured:
        route = "structured"
        reasoning = "query contains country, year, and metric constraints"
    elif use_visual:
        route = "visual"
        reasoning = "query asks for charts, figures, tables, trends, or visuals"
    else:
        route = "pdf"
        reasoning = "query is explanatory or lacks complete numeric constraints"

    decision = RouteDecision(
        route=route,
        use_structured=use_structured,
        use_pdf_retrieval=use_pdf,
        use_visual_retrieval=use_visual,
        constraints=constraints,
        reasoning=reasoning,
    )
    log_event(
        logger,
        logging.INFO,
        "router_agent_decision",
        question=question,
        route=decision.route,
        use_structured=decision.use_structured,
        use_pdf_retrieval=decision.use_pdf_retrieval,
        use_visual_retrieval=decision.use_visual_retrieval,
        constraints=[constraint.__dict__ for constraint in decision.constraints],
        reasoning=decision.reasoning,
    )
    return decision
