import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.ingestion import infer_metric_family
from app.retriever import RetrievalHints
from app.structured_query import extract_countries, extract_indicators, extract_year
from app.utils import log_event


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelfQueryResult:
    hints: Optional[RetrievalHints]
    confidence: float
    applied: bool
    reason: str


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def infer_topic(query: str) -> str:
    normalized = _normalize(query)
    if "regulation" in normalized or "regulatory" in normalized:
        return "regulations"
    if "standard" in normalized:
        return "standards"
    if "emission" in normalized or "co2" in normalized:
        return "emissions"
    if "growth" in normalized or "gdp" in normalized:
        return "growth"
    return ""


def extract_figure_id(query: str) -> str:
    match = re.search(r"\b(Fig\.?|Figure|Table|Chart|Panel)\s+(\d+(?:[.\s]\d+)?[A-Za-z]?)\b", str(query or ""), re.IGNORECASE)
    if not match:
        return ""
    kind, number = match.groups()
    kind = "Figure" if kind.lower().startswith("fig") else kind.title()
    number = re.sub(r"\s+", ".", number.strip())
    return f"{kind} {number}"


def build_self_query_hints(
    query: str,
    *,
    source_type: str = "pdf",
    visual_only: bool = False,
) -> SelfQueryResult:
    countries = extract_countries(query)
    country_name, country_iso3 = countries[0] if len(countries) == 1 else ("", None)
    year = extract_year(query)
    indicators = extract_indicators(query)
    indicator_family = indicators[0] if len(indicators) == 1 else ""
    topic = infer_topic(query)
    figure_id = extract_figure_id(query)
    normalized = _normalize(query)
    page_match = re.search(r"\bpage\s+(\d{1,4})\b", normalized)
    source_match = re.search(r"\b([\w ._-]+\.pdf)\b", str(query or ""), flags=re.IGNORECASE)

    signals = [
        bool(source_type),
        bool(country_name or country_iso3),
        bool(year),
        bool(indicator_family),
        bool(topic),
        bool(figure_id),
        bool(page_match),
        bool(source_match),
        bool(visual_only),
    ]
    confidence = sum(1 for signal in signals if signal) / len(signals)
    if not any(signals[1:]):
        result = SelfQueryResult(
            hints=RetrievalHints(source_type=source_type),
            confidence=0.25,
            applied=True,
            reason="source_type_only",
        )
    else:
        result = SelfQueryResult(
            hints=RetrievalHints(
                source_type=source_type,
                country_iso3=country_iso3,
                country_name=country_name or None,
                year=year,
                indicator_family=infer_metric_family(indicator_family),
                page=page_match.group(1) if page_match else None,
                source_filename=source_match.group(1).strip() if source_match else None,
                topic=topic or None,
                figure_id=figure_id or None,
                visual_type="visual" if visual_only else None,
            ),
            confidence=round(max(confidence, 0.35), 2),
            applied=True,
            reason="metadata_signals_extracted",
        )

    log_event(
        logger,
        logging.INFO,
        "self_query_filter_generated",
        query=query,
        hints=result.hints.__dict__ if result.hints is not None else None,
        confidence=result.confidence,
        applied=result.applied,
        reason=result.reason,
    )
    return result
