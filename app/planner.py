import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.llm import get_hybrid_llm
from app.utils import log_event


logger = logging.getLogger(__name__)

COUNTRY_CANONICAL = {
    "india": "India",
    "china": "China",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "uae": "United Arab Emirates",
}
YEAR_PATTERN = re.compile(r"\b\d{4}\b")


@dataclass(frozen=True)
class QueryPlanStep:
    subquestion: str
    purpose: str


@dataclass(frozen=True)
class QueryPlan:
    strategy: str
    steps: List[QueryPlanStep]
    used_llm: bool = False


COMPLEXITY_PATTERN = re.compile(
    r"\b(compare|contrast|versus|vs\.?|difference|differences|compared)\b",
    re.IGNORECASE,
)
COUNTRY_PATTERN = re.compile(
    r"\b(india|china|united states|usa|us|uk|united kingdom|uae)\b",
    re.IGNORECASE,
)
INDICATOR_PATTERN = re.compile(r"\b(gdp|co2|carbon dioxide|emissions)\b", re.IGNORECASE)


def is_complex_question(question: str) -> bool:
    normalized = str(question or "").strip()
    if not normalized:
        return False
    countries = set(COUNTRY_PATTERN.findall(normalized))
    if len(countries) >= 2:
        return True
    return bool(COMPLEXITY_PATTERN.search(normalized))


def _split_heuristically(question: str) -> List[QueryPlanStep]:
    normalized = str(question or "").strip()
    if not normalized:
        return []

    raw_parts = re.split(r"\b(?:and|vs\.?|versus|compare|contrast)\b", normalized, flags=re.IGNORECASE)
    cleaned_parts = [part.strip(" ,?") for part in raw_parts if part.strip(" ,?")]

    steps: List[QueryPlanStep] = []
    seen = set()
    for part in cleaned_parts:
        step_question = part if part.endswith("?") else f"{part}?"
        key = step_question.lower()
        if key in seen:
            continue
        seen.add(key)
        steps.append(QueryPlanStep(subquestion=step_question, purpose="subquery"))

    return steps


def _compare_steps(question: str) -> List[QueryPlanStep]:
    normalized = str(question or "").strip()
    if not normalized:
        return []

    if not COMPLEXITY_PATTERN.search(normalized):
        return []

    found_countries: List[str] = []
    seen_countries = set()
    for match in COUNTRY_PATTERN.finditer(normalized):
        country_key = match.group(0).lower()
        canonical = COUNTRY_CANONICAL.get(country_key, match.group(0).title())
        if canonical.lower() in seen_countries:
            continue
        seen_countries.add(canonical.lower())
        found_countries.append(canonical)

    indicators = []
    seen_indicators = set()
    for match in INDICATOR_PATTERN.finditer(normalized):
        indicator = match.group(0).lower()
        canonical_indicator = "GDP" if indicator == "gdp" else "CO2 emissions"
        if canonical_indicator.lower() in seen_indicators:
            continue
        seen_indicators.add(canonical_indicator.lower())
        indicators.append(canonical_indicator)

    years = YEAR_PATTERN.findall(normalized)
    year = years[0] if len(set(years)) == 1 else None

    if len(found_countries) < 2 or not indicators:
        return []

    steps: List[QueryPlanStep] = []
    for country in found_countries:
        for indicator in indicators:
            if year:
                subquestion = f"What was {country} {indicator} in {year}?"
            else:
                subquestion = f"What was {country} {indicator}?"
            steps.append(QueryPlanStep(subquestion=subquestion, purpose="comparison"))
    return steps


def _enforce_shared_year(question: str, steps: Sequence[QueryPlanStep]) -> List[QueryPlanStep]:
    years = YEAR_PATTERN.findall(str(question or ""))
    unique_years = sorted(set(years))
    if len(unique_years) != 1:
        return list(steps)

    requested_year = unique_years[0]
    normalized_question = str(question or "").strip()
    if not COMPLEXITY_PATTERN.search(normalized_question):
        return list(steps)

    enforced_steps: List[QueryPlanStep] = []
    for step in steps:
        subquestion = str(step.subquestion or "").strip()
        if not YEAR_PATTERN.search(subquestion):
            subquestion = subquestion.rstrip(" ?")
            subquestion = f"{subquestion} in {requested_year}?"
        enforced_steps.append(QueryPlanStep(subquestion=subquestion, purpose=step.purpose))
    return enforced_steps


def _plan_with_llm(question: str) -> Optional[QueryPlan]:
    llm = get_hybrid_llm()
    if not llm.is_available():
        return None

    prompt = f"""Break the user question into a minimal set of independent retrieval sub-questions.

Rules:
- Keep simple single-hop questions as one step.
- Split multi-country, multi-indicator, compare/contrast, and factual+explanatory questions into separate steps.
- Return JSON only in this schema:
{{
  "strategy": "direct | decomposed",
  "steps": [
    {{"subquestion": "text", "purpose": "factual | explanatory | comparison"}}
  ]
}}

Question: {question}
"""
    try:
        result = llm.invoke(user_prompt=prompt, system_prompt="You are a precise query planner.", session_id=None)
        payload = json.loads(str(result.get("answer", "")).strip())
        raw_steps = payload.get("steps", [])
        steps = [
            QueryPlanStep(
                subquestion=str(step.get("subquestion", "")).strip(),
                purpose=str(step.get("purpose", "subquery")).strip() or "subquery",
            )
            for step in raw_steps
            if str(step.get("subquestion", "")).strip()
        ]
        if not steps:
            return None
        return QueryPlan(
            strategy=str(payload.get("strategy", "decomposed")).strip() or "decomposed",
            steps=steps,
            used_llm=True,
        )
    except Exception as exc:
        logger.warning("Planner LLM step failed: %s", exc)
        return None


def build_query_plan(question: str) -> QueryPlan:
    if not is_complex_question(question):
        plan = QueryPlan(
            strategy="direct",
            steps=[QueryPlanStep(subquestion=str(question or "").strip(), purpose="direct")],
            used_llm=False,
        )
        log_event(logger, logging.INFO, "query_planner_output", strategy=plan.strategy, used_llm=plan.used_llm, steps=[step.__dict__ for step in plan.steps])
        return plan

    compare_steps = _compare_steps(question)
    if compare_steps:
        plan = QueryPlan(strategy="decomposed", steps=compare_steps, used_llm=False)
        log_event(logger, logging.INFO, "query_planner_output", strategy=plan.strategy, used_llm=plan.used_llm, steps=[step.__dict__ for step in plan.steps])
        return plan

    llm_plan = _plan_with_llm(question)
    if llm_plan is not None:
        llm_plan = QueryPlan(
            strategy=llm_plan.strategy,
            steps=_enforce_shared_year(question, llm_plan.steps),
            used_llm=llm_plan.used_llm,
        )
        log_event(logger, logging.INFO, "query_planner_output", strategy=llm_plan.strategy, used_llm=llm_plan.used_llm, steps=[step.__dict__ for step in llm_plan.steps])
        return llm_plan

    heuristic_steps = _enforce_shared_year(question, _split_heuristically(question))
    if not heuristic_steps:
        heuristic_steps = [QueryPlanStep(subquestion=str(question or "").strip(), purpose="direct")]
    plan = QueryPlan(strategy="decomposed", steps=heuristic_steps, used_llm=False)
    log_event(logger, logging.INFO, "query_planner_output", strategy=plan.strategy, used_llm=plan.used_llm, steps=[step.__dict__ for step in plan.steps])
    return plan


def flattened_subquestions(plan: QueryPlan) -> Sequence[str]:
    return [step.subquestion for step in plan.steps if step.subquestion]
