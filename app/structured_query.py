import csv
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from app.ingestion import DEFAULT_CSV_DIR, infer_metric_family
from app.utils import log_event

try:
    import pandas as pd
except ImportError:  # pragma: no cover - production dependency, fallback keeps app importable
    pd = None


logger = logging.getLogger(__name__)

INSUFFICIENT_DATA_MESSAGE = "I do not have sufficient data to answer this question."
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
DOCUMENT_RETRIEVAL_MARKERS = re.compile(
    r"\b(?:pdf|report|document|chapter|section|subsection|page|figure|table|chart|diagram|image|map|box|spotlight|caption)\b",
    re.IGNORECASE,
)
CSV_ROUTE_MARKERS = re.compile(
    r"\b(?:csv|dataset|data file|spreadsheet|gdp|co2|carbon dioxide|emissions?)\b",
    re.IGNORECASE,
)

COUNTRY_ALIASES = {
    "us": ("United States", "USA"),
    "u s": ("United States", "USA"),
    "usa": ("United States", "USA"),
    "united states": ("United States", "USA"),
    "united states of america": ("United States", "USA"),
    "uk": ("United Kingdom", "GBR"),
    "united kingdom": ("United Kingdom", "GBR"),
    "uae": ("United Arab Emirates", "ARE"),
    "india": ("India", "IND"),
    "china": ("China", "CHN"),
}


@dataclass(frozen=True)
class StructuredConstraint:
    country_name: str
    country_iso3: Optional[str]
    year: str
    indicator: str


@dataclass(frozen=True)
class StructuredLookup:
    constraint: StructuredConstraint
    document: Optional[Document]
    source_csv: Optional[str]

    @property
    def found(self) -> bool:
        return self.document is not None


@dataclass(frozen=True)
class StructuredQueryResult:
    constraints: List[StructuredConstraint]
    lookups: List[StructuredLookup]
    answer_documents: List[Document]
    missing_constraints: List[StructuredConstraint]
    engine: str

    @property
    def has_complete_answer(self) -> bool:
        return bool(self.answer_documents) and not self.missing_constraints


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _read_world_bank_csv(csv_path: Path):
    if pd is not None:
        return pd.read_csv(csv_path, skiprows=4, dtype=str).fillna("")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for _ in range(4):
            next(handle, None)
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({str(key or "").strip(): str(value or "").strip() for key, value in row.items()})
    return rows


@lru_cache(maxsize=1)
def _country_aliases_from_tables() -> Dict[str, Tuple[str, str]]:
    aliases = dict(COUNTRY_ALIASES)
    if not DEFAULT_CSV_DIR.exists():
        return aliases

    for csv_path in sorted(DEFAULT_CSV_DIR.glob("*.csv")):
        try:
            table = _read_world_bank_csv(csv_path)
            if pd is not None and hasattr(table, "iterrows"):
                iterator = (row for _idx, row in table.iterrows())
            else:
                iterator = iter(table)
            for row in iterator:
                country_name = str(row.get("Country Name", "")).strip()
                country_iso3 = str(row.get("Country Code", "")).strip()
                if country_name:
                    aliases[_normalize(country_name)] = (country_name, country_iso3)
                if country_iso3:
                    aliases[_normalize(country_iso3)] = (country_name or country_iso3, country_iso3)
        except Exception as exc:
            logger.warning("Could not inspect countries from %s: %s", csv_path, exc)
    return aliases


def extract_year(question: str) -> Optional[str]:
    years = YEAR_PATTERN.findall(str(question or ""))
    unique_years = sorted(set(years))
    if len(unique_years) == 1:
        return unique_years[0]
    return None


def extract_indicators(question: str) -> List[str]:
    normalized = _normalize(question)
    indicators: List[str] = []
    if "gdp" in normalized:
        indicators.append("gdp")
    if "co2" in normalized or "carbon dioxide" in normalized or "emission" in normalized:
        indicators.append("co2")
    return indicators


def extract_countries(question: str) -> List[Tuple[str, Optional[str]]]:
    normalized_question = f" {_normalize(question)} "
    matches: List[Tuple[int, int, Tuple[str, Optional[str]]]] = []
    aliases = _country_aliases_from_tables()
    for alias, country in aliases.items():
        if not alias:
            continue
        if len(alias) < 4 and alias not in COUNTRY_ALIASES:
            continue
        position = normalized_question.find(f" {alias} ")
        if position >= 0:
            matches.append((position, -len(alias), country))

    ordered: List[Tuple[str, Optional[str]]] = []
    seen = set()
    for _position, _length, country in sorted(matches):
        key = (_normalize(country[0]), _normalize(country[1]))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(country)
    return ordered


def extract_structured_constraints(question: str) -> List[StructuredConstraint]:
    year = extract_year(question)
    indicators = extract_indicators(question)
    countries = extract_countries(question)
    if not year or not indicators or not countries:
        return []

    return [
        StructuredConstraint(
            country_name=country_name,
            country_iso3=country_iso3,
            year=year,
            indicator=indicator,
        )
        for country_name, country_iso3 in countries
        for indicator in indicators
    ]


def looks_like_structured_query(question: str) -> bool:
    return bool(extract_year(question) and extract_indicators(question))


def should_use_structured_csv_query(question: str) -> bool:
    normalized = str(question or "").strip()
    constraints = extract_structured_constraints(normalized)
    if not constraints:
        return False
    if DOCUMENT_RETRIEVAL_MARKERS.search(normalized):
        return False
    return bool(CSV_ROUTE_MARKERS.search(normalized) or looks_like_structured_query(normalized))


def _iter_rows(table):
    if pd is not None and hasattr(table, "iterrows"):
        for _idx, row in table.iterrows():
            yield row
    else:
        yield from table


def _row_matches_country(row: object, constraint: StructuredConstraint) -> bool:
    row_name = _normalize(row.get("Country Name", ""))
    row_iso3 = _normalize(row.get("Country Code", ""))
    expected_name = _normalize(constraint.country_name)
    expected_iso3 = _normalize(constraint.country_iso3)
    return bool(
        (expected_iso3 and row_iso3 == expected_iso3)
        or (expected_name and row_name == expected_name)
    )


def _document_from_row(csv_path: Path, row_index: int, row: object, constraint: StructuredConstraint) -> Optional[Document]:
    value = str(row.get(constraint.year, "")).strip()
    if not value:
        return None

    indicator = str(row.get("Indicator Name", "")).strip()
    indicator_code = str(row.get("Indicator Code", "")).strip()
    country_name = str(row.get("Country Name", "")).strip()
    country_iso3 = str(row.get("Country Code", "")).strip()
    metric_family = infer_metric_family(indicator, indicator_code)
    if metric_family != constraint.indicator:
        return None

    text = f"In {constraint.year}, {indicator} for {country_name} ({country_iso3}) was {value}."
    return Document(
        page_content=text,
        metadata={
            "source": str(csv_path),
            "source_files": csv_path.name,
            "source_type": "csv",
            "retrieval_source": "pandas_structured",
            "dataset_type": indicator_code or indicator,
            "country_name": country_name,
            "country_iso3": country_iso3,
            "indicator": indicator,
            "metric_family": metric_family,
            "year": constraint.year,
            "value": value,
            "row_index": row_index,
        },
    )


class PandasStructuredQueryEngine:
    def __init__(self, csv_dir: Path = DEFAULT_CSV_DIR) -> None:
        self.csv_dir = csv_dir

    @property
    def engine_name(self) -> str:
        return "pandas" if pd is not None else "csv-fallback"

    @lru_cache(maxsize=1)
    def _tables(self) -> Tuple[Tuple[Path, object], ...]:
        if not self.csv_dir.exists():
            return tuple()
        tables: List[Tuple[Path, object]] = []
        for csv_path in sorted(self.csv_dir.glob("*.csv")):
            try:
                tables.append((csv_path, _read_world_bank_csv(csv_path)))
            except Exception as exc:
                logger.warning("Structured CSV load skipped for %s: %s", csv_path, exc)
        return tuple(tables)

    def lookup(self, constraint: StructuredConstraint) -> StructuredLookup:
        for csv_path, table in self._tables():
            for row_index, row in enumerate(_iter_rows(table), start=1):
                if not _row_matches_country(row, constraint):
                    continue
                document = _document_from_row(csv_path, row_index, row, constraint)
                if document is not None:
                    return StructuredLookup(
                        constraint=constraint,
                        document=document,
                        source_csv=csv_path.name,
                    )
        return StructuredLookup(constraint=constraint, document=None, source_csv=None)

    def answer(self, question: str) -> StructuredQueryResult:
        constraints = extract_structured_constraints(question)
        lookups = [self.lookup(constraint) for constraint in constraints]
        answer_documents = [lookup.document for lookup in lookups if lookup.document is not None]
        missing_constraints = [lookup.constraint for lookup in lookups if lookup.document is None]
        log_event(
            logger,
            logging.INFO,
            "pandas_structured_query_completed",
            engine=self.engine_name,
            question=question,
            constraints=[constraint.__dict__ for constraint in constraints],
            answer_documents=len(answer_documents),
            missing_constraints=[constraint.__dict__ for constraint in missing_constraints],
        )
        return StructuredQueryResult(
            constraints=constraints,
            lookups=lookups,
            answer_documents=answer_documents,
            missing_constraints=missing_constraints,
            engine=self.engine_name,
        )


@lru_cache(maxsize=1)
def get_structured_query_engine() -> PandasStructuredQueryEngine:
    return PandasStructuredQueryEngine()
