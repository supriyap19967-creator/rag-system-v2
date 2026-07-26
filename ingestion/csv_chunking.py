from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MISSING_VALUE_TOKENS = {"", "nan", "none", "null", "..", "...", "n/a", "na"}
WORLD_BANK_REQUIRED_HEADERS = ("Country Name", "Country Code", "Indicator Name", "Indicator Code")
COUNTRY_METADATA_HEADERS = ("Country Code", "Region", "IncomeGroup")
INDICATOR_METADATA_HEADERS = ("INDICATOR_CODE", "INDICATOR_NAME")
YEAR_HEADER_PATTERN = re.compile(r"^(?:19|20)\d{2}$")
EXTRACTED_TABLE_FILENAME_PATTERN = re.compile(
    r"page_(?P<page>\d+)_Table_(?P<identifier>[A-Za-z]?\d+(?:\.\d+)*)\.csv$",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class CsvChunkRecord:
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ParsedCsvFile:
    csv_kind: str
    source_path: str
    source_file: str
    blocks: list[CsvChunkRecord]
    metadata: dict[str, Any]


def _stable_file_hash(path: Path) -> str:
    source = str(path.as_posix()).lower()
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def _normalize_string(value: object) -> str:
    return str(value or "").strip()


def _normalize_header(value: object) -> str:
    return re.sub(r"\s+", " ", _normalize_string(value)).strip('"')


def _make_unique_headers(headers: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: dict[str, int] = {}
    for index, header in enumerate(headers, start=1):
        clean = _normalize_header(header) or f"column_{index}"
        count = seen.get(clean, 0)
        seen[clean] = count + 1
        output.append(clean if count == 0 else f"{clean}_{count + 1}")
    return output


def _read_csv_matrix(csv_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            rows.append([_normalize_string(value) for value in row])
    return rows


def _score_header_kind(row: list[str]) -> tuple[str, int]:
    normalized = {_normalize_header(cell).lower() for cell in row if _normalize_header(cell)}
    if not normalized:
        return "generic", 0
    world_bank_score = sum(1 for header in WORLD_BANK_REQUIRED_HEADERS if header.lower() in normalized)
    if world_bank_score >= 3:
        return "world_bank_wide", world_bank_score
    country_metadata_score = sum(1 for header in COUNTRY_METADATA_HEADERS if header.lower() in normalized)
    if country_metadata_score >= 2:
        return "country_metadata", country_metadata_score
    indicator_metadata_score = sum(1 for header in INDICATOR_METADATA_HEADERS if header.lower() in normalized)
    if indicator_metadata_score >= 2:
        return "indicator_metadata", indicator_metadata_score
    return "generic", 0


def detect_csv_header(csv_path: Path) -> tuple[int, list[str], str]:
    rows = _read_csv_matrix(csv_path)
    default_kind = "extracted_table_csv" if "extracted_tables" in str(csv_path.as_posix()).lower() else "generic"
    best_index = 0
    best_headers = _make_unique_headers(rows[0] if rows else [])
    best_kind = default_kind
    best_score = -1
    for index, row in enumerate(rows[:25]):
        if not any(cell.strip() for cell in row):
            continue
        lowered_join = " ".join(cell.lower() for cell in row if cell)
        if lowered_join.startswith("data source") or lowered_join.startswith("last updated date"):
            continue
        kind, score = _score_header_kind(row)
        if score > best_score:
            best_index = index
            best_headers = _make_unique_headers(row)
            best_kind = kind if kind != "generic" else default_kind
            best_score = score
        if kind != "generic" and score >= 3:
            break
    return best_index, best_headers, best_kind


def _rows_as_dicts(rows: list[list[str]], headers: list[str], header_index: int) -> list[dict[str, str]]:
    width = len(headers)
    output: list[dict[str, str]] = []
    for row in rows[header_index + 1 :]:
        if not any(str(value or "").strip() for value in row):
            continue
        padded = list(row[:width]) + [""] * max(0, width - len(row))
        output.append({headers[index]: padded[index] for index in range(width)})
    return output


def _is_missing(value: object) -> bool:
    return _normalize_string(value).lower() in MISSING_VALUE_TOKENS


def _parse_numeric(value: object) -> int | float | None:
    text = _normalize_string(value)
    if _is_missing(text):
        return None
    normalized = text.replace(",", "")
    try:
        number = float(normalized)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _common_metadata(
    csv_path: Path,
    *,
    document_type: str,
    entity_type: str,
    chunk_id: str,
    entity_id: str,
    entity_ids: list[str],
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_type": document_type,
        "source_type": document_type,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "entity_ids": entity_ids,
        "source": str(csv_path),
        "source_path": str(csv_path),
        "source_file": csv_path.name,
        "csv_path": str(csv_path),
        "table_csv_path": str(csv_path) if document_type == "extracted_table_csv" else "",
        "table_image_path": "",
        "figure_image_path": "",
        "chart_image_path": "",
        "chapter_number": "",
        "chapter_title": "",
        "section_title": "",
        "contains_csv": True,
        "contains_table": document_type == "extracted_table_csv",
        "contains_figure": False,
        "contains_chart": False,
        "contains_image": False,
    }


def _entity_aliases(*values: object) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_string(value)
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _aggregate_country(country_name: str, country_code: str) -> bool:
    text = f"{country_name} {country_code}".lower()
    aggregate_markers = (
        "income",
        "world",
        "africa",
        "europe",
        "asia",
        "middle east",
        "north america",
        "latin america",
        "caribbean",
        "union",
        "fragile",
        "small states",
        "dividend",
        "ida",
        "ibrd",
        "total",
        "countries",
        "economies",
        "aggregate",
    )
    return any(marker in text for marker in aggregate_markers)


def _year_columns(headers: Iterable[str]) -> list[str]:
    return [header for header in headers if YEAR_HEADER_PATTERN.fullmatch(_normalize_header(header))]


def _range_windows(years: list[int]) -> list[tuple[int, int]]:
    if not years:
        return []
    output: list[tuple[int, int]] = []
    start = (min(years) // 10) * 10
    end = max(years)
    while start <= end:
        window_end = min(start + 9, end)
        output.append((start, window_end))
        start += 10
    return output


def _timeseries_summary_text(
    country_name: str,
    country_code: str,
    indicator_name: str,
    indicator_code: str,
    available_years: list[int],
    total_years: int,
) -> str:
    if available_years:
        return (
            f"Country: {country_name} ({country_code}). Indicator: {indicator_name}, code {indicator_code}. "
            f"Annual values are available for {len(available_years)} years from {available_years[0]} to {available_years[-1]} "
            f"within a source series spanning {total_years} years."
        )
    return (
        f"Country: {country_name} ({country_code}). Indicator: {indicator_name}, code {indicator_code}. "
        "The source CSV contains the row structure, but all annual values in the detected year columns are missing."
    )


def _timeseries_range_text(
    country_name: str,
    country_code: str,
    indicator_name: str,
    indicator_code: str,
    year_start: int,
    year_end: int,
    values: dict[str, str],
) -> str:
    facts = "; ".join(f"{year} = {values[year]}" for year in sorted(values, key=int))
    return (
        f"Country: {country_name} ({country_code}). Indicator: {indicator_name}, code {indicator_code}. "
        f"Year range {year_start}-{year_end}. Values: {facts}."
    )


def _build_world_bank_chunks(csv_path: Path, headers: list[str], rows: list[dict[str, str]]) -> list[CsvChunkRecord]:
    file_hash = _stable_file_hash(csv_path)
    year_headers = _year_columns(headers)
    header_years = [int(year) for year in year_headers]
    chunks: list[CsvChunkRecord] = []
    for row_index, row in enumerate(rows, start=1):
        country_name = _normalize_string(row.get("Country Name"))
        country_code = _normalize_string(row.get("Country Code"))
        indicator_name = _normalize_string(row.get("Indicator Name"))
        indicator_code = _normalize_string(row.get("Indicator Code"))
        if not (country_name and country_code and indicator_name):
            continue
        available_pairs = {year: _normalize_string(row.get(year)) for year in year_headers if not _is_missing(row.get(year))}
        available_years = sorted(int(year) for year in available_pairs)
        missing_years = [year for year in header_years if str(year) not in available_pairs]
        base_chunk_id = f"csvdata::{file_hash}::{indicator_code or 'unknown_indicator'}::{country_code or row_index}"
        entity_id = base_chunk_id
        entity_ids = _entity_aliases(base_chunk_id, country_code, country_name, indicator_code, indicator_name)
        common = _common_metadata(
            csv_path,
            document_type="csv",
            entity_type="csv_timeseries",
            chunk_id=base_chunk_id,
            entity_id=entity_id,
            entity_ids=entity_ids,
        )
        common.update(
            {
                "country_name": country_name,
                "country_code": country_code,
                "indicator_name": indicator_name,
                "indicator_code": indicator_code,
                "year_min": available_years[0] if available_years else None,
                "year_max": available_years[-1] if available_years else None,
                "available_years": available_years,
                "missing_years": missing_years,
                "is_aggregate": _aggregate_country(country_name, country_code),
                "entity_scope": "aggregate" if _aggregate_country(country_name, country_code) else "country",
                "row_index": row_index,
            }
        )
        chunks.append(
            CsvChunkRecord(
                text=_timeseries_summary_text(
                    country_name,
                    country_code,
                    indicator_name,
                    indicator_code,
                    available_years,
                    len(header_years),
                ),
                metadata=common,
            )
        )

        for year_start, year_end in _range_windows(header_years):
            range_values = {
                year: available_pairs[year]
                for year in available_pairs
                if year_start <= int(year) <= year_end
            }
            if not range_values:
                continue
            range_chunk_id = f"csvrange::{file_hash}::{indicator_code or 'unknown_indicator'}::{country_code or row_index}::{year_start}_{year_end}"
            range_metadata = _common_metadata(
                csv_path,
                document_type="csv",
                entity_type="csv_timeseries_range",
                chunk_id=range_chunk_id,
                entity_id=entity_id,
                entity_ids=entity_ids,
            )
            range_metadata.update(
                {
                    "country_name": country_name,
                    "country_code": country_code,
                    "indicator_name": indicator_name,
                    "indicator_code": indicator_code,
                    "year_start": year_start,
                    "year_end": year_end,
                    "year_min": year_start,
                    "year_max": year_end,
                    "available_years": sorted(int(year) for year in range_values),
                    "missing_years": [year for year in range(year_start, year_end + 1) if str(year) not in range_values],
                    "values_by_year": {year: _parse_numeric(value) for year, value in range_values.items()},
                    "is_aggregate": common["is_aggregate"],
                    "entity_scope": common["entity_scope"],
                    "row_index": row_index,
                }
            )
            chunks.append(
                CsvChunkRecord(
                    text=_timeseries_range_text(
                        country_name,
                        country_code,
                        indicator_name,
                        indicator_code,
                        year_start,
                        year_end,
                        range_values,
                    ),
                    metadata=range_metadata,
                )
            )
    return chunks


def _build_country_metadata_chunks(csv_path: Path, rows: list[dict[str, str]]) -> list[CsvChunkRecord]:
    file_hash = _stable_file_hash(csv_path)
    chunks: list[CsvChunkRecord] = []
    for row_index, row in enumerate(rows, start=1):
        country_code = _normalize_string(row.get("Country Code"))
        if not country_code:
            continue
        country_name = _normalize_string(row.get("Country Name") or row.get("TableName"))
        region = _normalize_string(row.get("Region"))
        income_group = _normalize_string(row.get("IncomeGroup"))
        special_notes = _normalize_string(row.get("SpecialNotes"))
        table_name = _normalize_string(row.get("TableName"))
        chunk_id = f"countrymeta::{file_hash}::{country_code}"
        entity_ids = _entity_aliases(chunk_id, country_code, country_name, table_name)
        metadata = _common_metadata(
            csv_path,
            document_type="csv",
            entity_type="country_metadata",
            chunk_id=chunk_id,
            entity_id=chunk_id,
            entity_ids=entity_ids,
        )
        metadata.update(
            {
                "country_code": country_code,
                "country_name": country_name,
                "region": region,
                "income_group": income_group,
                "special_notes": special_notes,
                "table_name": table_name,
                "row_index": row_index,
                "is_aggregate": _aggregate_country(country_name or table_name, country_code),
                "entity_scope": "aggregate" if _aggregate_country(country_name or table_name, country_code) else "country",
            }
        )
        text = (
            f"Country metadata for {country_name or table_name or country_code} ({country_code}): "
            f"Region = {region or 'unknown'}; Income group = {income_group or 'unknown'}."
        )
        chunks.append(CsvChunkRecord(text=text, metadata=metadata))
    return chunks


def _build_indicator_metadata_chunks(csv_path: Path, rows: list[dict[str, str]]) -> list[CsvChunkRecord]:
    chunks: list[CsvChunkRecord] = []
    for row_index, row in enumerate(rows, start=1):
        indicator_code = _normalize_string(row.get("INDICATOR_CODE"))
        indicator_name = _normalize_string(row.get("INDICATOR_NAME"))
        if not indicator_code:
            continue
        chunk_id = f"indicatormeta::{indicator_code}"
        metadata = _common_metadata(
            csv_path,
            document_type="csv",
            entity_type="indicator_metadata",
            chunk_id=chunk_id,
            entity_id=chunk_id,
            entity_ids=_entity_aliases(chunk_id, indicator_code, indicator_name),
        )
        metadata.update(
            {
                "indicator_code": indicator_code,
                "indicator_name": indicator_name,
                "source_note": _normalize_string(row.get("SOURCE_NOTE")),
                "source_organization": _normalize_string(row.get("SOURCE_ORGANIZATION")),
                "row_index": row_index,
            }
        )
        text = (
            f"Indicator metadata: {indicator_code} = {indicator_name}. "
            f"Source note: {metadata['source_note'] or 'not provided'}."
        )
        chunks.append(CsvChunkRecord(text=text, metadata=metadata))
    return chunks


def _table_entity_from_path(csv_path: Path) -> tuple[str, int | None]:
    match = EXTRACTED_TABLE_FILENAME_PATTERN.search(csv_path.name)
    if not match:
        return f"Table {csv_path.stem}", None
    identifier = match.group("identifier")
    page_no = int(match.group("page"))
    return f"Table {identifier}", page_no


def _row_to_pairs(row: dict[str, str]) -> list[tuple[str, str]]:
    return [
        (_normalize_header(key), _normalize_string(value))
        for key, value in row.items()
        if _normalize_header(key) and not _is_missing(value)
    ]


def _build_extracted_table_chunks(csv_path: Path, headers: list[str], rows: list[dict[str, str]]) -> list[CsvChunkRecord]:
    entity_id, page_no = _table_entity_from_path(csv_path)
    entity_ids = _entity_aliases(entity_id, entity_id.replace(" ", "_"))
    column_names = [_normalize_header(header) for header in headers]
    row_count = len(rows)
    chunks: list[CsvChunkRecord] = []

    summary_metadata = _common_metadata(
        csv_path,
        document_type="extracted_table_csv",
        entity_type="table",
        chunk_id=f"tablecsv::{entity_id.replace(' ', '_')}::summary",
        entity_id=entity_id,
        entity_ids=entity_ids,
    )
    summary_metadata.update({"page_no": page_no, "column_names": column_names, "row_count": row_count})
    summary_text = (
        f"Extracted table {entity_id}"
        f"{f' from page {page_no}' if page_no is not None else ''}. "
        f"Columns: {', '.join(column_names)}. Row count: {row_count}."
    )
    chunks.append(CsvChunkRecord(text=summary_text, metadata=summary_metadata))

    columns_metadata = _common_metadata(
        csv_path,
        document_type="extracted_table_csv",
        entity_type="table",
        chunk_id=f"tablecsv::{entity_id.replace(' ', '_')}::columns",
        entity_id=entity_id,
        entity_ids=entity_ids,
    )
    columns_metadata.update({"page_no": page_no, "column_names": column_names, "row_count": row_count})
    chunks.append(
        CsvChunkRecord(
            text=f"Extracted table {entity_id} column names: {', '.join(column_names)}.",
            metadata=columns_metadata,
        )
    )

    for row_index, row in enumerate(rows, start=1):
        pairs = _row_to_pairs(row)
        if not pairs:
            continue
        facts = "; ".join(f"{key} = {value}" for key, value in pairs)
        row_metadata = _common_metadata(
            csv_path,
            document_type="extracted_table_csv",
            entity_type="table",
            chunk_id=f"tablecsv::{entity_id.replace(' ', '_')}::row::{row_index}",
            entity_id=entity_id,
            entity_ids=entity_ids,
        )
        row_metadata.update(
            {
                "page_no": page_no,
                "row_index": row_index,
                "column_names": column_names,
                "row_count": row_count,
            }
        )
        chunks.append(
            CsvChunkRecord(
                text=f"Extracted table {entity_id} row {row_index}: {facts}.",
                metadata=row_metadata,
            )
        )
    return chunks


def parse_csv_file(csv_path: Path) -> ParsedCsvFile:
    csv_path = Path(csv_path)
    rows_matrix = _read_csv_matrix(csv_path)
    header_index, headers, detected_kind = detect_csv_header(csv_path)
    row_dicts = _rows_as_dicts(rows_matrix, headers, header_index)

    if detected_kind == "world_bank_wide":
        blocks = _build_world_bank_chunks(csv_path, headers, row_dicts)
        document_type = "csv"
    elif detected_kind == "country_metadata":
        blocks = _build_country_metadata_chunks(csv_path, row_dicts)
        document_type = "csv"
    elif detected_kind == "indicator_metadata":
        blocks = _build_indicator_metadata_chunks(csv_path, row_dicts)
        document_type = "csv"
    else:
        detected_kind = "extracted_table_csv" if "extracted_tables" in str(csv_path.as_posix()).lower() else detected_kind
        blocks = _build_extracted_table_chunks(csv_path, headers, row_dicts) if detected_kind == "extracted_table_csv" else []
        document_type = "extracted_table_csv" if detected_kind == "extracted_table_csv" else "csv"

    metadata = {
        "source_type": document_type,
        "source_file": csv_path.name,
        "source_path": str(csv_path),
        "csv_kind": detected_kind,
        "header_row_index": header_index,
        "columns": headers,
        "row_count": len(row_dicts),
        "chunk_count": len(blocks),
    }
    return ParsedCsvFile(
        csv_kind=detected_kind,
        source_path=str(csv_path),
        source_file=csv_path.name,
        blocks=blocks,
        metadata=metadata,
    )
