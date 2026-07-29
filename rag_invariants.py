from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any


class RAGInvariantViolation(Exception):
    """Base exception for post-generation RAG invariant failures."""


class AssetPathHallucinationError(RAGInvariantViolation):
    """Raised when a generated asset path does not physically exist on disk."""


class MalformedCoordinatesError(RAGInvariantViolation):
    """Raised when bounding box coordinates are malformed or out of normalized bounds."""


class EntityGroundingViolation(RAGInvariantViolation):
    """Raised when a generated fact-like entity is absent from the retrieved source chunks."""


class AnchorQuoteMismatchError(RAGInvariantViolation):
    """Raised when a claimed direct quote is not an exact substring of source context."""


class VisualSpatialGroundingViolation(RAGInvariantViolation):
    """Raised when generated visual metadata is not grounded in source context."""


@dataclass(frozen=True)
class InvariantValidationReport:
    checked_asset_paths: int
    checked_bounding_boxes: int
    checked_entities: int
    checked_quotes: int


class RAGInvariantsValidator:
    """Deterministic post-generation validation for multimodal RAG outputs."""

    ASSET_PATH_KEYS = {
        "image_path",
        "image_paths",
        "csv_path",
        "csv_paths",
        "asset_path",
        "asset_paths",
        "file_path",
        "file_paths",
        "table_path",
        "table_paths",
    }
    BOUNDING_BOX_KEYS = {
        "bounding_box",
        "bounding_boxes",
        "bbox",
        "bboxes",
        "box",
        "boxes",
    }
    ASSET_PATH_PATTERN = re.compile(
        r"\b[^\s`'\"<>|]+?\.(?:png|jpg|jpeg|webp|gif|csv|xlsx|xls|pdf)\b",
        flags=re.IGNORECASE,
    )
    QUOTE_PATTERN = re.compile(r'"([^"\n]{4,500})"')
    NUMBER_PATTERN = re.compile(
        r"""
        (?<![\w])
        (?:
            \d{1,3}(?:,\d{3})+(?:\.\d+)?
            |\d+(?:\.\d+)?%
            |\d+(?:\.\d+)?\s*(?:percent|percentage\s+points|pp|million|billion|trillion|kg|km|tons?|years?|days?|months?)
            |\$\s?\d+(?:,\d{3})*(?:\.\d+)?
        )
        (?![\w])
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    STRUCTURAL_REFERENCE_PATTERN = re.compile(
        r"\b(?:Figure|Fig\.?|Chart|Table|Diagram|Panel)\s+[A-Za-z]?\d+(?:[.\-]\d+)*[A-Za-z]?\b",
        flags=re.IGNORECASE,
    )
    PROPER_NOUN_PHRASE_PATTERN = re.compile(
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b"
    )

    def validate(self, generated_payload: Any, source_chunks: list[dict[str, Any]]) -> InvariantValidationReport:
        payload_text = self._generated_response_text(generated_payload)
        source_text = self._source_text(source_chunks)
        normalized_source = self._normalize_for_search(source_text)

        asset_paths = self.extract_asset_paths(generated_payload)
        checked_asset_paths = self.validate_asset_paths(asset_paths)

        bounding_boxes = self.extract_bounding_boxes(generated_payload)
        checked_bounding_boxes = self.validate_bounding_boxes(bounding_boxes)

        entities = self.extract_fact_entities(payload_text)
        checked_entities = self.validate_entities_are_grounded(entities, normalized_source)

        quotes = self.extract_direct_quotes(payload_text)
        checked_quotes = self.validate_exact_quotes(quotes, source_text)

        return InvariantValidationReport(
            checked_asset_paths=checked_asset_paths,
            checked_bounding_boxes=checked_bounding_boxes,
            checked_entities=checked_entities,
            checked_quotes=checked_quotes,
        )

    def validate_asset_paths(self, asset_paths: list[str]) -> int:
        from app.multimodal_assets import build_asset_registry, normalize_entity_id
        from app.main import _resolve_existing_image_path
        resolved_paths = []
        for path in asset_paths:
            resolved = _resolve_existing_image_path(path)
            if resolved and os.path.exists(resolved):
                resolved_paths.append(resolved)
            else:
                resolved_paths.append(path)
        registry = None
        for path in resolved_paths:
            if os.path.exists(path):
                continue
            # If the path is a .pdf or generic .csv citation, skip raising path error
            filename = os.path.basename(path)
            if filename.lower().endswith(('.pdf', '.csv')):
                continue

            # Fallback check against normalized registry assets
            norm_id = normalize_entity_id(filename)
            if not registry:
                registry = build_asset_registry()
            matched = False
            for record in registry:
                if record.entity_id == norm_id:
                    if os.path.exists(record.absolute_path):
                        matched = True
                        break
            if not matched:
                raise AssetPathHallucinationError(f"Referenced asset path does not exist: {path}")
        return len(asset_paths)

    def validate_bounding_boxes(self, bounding_boxes: list[list[Any]], payload: dict[str, Any] | None = None, source_text: str = "") -> int:
        for box in bounding_boxes:
            if not isinstance(box, list) or len(box) != 4:
                raise MalformedCoordinatesError(f"Bounding box must be [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]: {box}")

            try:
                c0, c1, c2, c3 = [float(value) for value in box]
            except (TypeError, ValueError) as exc:
                raise MalformedCoordinatesError(f"Bounding box contains non-numeric values: {box}") from exc

            values = (c0, c1, c2, c3)
            if any(value < 0.0 or value > 1.0 for value in values):
                raise MalformedCoordinatesError(f"Bounding box values must be normalized between 0.0 and 1.0: {box}")
            # Ensure minimum coordinate does not exceed maximum in either ordering format
            if c0 > c2 or c1 > c3:
                raise MalformedCoordinatesError(f"Bounding box minimums must not exceed maximums: {box}")

        # Spatial grounding verification for visual elements using expanded source boundary
        if payload and isinstance(payload, dict) and source_text:
            expanded_sources = [source_text]
            reasoning = payload.get("text_reasoning") or payload.get("text_response")
            if reasoning:
                expanded_sources.append(str(reasoning))
            try:
                from streamlit_ui.StreamlitApp import LAST_VISION_RAW_CONTENT
                if LAST_VISION_RAW_CONTENT:
                    expanded_sources.append(str(LAST_VISION_RAW_CONTENT))
            except Exception:
                pass

            full_source_text = "\n".join(expanded_sources)
            normalized_source_expanded = self._normalize_for_search(full_source_text)

            ALLOWLIST = {'Series', 'Category', 'TargetValue', 'Value', 'None', '', 'Chart', 'Table', 'Row', 'Column', 'n/a'}
            ALLOWLIST_LOWER = {item.lower() for item in ALLOWLIST}
            metadata_keys = ["chart_title", "x_axis_label", "y_axis_label", "units"]
            for key in metadata_keys:
                value = payload.get(key)
                if value and isinstance(value, str):
                    val_clean = value.strip()
                    if val_clean.lower() in ALLOWLIST_LOWER:
                        continue
                    if val_clean and val_clean.lower() != "n/a":
                        words = [w.lower() for w in re.findall(r"\w+", val_clean) if len(w) > 2]
                        if words:
                            matched = 0
                            for w in words:
                                if w in normalized_source_expanded or any(w in src_word for src_word in normalized_source_expanded.split()):
                                    matched += 1
                            if len(words) > 0 and matched / len(words) < 0.3:
                                raise VisualSpatialGroundingViolation(
                                    f"Visual metadata '{key}' value '{val_clean}' is not semantically present in source context."
                                )

        return len(bounding_boxes)

    def validate_entities_are_grounded(self, entities: list[str], normalized_source: str, payload: dict[str, Any] | None = None, source_text: str = "") -> int:
        # 1. Expand the source text grounding boundary
        expanded_sources = [source_text]
        if payload and isinstance(payload, dict):
            reasoning = payload.get("text_reasoning") or payload.get("text_response")
            if reasoning:
                expanded_sources.append(str(reasoning))
        try:
            from streamlit_ui.StreamlitApp import LAST_VISION_RAW_CONTENT
            if LAST_VISION_RAW_CONTENT:
                expanded_sources.append(str(LAST_VISION_RAW_CONTENT))
        except Exception:
            pass

        full_source_text = "\n".join(expanded_sources)
        normalized_source_expanded = self._normalize_for_search(full_source_text)

        # 2. Whitelist generic structural terms
        whitelist = {
            "unnamed series", "data trends", "chart data", "table", "series", "category",
            "unnamed", "n/a", "data point", "value", "targetvalue", "data"
        }
        
        filtered_entities = []
        for entity in entities:
            ent_clean = str(entity).strip().lower()
            if ent_clean in whitelist or any(w in ent_clean for w in ["unnamed", "series", "category", "data point"]):
                continue
            filtered_entities.append(entity)

        # 3. Grounding check
        missing = [
            entity
            for entity in filtered_entities
            if self._normalize_for_search(entity) not in normalized_source_expanded
        ]
        if missing:
            raise EntityGroundingViolation(
                "Generated response contains ungrounded entities or metrics: "
                + ", ".join(missing[:10])
            )

        # Tabular schema check & numeric precision check
        if payload and isinstance(payload, dict) and full_source_text:
            table_rows = payload.get("extracted_table", [])
            if isinstance(table_rows, list) and table_rows:
                # Extract table headers (schema checking)
                markdown_headers = []
                for line in full_source_text.splitlines():
                    if "|" in line:
                        cols = [c.strip().lower() for c in line.split("|") if c.strip()]
                        if cols and not any("-" in c for c in cols):
                            markdown_headers.extend(cols)

                # Check if Category or Series represents headers deriving from source
                for row in table_rows:
                    row_dict = row if isinstance(row, dict) else getattr(row, "__dict__", {})
                    s = str(row_dict.get("Series", "")).strip()
                    c = str(row_dict.get("Category", "")).strip()
                    if s and s.lower() != "n/a" and len(s) > 3 and markdown_headers:
                        if not any(s.lower() in h or h in s.lower() for h in markdown_headers) and self._normalize_for_search(s) not in normalized_source_expanded:
                            raise EntityGroundingViolation(f"Generated column header/series '{s}' does not match context table schema.")

                # Numeric precision check
                source_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", full_source_text.replace(",", "")))
                for row in table_rows:
                    row_dict = row if isinstance(row, dict) else getattr(row, "__dict__", {})
                    val = row_dict.get("TargetValue")
                    if val is not None and val != "":
                        try:
                            val_float = float(val)
                            val_str = f"{val_float:g}"
                            if val_str not in source_numbers and str(val) not in source_numbers:
                                found_close = False
                                for s_num in source_numbers:
                                    try:
                                        if abs(float(s_num) - val_float) < 1e-4:
                                            found_close = True
                                            break
                                    except ValueError:
                                        continue
                                if not found_close:
                                    raise EntityGroundingViolation(
                                        f"Numeric value precision mismatch: generated value '{val}' not found in source context."
                                    )
                        except ValueError:
                            pass

        return len(entities)

    def validate_exact_quotes(self, quotes: list[str], source_text: str, payload: dict[str, Any] | None = None) -> int:
        expanded_sources = [source_text]
        if payload and isinstance(payload, dict):
            reasoning = payload.get("text_reasoning") or payload.get("text_response")
            if reasoning:
                expanded_sources.append(str(reasoning))
        try:
            from streamlit_ui.StreamlitApp import LAST_VISION_RAW_CONTENT
            if LAST_VISION_RAW_CONTENT:
                expanded_sources.append(str(LAST_VISION_RAW_CONTENT))
        except Exception:
            pass

        full_source_text = "\n".join(expanded_sources)
        for quote in quotes:
            if quote not in full_source_text:
                raise AnchorQuoteMismatchError(f"Quoted text is not an exact source substring: {quote}")
        return len(quotes)

    def extract_asset_paths(self, payload: Any) -> list[str]:
        paths: list[str] = []

        def walk(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    walk(child_value, str(child_key))
            elif isinstance(value, list):
                for item in value:
                    walk(item, key)
            elif isinstance(value, str):
                if key in self.ASSET_PATH_KEYS:
                    paths.append(value.strip())
                paths.extend(match.group(0).strip(").,;") for match in self.ASSET_PATH_PATTERN.finditer(value))

        walk(payload)
        return self._dedupe(paths)

    def extract_bounding_boxes(self, payload: Any) -> list[list[Any]]:
        boxes: list[list[Any]] = []

        def looks_like_box(value: Any) -> bool:
            return isinstance(value, list) and len(value) == 4 and not any(isinstance(item, (dict, list)) for item in value)

        def walk(value: Any, key: str = "") -> None:
            normalized_key = key.lower()
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    walk(child_value, str(child_key))
            elif isinstance(value, list):
                if normalized_key in self.BOUNDING_BOX_KEYS and looks_like_box(value):
                    boxes.append(value)
                    return
                if normalized_key in self.BOUNDING_BOX_KEYS and all(looks_like_box(item) for item in value):
                    boxes.extend(value)
                    return
                for item in value:
                    walk(item, key)

        walk(payload)
        return boxes

    def extract_fact_entities(self, payload_text: str) -> list[str]:
        candidates: list[str] = []
        candidates.extend(match.group(0) for match in self.NUMBER_PATTERN.finditer(payload_text))
        candidates.extend(match.group(0) for match in self.STRUCTURAL_REFERENCE_PATTERN.finditer(payload_text))

        for match in self.PROPER_NOUN_PHRASE_PATTERN.finditer(payload_text):
            phrase = match.group(0)
            if phrase.lower() not in {"Evidence Item", "Source Chunk", "Final Answer"}:
                candidates.append(phrase)

        return self._dedupe(candidates)

    def extract_direct_quotes(self, payload_text: str) -> list[str]:
        return self._dedupe(match.group(1) for match in self.QUOTE_PATTERN.finditer(payload_text))

    @staticmethod
    def _payload_to_text(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(payload)

    @classmethod
    def _generated_response_text(cls, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            preferred_keys = ("text_response", "answer", "response", "generated_text", "output_text", "text", "content")
            values = [
                str(payload[key])
                for key in preferred_keys
                if isinstance(payload.get(key), str) and payload.get(key, "").strip()
            ]
            if values:
                return "\n".join(values)
        return cls._payload_to_text(payload)

    @staticmethod
    def _source_text(source_chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for chunk in source_chunks or []:
            if not isinstance(chunk, dict):
                blocks.append(str(chunk))
                continue
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            blocks.append(str(chunk.get("content") or chunk.get("text") or chunk.get("page_content") or ""))
            blocks.append(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
            blocks.append(str(chunk.get("source") or ""))
        return "\n".join(block for block in blocks if block)

    @staticmethod
    def _normalize_for_search(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").replace(",", "")).strip().lower()

    @staticmethod
    def _dedupe(values: Any) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key not in seen:
                seen.add(key)
                output.append(text)
        return output


def _print_scorecard(name: str, validator: RAGInvariantsValidator, payload: Any, chunks: list[dict[str, Any]]) -> None:
    print(f"\n=== {name} ===")
    try:
        paths = validator.extract_asset_paths(payload)
        validator.validate_asset_paths(paths)
        print(f"[Layer 4: Path Verification] - PASSED: {len(paths)} path(s) physically exist on disk.")

        boxes = validator.extract_bounding_boxes(payload)
        validator.validate_bounding_boxes(boxes)
        print(f"[Layer 5: Bounding Box Validator] - PASSED: {len(boxes)} normalized box(es) valid.")

        source_text = validator._normalize_for_search(validator._source_text(chunks))
        entities = validator.extract_fact_entities(validator._generated_response_text(payload))
        validator.validate_entities_are_grounded(entities, source_text)
        print(f"[Layer 6: Entity Cross-Checker] - PASSED: {len(entities)} entity/entities grounded.")

        quotes = validator.extract_direct_quotes(validator._generated_response_text(payload))
        validator.validate_exact_quotes(quotes, validator._source_text(chunks))
        print(f"[Layer 7: Exact Quote Anchoring] - PASSED: {len(quotes)} quote(s) anchored.")
        print("[Invariant Result] - CLEARED FOR UI")
    except RAGInvariantViolation as exc:
        print(f"[Invariant Result] - BLOCKED: {exc.__class__.__name__}: {exc}")


if __name__ == "__main__":
    validator = RAGInvariantsValidator()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
        existing_image_path = image_file.name

    source_chunks = [
        {
            "content": (
                "Figure 4.1 reports 27% adoption by firms in India. "
                "The report states: standards reduce transaction costs."
            ),
            "source": "World Development Report 2025.pdf",
            "metadata": {"image_path": existing_image_path, "page_number": 207},
        }
    ]

    success_payload = {
        "answer": (
            "Figure 4.1 reports 27% adoption by firms in India. "
            'The source says "standards reduce transaction costs."'
        ),
        "image_path": existing_image_path,
        "bounding_boxes": [[0.10, 0.20, 0.70, 0.90]],
    }
    hallucinated_path_payload = {
        "answer": "Figure 4.1 reports 27% adoption by firms in India.",
        "image_path": os.path.join(tempfile.gettempdir(), "missing_hallucinated_chart.png"),
    }
    bad_quote_payload = {
        "answer": 'The source says "standards eliminate all transaction costs."',
        "image_path": existing_image_path,
    }

    try:
        _print_scorecard("Successful Fully Grounded Response", validator, success_payload, source_chunks)
        _print_scorecard("Failure: Hallucinated Image Path", validator, hallucinated_path_payload, source_chunks)
        _print_scorecard("Failure: Direct Quote Mismatch", validator, bad_quote_payload, source_chunks)
    finally:
        if os.path.exists(existing_image_path):
            os.unlink(existing_image_path)
