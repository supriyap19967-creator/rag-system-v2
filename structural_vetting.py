from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class StructuralVettingError(Exception):
    """Base exception for structural output vetting failures."""


class SchemaContractViolation(StructuralVettingError):
    """Raised when a generated payload violates the API response schema."""


class EmptyAssetPayloadError(StructuralVettingError):
    """Raised when an asserted asset field is present but empty/null."""


@dataclass(frozen=True)
class VettingResult:
    payload: dict[str, Any]
    healed_markdown: bool
    repairs: list[str]


class StructuralOutputVetter:
    """Fast deterministic validator for final API/UI payload shape."""

    REQUIRED_SCHEMA = {
        "text_response": str,
        "confidence_score": float,
        "metadata": dict,
    }
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

    def vet(self, output_payload: str | dict[str, Any]) -> VettingResult:
        payload = self._coerce_payload(output_payload)
        self._validate_schema(payload)
        self._enforce_non_null_asset_paths(payload)

        repaired_text, repairs = self._sanitize_markdown(payload["text_response"])
        payload["text_response"] = repaired_text

        return VettingResult(
            payload=payload,
            healed_markdown=bool(repairs),
            repairs=repairs,
        )

    def _coerce_payload(self, output_payload: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(output_payload, dict):
            return dict(output_payload)

        if isinstance(output_payload, str):
            try:
                parsed = json.loads(output_payload)
            except json.JSONDecodeError as exc:
                raise SchemaContractViolation("Output must be a valid JSON dictionary or dict object.") from exc
            if not isinstance(parsed, dict):
                raise SchemaContractViolation("Parsed JSON output must be a dictionary.")
            return parsed

        raise SchemaContractViolation("Output must be a dictionary or JSON dictionary string.")

    def _validate_schema(self, payload: dict[str, Any]) -> None:
        for key, expected_type in self.REQUIRED_SCHEMA.items():
            if key not in payload:
                raise SchemaContractViolation(f"Missing required key: {key}")
            value = payload[key]
            if expected_type is float:
                if not isinstance(value, float):
                    raise SchemaContractViolation(f"Key '{key}' must be a float.")
                continue
            if not isinstance(value, expected_type):
                raise SchemaContractViolation(f"Key '{key}' must be {expected_type.__name__}.")

    def _enforce_non_null_asset_paths(self, payload: dict[str, Any]) -> None:
        for key in list(payload.keys()):
            if key not in self.ASSET_PATH_KEYS:
                continue
            value = payload[key]
            if value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, list) and not value):
                payload.pop(key, None)
                continue
            self._validate_asset_value(key, value)

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in list(metadata.keys()):
                if key in self.ASSET_PATH_KEYS:
                    value = metadata[key]
                    if value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, list) and not value):
                        metadata.pop(key, None)
                        continue
                    self._validate_asset_value(f"metadata.{key}", value)

    def _validate_asset_value(self, key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        if isinstance(value, list):
            if not value:
                return
            for index, item in enumerate(value):
                if item is None or (isinstance(item, str) and not item.strip()):
                    raise EmptyAssetPayloadError(f"Asset field '{key}[{index}]' is empty.")

    def _sanitize_markdown(self, text: str) -> tuple[str, list[str]]:
        repaired = str(text or "")
        repairs: list[str] = []

        if repaired.count("**") % 2 == 1:
            repaired += "**"
            repairs.append("closed_unbalanced_bold_marker")

        if repaired.count("`") % 2 == 1:
            repaired += "`"
            repairs.append("closed_unbalanced_inline_code_marker")

        square_delta = repaired.count("[") - repaired.count("]")
        if square_delta > 0:
            repaired += "]" * square_delta
            repairs.append("closed_unbalanced_square_brackets")

        paren_delta = repaired.count("(") - repaired.count(")")
        if paren_delta > 0:
            repaired += ")" * paren_delta
            repairs.append("closed_unbalanced_parentheses")

        table_repaired = self._repair_markdown_tables(repaired)
        if table_repaired != repaired:
            repaired = table_repaired
            repairs.append("normalized_broken_table_pipes")

        return repaired, repairs

    @staticmethod
    def _repair_markdown_tables(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text

        repaired_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if "|" not in stripped:
                repaired_lines.append(line)
                continue

            # Do not alter ordinary prose with a single inline pipe.
            if stripped.count("|") < 2 and not stripped.startswith("|"):
                repaired_lines.append(line)
                continue

            leading = "" if stripped.startswith("|") else "| "
            trailing = "" if stripped.endswith("|") else " |"
            normalized = f"{leading}{stripped}{trailing}"
            normalized = re.sub(r"\s*\|\s*", " | ", normalized).strip()
            if not normalized.startswith("|"):
                normalized = f"| {normalized}"
            if not normalized.endswith("|"):
                normalized = f"{normalized} |"
            repaired_lines.append(normalized)

        return "\n".join(repaired_lines)


def _print_scorecard(name: str, vetter: StructuralOutputVetter, payload: Any) -> None:
    print(f"\n=== {name} ===")
    try:
        coerced = vetter._coerce_payload(payload)
        print("[Layer 8: Schema Integrity] - PASSED: Payload parsed as dictionary.")
        vetter._validate_schema(coerced)
        print("[Layer 8: Type Assertions] - PASSED: Required fields match strict types.")
        vetter._enforce_non_null_asset_paths(coerced)
        print("[Layer 9: Asset Path Enforcer] - PASSED: Declared asset paths are non-empty.")
        result = vetter.vet(payload)
        if result.healed_markdown:
            print(f"[Layer 10: Markdown Sanitizer] - HEALED: {', '.join(result.repairs)}.")
        else:
            print("[Layer 10: Markdown Sanitizer] - PASSED: No repairs required.")
        print("[Structural Result] - CLEARED FOR API RESPONSE")
        print(f"[Final text_response] - {result.payload['text_response']}")
    except StructuralVettingError as exc:
        print(f"[Structural Result] - BLOCKED: {exc.__class__.__name__}: {exc}")


if __name__ == "__main__":
    vetter = StructuralOutputVetter()

    clean_payload = {
        "text_response": "**Answer:** Table 4.1 is grounded and ready.",
        "confidence_score": 0.94,
        "metadata": {"source_file": "report.pdf", "image_path": "assets/extracted_images/page154_table1.png"},
    }
    bad_type_payload = {
        "text_response": "This payload has the wrong confidence type.",
        "confidence_score": "0.94",
        "metadata": {},
    }
    broken_markdown_payload = {
        "text_response": "**This text is broken",
        "confidence_score": 0.72,
        "metadata": {},
    }

    _print_scorecard("Successful Clean Payload", vetter, clean_payload)
    _print_scorecard("Failure: confidence_score Is String", vetter, bad_type_payload)
    _print_scorecard("Healing: Broken Markdown Bold", vetter, broken_markdown_payload)
