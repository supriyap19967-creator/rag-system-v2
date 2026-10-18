"""
Centralized Vision & Table Extraction Prompts with Automated SHA-256 Fingerprinting.

This module provides deterministic fingerprinting for visual and tabular extraction pipelines.
Whenever a prompt or model specification changes, the SHA-256 fingerprint automatically updates,
guaranteeing automatic cache invalidation and self-healing across:
- SemanticCacheManager (LRU Memory)
- In-memory RAM caches (_GLOBAL_VISION_OCR_CACHE, _IN_MEMORY_TRANSCRIPTION_CACHE)
- Local disk JSON caches (data_cache/transcriptions/*.json)
- Qdrant Cloud point payloads
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VISION_MODEL_NAME: str = "gemini-2.5-flash"

TABLE_EXTRACTION_PROMPT: str = """You are a precision-focused Document Intelligence and Vision-Language Agent specializing in tabular data extraction.

Analyze the provided image. Confirm it contains a table. If it is a standard figure/chart instead of a structured table, state: "Not a table image."

If it IS a table, execute the extraction following these strict rules with 100% accuracy:

1. STRUCTURAL BREAKDOWN:
   - Identify and list all column headers explicitly.
   - Explain the primary axis or category represented by the rows.
   - Describe what the intersection (values) represents in the context of this table.

2. EXHAUSTIVE DATA EXTRACTION:
   - Extract every single row and cell value completely. Do NOT truncate, summarize, or skip any rows or columns.
   - If a cell is genuinely blank or empty in the image, represent it explicitly as "[Empty]" rather than guessing, hallucinating, or outputting "N/A" arbitrarily.
   - Preserve exact numbers, units, decimal points, and text strings without altering formatting.

3. OUTPUT FORMAT:
   - Provide a clear structural explanation of the columns and rows first.
   - Present the extracted table data cleanly using a Markdown table ensuring zero data discrepancy compared to the source image."""

FIGURE_EXTRACTION_PROMPT: str = """### SYSTEM PROMPT: Complete Multi-Panel Visual Extraction & Analytical Summary Engine

You are a precise data extraction system. For any given image/figure asset, you MUST strictly output a single structured payload with the following mandatory sections:

0. MANDATORY DISAMBIGUATION RULE: Always strictly separate single sub-groups (e.g. Low Income Aggregate) from combined aggregate groups (e.g. Low & Middle Income Aggregate). Do not report combined aggregate limits as single sub-group values.

1. SECTION 1: MANDATORY TEXT SUMMARY & TREND ANALYSIS
   - Executive Overview (Title, Figure/Table ID, Core Objective)
   - Key Trends Narrative (Itemized key findings/groups, sub-panel trend analysis)
   - Conclusion (1-2 sentences summarizing core analytical takeaways)

2. SECTION 2: EXHAUSTIVE MULTI-PANEL DATA EXTRACTION TABLE
   - Format: Clean Markdown Table with populated headers and populated rows ONLY. No trailing empty pipes ('||' or '| |').
   - Columns: Mandatory mapping of [Panel / Sub-Chart, Category / Label, Data Value, Units, Conditions / Errors].
   - Rule: Do NOT return fallback, dummy, or mockup data (e.g., 'Series 1', 'Category 1', 'Unspecified Series 1', '10', '20'). If specific quantitative coordinates are ambiguous or dense (e.g., scatter plots), extract visible trendline boundaries, key outlier points, or explicit sample coordinates directly from the axes.

3. SECTION 3: SOURCE TRAIL
   - Pipeline source metadata string (e.g., Figure ID, Page number, Document source)."""


def get_pipeline_fingerprint(asset_kind: str) -> str:
    """
    Computes a deterministic 16-character SHA-256 fingerprint for the extraction pipeline.
    If the prompt, model, or instructions change, this fingerprint changes immediately,
    invalidating stale cache entries automatically without requiring manual clearing.
    """
    kind = "table" if "tab" in str(asset_kind).lower() else "figure"
    base_prompt = TABLE_EXTRACTION_PROMPT if kind == "table" else FIGURE_EXTRACTION_PROMPT
    fingerprint_seed = f"{kind}:{VISION_MODEL_NAME}:{base_prompt.strip()}"
    return hashlib.sha256(fingerprint_seed.encode("utf-8")).hexdigest()[:16]


def get_extraction_prompt(asset_kind: str, target_cat: str = "", target_id: str = "") -> str:
    """
    Returns the specialized prompt corresponding to the asset category.
    Tables receive the precision Document Intelligence tabular prompt.
    Figures receive the multi-panel trend and chart extraction prompt.
    """
    kind = "table" if "tab" in str(asset_kind).lower() or (target_cat and "tab" in str(target_cat).lower()) else "figure"
    if kind == "table":
        header_context = f"Target table: {target_cat} {target_id}\n\n" if (target_cat and target_id) else ""
        return f"{header_context}{TABLE_EXTRACTION_PROMPT}"
    else:
        header_context = f"Target figure: {target_cat} {target_id}\n\n" if (target_cat and target_id) else ""
        return f"{header_context}{FIGURE_EXTRACTION_PROMPT}"


def is_cache_fingerprint_valid(cached_entry: Any, asset_kind: str) -> bool:
    """
    Validates whether a cached payload (from RAM, disk, or Qdrant) matches the active
    pipeline fingerprint.
    
    - For tables: STRICT validation. If fingerprint is missing or outdated, returns False.
    - For figures: If fingerprint is present, it must match. If legacy without fingerprint,
      accepts only if it contains a valid multi-panel markdown table.
    """
    if not isinstance(cached_entry, dict):
        return False

    kind = "table" if "tab" in str(asset_kind).lower() else "figure"
    current_fp = get_pipeline_fingerprint(kind)
    cached_fp = cached_entry.get("fingerprint") or cached_entry.get("prompt_hash")

    if kind == "table":
        # Strict validation for tables: must match active fingerprint
        if not cached_fp or cached_fp != current_fp:
            return False
        # Also ensure content is not empty or refusal
        text = cached_entry.get("markdown_table") or cached_entry.get("content") or ""
        if not text or len(text.strip()) < 15:
            return False
        return True
    else:
        # For figures: if fingerprint exists, enforce match
        if cached_fp:
            return cached_fp == current_fp
        # Legacy figure entries without fingerprint
        text = cached_entry.get("markdown_table") or cached_entry.get("content") or ""
        return bool(text and "|" in text and not any(err in text.lower() for err in ["i'm sorry", "cannot extract", "no data", "too blurry"]))
