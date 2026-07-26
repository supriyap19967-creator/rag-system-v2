from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from ingestion.config import IngestionSettings
from ingestion.schemas import ExtractedImage, VisionDescription


load_dotenv()

logger = logging.getLogger(__name__)

PROMPT_VERSION = "gemini-2.5-flash-v2"
DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_CACHE_DIR = Path("./data_cache/visual_captions")

CHART_PROMPT = """You are a data analyst extracting structured information from a chart image for RAG retrieval.

Analyze the chart and extract ALL of the following that are visible:

1. CHART TYPE: (bar / line / pie / scatter / area / histogram / combo / other)
2. TITLE: Exact chart title if present
3. X-AXIS: Label name + unit + value range
4. Y-AXIS: Label name + unit + value range
5. LEGEND: All series/category names exactly as shown
6. DATA POINTS: Every visible data value (numbers, percentages, dates)
7. TRENDS: Direction of change (rising/falling/stable/cyclical)
8. PEAK/TROUGH: Highest and lowest values with their labels
9. COMPARISONS: Key differences between series or categories
10. ANNOTATIONS: Any callouts, footnotes, or source labels visible

OUTPUT FORMAT:
Write one dense retrieval-optimized paragraph followed by a compact data table of key values.
Ground every claim in what is visibly shown. Do NOT invent values.
"""

TABLE_PROMPT = """You are a data extraction expert reading a table image for RAG retrieval.

Extract ALL of the following from the table:

1. TABLE TITLE: Exact title if present
2. COLUMN HEADERS: Every column name exactly as shown
3. ROW LABELS: Every row identifier or index label
4. ALL CELL VALUES: Read every cell value systematically row by row
5. UNITS: Any units shown (%, $, kg, years, etc.)
6. FOOTNOTES: Any notes, asterisks, or source references at the bottom
7. TOTALS/SUBTOTALS: Any summary rows or columns

OUTPUT FORMAT:
First reproduce the table in plain text using | column | separators.
Then write a dense paragraph summarizing the key insights and notable values.
Preserve all numbers exactly. Do NOT skip cells or invent values.
"""

DIAGRAM_PROMPT = """You are a technical analyst extracting information from a diagram image for RAG retrieval.

Analyze the diagram and extract ALL of the following:

1. DIAGRAM TYPE: (flowchart / architecture / process flow / network / org chart / concept map / other)
2. TITLE: Exact diagram title if present
3. COMPONENTS: Every labeled box, circle, node, or entity — with exact labels
4. CONNECTIONS: Every arrow or line — describe source → destination and any label on the connection
5. FLOW DIRECTION: Overall direction (top-down / left-right / circular / branching)
6. DECISION POINTS: Any diamond shapes or conditional branches with their conditions
7. GROUPINGS: Any regions, swim lanes, or enclosing boundaries with their labels
8. ANNOTATIONS: Any callout text, legends, or footnotes visible

OUTPUT FORMAT:
Write a dense structured description starting with diagram type, then components, then connections as a list, 
then a summary paragraph. Every label must be quoted exactly as shown. Do NOT invent components.
"""

FIGURE_PROMPT = """You are analyzing an image or figure from a document for RAG retrieval.

Extract ALL of the following that are visible:

1. FIGURE TYPE: (photograph / illustration / screenshot / map / schematic / other)
2. TITLE / CAPTION: Any label or caption text shown on or near the image
3. MAIN SUBJECT: What is the primary subject or focus of the image
4. KEY ELEMENTS: All visible labeled or notable elements
5. TEXT IN IMAGE: Any text, numbers, or labels embedded in the image — transcribe exactly
6. SPATIAL LAYOUT: Describe relative positions of key elements (left/right/top/bottom/center)
7. COLOR CODING: Any color-based information conveying meaning
8. SOURCE / COPYRIGHT: Any source or attribution text visible

OUTPUT FORMAT:
Write a retrieval-optimized paragraph describing this figure for document search.
Be factual and specific. Quote all visible text exactly. Do NOT invent content.
"""

UNIVERSAL_VISUAL_PROMPT = """You are a multimodal document analyst extracting information from a visual element for RAG retrieval.

Step 1 — Identify the visual type:
Choose one: chart | table | diagram | figure | map | schematic | screenshot | other

Step 2 — Extract based on type:

IF CHART: Extract chart type, axes, all data values, legends, trends, peak/trough values.
IF TABLE: Reproduce all headers and cell values. Write as plain-text table then summarize.
IF DIAGRAM: List all labeled components and connections. Describe flow and structure.
IF FIGURE/IMAGE: Describe main subject, all visible text, spatial layout, key elements.

RULES:
- Quote all visible text and numbers exactly as shown
- Do NOT hallucinate or invent values not visible in the image
- Be information-dense and retrieval-optimized
- Include every number, label, percentage, and unit you can see
- Output must be self-contained (no references like "as shown above")
"""


def _select_prompt(image_type: str | None) -> str:
    if not image_type:
        return UNIVERSAL_VISUAL_PROMPT
    normalized = str(image_type).lower().strip()
    if "chart" in normalized or "graph" in normalized:
        return CHART_PROMPT
    elif "table" in normalized:
        return TABLE_PROMPT
    elif "diagram" in normalized or "flow" in normalized or "network" in normalized:
        return DIAGRAM_PROMPT
    elif "figure" in normalized or "image" in normalized or "photo" in normalized:
        return FIGURE_PROMPT
    return UNIVERSAL_VISUAL_PROMPT


@dataclass(frozen=True, slots=True)
class CaptionCacheEntry:
    image_path: str
    image_hash: str
    model: str
    created_at: str
    prompt_version: str
    caption: str


def _resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    configured = cache_dir or os.getenv("CACHE_DIR") or DEFAULT_CACHE_DIR
    return Path(configured).expanduser().resolve()


def image_md5(image_path: str | Path) -> str:
    """Return a deterministic MD5 hash of an image's raw bytes."""

    path = Path(image_path)
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image_bytes(image_path: Path) -> bytes:
    return image_path.read_bytes()


def _mime_type(image_path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(str(image_path))
    return guessed or "image/png"


def _cache_path(cache_dir: Path, image_hash: str) -> Path:
    return cache_dir / f"{image_hash}.json"


def _load_cache(cache_file: Path) -> CaptionCacheEntry | None:
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        caption = str(data.get("caption") or "").strip()
        if not caption:
            logger.warning("Ignoring caption cache with empty caption: %s", cache_file)
            return None
        prompt_version = str(data.get("prompt_version") or "")
        if prompt_version != PROMPT_VERSION:
            logger.info("Ignoring caption cache due to prompt version mismatch: got %s, expected %s", prompt_version, PROMPT_VERSION)
            return None
        return CaptionCacheEntry(
            image_path=str(data.get("image_path") or ""),
            image_hash=str(data.get("image_hash") or cache_file.stem),
            model=str(data.get("model") or DEFAULT_MODEL_NAME),
            created_at=str(data.get("created_at") or ""),
            prompt_version=prompt_version,
            caption=caption,
        )
    except Exception as exc:
        logger.warning("Could not read caption cache %s: %s", cache_file, exc)
        return None


def _atomic_write_json(cache_file: Path, payload: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    temp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_file, cache_file)


def _wrap_chart_description(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("[CHART DESCRIPTION]"):
        return stripped
    return f"[CHART DESCRIPTION]\n{stripped}\n[/CHART DESCRIPTION]"


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    response_json = getattr(exc, "response_json", None)
    details = []
    if isinstance(response_json, dict):
        details = response_json.get("error", {}).get("details", []) or response_json.get("details", [])
    for detail in details:
        retry_delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if not retry_delay:
            continue
        match = re.match(r"^(\d+(?:\.\d+)?)s$", str(retry_delay))
        if match:
            return float(match.group(1))
    message = str(exc)
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", message, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _is_rate_limit_error(exc: Exception) -> bool:
    status = _status_code(exc)
    message = str(exc).lower()
    return status == 429 or "resourceexhausted" in message or "resource exhausted" in message or "quota" in message


def _is_hard_quota_exhausted(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "generate_content_free_tier_requests" in message
        or "generaterequestsperday" in message
        or "limit: 0" in message
        or "please check your plan and billing details" in message
    )


def _is_transient_error(exc: Exception) -> bool:
    status = _status_code(exc)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(term in message for term in ("timeout", "temporarily unavailable", "connection", "rate limit"))


class GeminiApiKeyRotator:
    """Thread-safe API key selector with rotation on quota/rate-limit failures."""

    def __init__(self, keys: Iterable[str]) -> None:
        self._keys = [key.strip() for key in keys if key and key.strip()]
        if not self._keys:
            raise RuntimeError("Set GEMINI_API_KEYS, GEMINI_API_KEY, or GOOGLE_API_KEY before vision ingestion.")
        random.shuffle(self._keys)
        self._index = 0
        self._lock = threading.Lock()

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def current(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def rotate(self) -> str:
        with self._lock:
            self._index = (self._index + 1) % len(self._keys)
            logger.info("Rotated Gemini API key; active key index is %s of %s", self._index + 1, len(self._keys))
            return self._keys[self._index]


def _load_api_keys(settings: IngestionSettings | None = None) -> list[str]:
    raw_keys = os.getenv("GEMINI_API_KEYS", "")
    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
    fallback = ""
    if settings is not None:
        fallback = settings.gemini_api_key
    fallback = fallback or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    if fallback and fallback not in keys:
        keys.append(fallback)
    return keys


class GeminiVisionCaptioner:
    """Cache-aware Gemini 2.0 Flash Vision captioner for multimodal RAG ingestion."""

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        *,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
        max_retries: int = 5,
        max_concurrent_requests: int | None = None,
    ) -> None:
        self.settings = settings or IngestionSettings()
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME") or self.settings.gemini_vision_model or DEFAULT_MODEL_NAME
        self.max_retries = max(1, max_retries)
        self.max_concurrent_requests = max_concurrent_requests or self.settings.max_concurrent_requests
        self._key_rotator: GeminiApiKeyRotator | None = None
        self._clients: dict[str, Any] = {}
        self._clients_lock = threading.Lock()
        self._hard_quota_exhausted = False
        logger.info(
            "GeminiVisionCaptioner initialized with model=%s, cache_dir=%s, max_concurrent=%s",
            self.model_name,
            self.cache_dir,
            self.max_concurrent_requests,
        )

    def _rotator(self) -> GeminiApiKeyRotator:
        if self._key_rotator is None:
            self._key_rotator = GeminiApiKeyRotator(_load_api_keys(self.settings))
            logger.info("Loaded %s Gemini API key(s) for cache-miss captioning", self._key_rotator.key_count)
        return self._key_rotator

    def _client_for_key(self, api_key: str):
        with self._clients_lock:
            if api_key in self._clients:
                return self._clients[api_key]
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("Install google-genai to enable Gemini Vision captioning.") from exc

            client = genai.Client(api_key=api_key)
            self._clients[api_key] = client
            return client

    def warmup(self) -> None:
        """No-op warmup for Gemini Vision (API-based, no local models needed)."""
        logger.info("Gemini Vision Captioner is API-based; warmup is a no-op.")

    def _generate_caption(self, image_path: Path, image_bytes: bytes, prompt: str) -> str:
        from google.genai import types

        if self._hard_quota_exhausted:
            raise RuntimeError("Gemini hard quota exhausted for this run; skipping uncached visual captions.")

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=_mime_type(image_path))

        soft_retry_count = 0      # counts soft-429 retries  — unlimited
        transient_retry_count = 0  # counts other transient retries — limited to max_retries

        while True:
            api_key = self._rotator().current()
            client = self._client_for_key(api_key)
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image_part],
                )
                caption = str(getattr(response, "text", "") or "").strip()
                if not caption:
                    raise RuntimeError("Gemini returned an empty caption.")
                return caption

            except Exception as exc:
                if _is_rate_limit_error(exc):
                    # Hard daily quota — stop immediately, skip this image
                    if _is_hard_quota_exhausted(exc):
                        self._hard_quota_exhausted = True
                        logger.warning(
                            "Gemini hard daily quota exhausted while captioning %s; "
                            "remaining uncached visuals will be skipped for this run.",
                            image_path,
                        )
                        raise

                    # Soft 429 (per-minute / per-second rate-limit) — retry forever
                    soft_retry_count += 1
                    suggested = _extract_retry_delay_seconds(exc)
                    wait_seconds = suggested if suggested is not None else min(300.0, 2.0 ** soft_retry_count)
                    logger.warning(
                        "Gemini soft rate-limit (429) for %s — retry #%s, "
                        "waiting %.1f s before next attempt (exponential backoff: 2^n).",
                        image_path,
                        soft_retry_count,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)

                elif _is_transient_error(exc):
                    # Network / server error — limited retries
                    transient_retry_count += 1
                    if transient_retry_count >= self.max_retries:
                        raise RuntimeError(
                            f"Gemini transient error for {image_path} after "
                            f"{transient_retry_count} retries: {exc}"
                        ) from exc
                    wait_seconds = min(60.0, 2.0 ** transient_retry_count)
                    logger.warning(
                        "Transient Gemini error for %s — retry #%s/%s, waiting %.1f s: %s",
                        image_path,
                        transient_retry_count,
                        self.max_retries,
                        wait_seconds,
                        exc,
                    )
                    time.sleep(wait_seconds)

                else:
                    # Non-retryable error — raise immediately
                    raise

    def caption_image_path(self, image_path: str | Path, image_type: str = "figure") -> str | None:
        """Return a wrapped chart description, using cache before any API call."""

        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            logger.warning("Skipping missing image: %s", path)
            return None
        if not path.is_file():
            logger.warning("Skipping non-file image path: %s", path)
            return None

        try:
            image_hash = image_md5(path)
            cache_file = _cache_path(self.cache_dir, image_hash)
            cached = _load_cache(cache_file)
            if cached:
                logger.info("CACHE HIT visual caption: %s", path)
                return _wrap_chart_description(cached.caption)

            if self._hard_quota_exhausted:
                logger.warning("Skipping uncached visual caption because Gemini hard quota is exhausted: %s", path)
                return None

            logger.info("CACHE MISS visual caption: %s", path)
            image_bytes = _read_image_bytes(path)
            prompt = _select_prompt(image_type)
            caption = self._generate_caption(path, image_bytes, prompt=prompt)
            payload = CaptionCacheEntry(
                image_path=str(path),
                image_hash=image_hash,
                model=self.model_name,
                created_at=datetime.now(timezone.utc).isoformat(),
                prompt_version=PROMPT_VERSION,
                caption=caption,
            )
            _atomic_write_json(cache_file, payload.__dict__)
            logger.info("Cached Gemini visual caption for %s at %s", path, cache_file)
            return _wrap_chart_description(caption)
        except Exception as exc:
            logger.warning("Skipping image after Gemini captioning failure for %s: %s", path, exc)
            return None

    def describe_image(self, image: ExtractedImage) -> VisionDescription | None:
        description = self.caption_image_path(image.image_path, image_type=image.type)
        if not description:
            return None

        return VisionDescription(
            image_path=Path(image.image_path),
            page=image.page,
            type=image.type if image.type in {"chart", "diagram", "figure", "image"} else "figure",
            description=description,
            metadata={
                "source": image.source_path,
                "image_path": str(image.image_path),
                "element_id": image.element_id,
                "vision_model": self.model_name,
                "caption_cache_dir": str(self.cache_dir),
                **image.metadata,
            },
        )

    async def describe_images(self, images: Iterable[ExtractedImage]) -> list[VisionDescription]:
        semaphore = asyncio.Semaphore(max(1, int(self.max_concurrent_requests)))

        async def run_one(image: ExtractedImage) -> VisionDescription | None:
            async with semaphore:
                try:
                    return await asyncio.to_thread(self.describe_image, image)
                except Exception as exc:
                    logger.warning("Gemini vision captioning failed for %s: %s", image.image_path, exc)
                    return None

        results = await asyncio.gather(*(run_one(image) for image in images))
        return [result for result in results if result is not None]
