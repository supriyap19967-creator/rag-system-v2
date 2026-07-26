from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ingestion.config import IngestionSettings
from ingestion.model_loading import (
    LoadingProgressReporter,
    download_model_shards,
    ensure_cuda_available,
    resolve_cached_snapshot_path,
)
from ingestion.paddle_ocr import PaddleOcrExtractor
from ingestion.schemas import ExtractedImage, VisionDescription


logger = logging.getLogger(__name__)

# Bump this version any time prompts change so old cache entries are discarded.
PROMPT_VERSION = "paddleocr-qwen2.5-vl-3b-awq-v3"

# ---------------------------------------------------------------------------
# Type-aware Qwen visual analysis prompts
# ---------------------------------------------------------------------------

QWEN_CHART_PROMPT = """You are a data analyst extracting structured information from a chart image for RAG retrieval.
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
10. ANNOTATIONS: Any callouts, footnotes, or source labels
Use the OCR literals provided to ground all numeric values exactly."""

QWEN_TABLE_PROMPT = """You are a data extraction specialist analyzing a table image for RAG retrieval.
Extract ALL of the following:
1. TABLE TITLE: Exact title if present
2. COLUMN HEADERS: Every column name exactly as shown
3. ROW LABELS: Every row label or index
4. ALL DATA CELLS: Every value in every cell (numbers, text, symbols)
5. TOTALS/SUBTOTALS: Any summary rows or columns
6. FOOTNOTES: Any notes, source references, or methodology text below the table
7. UNITS: Units of measurement for each column
8. MISSING VALUES: Note any blank or N/A cells
Use the OCR literals to confirm exact numbers and spellings."""

QWEN_DIAGRAM_PROMPT = """You are a technical analyst describing a diagram or flowchart for RAG retrieval.
Extract ALL of the following:
1. DIAGRAM TYPE: (flowchart / org chart / process diagram / concept map / network / other)
2. TITLE: Exact title if present
3. NODES/BOXES: Every labeled element and its text
4. CONNECTIONS/ARROWS: All relationships between elements (direction + label if any)
5. FLOW/SEQUENCE: The logical sequence or hierarchy
6. LEGEND: Any legend or key explaining symbols/colors
7. ANNOTATIONS: Any callouts, labels, or explanatory text
8. SOURCE/FOOTNOTE: Any source references"""

QWEN_FIGURE_PROMPT = """You are an analyst describing a visual figure or image for RAG retrieval.
Extract ALL of the following:
1. FIGURE TYPE: What kind of visual is this (map, photograph, illustration, mixed, etc.)
2. TITLE/CAPTION: Exact title or caption text if present
3. MAIN SUBJECT: What is the primary subject or content shown
4. KEY ELEMENTS: All labeled regions, callouts, or highlighted areas
5. TEXT/NUMBERS: Every piece of text visible in the image
6. LEGEND: Any legend or color key
7. SPATIAL RELATIONSHIPS: How elements are arranged relative to each other
8. SOURCE/FOOTNOTE: Any source references or attribution text"""

QWEN_UNIVERSAL_PROMPT = """You are an expert data analyst. Use the attached image and the raw OCR text below.
Write a dense structured description covering:
- Visual type and title
- All data values, labels, and text visible in the image
- Trends, relationships, and key insights
- Any source, footnote, or methodology notes
Ground every statement in what is visible. Preserve exact numbers and labels from OCR literals."""

_PROMPT_BY_TYPE: dict[str, str] = {
    "chart": QWEN_CHART_PROMPT,
    "table": QWEN_TABLE_PROMPT,
    "diagram": QWEN_DIAGRAM_PROMPT,
    "figure": QWEN_FIGURE_PROMPT,
    "image": QWEN_FIGURE_PROMPT,
    "map": QWEN_DIAGRAM_PROMPT,
}


def _select_qwen_prompt(image_type: str) -> str:
    """Return the most appropriate Qwen analysis prompt for the given visual type."""
    return _PROMPT_BY_TYPE.get(str(image_type or "").lower(), QWEN_UNIVERSAL_PROMPT)


@dataclass(frozen=True, slots=True)
class QwenCacheEntry:
    image_path: str
    image_hash: str
    model: str
    created_at: str
    prompt_version: str
    description: str
    ocr_text: str = ""
    ocr_engine: str = "paddle"


def _image_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(cache_dir: Path, image_hash: str) -> Path:
    return cache_dir / f"{image_hash}.json"


def _load_cache(cache_file: Path) -> QwenCacheEntry | None:
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read Qwen cache %s: %s", cache_file, exc)
        return None
    description = str(payload.get("description") or "").strip()
    prompt_version = str(payload.get("prompt_version") or "")
    if not description or prompt_version != PROMPT_VERSION:
        return None
    return QwenCacheEntry(
        image_path=str(payload.get("image_path") or ""),
        image_hash=str(payload.get("image_hash") or cache_file.stem),
        model=str(payload.get("model") or ""),
        created_at=str(payload.get("created_at") or ""),
        prompt_version=prompt_version,
        description=description,
        ocr_text=str(payload.get("ocr_text") or ""),
        ocr_engine=str(payload.get("ocr_engine") or "paddle"),
    )


def _atomic_write_json(cache_file: Path, payload: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_file, cache_file)


class QwenVisionCaptioner:
    """PaddleOCR (CPU) + Qwen2.5-VL-3B AWQ (GPU) visual enrichment for charts and figures."""

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        *,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
        prompt: str = QWEN_UNIVERSAL_PROMPT,
        max_new_tokens: int = 768,
        max_concurrent_requests: int | None = None,
    ) -> None:
        self.settings = settings or IngestionSettings()
        self.cache_dir = Path(cache_dir or self.settings.caption_cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or self.settings.qwen_vl_model_id
        self.prompt = prompt  # default; overridden per-image by _select_qwen_prompt
        self.max_new_tokens = max_new_tokens
        self.max_concurrent_requests = 1  # Force serialization to prevent CUDA lockups on limited GPUs
        self._ocr = PaddleOcrExtractor(self.settings, cache_dir=self.cache_dir)
        self._model = None
        self._processor = None
        self._device = "cuda"
        import threading
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()
        logger.info(
            "QwenVisionCaptioner initialized with model=%s, cache_dir=%s, max_concurrent=%s",
            self.model_name,
            self.cache_dir,
            self.max_concurrent_requests,
        )

    def warmup(self) -> None:
        """Load PaddleOCR and Qwen AWQ with phased percentage progress."""
        reporter = LoadingProgressReporter("Loading")
        reporter.update(0.0, "Starting vision model warmup")

        reporter.phase(0.0, 5.0, "Checking CUDA availability", ensure_cuda_available)

        reporter.phase(5.0, 20.0, "Initializing PaddleOCR on CPU", self._ocr.warmup)

        self._ensure_model(reporter=reporter)
        reporter.update(100.0, "Vision models ready (PaddleOCR=CPU, Qwen AWQ=GPU)")

    def _ensure_model(self, reporter: LoadingProgressReporter | None = None) -> tuple[Any, Any]:
        with self._lock:
            return self._ensure_model_unlocked(reporter)

    def _ensure_model_unlocked(self, reporter: LoadingProgressReporter | None = None) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        progress = reporter or LoadingProgressReporter("Loading")

        hf_home = self.settings.workspace_hf_home.resolve()
        hf_cache = self.settings.workspace_hf_cache.resolve()
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
        hf_home.mkdir(parents=True, exist_ok=True)
        hf_cache.mkdir(parents=True, exist_ok=True)

        if reporter is None:
            progress.phase(0.0, 5.0, "Checking CUDA availability", ensure_cuda_available)
        else:
            ensure_cuda_available()

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_source = resolve_cached_snapshot_path(self.model_name, hf_cache) or self.model_name

        def _load_processor() -> None:
            self._processor = AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=True,
            )

        progress.phase(20.0, 30.0, "Loading Qwen processor", _load_processor)

        download_model_shards(
            self.model_name,
            hf_cache,
            progress,
            start_pct=30.0,
            end_pct=70.0,
        )

        def _load_model() -> None:
            import torch as _torch
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_source,
                torch_dtype=_torch.bfloat16,  # bfloat16 avoids rshift_cuda/Half errors
                device_map={"": 0},
                local_files_only=True,
                trust_remote_code=True,
            )
            self._model.eval()

            # -------------------------------------------------------------------
            # AWQ dtype bridge
            # Find and hook ALL AWQ quantized linear layers so their float16
            # output (from the AWQ CUDA kernel) is immediately upcast to
            # bfloat16 before the next layer sees it.
            # We search by class module path ('awq' package) AND class name.
            # -------------------------------------------------------------------
            def _awq_bf16_hook(module: Any, inp: Any, out: Any) -> Any:
                if isinstance(out, _torch.Tensor) and out.dtype == _torch.float16:
                    return out.to(_torch.bfloat16)
                if isinstance(out, (tuple, list)):
                    casted = [
                        v.to(_torch.bfloat16)
                        if isinstance(v, _torch.Tensor) and v.dtype == _torch.float16
                        else v
                        for v in out
                    ]
                    return type(out)(casted)
                return out

            # Diagnostic: print unique non-standard module types so we can
            # confirm which class names the AWQ layers use.
            seen_types: set[str] = set()
            for _m in self._model.modules():
                _cls = f"{_m.__class__.__module__}.{_m.__class__.__name__}"
                if (
                    "awq" in _cls.lower()
                    or "wqlinear" in _cls.lower()
                    or "quantlinear" in _cls.lower()
                ):
                    seen_types.add(_cls)
            if seen_types:
                logger.info("AWQ layer class names found in model: %s", seen_types)
            else:
                logger.warning(
                    "No AWQ layer class names found by name search; "
                    "falling back to tensor-attribute scan."
                )
                # Fallback: log any module that holds non-standard tensor attrs
                _sample_attrs: dict[str, list[str]] = {}
                for _name, _m in list(self._model.named_modules())[:200]:
                    _tensor_attrs = [
                        a for a in vars(_m)
                        if isinstance(getattr(_m, a, None), _torch.Tensor)
                        and a not in ("weight", "bias")
                    ]
                    if _tensor_attrs:
                        _key = _m.__class__.__name__
                        _sample_attrs.setdefault(_key, _tensor_attrs[:4])
                logger.info("Modules with extra tensor attrs (sample): %s", _sample_attrs)

            # Register hook on every module whose class comes from the awq package
            # OR whose class name looks like an AWQ quantized linear.
            _AWQ_KEYWORDS = ("awq", "wqlinear", "quantlinear", "gptq")
            patched = 0
            for module in self._model.modules():
                _cls_path = (
                    module.__class__.__module__ + "." + module.__class__.__name__
                ).lower()
                if any(kw in _cls_path for kw in _AWQ_KEYWORDS):
                    module.register_forward_hook(_awq_bf16_hook)
                    patched += 1
            # Log all module classes for diagnostics
            all_classes = {m.__class__.__module__ + "." + m.__class__.__name__ for m in self._model.modules()}
            logger.info("ALL MODULE CLASSES IN MODEL: %s", sorted(all_classes))

            if patched:
                logger.info(
                    "Registered bfloat16 bridge hooks on %d AWQ/quantized layers",
                    patched,
                )
            else:
                logger.warning(
                    "Still found 0 quantized layers to hook — dtype mismatch may persist."
                )



        progress.phase(70.0, 100.0, "Loading Qwen2.5-VL-3B AWQ on GPU", _load_model)
        return self._model, self._processor

    def _generate_description(self, image: ExtractedImage) -> tuple[str, str]:
        image_path = Path(image.image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Visual asset not found for Qwen parsing: {image_path}")

        raw_ocr = self._ocr.extract_text(image_path)
        return self._generate_description_with_context(
            image,
            raw_ocr_literals=raw_ocr,
            nearby_context_paragraphs="",
        )

    def _generate_description_with_context(
        self,
        image: ExtractedImage,
        *,
        raw_ocr_literals: str,
        nearby_context_paragraphs: str,
    ) -> tuple[str, str]:
        image_path = Path(image.image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Visual asset not found for Qwen parsing: {image_path}")

        model, processor = self._ensure_model()
        pil_image = Image.open(image_path).convert("RGB")
        # Pick the right prompt for this image type; fall back to self.prompt if no type set.
        image_type = str(getattr(image, "type", "") or "").lower()
        selected_prompt = _select_qwen_prompt(image_type) if image_type else self.prompt

        user_text = (
            f"{selected_prompt}\n\n"
            f"Nearby page context:\n{nearby_context_paragraphs.strip() or '(no nearby page context detected)'}\n\n"
            f"Raw OCR literals (PaddleOCR — axis labels, tick values, legend items, numbers):\n"
            f"{raw_ocr_literals.strip() or '(no OCR literals detected)'}\n\n"
            "Produce a complete structured analysis that preserves all numbers, labels, "
            "legend text, and source/footer references visible in the image."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        # Cast all floating-point inputs to bfloat16 to match the model's dtype.
        # (The processor may output float32 tensors; pixel_values especially.)
        inputs = {
            key: value.to(__import__("torch").bfloat16)
            if isinstance(value, __import__("torch").Tensor) and value.is_floating_point()
            else value
            for key, value in inputs.items()
        }

        with self._generation_lock:
            with __import__("torch").inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output.strip(), raw_ocr_literals

    def analyze_image(
        self,
        image: ExtractedImage,
        *,
        raw_ocr_literals: str = "",
        nearby_context_paragraphs: str = "",
    ) -> tuple[str, str]:
        image_path = Path(image.image_path).expanduser().resolve()
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Visual asset not found for Qwen parsing: {image_path}")
        if not raw_ocr_literals.strip():
            raw_ocr_literals = self._ocr.extract_text(image_path)
        return self._generate_description_with_context(
            image,
            raw_ocr_literals=raw_ocr_literals,
            nearby_context_paragraphs=nearby_context_paragraphs,
        )

    def describe_image(self, image: ExtractedImage) -> VisionDescription | None:
        """Describe a single image (no surrounding context). Used by pipeline.py."""
        image_path = Path(image.image_path).expanduser().resolve()
        if not image_path.exists() or not image_path.is_file():
            logger.warning("Skipping missing Qwen image asset: %s", image_path)
            return None

        image_hash = _image_hash(image_path)
        cache_file = _cache_path(self.cache_dir, image_hash)
        cached = _load_cache(cache_file)
        if cached:
            description = cached.description
            ocr_text = cached.ocr_text
        else:
            description, ocr_text = self._generate_description(image)
            payload = QwenCacheEntry(
                image_path=str(image_path),
                image_hash=image_hash,
                model=self.model_name,
                created_at=datetime.now(timezone.utc).isoformat(),
                prompt_version=PROMPT_VERSION,
                description=description,
                ocr_text=ocr_text,
                ocr_engine="paddle",
            )
            _atomic_write_json(cache_file, asdict(payload))

        return VisionDescription(
            image_path=image_path,
            page=image.page,
            type=image.type if image.type in {"chart", "diagram", "figure", "image"} else "figure",
            description=description,
            metadata={
                "source": image.source_path,
                "image_path": str(image_path),
                "element_id": image.element_id,
                "vision_model": self.model_name,
                "ocr_engine": "paddle",
                "ocr_text": ocr_text,
                "caption_cache_dir": str(self.cache_dir),
                **image.metadata,
            },
        )

    def describe_image_with_context(
        self,
        image: ExtractedImage,
        *,
        context_before: str = "",
        context_after: str = "",
    ) -> tuple[str, str] | None:
        """Describe a single image with surrounding PDF paragraph context.

        Returns (qwen_description, ocr_text) or None if the image file is missing.
        Results are cached keyed by (image_hash + context_hash) so changing the
        surrounding context invalidates the entry automatically.
        """
        image_path = Path(image.image_path).expanduser().resolve()
        if not image_path.exists() or not image_path.is_file():
            logger.warning("Skipping missing Qwen image asset: %s", image_path)
            return None

        # Minimum image size guard — Qwen's visual processor requires images > 28×28 px.
        # Skip tiny icons/decorations that would cause a 'must be larger than factor:28' error.
        try:
            from PIL import Image as _PIL_Image
            _w, _h = _PIL_Image.open(image_path).size
            if _w < 32 or _h < 32:
                logger.warning(
                    "Skipping too-small image (%dx%d px, min 32x32): %s",
                    _w, _h, image_path.name,
                )
                return None
        except Exception as _size_exc:
            logger.debug("Could not check image dimensions for %s: %s", image_path.name, _size_exc)

        # PaddleOCR always has its own sub-cache; this call is cheap on repeat.
        ocr_text = self._ocr.extract_text(image_path)

        nearby = "\n\n".join(part.strip() for part in [context_before, context_after] if part.strip())
        context_hash = hashlib.md5(nearby.encode("utf-8")).hexdigest()[:8]
        image_hash = _image_hash(image_path)
        cache_file = _cache_path(self.cache_dir, f"{image_hash}_{context_hash}")

        cached = _load_cache(cache_file)
        if cached:
            logger.debug("Cache hit (with context) for %s", image_path.name)
            return cached.description, cached.ocr_text

        qwen_description, _ = self._generate_description_with_context(
            image,
            raw_ocr_literals=ocr_text,
            nearby_context_paragraphs=nearby,
        )

        payload = QwenCacheEntry(
            image_path=str(image_path),
            image_hash=image_hash,
            model=self.model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            prompt_version=PROMPT_VERSION,
            description=qwen_description,
            ocr_text=ocr_text,
            ocr_engine="paddle",
        )
        _atomic_write_json(cache_file, asdict(payload))
        return qwen_description, ocr_text

    async def describe_images(self, images: Iterable[ExtractedImage]) -> list[VisionDescription]:
        semaphore = asyncio.Semaphore(max(1, int(self.max_concurrent_requests)))

        async def run_one(image: ExtractedImage) -> VisionDescription | None:
            async with semaphore:
                try:
                    return await asyncio.to_thread(self.describe_image, image)
                except Exception as exc:
                    import traceback
                    logger.warning("Qwen parsing failed for %s: %s\n%s", image.image_path, exc, traceback.format_exc())
                    return None

        results = await asyncio.gather(*(run_one(image) for image in images))
        return [result for result in results if result is not None]
