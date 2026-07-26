from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from dotenv import load_dotenv

from ingestion.config import IngestionSettings
from ingestion.schemas import ExtractedImage, VisionDescription


load_dotenv()

logger = logging.getLogger(__name__)

PROMPT_VERSION = "florence-v3-ocr-llm-narrative"
GEMINI_VISION_MODEL = os.getenv("GEMINI_MODEL_NAME") or os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
NVIDIA_VISION_MODEL = os.getenv("NVIDIA_LLAMA_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
VISUAL_SYNTHESIS_PROMPT = """You are an expert data analyst. Read the provided raw OCR text dump and look closely at the attached image. Write a highly detailed, comprehensive description of this visual asset in full sentence format.
- Do not summarize or omit data points.
- Transcribe every single chart trend, exact numerical value (like 7.5, 6.2, etc.), axis label, legend entry, and row item into clear, explanatory paragraphs.
- Ensure the output reads as a dense, narrative data transcription."""


@dataclass(frozen=True, slots=True)
class FlorenceCacheEntry:
    image_path: str
    image_hash: str
    model: str
    created_at: str
    prompt_version: str
    description: str
    task_outputs: dict[str, Any] | None = None


def _image_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(cache_dir: Path, image_hash: str) -> Path:
    return cache_dir / f"{image_hash}.json"


def _load_cache(cache_file: Path) -> FlorenceCacheEntry | None:
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read Florence cache %s: %s", cache_file, exc)
        return None
    description = str(payload.get("description") or "").strip()
    if not description:
        return None
    prompt_version = str(payload.get("prompt_version") or "")
    if prompt_version != PROMPT_VERSION:
        logger.info(
            "Ignoring stale Florence cache %s because prompt_version=%s != %s",
            cache_file,
            prompt_version or "<missing>",
            PROMPT_VERSION,
        )
        return None
    return FlorenceCacheEntry(
        image_path=str(payload.get("image_path") or ""),
        image_hash=str(payload.get("image_hash") or cache_file.stem),
        model=str(payload.get("model") or ""),
        created_at=str(payload.get("created_at") or ""),
        prompt_version=prompt_version,
        description=description,
        task_outputs=payload.get("task_outputs") if isinstance(payload.get("task_outputs"), dict) else None,
    )


def _atomic_write_json(cache_file: Path, payload: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_file, cache_file)


def _mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/png"


def _flatten_task_output(value: Any, *, prefix: str = "") -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [f"{prefix}{stripped}" if prefix else stripped] if stripped else []
    if isinstance(value, (int, float, bool)):
        rendered = str(value)
        return [f"{prefix}{rendered}" if prefix else rendered]
    if isinstance(value, list):
        lines: list[str] = []
        for index, item in enumerate(value, start=1):
            item_prefix = f"{prefix}{index}. " if prefix else ""
            flattened = _flatten_task_output(item, prefix=item_prefix)
            if flattened:
                lines.extend(flattened)
        return lines
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            key_prefix = f"{prefix}{key}: " if prefix else f"{key}: "
            flattened = _flatten_task_output(item, prefix=key_prefix)
            if flattened:
                lines.extend(flattened)
        return lines
    rendered = str(value).strip()
    return [f"{prefix}{rendered}" if prefix else rendered] if rendered else []


class FlorenceVisionCaptioner:
    """Local Florence-2-Large visual parser for charts, diagrams, tables, and figures."""

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        *,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
        max_concurrent_requests: int | None = None,
    ) -> None:
        self.settings = settings or IngestionSettings()
        self.cache_dir = Path(cache_dir or self.settings.caption_cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or self.settings.florence_model_id
        self.max_concurrent_requests = max_concurrent_requests or self.settings.max_concurrent_requests
        self._model = None
        self._processor = None
        self._device = None
        self._torch_dtype = None
        self._gemini_client = None
        self._nvidia_client = None
        logger.info(
            "FlorenceVisionCaptioner initialized with model=%s, cache_dir=%s, max_concurrent=%s",
            self.model_name,
            self.cache_dir,
            self.max_concurrent_requests,
        )

    def _ensure_model(self) -> tuple[Any, Any, Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor, self._device, self._torch_dtype

        hf_home = self.settings.workspace_hf_home.resolve()
        hf_cache = self.settings.workspace_hf_cache.resolve()
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
        hf_home.mkdir(parents=True, exist_ok=True)
        hf_cache.mkdir(parents=True, exist_ok=True)

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        model.eval()
        self._model = model
        self._processor = processor
        self._device = device
        self._torch_dtype = torch_dtype
        return model, processor, device, torch_dtype

    def _gemini_client_or_none(self):
        if self._gemini_client is not None:
            return self._gemini_client
        api_key = (
            self.settings.gemini_api_key
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
        if not api_key:
            return None
        from google import genai
        from google.genai import types

        self._gemini_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta", timeout=60_000),
        )
        return self._gemini_client

    def _nvidia_client_or_none(self):
        if self._nvidia_client is not None:
            return self._nvidia_client
        if not NVIDIA_API_KEY:
            return None
        from openai import OpenAI

        self._nvidia_client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL, timeout=120.0)
        return self._nvidia_client

    def _run_task(self, image: Image.Image, task: str, extra_text: str = "") -> str:
        import torch

        model, processor, device, torch_dtype = self._ensure_model()
        prompt = task if not extra_text else f"{task} {extra_text}"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=torch_dtype)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].to(device)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                num_beams=3,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        try:
            parsed = processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height),
            )
            if isinstance(parsed, dict):
                value = parsed.get(task)
                flattened = _flatten_task_output(value)
                if flattened:
                    return "\n".join(flattened).strip()
                if value is not None:
                    return str(value).strip()
        except Exception:
            pass
        return str(generated_text).strip()

    def _synthesize_with_gemini(self, image_path: Path, raw_ocr_text: str) -> tuple[str, str]:
        client = self._gemini_client_or_none()
        if client is None:
            raise RuntimeError("Gemini API key is not configured.")
        from google.genai import types

        image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=_mime_type(image_path))
        response = client.models.generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[
                VISUAL_SYNTHESIS_PROMPT,
                f"Raw OCR text dump:\n{raw_ocr_text}",
                image_part,
            ],
        )
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty visual synthesis response.")
        return text, GEMINI_VISION_MODEL

    def _synthesize_with_nvidia(self, image_path: Path, raw_ocr_text: str) -> tuple[str, str]:
        client = self._nvidia_client_or_none()
        if client is None:
            raise RuntimeError("NVIDIA vision API key is not configured.")
        image_bytes = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:{_mime_type(image_path)};base64,{image_bytes}"
        response = client.chat.completions.create(
            model=NVIDIA_VISION_MODEL,
            messages=[
                {"role": "system", "content": VISUAL_SYNTHESIS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Raw OCR text dump:\n{raw_ocr_text}"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.0,
            timeout=120.0,
        )
        text = str(response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("NVIDIA vision model returned an empty visual synthesis response.")
        return text, NVIDIA_VISION_MODEL

    def _synthesize_narrative_description(self, image_path: Path, raw_ocr_text: str) -> tuple[str, str]:
        errors: list[str] = []
        try:
            return self._synthesize_with_gemini(image_path, raw_ocr_text)
        except Exception as exc:
            errors.append(f"Gemini failed: {exc}")
        try:
            return self._synthesize_with_nvidia(image_path, raw_ocr_text)
        except Exception as exc:
            errors.append(f"NVIDIA failed: {exc}")
        raise RuntimeError(" | ".join(errors) or "No visual synthesis backend available.")

    def _generate_description(self, image: ExtractedImage) -> tuple[str, dict[str, str]]:
        image_path = Path(image.image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Visual asset not found for Florence parsing: {image_path}")
        with Image.open(image_path) as loaded_image:
            pil_image = loaded_image.convert("RGB")
            raw_ocr_text = self._run_task(pil_image, "<OCR>")
        final_description, synthesis_model = self._synthesize_narrative_description(image_path, raw_ocr_text)
        task_outputs = {
            "ocr_task": "<OCR>",
            "ocr_text": raw_ocr_text.strip(),
            "synthesis_prompt": VISUAL_SYNTHESIS_PROMPT,
            "synthesis_model": synthesis_model,
            "final_description": final_description.strip(),
        }
        return final_description.strip(), task_outputs

    def describe_image(self, image: ExtractedImage) -> VisionDescription | None:
        image_path = Path(image.image_path).expanduser().resolve()
        if not image_path.exists() or not image_path.is_file():
            logger.warning("Skipping missing Florence image asset: %s", image_path)
            return None

        image_hash = _image_hash(image_path)
        cache_file = _cache_path(self.cache_dir, image_hash)
        cached = _load_cache(cache_file)
        if cached:
            description = cached.description
            task_outputs = cached.task_outputs or {}
        else:
            description, task_outputs = self._generate_description(image)
            payload = FlorenceCacheEntry(
                image_path=str(image_path),
                image_hash=image_hash,
                model=self.model_name,
                created_at=datetime.now(timezone.utc).isoformat(),
                prompt_version=PROMPT_VERSION,
                description=description,
                task_outputs=task_outputs,
            )
            _atomic_write_json(cache_file, asdict(payload))
            print(f"\n=== VISUAL DESCRIPTION: {image_path.name} ===\n{description}\n", flush=True)

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
                "caption_cache_dir": str(self.cache_dir),
                "vision_task_outputs": task_outputs,
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
                    logger.warning("Florence parsing failed for %s: %s", image.image_path, exc)
                    return None

        results = await asyncio.gather(*(run_one(image) for image in images))
        return [result for result in results if result is not None]
