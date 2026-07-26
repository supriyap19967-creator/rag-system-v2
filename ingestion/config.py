from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


def _first_api_key() -> str:
    keys = [key.strip() for key in os.getenv("GEMINI_API_KEYS", "").split(",") if key.strip()]
    return keys[0] if keys else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""))


@dataclass(frozen=True, slots=True)
class IngestionSettings:
    """Environment-backed settings for multimodal ingestion."""

    workspace_hf_home: Path = _path_from_env("HF_HOME", ".hf_home")
    workspace_hf_cache: Path = _path_from_env("HUGGINGFACE_HUB_CACHE", ".hf_home/hub")
    docling_artifacts_dir: Path = _path_from_env("DOCLING_ARTIFACTS_PATH", "docling_models")
    figure_output_dir: Path = _path_from_env("INGESTION_FIGURE_OUTPUT_DIR", "assets/extracted_images")
    gemini_api_key: str = _first_api_key()
    gemini_vision_model: str = os.getenv("GEMINI_MODEL_NAME") or os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    florence_model_id: str = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large")
    qwen_vl_model_id: str = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    paddle_ocr_lang: str = os.getenv("PADDLE_OCR_LANG", "en")
    paddle_ocr_use_gpu: bool = False
    florence_prompt_task: str = os.getenv("FLORENCE_PROMPT_TASK", "<MORE_DETAILED_CAPTION>")
    florence_ocr_task: str = os.getenv("FLORENCE_OCR_TASK", "<OCR>")
    vision_backend: str = os.getenv("INGESTION_VISION_BACKEND", "gemini").lower()
    caption_cache_dir: Path = _path_from_env("CACHE_DIR", "data_cache/visual_captions")
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
    chunk_size: int = int(os.getenv("INGESTION_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("INGESTION_CHUNK_OVERLAP", "180"))
    max_concurrent_vision_tasks: int = int(os.getenv("INGESTION_MAX_CONCURRENT_VISION_TASKS", "2"))
    use_vision: bool = os.getenv("INGESTION_USE_VISION", "true").lower() not in {"0", "false", "no"}
    extract_figures: bool = os.getenv("INGESTION_EXTRACT_FIGURES", "true").lower() not in {"0", "false", "no"}
    pdf_strategy: str = os.getenv("INGESTION_PDF_STRATEGY", "hi_res")
    csv_backend: str = os.getenv("INGESTION_CSV_BACKEND", "auto").lower()
    llama_parse_api_key: str = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_PARSE_API_KEY") or os.getenv("LLAMAPARSE_API_KEY", "")
