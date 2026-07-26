from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestion.config import IngestionSettings


logger = logging.getLogger(__name__)

OCR_CACHE_VERSION = "paddle-v1"


def _configure_paddle_env(settings: IngestionSettings) -> None:
    workspace_root = settings.workspace_hf_home.parent.resolve()
    cache_root = (workspace_root / ".cache").resolve()
    paddle_home = (workspace_root / ".paddle_home").resolve()
    paddlex_home = (workspace_root / ".paddlex_home").resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    paddle_home.mkdir(parents=True, exist_ok=True)
    paddlex_home.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(workspace_root)
    os.environ["USERPROFILE"] = str(workspace_root)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["PADDLE_HOME"] = str(paddle_home)
    os.environ["PADDLEX_HOME"] = str(paddlex_home)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass(frozen=True, slots=True)
class PaddleOcrCacheEntry:
    image_path: str
    image_hash: str
    ocr_engine: str
    lang: str
    created_at: str
    cache_version: str
    ocr_text: str


def _image_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(cache_dir: Path, image_hash: str) -> Path:
    return cache_dir / f"{image_hash}.json"


def _atomic_write_json(cache_file: Path, payload: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_file, cache_file)


def _sort_key_from_box(box: list[list[float]]) -> tuple[float, float]:
    ys = [point[1] for point in box]
    xs = [point[0] for point in box]
    return (min(ys), min(xs))


class PaddleOcrExtractor:
    """CPU PaddleOCR extractor for text, numbers, and small labels in visual assets."""

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        *,
        cache_dir: str | Path | None = None,
        lang: str | None = None,
    ) -> None:
        self.settings = settings or IngestionSettings()
        _configure_paddle_env(self.settings)
        base_cache = Path(cache_dir or self.settings.caption_cache_dir).expanduser().resolve()
        self.cache_dir = (base_cache / "paddle_ocr").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lang = lang or self.settings.paddle_ocr_lang
        self._ocr = None
        logger.info("PaddleOcrExtractor initialized (lang=%s, device=cpu)", self.lang)

    def _ensure_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self.lang,
            use_gpu=False,
            show_log=False,
        )
        logger.info("PaddleOCR loaded on CPU (lang=%s)", self.lang)
        return self._ocr

    def warmup(self) -> None:
        self._ensure_ocr()

    @staticmethod
    def _parse_ocr_result(result: Any) -> str:
        if not result:
            return ""

        lines: list[tuple[tuple[float, float], str]] = []
        for page in result:
            if not page:
                continue
            for item in page:
                if not item or len(item) < 2:
                    continue
                box = item[0]
                text_payload = item[1]
                if isinstance(text_payload, (list, tuple)):
                    text = str(text_payload[0] or "").strip()
                else:
                    text = str(text_payload or "").strip()
                if not text:
                    continue
                lines.append((_sort_key_from_box(box), text))

        lines.sort(key=lambda entry: entry[0])
        return "\n".join(text for _key, text in lines).strip()

    def _load_cache(self, cache_file: Path) -> PaddleOcrCacheEntry | None:
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read PaddleOCR cache %s: %s", cache_file, exc)
            return None
        cache_version = str(payload.get("cache_version") or "")
        ocr_text = str(payload.get("ocr_text") or "")
        if cache_version != OCR_CACHE_VERSION or not ocr_text:
            return None
        return PaddleOcrCacheEntry(
            image_path=str(payload.get("image_path") or ""),
            image_hash=str(payload.get("image_hash") or cache_file.stem),
            ocr_engine=str(payload.get("ocr_engine") or "paddle"),
            lang=str(payload.get("lang") or self.lang),
            created_at=str(payload.get("created_at") or ""),
            cache_version=cache_version,
            ocr_text=ocr_text,
        )

    def extract_text(self, image_path: Path) -> str:
        image_path = image_path.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Visual asset not found for PaddleOCR: {image_path}")

        image_hash = _image_hash(image_path)
        cache_file = _cache_path(self.cache_dir, image_hash)
        cached = self._load_cache(cache_file)
        if cached:
            return cached.ocr_text

        ocr = self._ensure_ocr()
        result = ocr.ocr(str(image_path), cls=True)
        ocr_text = self._parse_ocr_result(result)

        payload = PaddleOcrCacheEntry(
            image_path=str(image_path),
            image_hash=image_hash,
            ocr_engine="paddle",
            lang=self.lang,
            created_at=datetime.now(timezone.utc).isoformat(),
            cache_version=OCR_CACHE_VERSION,
            ocr_text=ocr_text,
        )
        _atomic_write_json(cache_file, asdict(payload))
        return ocr_text
