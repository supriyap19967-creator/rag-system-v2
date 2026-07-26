import base64
import hashlib
import json
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

INPUT_DIR = Path("./extracted_charts")
CACHE_DIR = Path("./data_cache/visual_captions")
PROGRESS_FILE = Path("./visual_progress.json")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.2-11b-vision-instruct"
MAX_RETRIES = 10
INITIAL_BACKOFF_SECONDS = 10
POST_SUCCESS_SLEEP_SECONDS = 4
PROMPT = """You are a specialized document extraction engine. Analyze this chart/figure image and output a clean, detailed text description for a RAG database.

Your response MUST follow this structure:
1. FIGURE HEADLINE/TITLE: Read the exact text title, heading, and figure number printed at the top or bottom of the chart image.
2. CHART TYPE & TOPIC: State exactly what kind of chart it is and what metric it measures.
3. DATA BREAKDOWN: Describe the core trend lines, data points, percentages, and axis groups clearly so they can be easily searched via keyword vector search.

At the very end of your response, on a brand new line, output the exact figure or table label found inside the image in this format: [LABEL: Figure X.Y] or [LABEL: Figure O.X]. If no clear figure number is visible, output [LABEL: Unknown]."""
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_md5(image_path: Path) -> str:
    digest = hashlib.md5()
    with image_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path_for(image_hash: str) -> Path:
    return CACHE_DIR / f"{image_hash}.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def load_progress() -> set[str]:
    if not PROGRESS_FILE.exists():
        return set()
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(item) for item in data}
        if isinstance(data, dict):
            return {str(item) for item in data.get("processed_images", [])}
    except Exception as exc:
        print(f"[WARNING] Could not read {PROGRESS_FILE}: {exc}. Starting with empty progress.")
    return set()


def save_progress(processed_images: set[str]) -> None:
    atomic_write_json(
        PROGRESS_FILE,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed_images": sorted(processed_images),
        },
    )


def mime_type_for(image_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(image_path))
    return guessed or "image/png"


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return str(status) == "429" or "rate limit" in message or "too many requests" in message or "quota" in message


def list_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir.resolve()}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def image_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type_for(image_path)};base64,{encoded}"


def extract_label_and_clean_caption(caption: str) -> tuple[str, str]:
    match = re.search(r"\[LABEL:\s*([^\]]+?)\s*\]\s*$", caption.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return caption.strip(), "Unknown"

    label = match.group(1).strip() or "Unknown"
    caption_without_label = caption[: match.start()].strip()
    return caption_without_label, label


def generate_caption(client: OpenAI, image_path: Path) -> str:
    image_url = image_data_url(image_path)
    while True:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(
                    f"--> Debug: Sending image to NVIDIA NIM API... "
                    f"image={image_path.name}, attempt={attempt}/{MAX_RETRIES}",
                    flush=True,
                )
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    timeout=60.0,
                )
                print("--> Debug: Received response from NVIDIA NIM successfully!", flush=True)
                caption = str(response.choices[0].message.content or "").strip()
                if not caption:
                    raise RuntimeError("NVIDIA NIM returned an empty caption.")
                return caption
            except Exception as exc:
                print(f"--> Debug: Caught error: {exc}", flush=True)
                delay = 2**attempt
                print(
                    f"[WARNING] API/timeout error. Sleeping for {delay} seconds before retrying... "
                    f"Image={image_path.name}, attempt={attempt}/{MAX_RETRIES}",
                    flush=True,
                )
                time.sleep(delay)

        print(
            f"[WARNING] Exhausted {MAX_RETRIES} retries for {image_path.name}. "
            "Sleeping 30 seconds, then retrying the same image again.",
            flush=True,
        )
        time.sleep(30)


def save_caption(image_path: Path, image_hash: str, caption: str) -> None:
    caption_text_without_label_tag, extracted_label_value = extract_label_and_clean_caption(caption)
    qdrant_payload = {
        "text": caption_text_without_label_tag,
        "metadata": {
            "type": "visual_caption",
            "file_name": image_path.name,
            "figure_id": extracted_label_value,
        },
    }
    cache_payload = {
        "image_path": str(image_path.resolve()),
        "image_name": image_path.name,
        "image_hash": image_hash,
        "model": MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": PROMPT,
        "caption": caption_text_without_label_tag,
        "raw_caption": caption,
        "figure_id": extracted_label_value,
        "payload": qdrant_payload,
    }
    atomic_write_json(cache_path_for(image_hash), cache_payload)


def process_images() -> None:
    if not NVIDIA_API_KEY:
        raise RuntimeError("Set NVIDIA_API_KEY before running visual extraction.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    images = list_images(INPUT_DIR)
    processed_images = load_progress()

    if not images:
        print(f"[INFO] No supported images found in {INPUT_DIR.resolve()}")
        return

    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL, timeout=60.0)
    print(f"[INFO] Found {len(images)} image(s). Cache directory: {CACHE_DIR.resolve()}")
    print(f"[INFO] Loaded {len(processed_images)} processed image(s) from {PROGRESS_FILE.resolve()}")

    for index, image_path in enumerate(images, start=1):
        print(f"Processing image [{index}/{len(images)}]: {image_path.name}...")
        image_hash = image_md5(image_path)
        cache_file = cache_path_for(image_hash)
        image_id = image_path.name

        if cache_file.exists():
            print(f"[CACHE HIT] Skipping {image_path.name}")
            processed_images.add(image_id)
            save_progress(processed_images)
            continue

        if image_id in processed_images:
            print(f"[SKIPPING] Progress checkpoint hit for {image_path.name}, but cache is missing; retrying caption.")

        print(f"[CACHE MISS] Calling NVIDIA NIM for {image_path.name}")
        caption = generate_caption(client, image_path)
        save_caption(image_path, image_hash, caption)
        processed_images.add(image_id)
        save_progress(processed_images)
        print(f"[SUCCESS] Caption cached for {image_path.name}")
        time.sleep(POST_SUCCESS_SLEEP_SECONDS)


if __name__ == "__main__":
    process_images()
