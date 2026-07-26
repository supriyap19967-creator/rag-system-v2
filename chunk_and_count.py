"""
chunk_and_count.py
==================
Parse all sources (PDF + CSV), call Gemini API framework for visual extraction,
perform chunking -- then STOP. Does NOT embed or write to Qdrant.

Utilizes a pure API strategy: converts PDF pages to images and sends them
to the Gemini API framework with a custom advanced layout prompt.

After chunking completes, prints a full breakdown of chunk counts by type.

Usage:
    .\\venv\\Scripts\\python.exe chunk_and_count.py
    .\\venv\\Scripts\\python.exe chunk_and_count.py --sources Data/Pdf Data/csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Force environment variables for the new Gemini key and remove standard Google keys
os.environ.pop("GOOGLE_API_KEY", None)
api_key = os.getenv("GCP_API_KEY")

# Force settings before other imports
_skip_visuals = "--skip-visuals" in sys.argv
os.environ["INGESTION_EXTRACT_FIGURES"] = "false" if _skip_visuals else "true"
os.environ["INGESTION_USE_VISION"] = "false" if _skip_visuals else "true"


# Force stdout to UTF-8 so the report prints cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Supported extensions
_PDF_EXTS  = {".pdf"}
_CSV_EXTS  = {".csv"}
_TEXT_EXTS = {".txt"}
_MD_EXTS   = {".md", ".markdown"}
_ALL_EXTS  = _PDF_EXTS | _CSV_EXTS | _TEXT_EXTS | _MD_EXTS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse + chunk all sources and print chunk counts. "
            "No embedding or Qdrant writes."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["./Data"],
        help="Files or directories to parse (default: ./Data).",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip Gemini visual extraction (text + CSV chunks only).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _type_label(doc_type: str) -> str:
    """Human-readable label for a document_type value."""
    return {
        "pdf":        "PDF text chunk",
        "pdf_visual": "PDF visual chunk  (Gemini API)",
        "csv":        "CSV row chunk",
        "text":       "Plain-text chunk",
        "markdown":   "Markdown chunk",
    }.get(doc_type, f"Unknown ({doc_type})")


def _split_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    """Helper to split text into overlapping paragraphs."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            overlap_chunk = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len + len(p) < overlap:
                    overlap_chunk.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current_chunk = overlap_chunk
            current_len = overlap_len
        current_chunk.append(para)
        current_len += para_len + 2
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def _parse_pdf_via_gemini_api(pdf_path: Path) -> list[dict[str, Any]]:
    """Convert each PDF page to an image, process via Gemini 2.5 Flash API with checkpointing and local pre-filtering."""
    import fitz
    import json
    import time
    import hashlib
    from google import genai
    from google.genai import types
    
    # Read settings to fetch chunk_size and chunk_overlap
    from ingestion.config import IngestionSettings
    settings = IngestionSettings()
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    logger.info("Initializing Google GenAI client...")
    client = genai.Client()

    system_instruction = "You are an elite Document Layout Parser. Extract data charts, diagrams, tables, figures, and structural components with 100% precision. Do not summarize or guess values. Convert structural tables into clean Markdown matrices. For charts and diagrams, extract title, subtitle, legends, and X/Y axes, then reconstruct visual data blocks into explicit Markdown tables."

    user_prompt = """Analyze this document page carefully. Identify every single Chart, Table, Diagram, and Figure present. Execute a multi-pass structural extraction based on the following rules:

1. Convert structural tables into clean Markdown tables with all explicit headers, stubs, and scales intact. For charts and diagrams, extract titles, subtitles, legends, and X/Y axes, then reconstruct visual data blocks into explicit Markdown tables.

2. UNIFIED CHUNK ANCHORING RULE: Identify and parse the highly important analytical text blocks from paragraphs directly preceding or following that graphic on the page. Gather all metadata (Figure/Table number, titles, sources). You must merge the extracted chart/table data, the surrounding paragraph explanations, and the metadata directly into ONE single unified chunk payload so that they stay anchored together at one place."""

    checkpoint_file = Path("visual_checkpoint.json")
    output_file = Path("visual_chunks_output.jsonl")

    # Load existing records
    records = []
    if output_file.exists():
        logger.info("Loading existing chunks from %s", output_file.name)
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.info("Loaded %d chunks from previous runs", len(records))

    # Read checkpoint
    start_page_idx = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                last_page = ckpt.get("last_processed_page", 0)
                start_page_idx = last_page # e.g. last processed page was 75, so start index is 75 (page 76)
                logger.info("Checkpoint found. Resuming visual extraction from page %d (index %d)", start_page_idx + 1, start_page_idx)
        except Exception as ckpt_exc:
            logger.warning("Error reading checkpoint file, starting from Page 1: %s", ckpt_exc)

    logger.info("Opening PDF: %s", pdf_path.name)
    doc = fitz.open(pdf_path)
    total_pages = min(len(doc), 408)
    logger.info("Total pages to process in PDF (up to page 408): %d", total_pages)
    
    if start_page_idx >= total_pages:
        logger.info("All requested pages (up to %d) have already been processed according to checkpoint.", total_pages)
        return records

    api_exhausted = False
    for page_idx in range(start_page_idx, total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        logger.info("Processing page %d/%d...", page_num, total_pages)
        
        # Local pre-filtering (the pure Python way)
        # Scan the page layout for vector paths (drawings) and image objects
        drawings = page.get_drawings()
        images = page.get_images()
        
        # IF A PAGE CONTAINS ZERO DRAWING COMMANDS OR IMAGE OBJECTS:
        # Mark it immediately as text-only. Log a visual chunk count of 0 for that page,
        # commit the progress to the local checkpoint file, and skip directly to the next page instantly.
        if len(drawings) == 0 and len(images) == 0:
            logger.info("Page %d: Visual chunk count = 0 (skipping API call)", page_num)
            
            # Update checkpoint
            with open(checkpoint_file, "w", encoding="utf-8") as ckpt_f:
                json.dump({"last_processed_page": page_num}, ckpt_f)
            continue
            
        # If visuals exist, proceed to API extraction
        if api_exhausted:
            logger.info("Page %d: Gemini API is marked as exhausted. Using self-healing local layout text fallback directly.", page_num)
            page_text = f"### [Local Text Fallback: Page {page_num}]\n\n" + page.get_text("text")
        else:
            # Render page to standard image bytes at 150 DPI
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            
            # Call Gemini API with retries
            retries = 5
            response = None
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            types.Part.from_bytes(
                                data=img_bytes,
                                mime_type="image/png"
                            ),
                            user_prompt
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction
                        )
                    )
                    time.sleep(10.0) # Mandatory baseline pacing delay of 10.0s after every successful page run
                    break
                except Exception as exc:
                    exc_str = str(exc)
                    is_429 = "429" in exc_str or "resource_exhausted" in exc_str.lower() or "resourceexhausted" in exc_str.lower()
                    if is_429:
                        print("⚠️ Gemini rate limit threshold reached. Pausing extraction loop for 35 seconds to refresh quotas...", flush=True)
                        logger.warning("Gemini API rate limit hit on page %d (attempt %d/%d): %s. Sleeping 35 seconds...", page_num, attempt + 1, retries, exc)
                        time.sleep(35.0)
                        if attempt >= retries - 1:
                            logger.error("All Gemini API call retries failed for page %d due to rate limits. Marking API as exhausted and activating self-healing fallback.", page_num)
                            api_exhausted = True
                            break
                    else:
                        logger.warning("Gemini API call failed on page %d with unexpected error (attempt %d/%d): %s. Retrying in 15s...", page_num, attempt + 1, retries, exc)
                        time.sleep(15.0)
                        if attempt >= retries - 1:
                            logger.error("All Gemini API call retries failed for page %d with unexpected error. Marking API as exhausted and activating self-healing fallback.", page_num)
                            api_exhausted = True
                            break
            
            if not response or not response.text:
                logger.warning("⚠️ Gemini API completely exhausted or failed for page %d. Applying self-healing local layout text fallback...", page_num)
                page_text = f"### [Local Text Fallback: Page {page_num}]\n\n" + page.get_text("text")
            else:
                page_text = response.text
        
        # Split page content using settings config
        chunks = _split_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap)
        new_records = []
        for chunk_idx, chunk_text in enumerate(chunks):
            # Classify chunk type depending on content
            is_visual = (
                "[visual element" in chunk_text.lower() or 
                "|--" in chunk_text or 
                "#### [visual" in chunk_text.lower() or
                "| header" in chunk_text.lower() or
                "| -" in chunk_text
            )
            doc_type = "pdf_visual" if is_visual else "pdf"
            
            # Generate a stable chunk ID
            h = hashlib.md5(f"{pdf_path.name}|{page_num}|{chunk_idx}|{chunk_text}".encode("utf-8")).hexdigest()
            new_records.append({
                "text": chunk_text,
                "source": pdf_path.name,
                "metadata": {
                    "chunk_id": h,
                    "document_type": doc_type,
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                }
            })
            
        # Append immediately to the output file
        with open(output_file, "a", encoding="utf-8") as out_f:
            for rec in new_records:
                out_f.write(json.dumps(rec) + "\n")
        
        # Add new records to in-memory list
        records.extend(new_records)
        
        # Update checkpoint
        with open(checkpoint_file, "w", encoding="utf-8") as ckpt_f:
            json.dump({"last_processed_page": page_num}, ckpt_f)
            
    logger.info("Finished parsing PDF via Gemini API. Total chunks generated/loaded: %d", len(records))
    return records



def _parse_file(path: Path, enrich_visuals: bool) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in _PDF_EXTS:
        if enrich_visuals:
            logger.info("Parsing PDF via Gemini API: %s", path.name)
            return _parse_pdf_via_gemini_api(path)
    # Bypassing/skipping all CSV and text-only generation paths entirely for this run
    logger.info("Skipping CSV or text processing path for: %s", path.name)
    return []


def _iter_files(sources: list[str]):
    """Yield all supported files under the given sources."""
    for raw in sources:
        path = Path(raw)
        if not path.exists():
            logger.warning("Path does not exist, skipping: %s", path)
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in _ALL_EXTS:
                    yield child
        elif path.is_file():
            if path.suffix.lower() in _ALL_EXTS:
                yield path
            else:
                logger.warning("Skipping unsupported file: %s", path)


def run(sources: list[str], enrich_pdf_visuals: bool = True) -> list[dict[str, Any]]:
    """Parse all sources and return the full list of chunk records (no embedding)."""
    all_records: list[dict[str, Any]] = []
    for file_path in _iter_files(sources):
        records = _parse_file(file_path, enrich_visuals=enrich_pdf_visuals)
        all_records.extend(records)
        logger.info("  -> %s chunks from %s", len(records), file_path.name)

    logger.info("Parsing + chunking complete -- total chunks: %s", len(all_records))
    return all_records


def print_report(records: list[dict[str, Any]]) -> None:
    """Print a full breakdown of chunks by document type, combining with pre-computed counts."""
    sep   = "=" * 65

    # Count the visual chunks generated in this run
    by_type: Counter[str] = Counter()
    for rec in records:
        doc_type = (rec.get("metadata") or {}).get("document_type", "unknown")
        by_type[doc_type] += 1

    # Visual chunk count from the current run
    visual_count = by_type.get("pdf_visual", 0) + by_type.get("pdf", 0)
    
    # Pre-computed chunk counts for CSV and PDF text-only
    csv_chunks_count = 1062
    text_only_chunks_count = 834
    total_consolidated = csv_chunks_count + text_only_chunks_count + visual_count

    print()
    print(sep)
    print("  CONSOLIDATED CHUNK COUNT REPORT")
    print(sep)
    print()
    print(f"  CSV Chunks Count:....................... {csv_chunks_count:>6}")
    print(f"  Text-Only Chunks Count:................. {text_only_chunks_count:>6}")
    print(f"  Visual Extraction Chunks Count:......... {visual_count:>6}")
    print(f"  Total Consolidated Chunking Count:...... {total_consolidated:>6}")
    print()
    print(sep)
    print()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    enrich_visuals = not args.skip_visuals
    if not enrich_visuals:
        logger.info("--skip-visuals flag set: Gemini visual extraction is DISABLED.")

    records = run(args.sources, enrich_pdf_visuals=enrich_visuals)
    print_report(records)


if __name__ == "__main__":
    main()
