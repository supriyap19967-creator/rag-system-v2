import os
import sys
import time
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recrop_all_visual_assets")

def recrop_all_visual_assets():
    """
    Iterates through all PDF files in Data/Pdf or assets/extracted_images, 
    re-detecting visual figure/table layout bounding boxes using OpenRouter VLM, 
    and overwrites static PNG image files with precision visual crops.
    """
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF (fitz) is required to re-crop visual assets.")
        return

    from app.pdf_visual_extraction import (
        _vlm_detect_crop_rect,
        _fallback_crop_rect,
        _render_page_crop,
        _extract_page_text_lines,
        _caption_candidates_from_lines
    )

    pdf_dir = PROJECT_ROOT / "Data" / "Pdf"
    if not pdf_dir.exists():
        pdf_dir = PROJECT_ROOT / "Data"
        
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}. Checking project root...")
        pdf_files = list(PROJECT_ROOT.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDF files to scan for visual re-cropping.")
    output_dirs = [
        PROJECT_ROOT / "assets" / "extracted_images",
        PROJECT_ROOT / "extracted_charts"
    ]
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    total_recropped = 0

    for pdf_path in pdf_files:
        logger.info(f"Processing PDF: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            lines = _extract_page_text_lines(page)
            captions = _caption_candidates_from_lines(lines)

            if not captions:
                continue

            for cap_item in captions:
                caption = str(cap_item.get("caption") or "").strip()
                caption_bbox = cap_item.get("bbox")
                if not caption or not caption_bbox:
                    continue

                # Build target filenames to match potential disk representations
                import re
                fig_match = re.search(r"(figure|table)\s*(\d+(?:[\._]\d+)?)", caption, re.IGNORECASE)
                is_table_caption = "table" in caption.lower()
                is_fig_caption = "figure" in caption.lower()

                matching_files = []
                if fig_match:
                    kind, num = fig_match.groups()
                    kind_clean = kind.capitalize()
                    num_clean = num.replace(".", "_")
                    
                    for out_d in output_dirs:
                        exact_target = out_d / f"page_{page_num}_{kind_clean}_{num}.png"
                        if exact_target.exists():
                            matching_files.append(exact_target)
                        exact_target_underscore = out_d / f"page_{page_num}_{kind_clean}_{num_clean}.png"
                        if exact_target_underscore.exists():
                            matching_files.append(exact_target_underscore)

                        pattern = f"*{kind.lower()}*{num_clean}*.png"
                        for candidate in out_d.glob(pattern):
                            if candidate.name.endswith(".raw.png"):
                                continue
                            cand_name = candidate.name.lower()
                            if is_table_caption and "figure" in cand_name:
                                continue
                            if is_fig_caption and "table" in cand_name:
                                continue
                            matching_files.append(candidate)

                matching_files = list(set(matching_files))
                
                if not matching_files:
                    continue

                logger.info(f"Re-cropping visual target '{caption}' on page {page_num} ({len(matching_files)} matching files)...")

                # Run VLM / dynamic precision crop detector
                crop_rect, crop_quality, extraction_pass, visual_type = _fallback_crop_rect(
                    page,
                    caption_bbox,
                    caption=caption,
                    lines=lines,
                    force_pass=1
                )

                for target_path in matching_files:
                    render_meta = _render_page_crop(
                        page,
                        crop_rect,
                        target_path,
                        trim_right_text=(visual_type != "chart")
                    )
                    if render_meta:
                        total_recropped += 1
                        logger.info(f"Successfully re-cropped and updated visual asset: {target_path.name} (Quality: {crop_quality})")

        doc.close()

    logger.info(f"Re-cropping batch completed: {total_recropped} visual PNG assets successfully updated.")

if __name__ == "__main__":
    recrop_all_visual_assets()
