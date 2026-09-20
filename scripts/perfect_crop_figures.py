import re
import fitz
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perfect_crop_figures")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_ROOT / "assets" / "extracted_images"
PDF_PATH = PROJECT_ROOT / "Data" / "Pdf" / "World Development Report 2025.pdf"

doc = fitz.open(str(PDF_PATH))

def perfect_crop_figure(page_no: int, target_filename: str):
    """
    Finds exact span from Figure Header (top) down to Source/Note footer (bottom)
    and crops the figure perfectly with 15pt clean outer margins.
    """
    page = doc.load_page(page_no - 1)
    page_rect = page.rect
    blocks = page.get_text("blocks") or []
    drawings = page.get_drawings() or []
    
    # Extract figure number e.g. "Figure 6.1" or "Figure 4.2"
    m_fig = re.search(r'Figure_?(\d+(?:\.\d+)?)', target_filename, re.IGNORECASE)
    fig_token = f"Figure {m_fig.group(1)}" if m_fig else None
    
    header_block = None
    footer_block = None
    
    if fig_token:
        for b in blocks:
            text = b[4].strip()
            if fig_token.lower() in text.lower():
                header_block = b
                break

    # If header found, look for Source / Note block below header
    if header_block:
        h_y0 = header_block[1]
        below_blocks = [b for b in blocks if b[1] > h_y0 and b[1] < page_rect.height - 30]
        
        # Look for Source / Note / next Figure header
        for b in below_blocks:
            txt = b[4].strip().lower()
            if txt.startswith("source:") or txt.startswith("note:") or txt.startswith("sources:"):
                footer_block = b
            elif txt.startswith("figure ") and b[1] > h_y0 + 50:
                # Stopped at next figure header on same page
                break

    # Determine bounding box
    if header_block and footer_block:
        y0 = header_block[1] - 10.0
        y1 = footer_block[3] + 10.0
        x0 = min(header_block[0], footer_block[0]) - 15.0
        x1 = max(header_block[2], footer_block[2]) + 15.0
        
        # Expand x bounds to cover full graphic width
        graphic_blocks = [b for b in blocks if b[1] >= y0 and b[3] <= y1]
        if graphic_blocks:
            x0 = min(x0, min(b[0] for b in graphic_blocks) - 15.0)
            x1 = max(x1, max(b[2] for b in graphic_blocks) + 15.0)
            
        if drawings:
            fig_drawings = [d["rect"] for d in drawings if d["rect"].y0 >= y0 - 10 and d["rect"].y1 <= y1 + 10]
            if fig_drawings:
                x0 = min(x0, min(r.x0 for r in fig_drawings) - 15.0)
                x1 = max(x1, max(r.x1 for r in fig_drawings) + 15.0)
    else:
        # Fallback to smart drawing layout span
        if drawings:
            d_rects = [d["rect"] for d in drawings if d["rect"].height > 15 and d["rect"].width > 15]
            if d_rects:
                x0 = min(r.x0 for r in d_rects) - 20.0
                y0 = min(r.y0 for r in d_rects) - 20.0
                x1 = max(r.x1 for r in d_rects) + 20.0
                y1 = max(r.y1 for r in d_rects) + 20.0
            else:
                x0, y0, x1, y1 = 30.0, 50.0, page_rect.width - 30.0, page_rect.height - 50.0
        else:
            x0, y0, x1, y1 = 30.0, 50.0, page_rect.width - 30.0, page_rect.height - 50.0

    # Ensure box stays within page boundaries
    crop_box = fitz.Rect(
        max(0.0, x0),
        max(0.0, y0),
        min(page_rect.width, x1),
        min(page_rect.height, y1)
    )

    # Render high-resolution 300 DPI crop
    scale_factor = 300.0 / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor), clip=crop_box, alpha=False)
    
    out_file = ASSET_DIR / target_filename
    pix.save(str(out_file))
    logger.info("Successfully perfectly recropped: %s | Page %s | Box: %s", target_filename, page_no, crop_box)
    return True

# 1. Perfectly crop Figure 6.1 on page 302
perfect_crop_figure(302, "page_302_Figure_6.1.png")

# 2. Perfect crop all figures across entire asset directory
processed = 0
for img in ASSET_DIR.glob("*.png"):
    if "Figure" in img.name and not img.name.endswith(".raw.png") and img.name != "page_302_Figure_6.1.png":
        m_p = re.search(r'page_?(\d+)', img.name, re.IGNORECASE)
        if m_p:
            p_num = int(m_p.group(1))
            if 1 <= p_num <= len(doc):
                perfect_crop_figure(p_num, img.name)
                processed += 1

logger.info("Perfect figure recropping complete. Re-cropped %d figures spanning from Header to Footer.", processed + 1)
doc.close()
