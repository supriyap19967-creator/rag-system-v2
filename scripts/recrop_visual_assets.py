import os
import sys
import re
import logging
from pathlib import Path
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recrop_visual_assets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_ROOT / "assets" / "extracted_images"
PDF_PATH = PROJECT_ROOT / "Data" / "Pdf" / "World Development Report 2025.pdf"

if not PDF_PATH.exists():
    # Search recursively for PDF
    pdfs = list(PROJECT_ROOT.rglob("*.pdf"))
    if pdfs:
        PDF_PATH = pdfs[0]
    else:
        logger.error("PDF file not found in project root or subdirectories.")
        sys.exit(1)

logger.info("Opening PDF: %s", PDF_PATH)
doc = fitz.open(str(PDF_PATH))

def recrop_image(img_file: Path):
    """
    Recrop image file from PDF with generous padding (40pt margin) to prevent inner content clipping.
    """
    filename = img_file.name
    # Extract page_no from filename e.g. page_302_Figure_6.1.png or page302_figure1.png
    m_page = re.search(r'page_?(\d+)', filename, re.IGNORECASE)
    if not m_page:
        return False
    
    page_no = int(m_page.group(1))
    if page_no < 1 or page_no > len(doc):
        return False
    
    page = doc.load_page(page_no - 1)
    page_rect = page.rect
    
    # Search for target figure number e.g. "Figure 6.1" or "6.1"
    m_fig = re.search(r'Figure_?(\d+(?:\.\d+)?)', filename, re.IGNORECASE)
    fig_name = f"Figure {m_fig.group(1)}" if m_fig else None

    # Search text instances for title / caption
    rects = []
    if fig_name:
        text_instances = page.get_text("blocks")
        for b in text_instances:
            # b = (x0, y0, x1, y1, text, block_no, block_type)
            b_text = b[4]
            if fig_name.lower() in b_text.lower():
                rects.append(fitz.Rect(b[0], b[1], b[2], b[3]))

    # Also check drawings bounding box
    drawings = page.get_drawings()
    if drawings:
        for d in drawings:
            # d is a dict with 'rect' key
            r = d.get("rect")
            if r and r.height > 20 and r.width > 20:
                rects.append(r)

    if not rects:
        # Fallback: crop full page with generous margin
        crop_box = fitz.Rect(
            page_rect.width * 0.05,
            page_rect.height * 0.05,
            page_rect.width * 0.95,
            page_rect.height * 0.95
        )
    else:
        # Merge bounding boxes of drawing & text elements
        x0 = min(r.x0 for r in rects)
        y0 = min(r.y0 for r in rects)
        x1 = max(r.x1 for r in rects)
        y1 = max(r.y1 for r in rects)
        
        # Apply generous padding: 40pt horizontal and vertical padding
        pad_x = 45.0
        pad_y = 45.0
        
        crop_box = fitz.Rect(
            max(0.0, x0 - pad_x),
            max(0.0, y0 - pad_y),
            min(page_rect.width, x1 + pad_x),
            min(page_rect.height, y1 + pad_y)
        )

    # Render at 300 DPI (scale 300/72 = 4.1667)
    scale_factor = 300.0 / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor), clip=crop_box, alpha=False)
    
    # Save recropped high-res image
    pix.save(str(img_file))
    logger.info("Successfully recropped: %s | Page: %s | Box: %s", filename, page_no, crop_box)
    return True

# 1. Specifically recrop Figure 6.1
fig_61 = ASSET_DIR / "page_302_Figure_6.1.png"
if fig_61.exists():
    recrop_image(fig_61)

# 2. Recrop all Figure_*.png files in extracted_images
count = 0
for img in ASSET_DIR.glob("*.png"):
    if "Figure" in img.name and not img.name.endswith(".raw.png"):
        if recrop_image(img):
            count += 1

logger.info("Recropping complete. Processed %d visual figure images with generous 45pt padding.", count)
doc.close()
