import fitz
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "Data" / "Pdf" / "World Development Report 2025.pdf"

doc = fitz.open(str(PDF_PATH))
page = doc.load_page(301) # 0-indexed page 302

print("=== PAGE 302 RECT ===")
print(page.rect)

print("\n=== PAGE 302 TEXT BLOCKS ===")
blocks = page.get_text("blocks")
for i, b in enumerate(blocks):
    # b = (x0, y0, x1, y1, text, block_no, block_type)
    text_snippet = b[4].replace('\n', ' ').encode('ascii', 'ignore').decode('ascii')[:80]
    print(f"Block {i}: [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}] -> '{text_snippet}'")

print("\n=== PAGE 302 DRAWINGS ===")
drawings = page.get_drawings()
print(f"Total drawings: {len(drawings)}")
for i, d in enumerate(drawings[:10]):
    r = d["rect"]
    print(f"Drawing {i}: [{r.x0:.1f}, {r.y0:.1f}, {r.x1:.1f}, {r.y1:.1f}]")

# Render full page 302 at 300 DPI
pix = page.get_pixmap(dpi=300)
out_full = PROJECT_ROOT / "assets" / "extracted_images" / "page_302_full_page.png"
pix.save(str(out_full))
print(f"\nSaved full page 302 to {out_full}")
doc.close()
