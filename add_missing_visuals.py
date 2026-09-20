import fitz
import os
import sys
from openai import OpenAI
import json
import base64
import time
from dotenv import load_dotenv

load_dotenv()

failed_missing = [
    "Table O.1", "Figure 3.3", "Figure 3.6", "Figure 6.6", 
    "Figure 7.1", "Figure 7.9", "Figure O.1", "Figure O.2", 
    "Figure O.5", "Figure O.9", "Figure B4.6.1"
]

pdf_path = "data/Pdf/World Development Report 2025.pdf"
out_dir = "data_cache/transcriptions"
os.makedirs(out_dir, exist_ok=True)

api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
if not api_key:
    print("OPENROUTER_API_KEY not found.", flush=True)
    sys.exit(1)

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key
)

doc = fitz.open(pdf_path)

asset_pages = {asset: None for asset in failed_missing}
print("Pre-scanning PDF for assets...", flush=True)
for i, page in enumerate(doc):
    text = page.get_text()
    for asset in list(asset_pages.keys()):
        if asset_pages[asset] is None and asset in text:
            asset_pages[asset] = page
            print(f"Found {asset} on page {i+1}", flush=True)

for asset, found_page in asset_pages.items():
    if not found_page:
        print(f"Could not find {asset} in PDF text.", flush=True)
        continue
        
    print(f"Processing {asset}...", flush=True)
    try:
        pix = found_page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
        
        prompt = (
            f"You are a precise technical document parser. Extract the specific data for '{asset}' from this page. "
            f"You MUST return a JSON object with these EXACT keys: 'markdown_table', 'text_reasoning', and 'extracted_table'.\n\n"
            f"Rules for 'markdown_table':\n"
            f"- It MUST be a strictly formatted markdown table with exactly these 5 column headers: | Category | Series | TargetValue | Unit | Summary |\n"
            f"- Map the table's data intelligently into these 5 columns. If a column like Unit or TargetValue is not applicable, leave it as a blank space ' '.\n"
            f"- DO NOT use the word 'N/A' or any HTML tags like <br>.\n\n"
            f"Rules for 'text_reasoning':\n"
            f"- Provide a VERY comprehensive, highly detailed, multi-paragraph textual narrative explaining all rows, columns, and conclusions shown in the image. Extract every piece of knowledge.\n\n"
            f"Rules for 'extracted_table':\n"
            f"- This MUST be an empty JSON array `[]`.\n"
            f"Output ONLY valid JSON, with no other text."
        )
        
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=2000
        )
        
        res_text = response.choices[0].message.content.strip()
        if res_text.startswith("```json"):
            res_text = res_text[7:-3]
        elif res_text.startswith("```"):
            res_text = res_text[3:-3]
            
        data = json.loads(res_text.strip())
        
        asset_lower = asset.lower().replace(" ", "_")
        filename = f"table_{asset.split(' ')[1]}.json" if asset.startswith("Table") else f"figure_{asset.split(' ')[1]}.json"
        out_path = os.path.join(out_dir, filename)
        
        final_payload = {
            "asset_id": asset.split(" ")[1],
            "markdown_table": data.get("markdown_table", ""),
            "text_reasoning": data.get("text_reasoning", ""),
            "image_path": None,
            "extracted_table": data.get("extracted_table", [])
        }
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)
            
        print(f"Successfully saved {asset} to {filename}", flush=True)
        time.sleep(2)
        
    except Exception as e:
        print(f"Failed to process {asset}: {e}", flush=True)
