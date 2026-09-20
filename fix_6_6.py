import fitz
import os
import sys
from openai import OpenAI
import json
import base64
from dotenv import load_dotenv

load_dotenv()

asset = "Figure 6.6"
pdf_path = "data/Pdf/World Development Report 2025.pdf"
out_dir = "data_cache/transcriptions"

api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

doc = fitz.open(pdf_path)
found_page = None
for page in doc:
    if asset in page.get_text():
        found_page = page
        break

if found_page:
    pix = found_page.get_pixmap(dpi=150)
    b64_image = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    
    prompt = (
        f"You are a precise technical document parser. Extract the specific data for '{asset}' from this page. "
        f"You MUST return a JSON object with these EXACT keys: 'markdown_table', 'text_reasoning', and 'extracted_table'.\n\n"
        f"Rules for 'markdown_table':\n"
        f"- It MUST be a strictly formatted markdown table with exactly these 5 column headers: | Category | Series | TargetValue | Unit | Summary |\n"
        f"- Map the table's data intelligently into these 5 columns. If a column like Unit or TargetValue is not applicable, leave it as a blank space ' '.\n"
        f"Rules for 'text_reasoning':\n"
        f"- Provide a VERY comprehensive narrative explaining all rows, columns, and conclusions.\n\n"
        f"Rules for 'extracted_table':\n"
        f"- This MUST be an empty JSON array `[]`.\n"
        f"Output ONLY valid JSON, with no other text."
    )
    
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}] }],
        temperature=0.0,
        max_tokens=4000
    )
    
    res_text = response.choices[0].message.content.strip()
    if res_text.startswith("```json"): res_text = res_text[7:-3]
    elif res_text.startswith("```"): res_text = res_text[3:-3]
        
    data = json.loads(res_text.strip())
    filename = "figure_6.6.json"
    
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        json.dump({
            "asset_id": "6.6",
            "markdown_table": data.get("markdown_table", ""),
            "text_reasoning": data.get("text_reasoning", ""),
            "image_path": None,
            "extracted_table": data.get("extracted_table", [])
        }, f, indent=2)
    print("SUCCESS")
