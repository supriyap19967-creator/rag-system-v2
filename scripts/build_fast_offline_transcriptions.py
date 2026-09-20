import os
import sys
import json
import re
from pathlib import Path
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

# Load .env file manually or with dotenv
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and not os.getenv(k):
                    os.environ[k] = v

INPUT_DIRS = [
    ROOT / "extracted_charts",
    ROOT / "assets" / "extracted_images"
]
OUTPUT_DIR = ROOT / "data_cache" / "transcriptions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1") if api_key else None

def parse_entity_info(filename: str):
    m = re.search(r'(figure|fig|table|chart|diagram)[_\-\s]*([A-Za-z]?\d+(?:[\._]\d+)?)', filename, re.IGNORECASE)
    if m:
        raw_kind = m.group(1).lower()
        kind = "figure" if "fig" in raw_kind or "chart" in raw_kind or "diagram" in raw_kind else "table"
        asset_id = m.group(2)
        return kind, asset_id
    return None, None

import time
import concurrent.futures

def process_single_img(img_path: Path):
    filename = img_path.name
    kind, asset_id = parse_entity_info(filename)
    if not asset_id:
        return False

    id_variants = [asset_id, asset_id.replace('.', '_'), asset_id.replace('_', '.')]
    
    # Check if any variant already exists
    for var in id_variants:
        out_var = OUTPUT_DIR / f"{kind}_{var}.json"
        if out_var.exists():
            return False

    output_file = OUTPUT_DIR / f"{kind}_{asset_id}.json"
    print(f"📸 Transcribing visual asset: {filename} ({kind} {asset_id})...", flush=True)
    
    prompt = (
        f"Extract EVERY SINGLE data point, category, series, entity, metric, and value from this {kind} ({asset_id}). "
        "Do not omit, truncate, or skip any visible numbers or data series. Format completely as a clean Markdown table with columns: Category | Series | TargetValue | Unit | Summary."
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            import base64
            with open(img_path, "rb") as f_img:
                b64_img = base64.b64encode(f_img.read()).decode("utf-8")

            resp = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }],
                timeout=45.0
            )
            text_out = resp.choices[0].message.content if (resp.choices and resp.choices[0].message) else ""
            
            if text_out:
                record = {
                    "asset_type": kind,
                    "asset_id": asset_id,
                    "image_path": str(img_path.resolve()),
                    "markdown_table": text_out,
                    "status": "completed"
                }
                # Save main file
                with open(output_file, "w", encoding="utf-8") as out_f:
                    json.dump(record, out_f, indent=2)
                
                # Save variant aliases (e.g. 7_4 vs 7.4)
                for var in id_variants:
                    var_file = OUTPUT_DIR / f"{kind}_{var}.json"
                    if not var_file.exists():
                        with open(var_file, "w", encoding="utf-8") as var_f:
                            json.dump(record, var_f, indent=2)

                print(f"✅ Saved offline transcription: {output_file.name}", flush=True)
                return True
        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "429" in err_str or "in-flight" in err_str.lower():
                sleep_time = (attempt + 1) * 5
                print(f"⏳ OpenRouter rate limit hit for {filename}. Sleeping {sleep_time}s before retry...", flush=True)
                time.sleep(sleep_time)
            else:
                print(f"⚠️ Failed offline transcription for {filename}: {e}", flush=True)
                break
    return False

def process_all_assets():
    print("🚀 Starting Rate-Limited Offline Vision LLM Ingestion Processor...", flush=True)
    if not client:
        print("❌ Error: No OPENROUTER_API_KEY found in environment or .env!", flush=True)
        return

    all_images = []
    for input_dir in INPUT_DIRS:
        if input_dir.exists():
            for img_path in sorted(input_dir.glob("*.png")):
                if not img_path.name.endswith(".raw.png"):
                    all_images.append(img_path)

    print(f"📋 Found {len(all_images)} total visual assets to inspect.", flush=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_single_img, all_images))

    processed_count = sum(1 for r in results if r)
    print(f"🎉 Offline Vision Ingestion Complete! Processed {processed_count} new visual assets.", flush=True)

if __name__ == "__main__":
    process_all_assets()



