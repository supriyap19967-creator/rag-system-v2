import sys
import os
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from streamlit_ui.StreamlitApp import sanitize_axis_cross_blending_text, swap_and_clean_row


def sanitize_table_markdown(md_text: str) -> str:
    if not md_text or "|" not in md_text:
        return sanitize_axis_cross_blending_text(md_text)

    lines = md_text.splitlines()
    cleaned_lines = []
    
    # First pass: sanitize text
    for line in lines:
        cleaned_line = sanitize_axis_cross_blending_text(line)
        cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


def sanitize_json_obj(obj):
    if isinstance(obj, str):
        return sanitize_axis_cross_blending_text(obj)
    elif isinstance(obj, list):
        return [sanitize_json_obj(item) for item in obj]
    elif isinstance(obj, dict):
        new_dict = {}
        # Special check for table rows: Series, Category, TargetValue
        if "Series" in obj or "Category" in obj or "TargetValue" in obj:
            s_raw = obj.get("Series", "Data Point")
            c_raw = obj.get("Category", "")
            v_raw = obj.get("TargetValue")
            s_clean, c_clean, v_clean = swap_and_clean_row(s_raw, c_raw, v_raw)
            obj["Series"] = s_clean
            obj["Category"] = c_clean
            obj["TargetValue"] = v_clean

        for k, v in obj.items():
            if k in ["markdown_table", "table_markdown", "text", "raw_content", "anchor_text", "generated_description", "text_reasoning"] and isinstance(v, str):
                new_dict[k] = sanitize_table_markdown(v)
            else:
                new_dict[k] = sanitize_json_obj(v)
        return new_dict
    else:
        return obj


def main():
    cache_root = PROJECT_ROOT / "data_cache"
    if not cache_root.exists():
        print(f"Cache root {cache_root} does not exist.")
        return

    json_files = list(cache_root.glob("**/*.json"))
    jsonl_files = list(cache_root.glob("**/*.jsonl"))
    all_files = json_files + jsonl_files

    print(f"Auditing and sanitizing {len(all_files)} cache files in {cache_root}...")
    sanitized_count = 0

    for file_path in all_files:
        try:
            if file_path.suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cleaned_data = sanitize_json_obj(data)
                if json.dumps(data, sort_keys=True) != json.dumps(cleaned_data, sort_keys=True):
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
                    sanitized_count += 1
                    print(f"  Cleaned & sanitized: {file_path.relative_to(PROJECT_ROOT)}")

            elif file_path.suffix == ".jsonl":
                lines_cleaned = False
                cleaned_lines = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            cleaned_lines.append("")
                            continue
                        try:
                            data = json.loads(line_str)
                            cleaned_data = sanitize_json_obj(data)
                            if json.dumps(data, sort_keys=True) != json.dumps(cleaned_data, sort_keys=True):
                                lines_cleaned = True
                            cleaned_lines.append(json.dumps(cleaned_data, ensure_ascii=False))
                        except Exception:
                            cleaned_lines.append(line_str)

                if lines_cleaned:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(cleaned_lines) + "\n")
                    sanitized_count += 1
                    print(f"  Cleaned & sanitized JSONL: {file_path.relative_to(PROJECT_ROOT)}")

        except Exception as e:
            print(f"Skipping file {file_path.name} due to error: {e}")

    print(f"\nSuccessfully completed disk cache sanitization! Cleaned {sanitized_count} files.")


if __name__ == "__main__":
    main()
