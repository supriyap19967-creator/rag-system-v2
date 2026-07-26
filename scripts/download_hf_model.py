from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot file-by-file with simple progress.")
    parser.add_argument("model_id")
    parser.add_argument("--cache-dir", default=".hf_home/hub")
    args = parser.parse_args()

    api = HfApi()
    info = api.model_info(args.model_id, files_metadata=True)
    siblings = [s for s in info.siblings if s.rfilename]
    total_bytes = sum((s.size or 0) for s in siblings)
    downloaded = 0

    print(f"MODEL={args.model_id}")
    print(f"TOTAL_BYTES={total_bytes}")

    for sibling in siblings:
        hf_hub_download(
            repo_id=args.model_id,
            filename=sibling.rfilename,
            cache_dir=args.cache_dir,
            resume_download=True,
        )
        downloaded += sibling.size or 0
        pct = round((downloaded / total_bytes) * 100, 2) if total_bytes else 100.0
        print(f"DOWNLOADED={downloaded} PROGRESS={pct}% FILE={sibling.rfilename}", flush=True)


if __name__ == "__main__":
    main()
