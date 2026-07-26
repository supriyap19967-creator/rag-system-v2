import os
import sys
import traceback

print("Starting diag.py...", flush=True)

try:
    print("Importing torch...", flush=True)
    import torch
    
    print("Importing config and snapshot utils...", flush=True)
    from ingestion.config import IngestionSettings
    from ingestion.model_loading import resolve_cached_snapshot_path
    
    print("Importing Qwen model...", flush=True)
    from transformers import Qwen2_5_VLForConditionalGeneration

    settings = IngestionSettings()
    hf_cache = settings.workspace_hf_cache.resolve()

    os.environ["HF_HOME"] = str(settings.workspace_hf_home.resolve())
    os.environ["HF_HUB_CACHE"] = str(hf_cache)

    path = resolve_cached_snapshot_path(settings.qwen_vl_model_id, hf_cache)
    print(f"Resolved model path: {path}", flush=True)

    print("Calling from_pretrained (this may take a few seconds)...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True
    )

    print("Model loaded successfully! Printing classes...", flush=True)
    classes = {m.__class__.__module__ + "." + m.__class__.__name__ for m in model.modules()}
    print("\n--- ALL MODULE CLASSES ---", flush=True)
    for c in sorted(classes):
        print(c, flush=True)

except Exception as e:
    print(f"\n!!! Exception occurred: {str(e)}", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()

print("Finished diag.py execution.", flush=True)
