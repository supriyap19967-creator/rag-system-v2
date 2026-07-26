import sys, os
sys.path.insert(0, "c:/Users/supri/recovered-rag-project")
os.chdir("c:/Users/supri/recovered-rag-project")
import warnings; warnings.filterwarnings("ignore")

import torch
from ingestion.config import IngestionSettings
from ingestion.model_loading import resolve_cached_snapshot_path
from transformers import Qwen2_5_VLForConditionalGeneration

s = IngestionSettings()
hf = s.workspace_hf_cache.resolve()
src = resolve_cached_snapshot_path(s.qwen_vl_model_id, hf) or s.qwen_vl_model_id

out_lines = [f"Model source: {src}"]
out_lines.append(f"PyTorch: {torch.__version__}")

import transformers
out_lines.append(f"Transformers: {transformers.__version__}")

out_lines.append("Loading model...")
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    src,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
    trust_remote_code=True,
)
out_lines.append("Model loaded. Counting module types...")

types: dict[str, int] = {}
for name, mod in m.named_modules():
    cls = mod.__class__.__module__ + "." + mod.__class__.__name__
    types[cls] = types.get(cls, 0) + 1

out_lines.append(f"Total unique module types: {len(types)}")
out_lines.append("Top 25 by count:")
for cls, cnt in sorted(types.items(), key=lambda x: -x[1])[:25]:
    out_lines.append(f"  {cnt:4d}  {cls}")

# Check specific tensor attributes across all modules
out_lines.append("\nModules with non-standard tensor attributes:")
attr_map: dict[str, list[str]] = {}
for name, mod in m.named_modules():
    tensor_attrs = [
        a for a in vars(mod)
        if not a.startswith("_")
        and isinstance(getattr(mod, a, None), torch.Tensor)
        and a not in ("weight", "bias")
    ]
    if tensor_attrs:
        key = mod.__class__.__name__
        if key not in attr_map:
            attr_map[key] = tensor_attrs
            dtype_info = {a: str(getattr(mod, a).dtype) for a in tensor_attrs[:4]}
            out_lines.append(f"  {key}: {dtype_info}")

result_path = r"c:\Users\supri\recovered-rag-project\awq_diag.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"Written to {result_path}")
