from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class LoadingProgressReporter:
    """Reports phased model-loading progress as a single 0-100% line."""

    def __init__(self, label: str = "Loading") -> None:
        self.label = label
        self._last_pct = -1.0

    def update(self, pct: float, message: str) -> None:
        clamped = max(0.0, min(100.0, pct))
        if clamped <= self._last_pct and self._last_pct >= 0:
            return
        self._last_pct = clamped
        line = f"[{self.label} {clamped:5.1f}%] {message}"
        logger.info(line)
        print(line, flush=True)

    def phase(self, start_pct: float, end_pct: float, message: str, fn: Callable[[], None]) -> None:
        self.update(start_pct, message)
        fn()
        self.update(end_pct, f"{message} — done")


def ensure_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Qwen2.5-VL-3B AWQ on GPU. "
            "Install CUDA-enabled PyTorch (see requirements-gpu.txt) and verify your GPU driver."
        )


def download_model_shards(
    model_id: str,
    cache_dir: Path,
    reporter: LoadingProgressReporter,
    start_pct: float = 30.0,
    end_pct: float = 70.0,
) -> None:
    """Pre-download weight shards with per-shard progress mapped into [start_pct, end_pct]."""
    repo_cache_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    snapshot_root = repo_cache_dir / "snapshots"
    if snapshot_root.exists():
        snapshot_dirs = [path for path in snapshot_root.iterdir() if path.is_dir()]
        if snapshot_dirs:
            weight_files = []
            for snapshot_dir in snapshot_dirs:
                weight_files.extend(snapshot_dir.glob("*.safetensors"))
                weight_files.extend(snapshot_dir.glob("*.bin"))
                weight_files.extend(snapshot_dir.glob("*.pt"))
            if weight_files:
                reporter.update(end_pct, f"Using cached model weights ({len(weight_files)} shard(s))")
                return

    from huggingface_hub import hf_hub_download, list_repo_files

    weight_suffixes = (".safetensors", ".bin", ".pt")
    try:
        repo_files = list_repo_files(model_id)
    except Exception as exc:
        logger.warning("Could not list repo files for %s: %s — skipping shard pre-download", model_id, exc)
        reporter.update(end_pct, "Using cached or inline model download")
        return

    shards = sorted(
        name
        for name in repo_files
        if name.endswith(weight_suffixes) and not name.startswith(".")
    )
    if not shards:
        reporter.update(end_pct, "No separate weight shards to pre-download")
        return

    span = end_pct - start_pct
    for index, shard_name in enumerate(shards, start=1):
        shard_start = start_pct + span * (index - 1) / len(shards)
        shard_end = start_pct + span * index / len(shards)
        reporter.update(shard_start, f"Downloading model weights ({index}/{len(shards)}): {shard_name}")

        def _download(name: str = shard_name) -> None:
            hf_hub_download(
                repo_id=model_id,
                filename=name,
                cache_dir=str(cache_dir),
            )

        reporter.phase(shard_start, shard_end, f"Downloaded {shard_name}", _download)


def resolve_cached_snapshot_path(model_id: str, cache_dir: Path) -> Path | None:
    repo_cache_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    snapshot_root = repo_cache_dir / "snapshots"
    if not snapshot_root.exists():
        return None

    snapshot_dirs = sorted(
        (path for path in snapshot_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot_dir in snapshot_dirs:
        config_file = snapshot_dir / "config.json"
        if config_file.exists():
            return snapshot_dir
    return None
