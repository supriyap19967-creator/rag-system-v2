import tempfile
import unittest
from pathlib import Path

from vectordb.fastembed_runtime import (
    FastEmbedRuntimeSettings,
    resolve_verified_model_path,
    verify_model_directory,
)


class FastEmbedRuntimeTests(unittest.TestCase):
    def test_verify_model_directory_accepts_onnx_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "bm25"
            model_dir.mkdir()
            (model_dir / "model.onnx").write_bytes(b"onnx")
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            self.assertTrue(verify_model_directory(model_dir))

    def test_resolve_verified_model_path_uses_specific_model_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "bm25"
            model_dir.mkdir()
            (model_dir / "model.onnx").write_bytes(b"onnx")
            settings = FastEmbedRuntimeSettings(
                model_name="Qdrant/bm25",
                cache_dir=Path(tmp_dir) / "cache",
                specific_model_path=str(model_dir),
            )
            resolved = resolve_verified_model_path(settings)
            self.assertEqual(resolved, model_dir.resolve())

    def test_resolve_verified_model_path_returns_none_when_cache_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = FastEmbedRuntimeSettings(
                model_name="Qdrant/bm25",
                cache_dir=Path(tmp_dir) / "cache",
                specific_model_path="",
                local_files_only=True,
                allow_network_download=False,
            )
            self.assertIsNone(resolve_verified_model_path(settings))


if __name__ == "__main__":
    unittest.main()
