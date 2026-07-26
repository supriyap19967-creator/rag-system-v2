import tempfile
import unittest
from pathlib import Path

from ingestion.visual_paths import absolute_asset_path, canonical_flat_image_path


class VisualPathTests(unittest.TestCase):
    def test_canonical_flat_image_path_creates_absolute_copy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "assets" / "extracted_images"
            scratch = Path(tmp_dir) / "scratch"
            scratch.mkdir(parents=True)
            source = scratch / "raw.png"
            source.write_bytes(b"png-bytes")

            destination = canonical_flat_image_path(
                root,
                source,
                page_number=208,
                visual_type="figure",
                entity_label="Figure 4.2",
            )

            self.assertTrue(destination.is_absolute())
            self.assertTrue(destination.exists())
            self.assertEqual(destination.name, "page_208_Figure_Figure_4.2.png")
            self.assertEqual(absolute_asset_path(destination), str(destination.resolve()))


if __name__ == "__main__":
    unittest.main()
