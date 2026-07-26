from __future__ import annotations

from pathlib import Path

from ingestion.pdf_chunking import PdfVisualCandidate, build_visual_blocks
from ingestion.schemas import ContentBlock, EnrichedDocument, ExtractedImage, VisionDescription


def _vision_block(description: VisionDescription) -> str:
    text = _combined_visual_text(description)
    if text.strip().startswith("[CHART DESCRIPTION]"):
        return text.strip()
    return (
        "[CHART DESCRIPTION]\n\n"
        f"{text.strip()}\n\n"
        "[/CHART DESCRIPTION]"
    )


def _combined_visual_text(description: VisionDescription) -> str:
    source_label = str(
        description.metadata.get("source_label")
        or description.metadata.get("caption")
        or description.metadata.get("text")
        or ""
    ).strip()
    visual_analysis = description.description.strip()
    if not source_label or visual_analysis.startswith("Source Label:"):
        return visual_analysis
    return f"Source Label: {source_label}\n\nVisual Analysis:\n{visual_analysis}"


def _table_summary(block: ContentBlock) -> str:
    lines = [line.strip() for line in block.text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return ""
    headers = [part.strip() for part in table_lines[0].strip("|").split("|")]
    row_count = max(len(table_lines) - 2, 0)
    return f"Table summary: this table has {row_count} data rows with columns: {', '.join(headers)}."


class ContentMerger:
    """Merges structured PDF text blocks with visual caption/asset/context children."""

    def merge(
        self,
        document: EnrichedDocument,
        images: list[ExtractedImage],
        descriptions: list[VisionDescription],
    ) -> EnrichedDocument:
        markdown_parts = [document.markdown.strip()]
        extra_blocks: list[ContentBlock] = []
        if document.metadata.get("source_type") == "pdf":
            visual_candidates = [
                PdfVisualCandidate(**candidate)
                for candidate in document.metadata.get("visual_candidates", [])
            ]
            extra_blocks.extend(
                build_visual_blocks(
                    pdf_path=Path(document.source_path),
                    document=document,
                    visual_candidates=visual_candidates,
                    images=images,
                    descriptions=descriptions,
                )
            )
        else:
            for block in document.blocks:
                if block.type == "table":
                    summary = _table_summary(block)
                    if summary:
                        extra_blocks.append(
                            ContentBlock(
                                text=summary,
                                type="table",
                                page=block.page,
                                source_path=block.source_path,
                                metadata={**block.metadata, "summary_generated": True},
                            )
                        )
                        markdown_parts.extend(["", summary])
        return EnrichedDocument(
            source_path=document.source_path,
            markdown="\n".join(markdown_parts).strip() + "\n",
            blocks=[*document.blocks, *extra_blocks],
            images=images,
            metadata={
                **document.metadata,
                "visual_count": len(images),
                "vision_description_count": len(descriptions),
                "chart_metadata": [
                    {
                        "type": description.type,
                        "page": description.page,
                        "image_path": str(description.image_path),
                        "description": description.description,
                    }
                    for description in descriptions
                ],
            },
        )
