import sys
import types

# Mock sentence_transformers trainer, training_args, cross_encoder, and sparse_encoder to bypass Trainer imports
sys.modules['sentence_transformers.trainer'] = types.ModuleType('sentence_transformers.trainer')
sys.modules['sentence_transformers.trainer'].SentenceTransformerTrainer = None

sys.modules['sentence_transformers.training_args'] = types.ModuleType('sentence_transformers.training_args')
sys.modules['sentence_transformers.training_args'].SentenceTransformerTrainingArguments = None
sys.modules['sentence_transformers.training_args'].BatchSamplers = None
sys.modules['sentence_transformers.training_args'].MultiDatasetBatchSamplers = None

sys.modules['sentence_transformers.sparse_encoder'] = types.ModuleType('sentence_transformers.sparse_encoder')
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoder = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderModelCardData = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderTrainer = None
sys.modules['sentence_transformers.sparse_encoder'].SparseEncoderTrainingArguments = None

sys.modules['sentence_transformers.cross_encoder'] = types.ModuleType('sentence_transformers.cross_encoder')
sys.modules['sentence_transformers.cross_encoder'].CrossEncoder = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderModelCardData = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderTrainer = None
sys.modules['sentence_transformers.cross_encoder'].CrossEncoderTrainingArguments = None

# Configure stdout to support UTF-8 (emojis) on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import logging
import os
import json
import uuid
import re
import time
from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import asyncio
import io
from PIL import Image
import fitz  # PyMuPDF
from google import genai
from google.genai import types as genai_types

# Add project root to path to resolve custom modules correctly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

# Force environment variables for the new Gemini key and remove standard Google keys
os.environ.pop("GOOGLE_API_KEY", None)
api_key = os.getenv("GCP_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "conversational_rag"
PDF_PATH = PROJECT_ROOT / "Data" / "Pdf" / "World Development Report 2025.pdf"
VISUAL_JSONL_PATH = PROJECT_ROOT / "Data" / "visual_rechunk_20260620.jsonl"
CACHE_DIR = PROJECT_ROOT / "data_cache" / "high_res_visual_anchor_texts"
OUTPUT_IMAGES_DIR = PROJECT_ROOT / "extracted_images"

def _stable_chunk_id(chunk_content: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_content)))

def extract_proximity_context(page, asset_bbox, max_words=300):
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return ""
    ax0, ay0, ax1, ay1 = asset_bbox
    above_blocks = []
    below_blocks = []
    for block in blocks:
        bx0, by0, bx1, by1, text, block_no, block_type = block
        text = text.strip()
        if not text:
            continue
        if by1 <= ay0 + 5:
            above_blocks.append((by1, text))
        elif by0 >= ay1 - 5:
            below_blocks.append((by0, text))
    
    above_blocks.sort(key=lambda x: x[0], reverse=True)
    below_blocks.sort(key=lambda x: x[0])
    
    above_text_list = []
    above_word_count = 0
    for _, text in above_blocks:
        words = text.split()
        if above_word_count + len(words) <= max_words:
            above_text_list.append(text)
            above_word_count += len(words)
        else:
            remaining = max_words - above_word_count
            if remaining > 0:
                above_text_list.append(" ".join(words[-remaining:]))
            break
            
    below_text_list = []
    below_word_count = 0
    for _, text in below_blocks:
        words = text.split()
        if below_word_count + len(words) <= max_words:
            below_text_list.append(text)
            below_word_count += len(words)
        else:
            remaining = max_words - below_word_count
            if remaining > 0:
                below_text_list.append(" ".join(words[:remaining]))
            break
            
    above_text = "\n".join(reversed(above_text_list)).strip()
    below_text = "\n".join(below_text_list).strip()
    
    parts = []
    if above_text:
        parts.append(f"[CONTEXT ABOVE]:\n{above_text}")
    if below_text:
        parts.append(f"[CONTEXT BELOW]:\n{below_text}")
        
    return "\n\n".join(parts)

def extract_asset_id(caption: str) -> str:
    match = re.search(r"\b(?:Figure|Table|Chart|Diagram|Graph|Spotlight|Box)\s*([A-Za-z]?\d+(?:\.\d+)*)\b", caption, re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown"

def main():
    logger.info("Starting Upgraded Visual PDF extraction & Qdrant Ingestion...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        logger.error(f"PDF file not found at {PDF_PATH}!")
        sys.exit(1)

    if not VISUAL_JSONL_PATH.exists():
        logger.error(f"Visual JSONL path not found at {VISUAL_JSONL_PATH}!")
        sys.exit(1)

    # 1. Parse JSONL file to get visual candidates
    logger.info("Loading visual candidates from JSONL...")
    visual_candidates = []
    seen_cand_ids = set()
    
    # We first build a helper map for descriptions associated with entity_id, page_no
    desc_map = {}
    with open(VISUAL_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            metadata = item.get("metadata") or {}
            ent_id = metadata.get("entity_id")
            page_no = metadata.get("page_no") or metadata.get("page_number")
            desc = metadata.get("description")
            if ent_id and page_no and desc:
                desc_map[(ent_id, page_no)] = desc

    with open(VISUAL_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            metadata = item.get("metadata") or {}
            
            # Chunk top-level candidate check
            ent_id = metadata.get("entity_id")
            page_no = metadata.get("page_no") or metadata.get("page_number")
            bbox = metadata.get("bbox")
            desc = metadata.get("description") or desc_map.get((ent_id, page_no), "")
            
            if ent_id and page_no:
                ent_lower = ent_id.lower()
                title = metadata.get("visual_title") or metadata.get("caption_text") or ent_id
                title_lower = title.lower()
                
                # Caption Validation Gate
                if "table" in title_lower or "table" in ent_lower:
                    asset_type = "table"
                elif "figure" in title_lower or "figure" in ent_lower:
                    asset_type = "figure"
                else:
                    continue
                
                key = (ent_id, page_no)
                if key not in seen_cand_ids:
                    seen_cand_ids.add(key)
                    visual_candidates.append({
                        "entity_id": ent_id,
                        "page_no": page_no,
                        "bbox": bbox,
                        "caption_text": title,
                        "asset_type": asset_type,
                        "description": desc
                    })
            
            # Array visual_candidates check
            cands = metadata.get("visual_candidates") or []
            for cand in cands:
                cand_id = cand.get("entity_id")
                c_page_no = cand.get("page_no")
                c_bbox = cand.get("bbox")
                c_title = cand.get("visual_title") or cand.get("caption_text") or cand_id or ""
                c_desc = desc_map.get((cand_id, c_page_no), "")
                
                if cand_id and c_page_no:
                    cand_lower = cand_id.lower()
                    c_title_lower = c_title.lower()
                    
                    # Caption Validation Gate
                    if "table" in c_title_lower or "table" in cand_lower:
                        c_asset_type = "table"
                    elif "figure" in c_title_lower or "figure" in cand_lower:
                        c_asset_type = "figure"
                    else:
                        continue
                    
                    key = (cand_id, c_page_no)
                    if key not in seen_cand_ids:
                        seen_cand_ids.add(key)
                        visual_candidates.append({
                            "entity_id": cand_id,
                            "page_no": c_page_no,
                            "bbox": c_bbox,
                            "caption_text": c_title,
                            "asset_type": c_asset_type,
                            "description": c_desc
                        })

    logger.info(f"Loaded {len(visual_candidates)} unique figures/tables from metadata.")

    # Group candidates by page_no for overlap calculations
    from collections import defaultdict
    page_candidates = defaultdict(list)
    for cand in visual_candidates:
        page_candidates[cand["page_no"]].append(cand)

    # Open PDF for page-level cropping
    doc = fitz.open(str(PDF_PATH))
    genai_client = genai.Client()

    # Precompute snapped bboxes for all candidates using Docling Layout
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import AcceleratorOptions

    DOCLING_ARTIFACTS_PATH = PROJECT_ROOT / "docling_models"
    os.environ["DOCLING_ARTIFACTS_PATH"] = str(DOCLING_ARTIFACTS_PATH)

    logger.info("Initializing Docling single-page layout analyzer...")
    docling_pipeline_options = PdfPipelineOptions(
        document_timeout=30.0,
        artifacts_path=DOCLING_ARTIFACTS_PATH,
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=2),
        do_ocr=False,                  # Keep it extremely fast and lightweight
        do_table_structure=True,        # Enable neural table-cell grouping
        generate_picture_images=True,   # Enable picture bounding boxes
        generate_table_images=True,
        force_backend_text=True,
    )
    docling_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=docling_pipeline_options)}
    )

    def docling_bbox_to_pymupdf(bbox, page_height):
        origin = getattr(bbox, 'coord_origin', 'BOTTOMLEFT')
        if str(origin).upper().endswith('BOTTOMLEFT'):
            y_coords = [page_height - bbox.t, page_height - bbox.b]
            return [bbox.l, min(y_coords), bbox.r, max(y_coords)]
        else:
            return [bbox.l, bbox.t, bbox.r, bbox.b]

    def find_closest_docling_bbox(cand, docling_doc, page_height):
        asset_type = cand["asset_type"]
        caption_bbox = cand["bbox"]
        if not caption_bbox:
            return None
        
        cx0, cy0, cx1, cy1 = caption_bbox
        
        tables = []
        pictures = []
        if docling_doc:
            for item, _ in docling_doc.iterate_items():
                cls_name = item.__class__.__name__
                prov = getattr(item, 'prov', None)
                if not prov:
                    continue
                p = prov[0] if isinstance(prov, list) else prov
                bbox = getattr(p, 'bbox', None)
                if not bbox:
                    continue
                
                pymupdf_box = docling_bbox_to_pymupdf(bbox, page_height)
                
                if cls_name == "TableItem":
                    tables.append(pymupdf_box)
                elif cls_name == "PictureItem":
                    pictures.append(pymupdf_box)
                    
        primary_list = tables if asset_type == "table" else pictures
        secondary_list = pictures if asset_type == "table" else tables
        
        def get_closest(bbox_list):
            best_box = None
            min_dist = float('inf')
            for box in bbox_list:
                bx0, by0, bx1, by1 = box
                dx = max(0, cx0 - bx1, bx0 - cx1)
                dy = max(0, cy0 - by1, by0 - cy1)
                dist = (dx*dx + dy*dy)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_box = box
            return best_box, min_dist

        best_box, min_dist = get_closest(primary_list)
        if best_box and min_dist < 250:
            return best_box
            
        best_fallback_box, fallback_dist = get_closest(secondary_list)
        if best_fallback_box and fallback_dist < 250:
            return best_fallback_box
            
        return best_box

    logger.info("Precomputing snapped bboxes with native Docling object boundaries...")
    for p_no, cands in page_candidates.items():
        if p_no - 1 < 0 or p_no - 1 >= len(doc):
            continue
        page = doc[p_no - 1]
        
        docling_doc = None
        try:
            logger.info(f"Running Docling page conversion for physical page {p_no}...")
            docling_result = docling_converter.convert(str(PDF_PATH), page_range=(p_no, p_no))
            docling_doc = docling_result.document
        except Exception as exc:
            logger.warning(f"Docling page {p_no} conversion failed: {exc}")
            
        for cand in cands:
            bbox = cand["bbox"]
            if bbox:
                x0, y0, x1, y1 = bbox
            else:
                x0, y0, x1, y1 = 50, 50, page.rect.width - 50, page.rect.height - 50
            
            snapped = find_closest_docling_bbox(cand, docling_doc, page.rect.height)
            
            if snapped:
                cand["snapped_bbox"] = list(snapped)
                # Pad by standard 20-pixel padding on all sides
                pad = 20
                cand["snapped_bbox"][0] = max(0.0, cand["snapped_bbox"][0] - pad)
                cand["snapped_bbox"][1] = max(0.0, cand["snapped_bbox"][1] - pad)
                cand["snapped_bbox"][2] = min(page.rect.width, cand["snapped_bbox"][2] + pad)
                cand["snapped_bbox"][3] = min(page.rect.height, cand["snapped_bbox"][3] + pad)
                logger.info(f"Snapped candidate {cand['entity_id']} to Docling bbox {cand['snapped_bbox']}")
            else:
                # Slicing defect fix: Fallback to generous vertical expansion to capture full graphic matrix
                if y0 < page.rect.height / 2:
                    y1_new = min(page.rect.height - 50, y1 + 350)
                    y0_new = max(50, y0 - 20)
                    x0_new = 50
                    x1_new = page.rect.width - 50
                else:
                    y0_new = max(50, y0 - 350)
                    y1_new = min(page.rect.height - 50, y1 + 20)
                    x0_new = 50
                    x1_new = page.rect.width - 50
                cand["snapped_bbox"] = [x0_new, y0_new, x1_new, y1_new]
                logger.info(f"Fallback bbox for candidate {cand['entity_id']}: {cand['snapped_bbox']}")

    async def process_candidate(idx, cand, doc, genai_client, sem):
        page_no = cand["page_no"]
        bbox = cand["bbox"]
        caption_text = cand["caption_text"]
        asset_type = cand["asset_type"]
        asset_id = extract_asset_id(caption_text)
        
        if asset_id == "unknown":
            asset_id = extract_asset_id(cand["entity_id"])
        
        if asset_id == "unknown":
            asset_id = f"{page_no}.{idx + 1}"

        image_filename = f"world_development_report_2025_{asset_type}_{asset_id.replace('.', '_')}.png"
        image_path = OUTPUT_IMAGES_DIR / image_filename

        logger.info(f"[{idx+1}/{len(visual_candidates)}] Processing {asset_type.upper()} {asset_id} on Page {page_no}...")

        # PyMuPDF Page-level extraction
        if page_no - 1 < 0 or page_no - 1 >= len(doc):
            logger.warning(f"Page number {page_no} is out of bounds for the PDF (length: {len(doc)}). Skipping.")
            return None
        page = doc[page_no - 1]
        
        # Get precomputed snapped bbox
        x0, y0, x1, y1 = cand["snapped_bbox"]

        # Enforce page rect limits
        page_rect = page.rect
        x0 = max(0, min(x0, page_rect.width))
        y0 = max(0, min(y0, page_rect.height))
        x1 = max(0, min(x1, page_rect.width))
        y1 = max(0, min(y1, page_rect.height))

        # Extract proximity context
        nearby_context = extract_proximity_context(page, (x0, y0, x1, y1), max_words=300)

        if x1 > x0 + 5 and y1 > y0 + 5:
            rect = fitz.Rect(x0, y0, x1, y1)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
            pix.save(str(image_path))
        else:
            # Fallback
            rect = page.rect
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
            pix.save(str(image_path))

        # Determine anchor_text
        cache_file = CACHE_DIR / f"{page_no}_{asset_type}_{asset_id.replace('.', '_')}.json"
        anchor_text = ""

        # Check local cache first (bypassed for fresh extraction)
        if False:
            try:
                with open(cache_file, "r", encoding="utf-8") as cf:
                    cached_data = json.load(cf)
                    anchor_text = cached_data.get("anchor_text", "")
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")

        # Fallback to Gemini if no valid cached description exists
        if not anchor_text:
            try:
                img_bytes_io = io.BytesIO()
                with Image.open(image_path) as img:
                    max_dim = 1024
                    if max(img.size) > max_dim:
                        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    img.save(img_bytes_io, format="PNG", optimize=True)
                img_data = img_bytes_io.getvalue()
            except Exception as img_err:
                logger.warning(f"Failed compressing image: {img_err}. Using original bytes.")
                img_data = image_path.read_bytes()

            prompt_vision = (
                "You are a precise technical document parser. Your task is to extract ALL information from the provided image and explain it entirely in clean, well-formed paragraphs and comprehensive sentences. You must NOT output lazy labels, raw numbers, dry axis lists, or markdown grid tables.\n\n"
                "Follow these strict formatting rules:\n"
                "1. FOR CHARTS AND VISUALS: Synthesize the data into a narrative explanation. Explain what the visual represents, the relationship between the trends, what the X and Y axes signify contextually, and provide a thorough, written breakdown of key findings, conclusions, and data points shown in clean paragraph form.\n"
                "2. FOR TABLES: Do NOT output a markdown grid or structured table. Instead, translate the tabular data into a highly detailed textual narrative, explaining the rows, column relationships, and values in structured paragraph form.\n"
                "3. Exhaustively transcribe all titles, subtitles, headers, data labels, and footnotes verbatim, but present them in clean, well-formed paragraphs and complete sentences."
            )

            gemini_success = False
            for attempt in range(1, 10):
                await sem.acquire()
                try:
                    user_prompt = "Analyze the provided image."
                    if caption_text:
                        user_prompt += f"\n\nCaption from PDF: {caption_text}"
                    if nearby_context:
                        user_prompt += f"\n\nNearby text context from the page:\n{nearby_context}"

                    response = await genai_client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            genai_types.Part.from_bytes(data=img_data, mime_type="image/png"),
                            user_prompt
                        ],
                        config=genai_types.GenerateContentConfig(
                            system_instruction=prompt_vision,
                            temperature=0.0,
                        )
                    )
                    anchor_text = response.text or ""
                    gemini_success = True
                    sem.release()
                    break
                except Exception as exc:
                    sem.release()
                    exc_str = str(exc).lower()
                    is_rate_limit = "429" in exc_str or "resource_exhausted" in exc_str or "503" in exc_str or "service unavailable" in exc_str
                    if is_rate_limit:
                        logger.warning(f"[{idx+1}] Gemini API rate limit/503 hit. Releasing slot and backing off exactly 5 seconds before retry (attempt {attempt})...")
                    else:
                        logger.warning(f"[{idx+1}] Gemini API call failed: {exc}. Releasing slot and backing off exactly 5 seconds before retry...")
                    await asyncio.sleep(5.0)

            if not gemini_success:
                anchor_text = f"[Anchor Text Fallback]: Clean data extraction for {asset_type} {asset_id}."

            # Save to cache
            try:
                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump({"anchor_text": anchor_text}, cf)
            except Exception as e:
                logger.warning(f"Failed to write cache file {cache_file}: {e}")

        return {
            "asset_type": asset_type,
            "asset_id": asset_id,
            "caption_text": caption_text,
            "anchor_text": anchor_text,
            "image_path": str(image_path.resolve()),
            "page_no": page_no,
            "nearby_context": nearby_context
        }

    async def run_pipeline():
        sem = asyncio.Semaphore(5)
        tasks = []
        for idx, cand in enumerate(visual_candidates):
            tasks.append(process_candidate(idx, cand, doc, genai_client, sem))
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    logger.info("Executing async visual extraction pipeline...")
    processed_points = asyncio.run(run_pipeline())
    doc.close()
    logger.info(f"Extraction and captioning complete. Total visual assets: {len(processed_points)}")

    # Setup Qdrant Client (managed client builder supporting local disk)
    from vectordb.qdrant_client_manager import get_qdrant_client as build_managed_qdrant_client
    client = build_managed_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        logger.error(f"Qdrant collection {COLLECTION_NAME} does not exist!")
        sys.exit(1)

    # Wipe ONLY previous visual extraction points
    logger.info("Purging old visual extraction points from Qdrant collection...")
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.Filter(
            should=[
                models.FieldCondition(key="contains_image", match=models.MatchValue(value=True)),
                models.FieldCondition(key="contains_figure", match=models.MatchValue(value=True)),
                models.FieldCondition(key="metadata.contains_image", match=models.MatchValue(value=True)),
                models.FieldCondition(key="metadata.contains_figure", match=models.MatchValue(value=True)),
            ]
        )
    )
    print("🧹 Old contaminated visual extraction vectors successfully wiped.")

    # Load local SentenceTransformer model
    logger.info("Loading local SentenceTransformer('all-MiniLM-L6-v2')...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate Embeddings & Construct Qdrant points
    points = []
    logger.info("Generating embeddings and constructing Qdrant payload points...")
    for idx, item in enumerate(processed_points):
        text_payload = f"Caption: {item['caption_text']}\n\nAnchor Data:\n{item['anchor_text']}"
        if item.get("nearby_context"):
            text_payload += f"\n\nNearby Context:\n{item['nearby_context']}"
        dense_vector = embed_model.encode(text_payload, convert_to_numpy=True).tolist()

        # METADATA DICTIONARY SCHEMA STRUCTURE
        payload = {
            "text": text_payload,
            "page_content": text_payload,
            "source": "World Development Report 2025.pdf",
            "image_path": item["image_path"],
            "contains_chart": item["asset_type"] == "figure",
            "contains_table": item["asset_type"] == "table",
            "contains_figure": item["asset_type"] == "figure",
            "contains_image": True,
            "contains_csv": False,
            "document_type": "pdf_visual",
            "metadata": {
                "asset_type": item["asset_type"],
                "asset_id": item["asset_id"],
                "caption_text": item["caption_text"],
                "anchor_text": item["anchor_text"],
                "image_path": item["image_path"],
                "nearby_context": item.get("nearby_context", ""),
                "document_type": "pdf_visual",
                "page_number": item["page_no"],
                "chunk_id": _stable_chunk_id(text_payload)
            }
        }

        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, payload["metadata"]["chunk_id"])),
                vector={
                    "dense": dense_vector,
                    "sparse": models.SparseVector(indices=[], values=[]),
                },
                payload=payload
            )
        )

    # Ingest points in batches
    batch_size = 64
    total_points = len(points)
    uploaded = 0
    logger.info(f"Upserting {total_points} new visual points...")
    for start in range(0, total_points, batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        uploaded += len(batch)
        logger.info(f"Upserted {uploaded}/{total_points} points.")

    # 4. COLLECTION POST-COUNT AUDIT
    csv_count = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            should=[
                models.FieldCondition(key="document_type", match=models.MatchValue(value="csv")),
                models.FieldCondition(key="metadata.document_type", match=models.MatchValue(value="csv"))
            ]
        )
    ).count

    text_count = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            should=[
                models.FieldCondition(key="document_type", match=models.MatchValue(value="pdf")),
                models.FieldCondition(key="metadata.document_type", match=models.MatchValue(value="pdf"))
            ]
        )
    ).count

    visual_count = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            should=[
                models.FieldCondition(key="document_type", match=models.MatchValue(value="pdf_visual")),
                models.FieldCondition(key="metadata.document_type", match=models.MatchValue(value="pdf_visual"))
            ]
        )
    ).count

    client.close()

    print("🧹 Old contaminated visual extraction vectors successfully wiped.")
    print(f"📈 Total Production Collection Counts:")
    print(f" - CSV Chunk Count: {csv_count}")
    print(f" - Text-Only Chunk Count: {text_count}")
    print(f" - Cleaned Visual Extraction Chunk Count: {visual_count}")

if __name__ == "__main__":
    main()
