---
title: Rag System V2
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app/main.py
pinned: false
---

# Multimodal Agentic RAG

An agentic, multi-modal RAG architecture built to extract, reason over, and retrieve unstructured document text, tabular CSV data, and visual charts.

### Key Capabilities

- Agentic Routing: Powered by Pydantic-AI to dynamically orchestrate queries across Qdrant vector search, pandas dataframe execution, and OpenRouter Gemini Vision.
- Visual Grounding: Features a dynamic path-resolution registry to map document figures directly to original raw crop images and bounding boxes.
- 13-Layer Guardrail Safety: Built-in validation gauntlet covering path safety, quote anchoring, rate limits, and faithfulness with automated Self-RAG loops.
- Interactive UI: Streamlit interface rendering grounded citations, Markdown tables, and exact visual assets inline.

### Tech Stack & System Architecture

Frontend & Presentation
- Streamlit : Interactive chat UI, dynamic source citations, extracted Markdown tables, and visual chart rendering

Core Agent Framework
- Pydantic-AI: Agent reasoning, tool calling, structured BaseModel output validation, and self-correction loops

Models & Orchestration
- Groq / NVIDIA NIM API: High-speed LLM reasoning engine for text-based synthesis
- Google Gemini 2.5 Flash (via OpenRouter): Vision-Language Model (VLM) for high-fidelity OCR, table parsing, and visual image analysis

Storage & Retrieval
- Qdrant: Vector database for document indexing and hybrid semantic/keyword search
- Pandas: Dynamic query execution engine for structured CSV data, math, and filtering

Embeddings & Reranking
- Sentence-Transformers: Dense vector embedding generation
- Cross-Encoder Rerankers: Top-K chunk relevance optimization before LLM generation

Security, Safety & Guardrails
- Custom RAGMasterSafetyGauntlet: 13-layer safety engine for PII redaction, prompt injection defense, rate limiting, path safety, quote anchoring, and faithfulness evaluation

Observability & Tracing
- Langfuse: Real-time execution tracing, latency tracking, token usage, and safety scoring
- OpenTelemetry: Standardized agent execution logging and telemetry

# Enterprise Multimodal Conversational RAG System: Flowcharts & Architecture

This document provides an end-to-end, interview-grade architectural specification and system flowcharts for our **Enterprise Multimodal Conversational RAG System**. It covers the complete lifecycle of data ingestion, contextual query rewriting, deterministic intent routing, multi-agent collaboration, parallel multi-threaded retrieval, and 14-layer compliance gauntlet validation.

---

## 1. High-Level Architecture Overview (7-Layer Multimodal Pipeline)

```mermaid
flowchart TD
    subgraph ClientGateway ["Layer 1: Client Interface & Gateway Control"]
        User(["👤 User Input Question"]) -->|"HTTP POST /query or WS"| StreamlitUI["Streamlit UI Container<br/>[StreamlitApp.py]"]
        StreamlitUI -->|"Raw Question + Session ID"| Gateway["Gateway Guardrails<br/>[gateway_guardrails.py]"]
        Gateway -->|"Layer 1-3 Pre-validation"| PIIRedact["PII Masking & History Redaction<br/>[gateway_guardrails.py]"]
    end

    subgraph MemoryContext ["Layer 2: Memory & Contextual Query Engine"]
        PIIRedact -->|"Sanitized Query"| MemoryManager["Multimodal Conversation Manager<br/>[conversation_manager.py]"]
        MemoryManager <-->|"Fetch Past 4 Turns & Active Assets"| SessionStore[("Session Memory & Active Asset Registry<br/>LAST_ACTIVE_IMAGE_PATH")]
        MemoryManager -->|"Anaphora Rewritten Standalone Query"| ContextualizedQuery["Contextualized Query<br/>(e.g., 'above figure' -> 'Figure 4.2')"]
    end

    subgraph IntentRouting ["Layer 3: Intent Classification & Fast-Path Dispatcher"]
        ContextualizedQuery -->|"Query String"| LRUCacheCheck{"LRU Semantic Cache Hit?<br/>[cache.py]"}
        LRUCacheCheck -- "Yes (<10ms Hit)" --> StreamlitUI
        LRUCacheCheck -- "No Cache Hit" --> IntentRouter["Deterministic Intent Router<br/>[intent_router.py]"]
        
        IntentRouter -->|"Intent Decision + Allowed Tools"| IntentBranch{"Classified Intent"}
        IntentBranch -- "VISUAL_SPECIFIC" --> VisionPath["Visual Specific Dispatcher<br/>(allow_pandas: False)"]
        IntentBranch -- "CSV_ONLY" --> FastCSVPath["Direct Parametric CSV Fast-Path<br/>(<5ms Lookup)"]
        IntentBranch -- "HYBRID / RESEARCH" --> AgentOrchestrator["Supervisor Orchestrator Agent<br/>[agents/supervisor.py]"]
    end

    subgraph MultiAgentCore ["Layer 4: Multi-Agent Collaboration Core"]
        AgentOrchestrator -->|"Task Plan & Sub-queries"| AgentTarget{"Target Agent Routing"}
        AgentTarget -- "RESEARCH_AGENT" --> ResearchAgent["Research Agent<br/>[agents/research.py]"]
        AgentTarget -- "VISION_AGENT" --> VisionAgent["Vision Agent<br/>[agents/vision.py]"]
        AgentTarget -- "DATA_AGENT" --> DataAgent["Data Agent<br/>[agents/data.py]"]
        AgentTarget -- "DIRECT_LLM" --> DirectLLM["Direct LLM Generator<br/>[Llama-3.3-70B / Gemini]"]
    end

    subgraph ParallelEngine ["Layer 5: Parallel Multi-Threaded Data Retrieval & Engine"]
        VisionPath & ResearchAgent & DataAgent & VisionAgent -->|"Concurrent Tasks"| ParallelPool["Parallel Worker Pool (ThreadPoolExecutor / asyncio)<br/>max_workers=3"]
        
        ParallelPool -->|"Worker 1: Dense + Sparse Query"| QdrantHybrid["Qdrant Hybrid Vector Search<br/>(BGE-M3 Dense + BM25 Sparse)<br/>[retriever.py]"]
        QdrantHybrid --> Reranker["Cross-Encoder Reranker<br/>(BGE-Reranker-v2-m3)<br/>[reranker.py]"]
        
        ParallelPool -->|"Worker 2: RAM Lookup / PIL Downscale"| VisionStore["In-Memory RAM Transcription Store & VLM<br/>(_IN_MEMORY_TRANSCRIPTION_CACHE)<br/>[schemas_and_agent.py]"]
        
        ParallelPool -->|"Worker 3: Pandas Sandbox"| PandasEngine["Pandas Sandbox Execution<br/>(gdp_df / co2_df DataFrames)<br/>[structured_query.py]"]
    end

    subgraph SynthesisValidation ["Layer 6: Answer Synthesis & 14-Phase Compliance Gauntlet"]
        Reranker & VisionStore & PandasEngine & FastCSVPath & DirectLLM -->|"Top-3 Chunks + Vision Table + CSV Data"| Synthesizer["LLM Synthesizer Engine<br/>[llm.py / query_rag.py]"]
        Synthesizer -->|"Draft Payload"| ValidatorAgent["Validator Agent & Gauntlet Gatekeeper<br/>[agents/validation.py]"]
        
        ValidatorAgent -->|"Payload + Context Chunks"| GauntletPhases["14-Layer Safety Gauntlet Evaluator<br/>[compliance_safety.py]"]
        
        subgraph GauntletPhases ["3-Phase Compliance Guardrail Pipeline"]
            Phase1["Phase 1: Input Security & Pre-Sanitization<br/>• Prompt Injection Filter<br/>• PII Redaction & Masking<br/>• Toxicity & System Leak Scanner"]
            Phase2["Phase 2: Contextual & Structural Alignment<br/>• Asset Path & File Check<br/>• Category & Legend Disambiguator<br/>• Structural JSON Schema Validation"]
            Phase3["Phase 3: Fidelity & Hallucination Defense<br/>• Line-Level Faithfulness Evaluator<br/>• Comma & Numeric Whitelist Normalizer<br/>• Safe Fallback Interceptor"]
            
            Phase1 --> Phase2 --> Phase3
        end
        
        Phase3 -->|"Validation Status"| CheckGauntlet{"Gauntlet Cleared?"}
        CheckGauntlet -- "Passed (Score >=0.90)" --> FinalPayload["Cleared Payload + Citation Card"]
        CheckGauntlet -- "Failed / Violation" --> FallbackHandler["Deterministic Safe Fallback Handler"]
    end

    subgraph Presentation ["Layer 7: UI Presentation & Telemetry"]
        FinalPayload & FallbackHandler -->|"Markdown Narrative + Table Cards"| Formatter["Collapsible Table & UI Formatter<br/>[StreamlitApp.py]"]
        Formatter -->|"Rendered HTML/Markdown + Asset Image"| StreamlitUI
        Formatter -->|"Async Background Log (Fire & Forget)"| Telemetry["Langfuse & OTEL Tracing Telemetry<br/>[StreamlitApp.py]"]
    end
```

---

## Technical Legend & Component Responsibilities

### Subsystem Component Matrix

| Subsystem / Node | Code File Location | Input Payload | Output Payload | Key Technical Responsibilities |
| :--- | :--- | :--- | :--- | :--- |
| **Streamlit UI** | [StreamlitApp.py](file:///C:/Users/supri/recovered-rag-project/streamlit_ui/StreamlitApp.py) | User prompt / Voice WAV | Rendered Markdown + Images | Main user interaction interface, session state maintenance (`LAST_ACTIVE_IMAGE_PATH`), voice transcription trigger, and response streaming. |
| **Gateway Guardrails** | [gateway_guardrails.py](file:///C:/Users/supri/recovered-rag-project/gateway_guardrails.py) | Raw user string | Sanitized query string | Enforces rate limits, token budgets, toxicity filters, PII masking, and Layer 1–3 security checks before retrieval. |
| **Multimodal Conversation Manager** | [conversation_manager.py](file:///C:/Users/supri/recovered-rag-project/app/conversation_manager.py) | Sanitized query + session ID | Contextualized standalone query | Evaluates relative references (`above figure`, `this table`, `it`). Resolves implicit follow-ups against past turns and active asset memory into explicit standalone queries. |
| **Deterministic Intent Router** | [intent_router.py](file:///C:/Users/supri/recovered-rag-project/app/intent_router.py) | Standalone query string | `(QueryIntent, allowed_tools)` | Sub-millisecond (<1ms) intent classifier. Prioritizes `VISUAL_SPECIFIC` for figure queries and bypasses unnecessary LLM supervisor calls. |
| **Supervisor Orchestrator Agent** | [supervisor.py](file:///C:/Users/supri/recovered-rag-project/app/agents/supervisor.py) | Contextualized query + history | Structured TaskPlan | Pydantic AI agent that analyzes complex multi-part queries and delegates sub-tasks to specialized sub-agents (`RESEARCH_AGENT`, `VISION_AGENT`, `DATA_AGENT`). |
| **Research Agent** | [research.py](file:///C:/Users/supri/recovered-rag-project/app/agents/research.py) | Textual sub-query | Key findings + Source notes | Specialized sub-agent for qualitative text search, semantic document retrieval, and report synthesis. |
| **Vision Agent** | [vision.py](file:///C:/Users/supri/recovered-rag-project/app/agents/vision.py) | Image path + user query | Structured VisionOutput payload | Handles visual charts, diagrams, and figures. Features **PIL thumbnail downscaling (max 1024px)** for 70% lighter base64 network payloads. |
| **Data Agent** | [data.py](file:///C:/Users/supri/recovered-rag-project/app/agents/data.py) | Quantitative sub-query | Executive metrics + Code ref | Executes Pandas dataframe code in a sandboxed execution environment for calculations, groupbys, and rankings. |
| **Qdrant Hybrid Retriever** | [retriever.py](file:///C:/Users/supri/recovered-rag-project/app/retriever.py) | Query string + Filters | Raw matching points | Performs dense vector search (BGE-M3, 1024-dim) combined with sparse token search (BM25) inside local Qdrant collections. |
| **Cross-Encoder Reranker** | [reranker.py](file:///C:/Users/supri/recovered-rag-project/app/reranker.py) | Query + Top-N candidates | Top-3 reranked documents | Uses `BAAI/bge-reranker-v2-m3` to compute deep cross-attention similarity scores between query and retrieved document text. |
| **In-Memory RAM Transcription Store** | [schemas_and_agent.py](file:///C:/Users/supri/recovered-rag-project/multimodal-rag-system/schemas_and_agent.py) | Figure / Table ID | Markdown table string | Eagerly pre-loads 215+ pre-computed visual table markdowns into RAM (`_IN_MEMORY_TRANSCRIPTION_CACHE`) on startup, serving extractions in **<10ms**. |
| **Validator Agent & Safety Gauntlet** | [validation.py](file:///C:/Users/supri/recovered-rag-project/app/agents/validation.py) & [compliance_safety.py](file:///C:/Users/supri/recovered-rag-project/compliance_safety.py) | Draft response + Source context | Cleared payload + Faithfulness score | 14-Layer gatekeeper running prompt injection filters, PII checks, system prompt scanners, exact quote anchoring, and line-level faithfulness evaluation. |

---

### Detailed 14-Layer Compliance Safety Gauntlet Breakdown

To ensure zero visual clutter in the high-level architecture diagram while providing full technical depth for technical interviews, the complete 14-layer compliance gauntlet implemented in [compliance_safety.py](file:///C:/Users/supri/recovered-rag-project/compliance_safety.py) is categorized below:

#### **Phase 1: Input Security & Pre-Sanitization (Layers 1–4)**
1. **Layer 1: Prompt Injection & Jailbreak Scanner**: Scans user input for adversarial instructions (e.g. `ignore previous rules`, `DAN mode`).
2. **Layer 2: PII Redaction & Masking Filter**: Masks sensitive personal data (emails, phone numbers, API keys) before query processing.
3. **Layer 3: Toxicity & Financial Tone Compliance**: Verifies queries adhere to regulatory communication standards.
4. **Layer 4: System Prompt Leak Interceptor**: Detects and blocks attempts to extract internal system prompt instructions.

#### **Phase 2: Contextual & Structural Alignment (Layers 5–9)**
5. **Layer 5: Asset Path & Image File Existence Check**: Verifies that referenced visual crops (`figure_4_2.png`) exist on disk before VLM calls.
6. **Layer 6: Bounding Box & Layout Anchor Resolver**: Ensures visual coordinates match original PDF page bounding boxes.
7. **Layer 7: Category & Legend Disambiguator**: Prevents conflating single categories with aggregate brackets (e.g., *Low Income* $\le \$1,135$ vs *Low & Middle Income* $< \$13,935$).
8. **Layer 8: Structural Pydantic Schema Validator**: Guarantees output payloads conform strictly to required JSON key types.
9. **Layer 9: Exact Quote & Context Anchoring**: Cross-checks verbatim text claims against retrieved vector context chunks.

#### **Phase 3: Fidelity & Hallucination Defense (Layers 10–14)**
10. **Layer 10: Comma & Numeric Token Whitelist Normalizer**: Normalizes numeric formatting (e.g., `$1,135` vs `1135`) for exact token comparison.
11. **Layer 11: In-Memory RAM Transcription Matcher**: Falls back to `_IN_MEMORY_TRANSCRIPTION_CACHE` tables when vector context lacks explicit numbers.
12. **Layer 12: Line-Level Faithfulness Evaluator**: Computes sentence-level maximum semantic similarity, enforcing a strict passing threshold ($\ge 0.90$).
13. **Layer 13: Hallucination Interceptor & Refinement**: Flags ungrounded quantitative claims and triggers self-correction loops.
14. **Layer 14: Deterministic Safe Fallback Interceptor**: Emits a clean, non-crashing fallback explanation when confidence thresholds fail.


---

## 2. Deep Dive: Ingestion Pipeline

The ingestion pipeline is designed to handle multimodal and structured documents (PDFs, CSVs) to extract textual and visual elements (tables, charts, figures) accurately.

```mermaid
flowchart TD
    Start([Start Ingestion]) --> CheckType{Document Type}
    
    %% CSV processing
    CheckType -- CSV File --> CSVProc[Parse rows & Metadata]
    CSVProc --> CSVChunk[Token-bounded Row Chunking]
    
    %% PDF processing
    CheckType -- PDF File --> PDFProc[PDF Visual & Text Extraction]
    PDFProc --> Docling[Docling Layout Analysis]
    PDFProc --> LayoutExtract[Extract Images, Charts & Tables]
    LayoutExtract --> Cropper[Generate High-Quality Visual Crops]
    
    %% Chunking
    CSVChunk --> ChunkCombine[Unify Textual & Multimodal Chunks]
    Docling --> ChunkCombine
    Cropper --> ChunkCombine
    
    %% Enrichment and Invariants
    ChunkCombine --> MetadataGen[Generate Page, Chapter & Source Metadata]
    MetadataGen --> SafetyVetting[Compliance Vetting & Structural Safety Verification]
    
    %% Vectorizing & Storage
    SafetyVetting --> DenseVector[Dense Embeddings: BGE-M3]
    SafetyVetting --> SparseVector[Sparse Embeddings: BM25 Tokenizer]
    
    DenseVector & SparseVector --> UploadQdrant[(Upsert to Qdrant Collection)]
    UploadQdrant --> EndIngest([Ingestion Complete])
```

- **Visual Extraction**: Uses [pdf_visual_extraction.py](file:///C:/Users/supri/recovered-rag-project/app/pdf_visual_extraction.py) to isolate visual elements and crops.
- **Data Ingestion Script**: Managed by [ingest_data.py](file:///C:/Users/supri/recovered-rag-project/ingest_data.py) and deployment scripts like [deploy_to_qdrant.py](file:///C:/Users/supri/recovered-rag-project/deploy_to_qdrant.py).

---

## 3. Deep Dive: Multi-Turn Execution Sequence & Follow-up Resolution

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit UI
    participant CM as Multimodal Conversation Manager
    participant IR as Intent Router
    participant TP as Parallel Thread Pool (Executor)
    participant QD as Qdrant Vector DB
    participant RAM as In-Memory RAM Store
    participant GA as Compliance Safety Gauntlet
    
    User->>UI: "in above figure what are the values of low income?"
    UI->>CM: contextualize_user_query(user_query, session_id)
    CM->>CM: Inspect past 4 turns & LAST_ACTIVE_TARGET_ID ("Figure 4.2")
    CM-->>UI: Rewritten Standalone Query: "in Figure 4.2 what are the values of low income?"
    
    UI->>IR: classify_query_intent("in Figure 4.2...")
    IR-->>UI: Intent: VISUAL_SPECIFIC (allow_pandas: False, allow_vision: True)
    
    rect rgb(240, 248, 255)
        note over UI, RAM: Parallel Concurrency Block (ThreadPoolExecutor max_workers=3)
        par Worker 1: Qdrant Text Search
            UI->>QD: Scroll points for metadata.asset_id="4.2" & asset_type="figure"
            QD-->>UI: Return Figure 4.2 context chunks
        and Worker 2: RAM Transcription Lookup
            UI->>RAM: Query _IN_MEMORY_TRANSCRIPTION_CACHE["figure_4_2"]
            RAM-->>UI: Instant return of Figure 4.2 table markdown (<10ms)
        end
    end
    
    UI->>UI: Synthesize answer: "Low-income economies are defined as $1,135 or less..."
    
    rect rgb(240, 255, 240)
        note over UI, GA: Asynchronous Non-Blocking Validation
        UI->>GA: evaluate_faithfulness(answer, source_chunks, payload)
        GA->>GA: Compute line-level semantic similarity & comma-normalized token overlap
        GA-->>UI: Faithfulness Score: 0.95 (Cleared)
    end
    
    UI-->>User: Render Answer + Figure 4.2 Crop + Sources Card
```


- **Guardrails**: Input and gateway guardrails are processed via [gateway_guardrails.py](file:///C:/Users/supri/recovered-rag-project/gateway_guardrails.py) and safety filters in [compliance_safety.py](file:///C:/Users/supri/recovered-rag-project/compliance_safety.py).
- **Retrieval & Reranking**: Conducted by [retriever.py](file:///C:/Users/supri/recovered-rag-project/app/retriever.py) and [reranker.py](file:///C:/Users/supri/recovered-rag-project/app/reranker.py).
- **Generation & Fallbacks**: Synthesized in [query_rag.py](file:///C:/Users/supri/recovered-rag-project/query_rag.py) with [llamaindex_brain.py](file:///C:/Users/supri/recovered-rag-project/app/llamaindex_brain.py).

```
---

## 2. Key Components

### 🖥️ A. Frontend: Streamlit Application
* *Primary Role:* Manages UI rendering, voice input processing, active chatbot session management, and chat history persistence.
* *display_image_robustly Helper:* A specialized utility that handles cross-platform path translation (Windows vs. Linux), filters out broken Git LFS pointer files (<1 KB), and dynamically loads visual assets from either root or mount directories.

### 🧠 B. Brain: Pydantic-AI Orchestrator
* *Primary Role:* Functions as the central routing and decision engine.
* *Model Integration Strategy:*
  * *Groq / NVIDIA APIs:* Utilized for rapid text reasoning, logic evaluation, and route classification.
  * *OpenRouter (Gemini 2.5 Flash):* Leveraged specifically for multimodal vision processing, diagram understanding, and extracting structured data points from visual figures.
* *Autonomous Routing:* Uses dynamic tool call schemas to inspect data structures on the fly, avoiding rigid heuristics.

### 📁 C. Indexing & Registries (Craft ID Mapping)
* *Multimodal Asset Registry & Craft ID Mapping:* A compiled Python catalog linking raw files, figures, tables, page numbers, and unique *Craft IDs / Entity IDs* directly to their absolute disk paths.
  * *Craft ID Cataloging:* Assigns deterministic entity identifiers (Craft IDs) to every extracted visual element, chart, and structured table during ingestion.
  * *Deterministic Mapping:* Ensures the orchestrator can perform fast, direct lookups by Craft ID rather than relying solely on fuzzy semantic searches, guaranteeing exact asset retrieval.
* *Vector Store:* Qdrant indexing standard text pages embedded using sentence-transformers.
* *Tabular Index:* Pre-loaded Pandas DataFrames representing structured document tables for exact, programmatic data manipulation.

### 🛡️ D. Security: 13-Layer Safety Gauntlet
An invariant safety pipeline executing strict sequential checks prior to payload dispatch:
* *Core Layers:* PII Redaction, Prompt Injection Scans, Craft ID / Path Verification, Bounding Box Region Alignment, Exact Quote Anchoring, and Faithfulness Evaluations.

---

## 3. Query Execution Lifecycle

### Step 1: Input Validation & Rate Limiting
* Scans incoming user queries for active prompt injections and sensitive PII leak vectors.
* Evaluates sliding-window rate limits (REQUEST_CAP = 5 requests per WINDOW_SECONDS = 60).

### Step 2: Intent Classification & Search Routing
The agent analyzes query semantics and dispatches execution to one of three optimal pathways:
* *Pathway A (Tabular):* Routed to Pandas for data aggregations, dynamic mathematical computations, and direct dataframe filtering.
* *Pathway B (Textual):* Routed to Qdrant vector search for semantic chunk retrieval, followed by a Transformer-based cross-encoder reranking pass.
* *Pathway C (Visual & Entity Lookups):* Triggered when a query references a figure, image, chart, or explicit Craft ID. Looks up the precise Craft ID in the registry and employs Annotated type definitions to compel the Gemini Vision model to output structured Markdown table representations of visual charts.

### Step 3: Self-Correction & Table Recovery Loop
* If a visual query fails or yields an empty table payload, the orchestrator intercepts the raw vision output.
* A programmatic fallback parser extracts Markdown table rows (|) directly from the model's intermediate reasoning string and injects them back into the structured response payload.

### Step 4: Output Guardrails & Safety Vetting
* *Path & Craft ID Alignment:* Verifies that any referenced visual asset matches its registered Craft ID, exists in the asset registry, and points to a valid binary image (filtering Git LFS pointers).
* *Bounding Box Matching:* Confirms extracted chart data boundaries map precisely back to source document page coordinates using entity metadata.
* *Faithfulness Evaluation:* Computes semantic similarity scores against retrieved context chunks to detect and eliminate hallucinations.

### Step 5: Frontend Rendering
* *Text Synthesis:* Streamlit renders the validated reasoning stream.
* *Tabular Data:* Reconstructs raw tabular outputs into clean interactive UI tables.
* *Visual Data:* Displays high-resolution binary image assets mapped directly from the Craft ID registry.
---

# Live Demo

🔗 **Streamlit App:**  
https:/rag-system-v2.streamlit.app

---

Endpoint:

```
POST /query
```



# Project Structure

```

    recovered-rag-project/
    │
    ├── .agents/                          # Customization configurations (hooks, config configs)
    │
    ├── app/                              # Core application backend
    │   ├── __init__.py
    │   ├── conversation_manager.py       # Manages session history and chat message memory
    │   ├── embeddings.py                 # Generates dense text vector embeddings
    │   ├── main.py                       # FastAPI server setup and core business logic
    │   ├── multimodal_assets.py          # Multimodal asset registry scanner and mapping logic
    │   └── reranker.py                   # Reranking layer utilizing transformers models
    │
    ├── assets/                           # Source document extract folders
    │   ├── extracted_images/             # Page images and visual charts (including LFS files)
    │   └── extracted_tables/             # Extracted tables formatted as raw CSV files
    │
    ├── extracted_images/                 # Real binary visual images folder (targets for resolution)
    │
    ├── multimodal-rag-system/            # Helper modules
    │   └── schemas_and_agent.py          # Pydantic schema configurations and tool schemas
    │
    ├── streamlit_ui/                     # Streamlit frontend app
    │   └── StreamlitApp.py               # Main UI rendering engine and validation controller
    │
    ├── tests/                            # Validation tests
    │   ├── __init__.py
    │   └── test_guardrail_eval.py        # System testing suite for gauntlet evaluations
    │
    ├── compliance_safety.py              # Decoupled 13-Layer safety gauntlet validation pipeline
    ├── gateway_guardrails.py             # Wallet protection, rate limiting, and PII gateway logic
    ├── pytest.ini                        # Pytest config options
    ├── requirements.txt                  # Python dependency list
    └── .env                              # Environment api keys (Groq, OpenRouter, Langfuse)
```

---

## 🛠️ Local Setup & Installation

### 1. Repository & Git LFS Setup
```bash
git clone [https://github.com/your-username/rag-system-v2.git](https://github.com/your-username/rag-system-v2.git)
cd rag-system-v2
git lfs install && git lfs pull
### Run FastAPI Server
uvicorn app.main:app --reload
```
### 2. Virtual Environment & Dependencies
python -m venv venv
.\venv\Scripts\activate  # Windows (or: source venv/bin/activate on Mac/Linux)
pip install --upgrade pip && pip install -r requirements.txt

### 3. Environment Variables (⁠.env⁠)
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
- Optional Logging
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)

### 4. Run Services & Launch App
- Start Qdrant Vector Store (Docker)
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
- Run Gauntlet Tests
pytest -v
- Launch Streamlit Interface
streamlit run streamlit_ui/StreamlitApp.py


---

# Deployment
Deploy to Streamlit Cloud


# Future Improvements

  ### 1. Vector Database Hybrid Search Upgrade
  Implement Sparse-Dense Hybrid Search in Qdrant (combining BM25 keyword matching with dense vectors) to improve document search precision,
  especially for specific section codes and numeric figures.

  ### 2. LLM Reranking Optimization
  Migrate to a hosted cloud reranking endpoint (like Cohere Rerank API or BGE-Reranker-Large). This will significantly reduce local latency
  and improve the accuracy of top retrieval contexts.

  ### 3. Dynamic Bounding Box Layout Parsing
  Integrate a layout-aware PDF parser like PyMuPDF / LayoutParser or Gemini Document Parsing to detect chart coordinates dynamically on-
  the-fly, allowing the system to handle any raw PDF without pre-cropped coordinates.

  ### 4. Semantic Caching Layer
  Introduce a semantic cache (e.g., using GPTCache or a Qdrant semantic matching index) to capture repeated or highly similar user queries,
  returning the cached response in milliseconds without hit costs.

  ### 5. Multi-Agent Collaboration Topology
   Upgrade to a hierarchical team of agents:
      • Research Agent: Specializes in retrieving text and cross-referencing.
      • Vision Agent: Specialized in reading complex chart layouts.
      • Validator Agent: Operates as a compiler to cross-check outputs before UI delivery.

# Example Questions

- How must suspicious transactions be reported?
- What penalties apply for delayed reporting?
- Under which rule should suspicious transactions be reported to FIU-IND?

---

# Author

**Supriya**  
AI / ML Engineer | Generative AI
