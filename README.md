# Financial Regulatory RAG System

A production-style Retrieval-Augmented Generation (RAG) system for financial regulatory Q&A that reduces hallucination using hybrid retrieval, cross-encoder reranking, and source-grounded responses. Achieved ~0.875 Recall@5 using hybrid retrieval + cross-encoder reranking on RBI regulatory data

## Why it matters
- Financial compliance requires accurate, verifiable answers — hallucinated responses can lead to regulatory and financial risk  
- This system ensures responses are grounded in official RBI regulatory documents  

## What makes it better
- Hybrid Retrieval (BM25 + FAISS) for keyword + semantic search  
- Cross-Encoder Reranking improving Recall@5 from ~0.75 → ~0.875  
- Source-grounded responses to reduce hallucination  
- Quantitative evaluation using Recall@k and RAGAS metrics  
- Production-ready deployment using FastAPI + Cloud Run + Streamlit UI  

---
# Live Demo

🔗 **Streamlit App:**  
https://financial-rag-ui-912628415543.us-central1.run.app/

🔗 **API Docs:**  
https://financial-rag-api-912628415543.us-central1.run.app/docs

---
# System Architecture

```
User Query
   ↓
Streamlit UI
   ↓
FastAPI API (/query endpoint)
   ↓
Hybrid Retrieval
   • BM25 Retriever
   • FAISS Vector Search
   ↓
Cross-Encoder Reranking
(cross-encoder/ms-marco-MiniLM-L-6-v2)
   ↓
Context Construction
   ↓
LLM Answer Generation
   ↓
Final Response
```

---

# Key Features

## Hybrid Retrieval
Combines **BM25 lexical search** and **FAISS dense vector search** to improve retrieval accuracy.

## Cross-Encoder Reranking
Documents are reranked using: cross-encoder/ms-marco-MiniLM-L-6-v2
This improves answer quality by selecting the most relevant context.

## Source Attribution
The system returns the **document source and page number** used to generate the answer.  
This ensures responses are grounded in the original financial regulatory documents and helps reduce hallucinations.

## FastAPI Backend
Provides a scalable REST API.

Endpoint:

```
POST /query
```

Returns:

- Generated answer
- Response latency
- Metadata

## Streamlit UI
Interactive interface to ask compliance-related questions.

## Cloud Deployment
The system is containerized using **Docker** and deployed on **Google Cloud Run**.

---

# Tech Stack

| Component | Technology |
|--------|--------|
Backend API | FastAPI |
Frontend | Streamlit |
Vector Database | FAISS |
Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
Retrieval | Hybrid (BM25 + FAISS) |
LLM | OpenRouter (Llama 3 / Open-source models)
Deployment | Docker + Google Cloud Run |

---

# Retrieval Evaluation

We evaluated retrieval performance using Recall@5 on a manually labeled dataset derived from RBI KYC guidelines.

| Method              | Recall@5 |
|---------------------|---------|
| BM25                | 0.50    |
| FAISS               | 0.875   |
| Hybrid              | 0.75    |
| Hybrid + Reranking  | 0.875   |

- Dense retrieval (FAISS) performed best for semantic regulatory data
- BM25 underperformed due to lack of strong keyword signals
- Hybrid improved baseline retrieval
- Cross-encoder reranking significantly improved result ordering

👉 Reranking improved Hybrid performance from ~0.75 → ~0.875

---

## Limitations

While the system performs well on structured regulatory queries, several limitations were observed:
   1. Sensitivity to Query Quality
   - The system struggles with vague or poorly phrased queries  
   - Retrieval performance depends heavily on how clearly the query matches document intent  

   2. Context Window Constraints
   - Only top-k retrieved chunks are passed to the LLM  
   - Important information may be missed if not retrieved in top results  

   3. Hallucination Risk
   - If retrieval fails or returns weak context, the LLM may generate partially incorrect answers  
   - This was observed in edge cases with ambiguous queries  

   4. Dataset Limitations
   - Performance is tied to the quality and coverage of the RBI document  
   - Missing or incomplete sections can lead to incomplete answers  

   5. Retrieval Bias
   - Dense retrieval (FAISS) dominates performance due to semantic nature of data  
   - BM25 contributes less in this domain, reducing hybrid effectiveness  

   6. Computational Overhead
   - Cross-encoder reranking improves accuracy but increases latency  
   - Not optimal for real-time high-throughput systems without optimization  

---

# Retrieval Pipeline

1. User submits a question
2. Hybrid retriever fetches candidate documents
3. BM25 search retrieves keyword matches
4. FAISS performs dense vector similarity search
5. Cross-Encoder reranks retrieved documents
6. Top documents are selected as context
7. LLM generates the final answer
8. API returns the response with latency metadata
9.  Response includes source citation for transparency

---

# API Usage

## Query Endpoint

```
POST /query
```

### Example Request

```json
{
 "question": "Under which Rule should suspicious transactions be reported to FIU-IND?"
}
```

### Example Response

```json
{
 "question": "...",
 "answer": "...",
 "sources": ["Finance_RBI.pdf (Page 14)"],
 "latency_seconds": 1.42
}
```

---

# Project Structure

```
rag-system-v2
│
├── app
│   ├── ingestion.py
│   ├── vector.py
│   ├── retriever.py
│   ├── llm.py
│   └── main.py
│
├── Data
│   └── Vector
│
├── streamlit_ui
│   └── StreamlitApp.py
│
├── evaluate.py
├── Dockerfile
├── Dockerfile.streamlit
├── requirements.txt
└── README.md
```

---

# Installation

### Clone Repository
git clone https://github.com/supriyap19967-creator/rag-systm-v2
cd rag-systm-v2

### Install Dependencies
pip install -r requirements.txt

### Run FastAPI Server
uvicorn app.main:app --reload

### Run Streamlit UI
streamlit run streamlit_ui/StreamlitApp.py

---

# Deployment

### Build Docker Image
docker build -t financial-rag-api .

### Deploy to Google Cloud Run
gcloud run deploy financial-rag-api

# Future Improvements

- Query rewriting for handling vague user inputs  
- Lightweight reranking models to reduce latency  
- Multi-document reasoning for complex queries  
- Evaluation on larger and more diverse datasets  

# Example Questions

- How must suspicious transactions be reported?
- What penalties apply for delayed reporting?
- Under which rule should suspicious transactions be reported to FIU-IND?

---

# Author

**Supriya**  
AI / ML Engineer | Generative AI
