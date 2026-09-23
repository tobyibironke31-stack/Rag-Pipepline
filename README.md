# Rag-Pipepline

# 📄 Hybrid RAG PDF Assistant

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built to extract, search, and answer questions from technical PDF documents with high accuracy. 

It combines **dense vector search** (semantic meaning) with **sparse BM25 retrieval** (exact keywords) and uses a **cross-encoder reranker** to select the most relevant context before generating answers with **Google Gemini**.

---

## 🌟 Features

- **Fast PDF Ingestion:** Uses PyMuPDF to extract text and preserve page numbers and document metadata.
- **Hybrid Retrieval:** Runs dense vector embeddings (`intfloat/e5-small-v2`) and keyword search (`BM25`) in parallel to capture both conceptual questions and specific part numbers/tolerances.
- **Cross-Encoder Reranking:** Re-scores retrieved passages with `ms-marco-MiniLM-L-6-v2` to filter out irrelevant chunks.
- **Grounded Answer Generation:** Passes top-ranked context to Google Gemini to generate hallucination-resistant answers.
- **Interactive UI:** Simple Gradio web app for drag-and-drop PDF uploads and chat-based Q&A.
