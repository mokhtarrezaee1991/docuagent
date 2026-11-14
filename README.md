# DocuAgent – AI-Powered Document Assistant

**Status: 🚧 Work in Progress | Active Development**

DocuAgent is an AI-powered document assistant built with Python, FastAPI, Streamlit, and Retrieval-Augmented Generation (RAG).

The application allows users to upload multiple PDF documents, ask questions in natural language, and receive AI-generated answers with references to the original documents.

This project is being developed as a practical portfolio project focused on **Generative AI, semantic search, document processing, and AI application development**.

## Current Features

- **Multi-PDF Upload:** Upload and index multiple PDF documents.
- **PDF Text Extraction:** Extract text and preserve page information.
- **Token-Based Chunking:** Split documents into manageable text sections with overlap.
- **Semantic Search:** Find relevant document sections using multilingual sentence embeddings and cosine similarity.
- **Retrieval-Augmented Generation (RAG):** Generate answers using OpenAI GPT-4.1-mini and retrieved document context.
- **Source References:** Display document names, page numbers, and chunk references.
- **Interactive Interface:** Ask questions through a Streamlit chat application.
- **Document Management:** List and remove indexed documents.
- **Automated Tests:** Unit and API tests for core functionality.

## Technology Stack

| Category             | Technologies          |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Backend              | FastAPI               |
| Frontend             | Streamlit             |
| AI / LLM             | OpenAI GPT-4.1-mini   |
| Embeddings           | Sentence Transformers |
| Semantic Search      | Cosine Similarity     |
| PDF Processing       | PyMuPDF               |
| Testing              | pytest                |
| Version Control      | Git & GitHub          |

## How It Works

1. **Document Upload:** Users upload PDF files through Streamlit.
2. **Text Extraction:** The backend extracts text and page information.
3. **Chunking:** Text is divided into token-based sections.
4. **Embedding Generation:** A local Sentence Transformers model converts text chunks into numerical vectors.
5. **Semantic Retrieval:** The system identifies relevant chunks based on the user's question.
6. **Answer Generation:** GPT-4.1-mini generates an answer using the retrieved context.
7. **Source Attribution:** The application displays references to the original documents.

## Current Architecture

```text
                  Streamlit UI
                       |
                  FastAPI Backend
                       |
              PDF Document Processing
                       |
               Token-Based Chunking
                       |
              Local Embedding Model
                       |
                Semantic Search
                       |
                 Relevant Chunks
                       |
                OpenAI GPT-4.1-mini
                       |
              Answer + PDF Sources
```

## Development Roadmap

- [x] FastAPI backend setup
- [x] PDF text extraction
- [x] Token-based text chunking
- [x] Local embedding generation
- [x] Semantic search
- [x] RAG answer generation
- [x] Streamlit chat interface
- [x] Multi-document support
- [x] Document and page references
- [x] Automated testing
- [ ] Persistent document and embedding storage with SQLite
- [ ] Advanced retrieval and answer-quality evaluation
- [ ] AI agent workflows with LangGraph
- [ ] Improved document analysis and information extraction
- [ ] Deployment and CI/CD

## Current Limitations

DocuAgent is currently a local development prototype.

- Indexed documents and embeddings are stored in memory and are lost when the backend restarts.
- The application is designed for local single-user use.
- OpenAI API access is required for answer generation.
- Source references do not guarantee that every generated answer is factually correct.
- Production authentication and access controls have not yet been implemented.

## Getting Started

A complete installation and configuration guide will be added as development progresses.

The project currently runs locally using FastAPI and Streamlit. An OpenAI API key is required for AI-generated answers and must be configured through environment variables.

**Never commit API keys or `.env` files to GitHub.**

## Project Goal

The goal is to develop DocuAgent into a more robust AI document intelligence application, with persistent storage, improved retrieval quality, and agent-based workflows.

The project also serves as a hands-on demonstration of Python backend development, RAG architecture, API integration, and AI application testing.

---

**Development Status:** Active development – features and architecture are continuously being improved.
