# Multi-Modal RAG System

A Retrieval-Augmented Generation (RAG) system for multi-modal document processing and question answering.

## Features

- PDF document processing with text and image extraction
- Multi-modal embeddings using sentence transformers
- FAISS vector store for efficient similarity search
- Question answering with LLM integration
- Streamlit web interface

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Process documents and create embeddings:
```bash
python run_pipeline.py
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

- `app.py` - Streamlit web interface
- `config.py` - Configuration settings
- `document_processor.py` - Document processing logic
- `create_embeddings.py` - Embedding creation
- `vector_store.py` - Vector store management
- `llm_qa.py` - Question answering system
- `data/` - Data directory for documents and vector store

## Requirements

See `requirements.txt` for detailed dependencies.
