# Text-Extraction-using-RAG
markdown
# Resume Information Extraction using NLP and FAISS

This project extracts relevant information (e.g., email, CGPA) from resumes in PDF format using NLP techniques, FAISS for similarity search, and a generative model for answering questions.

## Features
- Extract text from PDFs using `pdfplumber`
- Preprocess and chunk text for efficient search
- Generate embeddings using `SentenceTransformer`
- Create a FAISS index for fast retrieval
- Use `FLAN-T5` to answer questions based on extracted data

## Installation
