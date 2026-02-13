# RAG with OpenAI (Native Implementation)

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using OpenAI's native APIs. The goal of this project is to understand how RAG systems work internally without relying on external frameworks.

This project was completed as part of the IronHack AI Consulting & Integration program.

---

## Overview

Retrieval-Augmented Generation (RAG) combines semantic search with language model generation. Instead of relying only on model training data, the system retrieves relevant document content and uses it as context when generating answers.

The pipeline consists of two main phases:

### 1. Indexing Phase
- Load documents (PDF or text)
- Split documents into chunks
- Generate embeddings using OpenAI embedding models
- Store embeddings alongside text and metadata

### 2. Query Phase
- Embed the user query
- Calculate cosine similarity against stored embeddings
- Retrieve the most relevant chunks
- Generate answers using retrieved context
- Include source citations for traceability

---

## Tech Stack

- Python
- OpenAI API
- NumPy
- PyPDF
- python-dotenv

---
rag-openai-native/
│
├── rag_openai_native.py
├── docs/
│ └── podcast.txt
├── .env
└── README.md

---

## How to Run

### 1. Install dependencies

pip install openai numpy pypdf python-dotenv

### 2. Set environment variable
Create a `.env` file:
### 3. Run the application
python rag_openai_native.py

---

## Example Capabilities

- Answers questions based on provided documents
- Retrieves relevant document chunks using embeddings
- Provides citations referencing source chunks
- Demonstrates native RAG pipeline implementation

---

## Notes

This project is intended for educational purposes to demonstrate RAG fundamentals and OpenAI API usage.
1.Initialize git
git init

2️. Add files
git add .

3️. Commit
git commit -m "Initial commit - Native OpenAI RAG pipeline implementation"

4. Create repo on GitHub
Go to GitHub → New Repository:
Name:
rag-openai-native

Do NOT add README (you already have one).

5. Connect and push
GitHub will show something like:
git remote add origin https://github.com/aprilatkinson/rag-openai-native.git
git branch -M main
git push -u origin main

