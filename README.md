# Hybrid RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through natural language conversations.

The system combines semantic search (FAISS) and keyword search (BM25) using Reciprocal Rank Fusion (RRF), along with query rewriting and multi-query retrieval to improve retrieval quality and answer grounding.

## Features

* PDF Upload and Dynamic Indexing
* Semantic Chunking
* Hybrid Retrieval (FAISS + BM25)
* Reciprocal Rank Fusion (RRF)
* Multi-Query Retrieval
* Conversation Memory
* Query Rewriting
* Source Filtering
* Citation Generation
* Local LLM Inference using Ollama (Llama 3)
* Streamlit User Interface

## Tech Stack

* Python
* LangChain
* FAISS
* BM25
* Sentence Transformers
* Ollama
* Streamlit

## Retrieval Pipeline

User Query
→ Query Rewriting
→ Multi Query Generation
→ FAISS Retrieval
→ BM25 Retrieval
→ Reciprocal Rank Fusion
→ Top-K Context Selection
→ Llama 3
→ Answer with Citations

## Installation

git clone https://github.com/Bhargava1905/Hybrid-RAG-PDF-Chatbot.git

cd Hybrid-RAG-PDF-Chatbot

pip install -r requirements.txt

## Run Llama 3 Locally

Install Ollama and pull the model:

ollama pull llama3

## Usage

streamlit run app.py

## Key Highlights

- Implemented Hybrid Retrieval using FAISS and BM25
- Applied Reciprocal Rank Fusion (RRF) for result aggregation
- Built Multi-Query Retrieval to improve context recall
- Added Query Rewriting for conversational follow-up questions
- Implemented document-level filtering and citation tracking
- Developed an end-to-end local RAG pipeline using Llama 3 and Ollama

## Future Improvements

- Cross-Encoder Re-ranking
- Retrieval Evaluation Framework (Precision@K, Recall@K)
- Cloud Deployment (AWS/GCP)
- Advanced Metadata Filtering
- Parent-Child Retrieval
- Hybrid Search Weight Optimization
