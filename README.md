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

## Future Improvements

* Cross Encoder Re-ranking
* Evaluation Framework
* Cloud Deployment
* Advanced Metadata Filtering
