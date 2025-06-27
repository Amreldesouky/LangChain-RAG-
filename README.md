# 🧠 Local RAG System with LangChain, Phi-3, and CSV

This project implements a **fully local Retrieval-Augmented Generation (RAG) system** that allows you to **chat with structured CSV data** — no APIs, no cloud, just fast and private Q&A from your own machine.

## 📌 Description

This system uses:

- 🔗 **LangChain** for retrieval pipeline
- 🤖 **Phi-3** (via Ollama) as the local language model
- 🧠 **HuggingFace Embeddings** (all-MiniLM-L6-v2)
- 💾 **ChromaDB** for persistent vector storage
- 📂 **CSV** file as the knowledge base

Ask natural questions like:
> "What is Michael Brown's position?"  
> "Who works in the Engineering department?"  
> "When did Sara Lee join?"

And get answers instantly — grounded in your own data.

---

## 🛠️ How It Works

1. **Loads CSV** and converts rows to documents.
2. **Embeds text** using MiniLM from HuggingFace.
3. **Stores vectors** in ChromaDB locally.
4. **Retrieves relevant chunks** based on user query.
5. **Generates answer** using Phi-3 via Ollama.

---

## 🚀 Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install langchain langchain-community chromadb
   pip install sentence-transformers pandas
