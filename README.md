---
title: HackRx RAG QA API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# HackRx LLM+RAG API

## 🎯 Overview
REST API for answering questions from policy documents using Gemma LLM + Vector Search RAG pipeline.

## 🚀 Live Demo
- **API Endpoint**: `https://your-username-hackrx-llm-api.hf.space/hackrx/run`
- **Health Check**: `https://your-username-hackrx-llm-api.hf.space/health`

## 🔧 Key Features
- **LLM**: Google Gemma-2B-IT (via HuggingFace Inference API)
- **Embeddings**: all-MiniLM-L6-v2 (local)
- **Vector DB**: ChromaDB (in-memory)
- **Framework**: FastAPI
- **Hosting**: Hugging Face Spaces

## 📋 API Usage

### Authentication
