import os
import re
import json
import requests
import tempfile
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional

import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Create FastAPI app
app = FastAPI(
    title="HackRx RAG QA API",
    description="Extract answers from policy PDFs using LLM + Vector Search",
    version="1.0"
)

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
EMBED_MODEL = "all-MiniLM-L6-v2"
HACKRX_API_KEY = os.environ.get("HACKRX_API_KEY", "hackrx-default-key-2024")

# Initialize models
print("🚀 Loading embedding model...")
try:
    embedding_model = SentenceTransformer(EMBED_MODEL)
    print("✅ Embedding model loaded successfully")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

# Initialize ChromaDB
try:
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("policy_docs")
    print("✅ ChromaDB initialized successfully")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")
    collection = None

def query_huggingface_api(prompt: str) -> str:
    """Query Gemma via Hugging Face Inference API"""
    if not HF_TOKEN:
        return "Error: HF_TOKEN not configured"
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.1,
            "return_full_text": False,
            "do_sample": False
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response generated')
            elif isinstance(result, dict):
                return result.get('generated_text', 'No response generated')
            else:
                return str(result)
        elif response.status_code == 503:
            return "Model is loading, please try again in a few moments"
        else:
            print(f"HF API Error: {response.status_code} - {response.text}")
            return f"API Error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Request timeout - please try again"
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return f"Request failed: {str(e)}"

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return local file path"""
    try:
        print(f"📥 Downloading PDF from: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            print(f"✅ PDF downloaded successfully: {tmp_file.name}")
            return tmp_file.name
    except Exception as e:
        print(f"❌ Failed to download PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        print(f"📄 Extracting text from PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                print(f"Processed page {i+1}/{len(pdf.pages)}")
        
        print(f"✅ Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"❌ Failed to extract text from PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks for vector storage"""
    if not text.strip():
        return []
    
    print(f"🔪 Chunking text into pieces of max {chunk_size} characters")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    print(f"✅ Created {len(chunks)} text chunks")
    return chunks

def store_chunks_in_vectordb(chunks: List[str]):
    """Store text chunks in vector database"""
    if not collection or not embedding_model:
        raise HTTPException(status_code=500, detail="Vector DB or embedding model not initialized")
    
    try:
        print("🗃️ Storing chunks in vector database...")
        # Clear existing collection
        collection.delete(where={})
        
        if not chunks:
            return
        
        print("🔢 Generating embeddings...")
        embeddings = embedding_model.encode(chunks, batch_size=16, show_progress_bar=True)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )
        print(f"✅ Stored {len(chunks)} chunks in vector database")
    except Exception as e:
        print(f"❌ Error storing chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store document chunks: {str(e)}")

def search_relevant_chunks(question: str, top_k: int = 3) -> List[str]:
    """Search for relevant chunks using semantic similarity"""
    if not collection or not embedding_model:
        return []
    
    try:
        print(f"🔍 Searching for relevant chunks for: {question[:100]}...")
        query_embedding = embedding_model.encode([question])
        
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents"]
        )
        
        relevant_docs = results['documents'][0] if results['documents'] else []
        print(f"✅ Found {len(relevant_docs)} relevant chunks")
        return relevant_docs
    except Exception as e:
        print(f"❌ Error searching chunks: {str(e)}")
        return []

def generate_answer(question: str, context_chunks: List[str]) -> str:
    """Generate answer using HF Inference API"""
    if not context_chunks:
        return "No relevant information found in the policy document."
    
    print(f"🤖 Generating answer for: {question[:100]}...")
    context = "\n\n".join(context_chunks[:3])  # Limit context length
    
    prompt = f"""Context from policy document:
{context}

Question: {question}

Based only on the provided context, give a direct and concise answer. If the information is not available in the context, say "Information not available in the policy document."

Answer:"""

    try:
        response = query_huggingface_api(prompt)
        
        # Clean up the response
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response.strip()
        
        # Remove extra whitespace and newlines
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip()
        
        print(f"✅ Generated answer: {answer[:100]}...")
        return answer if answer else "Information not available in the policy document."
        
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return "Error processing the question."

# API Routes
@app.get("/")
def read_root():
    return {
        "message": "🚀 HackRx RAG QA API is running", 
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "main_api": "/hackrx/run"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "model": "google/gemma-2b-it (via HF Inference API)",
        "embedding_model": EMBED_MODEL,
        "vector_db": "ChromaDB",
        "hf_token_configured": bool(HF_TOKEN),
        "components": {
            "embedding_model": "loaded" if embedding_model else "error",
            "vector_db": "initialized" if collection else "error"
        }
    }

@app.post("/hackrx/run")
async def hackrx_run(
    request_data: dict,
    authorization: Optional[str] = Header(None)
):
    print(f"🎯 Received request for /hackrx/run")
    
    # Authentication
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    api_key = authorization.replace("Bearer ", "")
    if api_key != HACKRX_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Validate input
    documents_url = request_data.get("documents")
    questions = request_data.get("questions", [])
    
    if not documents_url or not questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' URL or 'questions' list")
    
    if not isinstance(questions, list) or len(questions) == 0:
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty list")
    
    print(f"📝 Processing {len(questions)} questions")
    
    try:
        # Download and process PDF
        pdf_path = download_pdf_from_url(documents_url)
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in the PDF document")
        
        chunks = chunk_text(text)
        store_chunks_in_vectordb(chunks)
        
        # Process each question
        answers = []
        for i, question in enumerate(questions):
            print(f"🔄 Processing question {i+1}/{len(questions)}")
            
            if not question or not isinstance(question, str):
                answers.append("Invalid question format")
                continue
                
            relevant_chunks = search_relevant_chunks(question.strip())
            answer = generate_answer(question.strip(), relevant_chunks)
            answers.append(answer)
        
        # Clean up temporary file
        try:
            os.unlink(pdf_path)
        except:
            pass  # Ignore cleanup errors
        
        print(f"✅ Successfully processed all questions")
        return JSONResponse(content={"answers": answers})
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# For Hugging Face Spaces compatibility
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting HackRx RAG QA API...")
    uvicorn.run(app, host="0.0.0.0", port=7860)  # Port 7860 is standard for HF Spaces
