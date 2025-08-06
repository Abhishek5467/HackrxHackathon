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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI(
    title="HackRx RAG QA API",
    description="Extract answers from policy PDFs using LLM + Vector Search",
    version="1.0"
)

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "google/gemma-2b-it"  # Change to "mistralai/Mistral-7B-Instruct-v0.1" if needed
EMBED_MODEL = "all-MiniLM-L6-v2"
HACKRX_API_KEY = os.environ.get("HACKRX_API_KEY", "your-secret-api-key")

# Initialize models
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
llm_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN, device_map="auto")
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

embedding_model = SentenceTransformer(EMBED_MODEL)
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("policy_docs")

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return local file path"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        return tmp_file.name

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks for vector storage"""
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
    
    return chunks

def store_chunks_in_vectordb(chunks: List[str]):
    """Store text chunks in vector database"""
    # Clear existing collection
    collection.delete(where={})
    
    if not chunks:
        return
    
    embeddings = embedding_model.encode(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

def search_relevant_chunks(question: str, top_k: int = 3) -> List[str]:
    """Search for relevant chunks using semantic similarity"""
    query_embedding = embedding_model.encode([question])
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents"]
    )
    
    return results['documents'][0] if results['documents'] else []

def generate_answer(question: str, context_chunks: List[str]) -> str:
    """Generate answer using LLM based on context"""
    context = "\n\n".join(context_chunks[:3])  # Limit context length
    
    prompt = f"""Context from policy document:
{context}

Question: {question}

Based only on the provided context, give a direct and concise answer. If the information is not available in the context, say "Information not available in the policy document."

Answer:"""

    try:
        response = llm_pipeline(
            prompt,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False
        )
        
        generated_text = response[0]['generated_text']
        # Extract only the answer part
        answer = generated_text.split("Answer:")[-1].strip()
        
        # Clean up the answer
        answer = re.sub(r'\n+', ' ', answer)
        answer = answer.split('\n')[0].strip()  # Take first line only
        
        return answer if answer else "Information not available in the policy document."
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error processing the question."

@app.get("/")
def read_root():
    return {"message": "HackRx RAG QA API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/hackrx/run")
async def hackrx_run(
    request_data: dict,
    authorization: Optional[str] = Header(None)
):
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
    
    try:
        # Download and process PDF
        pdf_path = download_pdf_from_url(documents_url)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        store_chunks_in_vectordb(chunks)
        
        # Process each question
        answers = []
        for question in questions:
            relevant_chunks = search_relevant_chunks(question)
            answer = generate_answer(question, relevant_chunks)
            answers.append(answer)
        
        # Clean up temporary file
        os.unlink(pdf_path)
        
        return JSONResponse(content={"answers": answers})
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
