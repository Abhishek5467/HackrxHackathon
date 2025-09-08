import os
import re
import json
import requests
import tempfile
import gradio as gr
from typing import List, Optional
import torch

import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
EMBED_MODEL = "all-MiniLM-L6-v2" 
HACKRX_API_KEY = os.environ.get("HACKRX_API_KEY", "hackrx-default-key-2024")
MODEL_NAME = "google/gemma-2b-it"

# Initialize local Gemma model with optimizations
print("🚀 Loading Gemma model locally...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        token=HF_TOKEN,
        torch_dtype=torch.float16,  # Memory optimization
        low_cpu_mem_usage=True,     # Efficient loading
        trust_remote_code=True
    )
    
    qa_model = pipeline(
        "text-generation", 
        model=llm_model, 
        tokenizer=tokenizer,
        return_full_text=False,
        device_map="auto"
    )
    print("✅ Gemma model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Gemma model: {e}")
    qa_model = None
    tokenizer = None

# Initialize embedding model
print("🚀 Loading embedding model...")
try:
    embedding_model = SentenceTransformer(EMBED_MODEL)
    print("✅ Embedding model loaded successfully")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

# Initialize ChromaDB
try:
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection("policy_docs")
    print("✅ ChromaDB initialized successfully")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")
    collection = None

def query_local_gemma(prompt: str) -> str:
    """Query Gemma using local pipeline"""
    if not qa_model or not tokenizer:
        return "Error: Gemma model not loaded"
    
    try:
        print(f"🤖 Querying local Gemma with prompt length: {len(prompt)}")
        
        # Truncate prompt if too long (Gemma has context limits)
        max_prompt_length = 1500
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
            print(f"⚠️ Truncated prompt to {max_prompt_length} characters")
        
        response = qa_model(
            prompt,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get('generated_text', '')
        else:
            generated_text = str(response)
        
        print(f"✅ Generated response length: {len(generated_text)}")
        return generated_text
        
    except Exception as e:
        error_msg = f"Local Gemma error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return local file path"""
    try:
        print(f"📥 Downloading PDF from: {url[:100]}...")
        
        # Add headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            print(f"✅ PDF downloaded successfully: {tmp_file.name}")
            return tmp_file.name
    except Exception as e:
        print(f"❌ Failed to download PDF: {str(e)}")
        raise Exception(f"Failed to download PDF: {str(e)}")

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
                if i % 10 == 0:  # Progress indicator
                    print(f"Processed page {i+1}/{len(pdf.pages)}")
        
        print(f"✅ Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"❌ Failed to extract text from PDF: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks for vector storage"""
    if not text.strip():
        return []
    
    print(f"🔪 Chunking text into pieces of max {chunk_size} characters")
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Clean sentence
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 50]
    
    print(f"✅ Created {len(chunks)} text chunks")
    return chunks

def store_chunks_in_vectordb(chunks: List[str]):
    """Store text chunks in vector database"""
    if not collection or not embedding_model:
        raise Exception("Vector DB or embedding model not initialized")
    
    try:
        print("🗃️ Storing chunks in vector database...")
        
        # Clear existing documents more safely
        try:
            all_items = collection.get()
            if all_items and all_items.get('ids') and len(all_items['ids']) > 0:
                collection.delete(ids=all_items['ids'])
                print(f"🧹 Cleared {len(all_items['ids'])} existing documents")
        except Exception as e:
            print(f"⚠️ Could not clear collection: {e}")
            # Try recreating collection
            try:
                chroma_client.delete_collection("policy_docs")
                collection = chroma_client.create_collection("policy_docs")
                print("🔄 Recreated collection")
            except:
                pass
        
        if not chunks:
            print("⚠️ No chunks to store")
            return
        
        print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, batch_size=8, show_progress_bar=True)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )
        print(f"✅ Stored {len(chunks)} chunks in vector database")
    except Exception as e:
        print(f"❌ Error storing chunks: {str(e)}")
        raise Exception(f"Failed to store document chunks: {str(e)}")

def search_relevant_chunks(question: str, top_k: int = 3) -> List[str]:
    """Search for relevant chunks using semantic similarity"""
    if not collection or not embedding_model:
        print("⚠️ Vector DB or embedding model not available")
        return []
    
    try:
        print(f"🔍 Searching for relevant chunks for: {question[:100]}...")
        query_embedding = embedding_model.encode([question])
        
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, 5),  # Limit to max 5
            include=["documents"]
        )
        
        relevant_docs = results.get('documents', [[]])[0] if results.get('documents') else []
        print(f"✅ Found {len(relevant_docs)} relevant chunks")
        return relevant_docs
    except Exception as e:
        print(f"❌ Error searching chunks: {str(e)}")
        return []

def generate_answer(question: str, context_chunks: List[str]) -> str:
    """Generate answer using local Gemma model"""
    if not context_chunks:
        return "No relevant information found in the policy document."
    
    print(f"🤖 Generating answer for: {question[:100]}...")
    
    # Limit context to avoid token limits
    context = "\n\n".join(context_chunks[:2])  # Use only top 2 chunks
    if len(context) > 800:  # Further limit context
        context = context[:800] + "..."
    
    # Optimized prompt for Gemma
    prompt = f"""Context: {context}

Question: {question}

Answer based only on the context provided:"""

    try:
        response = query_local_gemma(prompt)
        
        # Handle error responses
        if "error" in response.lower() or "Error:" in response:
            print(f"❌ Model returned error: {response}")
            return "Error processing the question with the model."
        
        # Clean up the response
        answer = response.strip()
        
        # Remove any prompt repetition
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        if "Context:" in answer:
            answer = answer.split("Context:")[-1].strip()
        
        # Clean whitespace and newlines
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip()
        
        # Limit answer length
        if len(answer) > 500:
            answer = answer[:500] + "..."
        
        print(f"✅ Generated answer: {answer[:100]}...")
        return answer if answer else "Information not available in the policy document."
        
    except Exception as e:
        error_msg = f"Answer generation failed: {str(e)}"
        print(f"❌ {error_msg}")
        return "Error generating answer from the model."

# Main processing function for Gradio
def process_hackrx_request(pdf_url: str, questions_text: str, api_key: str) -> str:
    """Main function to process HackRx request via Gradio interface"""
    
    # Validate API key
    if not api_key or api_key.strip() != HACKRX_API_KEY:
        return json.dumps({"error": "Invalid or missing API key"}, indent=2)
    
    # Validate inputs
    if not pdf_url or not pdf_url.strip():
        return json.dumps({"error": "PDF URL is required"}, indent=2)
    
    if not questions_text or not questions_text.strip():
        return json.dumps({"error": "Questions are required"}, indent=2)
    
    # Parse questions
    try:
        if questions_text.strip().startswith('['):
            questions = json.loads(questions_text.strip())
        else:
            questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
    except json.JSONDecodeError:
        questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
    
    if not questions:
        return json.dumps({"error": "No valid questions found"}, indent=2)
    
    # Limit number of questions to avoid timeouts
    if len(questions) > 10:
        questions = questions[:10]
        print(f"⚠️ Limited to first 10 questions")
    
    print(f"📝 Processing {len(questions)} questions for PDF: {pdf_url[:100]}...")
    
    try:
        # Download and process PDF
        pdf_path = download_pdf_from_url(pdf_url.strip())
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            return json.dumps({"error": "No text found in the PDF document"}, indent=2)
        
        chunks = chunk_text(text)
        if not chunks:
            return json.dumps({"error": "Failed to create text chunks from PDF"}, indent=2)
            
        store_chunks_in_vectordb(chunks)
        
        # Process each question
        answers = []
        for i, question in enumerate(questions):
            print(f"🔄 Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
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
            pass
        
        print(f"✅ Successfully processed all questions")
        
        # Return result in HackRx format
        result = {"answers": answers}
        return json.dumps(result, indent=2)
        
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return json.dumps({"error": f"Processing failed: {str(e)}"}, indent=2)

# Health check function
def health_check() -> str:
    """Health check function for Gradio"""
    health_status = {
        "status": "ok", 
        "model": f"{MODEL_NAME} (local pipeline)",
        "embedding_model": EMBED_MODEL,
        "vector_db": "ChromaDB",
        "hf_token_configured": bool(HF_TOKEN),
        "api_key_configured": bool(HACKRX_API_KEY != "hackrx-default-key-2024"),
        "components": {
            "gemma_model": "loaded" if qa_model else "error",
            "embedding_model": "loaded" if embedding_model else "error",
            "vector_db": "initialized" if collection else "error"
        },
        "system_info": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    return json.dumps(health_status, indent=2)

# Simple test function
def test_simple_qa():
    """Test function with a simple question"""
    try:
        if not qa_model:
            return "❌ Model not loaded"
        
        test_prompt = "What is 2 + 2? Answer:"
        response = query_local_gemma(test_prompt)
        return f"✅ Test successful: {response}"
    except Exception as e:
        return f"❌ Test failed: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="HackRx RAG QA API", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 HackRx LLM+RAG API
    
    **Extract answers from policy PDFs using Local Gemma + Vector Search**
    
    This API processes insurance policy documents and answers questions based on document content.
    
    Built for **HackRx 2024** submission.
    """)
    
    with gr.Tab("🎯 Main API"):
        gr.Markdown("### Process Policy Documents")
        
        with gr.Row():
            with gr.Column():
                pdf_url = gr.Textbox(
                    label="📄 PDF Document URL",
                    placeholder="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                    lines=3
                )
                
                questions_input = gr.Textbox(
                    label="❓ Questions (one per line, max 10)",
                    placeholder="What is the grace period for premium payment?\nWhat is the waiting period for pre-existing diseases?\nDoes this policy cover maternity expenses?",
                    lines=6
                )
                
                api_key_input = gr.Textbox(
                    label="🔑 API Key",
                    placeholder="Enter your API key",
                    type="password"
                )
                
                process_btn = gr.Button("🔄 Process Request", variant="primary", size="lg")
            
            with gr.Column():
                output = gr.Textbox(
                    label="📋 Response (JSON)",
                    lines=20,
                    interactive=False
                )
        
        process_btn.click(
            fn=process_hackrx_request,
            inputs=[pdf_url, questions_input, api_key_input],
            outputs=output
        )
        
        gr.Markdown("""
        ### 📝 Expected Response Format:
        ```
        {
          "answers": [
            "A grace period of thirty days is provided for premium payment...",
            "There is a waiting period of thirty-six (36) months...",
            "Yes, the policy covers maternity expenses with specific conditions..."
          ]
        }
        ```
        
        ### 💡 Tips:
        - Use the exact API key provided by your team
        - Questions should be clear and specific to the policy document  
        - Maximum 10 questions per request to avoid timeouts
        - PDF should be publicly accessible
        """)
    
    with gr.Tab("🏥 Health Check"):
        gr.Markdown("### System Status & Diagnostics")
        
        with gr.Row():
            with gr.Column():
                health_btn = gr.Button("🔍 Check System Health", variant="secondary")
                test_btn = gr.Button("🧪 Test Model", variant="secondary")
            
            with gr.Column():
                health_output = gr.Textbox(
                    label="System Status",
                    lines=15,
                    interactive=False
                )
                test_output = gr.Textbox(
                    label="Model Test Result",
                    lines=3,
                    interactive=False
                )
        
        health_btn.click(fn=health_check, outputs=health_output)
        test_btn.click(fn=test_simple_qa, outputs=test_output)
    
    with gr.Tab("📚 Documentation"):
        gr.Markdown("""
        ## 📋 API Usage Guide
        
        ### 🔐 Authentication
        Use your assigned API key in the "API Key" field above. Contact your team lead if you don't have one.
        
        ### 📄 Input Requirements
        - **PDF URL**: Must be a direct, publicly accessible link to the policy document
        - **Questions**: Enter one question per line, maximum 10 questions per request
        
        ### ❓ Example Questions for Insurance Policies:
        ```
        What is the grace period for premium payment under this policy?
        What is the waiting period for pre-existing diseases (PED) to be covered?
        Does this policy cover maternity expenses, and what are the conditions?
        What is the waiting period for cataract surgery?
        Are medical expenses for organ donors covered under this policy?
        What is the No Claim Discount (NCD) offered in this policy?
        Is there a benefit for preventive health check-ups?
        How does the policy define a 'Hospital'?
        What is the extent of coverage for AYUSH treatments?
        Are there any sub-limits on room rent and ICU charges?
        ```
        
        ### 🔧 Technical Specifications
        - **LLM**: Google Gemma-2B-IT (locally hosted pipeline)
        - **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
        - **Vector Database**: ChromaDB (in-memory)
        - **Framework**: Gradio + HuggingFace Spaces
        - **Processing**: RAG (Retrieval-Augmented Generation)
        
        ### 🚀 How It Works
        1. **Document Processing**: Downloads and extracts text from PDF
        2. **Text Chunking**: Splits document into searchable segments  
        3. **Vector Storage**: Creates embeddings and stores in ChromaDB
        4. **Question Processing**: For each question, finds relevant chunks
        5. **Answer Generation**: Uses Gemma to generate answers based on context
        
        ### ⚡ Performance Notes
        - First request may take 30-60 seconds (model initialization)
        - Subsequent requests: 10-30 seconds depending on document size
        - Optimized for policy documents up to 100 pages
        
        ### 🎯 HackRx 2024 Submission
        **Team**: [Your Team Name]  
        **Members**: [Team Member Names]  
        **Submission Date**: [Date]  
        
        This solution demonstrates advanced RAG architecture with local LLM deployment for enterprise document analysis.
        """)

# Launch the demo
if __name__ == "__main__":
    print("🚀 Starting HackRx RAG QA API with local Gemma model...")
    print(f"📊 Model: {MODEL_NAME}")
    print(f"🔐 API Key configured: {bool(HACKRX_API_KEY != 'hackrx-default-key-2024')}")
    print(f"🤗 HF Token configured: {bool(HF_TOKEN)}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
