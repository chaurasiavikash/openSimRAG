import os
import argparse
import time
from flask import Flask, render_template, request, jsonify
import markdown
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'opensim-rag-secret-key'

# Global variables to hold the RAG components
vectorstore = None
llm = None
embedding_model = None

def initialize_rag(db_path, model_name="HuggingFaceH4/zephyr-7b-beta"):
    """Initialize the RAG system components"""
    global vectorstore, llm, embedding_model
    
    print(f"Initializing RAG system...")
    print(f"Database path: {db_path}")
    print(f"Model: {model_name}")
    
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize vector database
    print(f"Loading vector database from: {db_path}")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model,
        collection_name="opensim_docs"
    )
    
    # Initialize language model
    print(f"Loading language model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure model loading options
        model_loading_options = {
            "device_map": "auto",
            "torch_dtype": torch.float16
        }
        
        # For 4-bit quantization if bitsandbytes is available
        try:
            from bitsandbytes.nn import Linear4bit
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            model_loading_options["quantization_config"] = quantization_config
            print("Using 4-bit quantization for model loading")
        except ImportError:
            print("bitsandbytes not available. Using default model loading.")
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_loading_options
        )
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        # Create LangChain pipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if you have enough memory and the model is accessible.")
        print("Continuing without language model (will use remote API if enabled).")
        llm = None
    
    print("RAG system initialized!")

def process_question(question):
    """Process a question and return the answer with sources"""
    if vectorstore is None:
        return {
            "error": "RAG system not initialized. Please check server logs."
        }
    
    start_time = time.time()
    
    try:
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(question, k=5)
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant information about that in the OpenSim documentation. Could you try rephrasing your question?",
                "sources": [],
                "time_taken": time.time() - start_time
            }
        
        # Build context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        if llm:
            # Create prompt based on model type
            if "zephyr" in str(llm).lower():
                prompt = f"<|system|>\nYou are an expert assistant for OpenSim biomechanical simulation software. Answer the question based only on the provided context.\n<|user|>\nContext from OpenSim documentation:\n\n{context}\n\nQuestion: {question}\n<|assistant|>"
            else:
                prompt = f"You are an expert assistant for OpenSim biomechanical simulation software. Answer the question based only on the provided context.\n\nContext from OpenSim documentation:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Generate response
            response = llm.pipeline(prompt, return_full_text=False)[0]['generated_text']
            
            # Clean up response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
        else:
            # Fallback if model not loaded
            response = "The language model is not loaded. Please check server logs or try again later."
        
        # Process sources
        sources = []
        for doc in docs[:3]:  # Limit to top 3 sources
            source = {
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source)
        
        return {
            "answer": response,
            "sources": sources,
            "time_taken": time.time() - start_time
        }
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "error": f"An error occurred while processing your question: {str(e)}",
            "time_taken": time.time() - start_time
        }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle the query API endpoint"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    result = process_question(question)
    return jsonify(result)

@app.route('/status')
def status():
    """Return the status of the RAG system"""
    is_initialized = vectorstore is not None
    has_llm = llm is not None
    
    return jsonify({
        "status": "ok" if is_initialized else "not_initialized",
        "vectorstore_initialized": is_initialized,
        "llm_initialized": has_llm,
        "embedding_model": "all-MiniLM-L6-v2"
    })

def main():
    """Main function to run the Flask app"""
    parser = argparse.ArgumentParser(description="OpenSim RAG Web Interface")
    
    parser.add_argument("--db_path", type=str, default="./chroma_db",
                       help="Path to the ChromaDB database")
    parser.add_argument("--model", type=str, default="HuggingFaceH4/zephyr-7b-beta",
                       help="Name of the language model to use")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run the server on")
    parser.add_argument("--debug", action="store_true",
                       help="Run Flask in debug mode")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    initialize_rag(db_path=args.db_path, model_name=args.model)
    
    # Start the Flask app
    print(f"Starting web server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()