"""
A simple modification for the Enhanced RAG system to fix the negative similarity scores issue.
"""

import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def fix_retrieval(db_path="./chroma_db"):
    """
    Test and fix the retrieval from the vector database.
    """
    print(f"Testing direct retrieval from: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Error: Database path {db_path} does not exist.")
        return
    
    # Load the embedding model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to the vector database
    print("Connecting to vector database...")
    try:
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name="opensim_docs"
        )
        
        # Count documents
        count = vectorstore._collection.count()
        print(f"Database contains {count} documents.")
        
        if count == 0:
            print("Error: The database is empty")
            return
        
        # Test a query directly
        print("\nTesting direct document retrieval without relevance scores...")
        query = "markers in opensim"
        docs = vectorstore.similarity_search(query, k=5)
        
        if docs:
            print(f"Successfully retrieved {len(docs)} documents!")
            for i, doc in enumerate(docs[:3]):
                print(f"\nDocument {i+1}:")
                print(f"Title: {doc.metadata.get('title', 'Unknown')}")
                print(f"Snippet: {doc.page_content[:150]}...")
            
            # Create a simple test script that uses direct document retrieval
            print("\nCreating a fixed_rag.py script...")
            with open("fixed_rag.py", "w") as f:
                f.write("""
import os
import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def main():
    parser = argparse.ArgumentParser(description="Simple Fixed OpenSim RAG")
    parser.add_argument("--db_path", type=str, default="./chroma_db", help="Path to the ChromaDB database")
    parser.add_argument("--question", type=str, help="Question to ask (optional)")
    args = parser.parse_args()
    
    # Load embeddings model
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to vector database
    print(f"Loading vector database from: {args.db_path}")
    vectorstore = Chroma(
        persist_directory=args.db_path,
        embedding_function=embeddings,
        collection_name="opensim_docs"
    )
    
    # Load Zephyr model
    print("Loading Zephyr model (this may take a minute)...")
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7
    )
    
    llm = HuggingFacePipeline(pipeline=text_gen)
    
    # Run in interactive mode
    print("\\nOpenSim RAG System Ready!")
    print("Ask questions about OpenSim (type 'exit' to quit)")
    
    while True:
        # Get question from argument or prompt
        if args.question:
            question = args.question
        else:
            question = input("\\nQuestion: ")
            
        if question.lower() in ["exit", "quit", "q"]:
            print("Exiting")
            break
        
        # Retrieve documents using simple similarity search (no thresholds)
        print(f"Retrieving information about: {question}")
        docs = vectorstore.similarity_search(question, k=5)
        
        if not docs:
            print("No relevant information found in the database.")
            continue
        
        # Build context from retrieved documents
        context = "\\n\\n".join([doc.page_content for doc in docs])
        
        # Create prompt 
        prompt = f"<|system|>\\nYou are an expert assistant for OpenSim biomechanical simulation software. Answer the question based only on the provided context.\\n<|user|>\\nContext from OpenSim documentation:\\n\\n{context}\\n\\nQuestion: {question}\\n<|assistant|>"
        
        # Generate answer
        print("Generating answer...")
        response = text_gen(prompt, return_full_text=False)[0]['generated_text']
        
        print("\\nAnswer:")
        print(response)
        
        # Print sources
        print("\\nSources:")
        for i, doc in enumerate(docs[:3]):
            title = doc.metadata.get('title', 'Unknown')
            source = doc.metadata.get('source', 'Unknown source')
            print(f"{i+1}. {title} - {source}")
        
        # Exit if question was provided as argument
        if args.question:
            break

if __name__ == "__main__":
    main()
""")
            
            print("\nFixed RAG script created! You can run it with:")
            print("python fixed_rag.py")
            print("\nThis script will bypass the similarity score thresholds and should work properly for your queries.")
            
        else:
            print("No documents retrieved with direct search either.")
            print("There might be a fundamental issue with the embeddings or database.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Use command line argument for database path if provided
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./chroma_db"
    fix_retrieval(db_path)