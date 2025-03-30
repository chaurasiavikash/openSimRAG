#!/usr/bin/env python
"""
This script processes the existing chunked data and creates a fixed vector database.
Run this if you encountered the ChromaDB metadata error.
"""

import os
import argparse
from vector_database import VectorDatabase

def main():
    parser = argparse.ArgumentParser(description="Fix OpenSim Vector Database")
    
    parser.add_argument("--chunked_dir", type=str, default="./chunked_data", 
                       help="Directory containing chunked data")
    parser.add_argument("--vector_dir", type=str, default="./vector_db_fixed", 
                       help="Directory to save fixed vector database")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", 
                       help="Name of the sentence transformer model")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.vector_dir, exist_ok=True)
    
    print("\n=== Creating Fixed Vector Database ===")
    
    # Initialize the vector database with the fixed implementation
    db = VectorDatabase(
        input_dir=args.chunked_dir,
        output_dir=args.vector_dir,
        model_name=args.model_name
    )
    
    # Process the chunks and create the vector database
    db.process_chunks()
    
    print("\nTesting the fixed database with sample queries...")
    
    # Test with some sample queries
    sample_queries = [
        "How do I install OpenSim?",
        "What is forward dynamics in OpenSim?",
        "How to create a muscle model in OpenSim?"
    ]
    
    for query in sample_queries:
        db.test_query(query)
    
    print("\nFixed vector database created successfully!")
    print(f"The database is stored in: {args.vector_dir}")
    print("\nTo use this fixed database with the RAG systems, specify the path with --vector_db_path:")
    print(f"python rag_system.py --vector_db_path {os.path.join(args.vector_dir, 'chroma_db')}")
    print(f"python opensim_openai_rag.py --vector_db_path {os.path.join(args.vector_dir, 'chroma_db')}")
    print(f"python web_interface.py --vector_db_path {os.path.join(args.vector_dir, 'chroma_db')}")

if __name__ == "__main__":
    main()