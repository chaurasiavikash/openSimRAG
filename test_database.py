import sys
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def test_database(vector_db_path):
    """
    Test a vector database by performing some simple queries.
    """
    print(f"Testing vector database at: {vector_db_path}")
    
    if not os.path.exists(vector_db_path):
        print(f"Error: Database path {vector_db_path} does not exist.")
        return False
    
    try:
        # Load the embedding model
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Connect to the vector database
        print("Connecting to vector database...")
        vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings,
            collection_name="opensim_docs"
        )
        
        # Get the collection size
        count = vectorstore._collection.count()
        print(f"Database contains {count} documents.")
        
        if count == 0:
            print("Error: Database exists but contains no documents.")
            return False
        
        # Perform some test queries
        test_queries = [
            "What is OpenSim?",
            "How to install OpenSim",
            "markers in OpenSim",
            "forward dynamics simulation",
            "Python API in OpenSim"
        ]
        
        print("\nPerforming test queries...\n")
        
        for query in test_queries:
            print(f"Query: '{query}'")
            # Get relevant documents
            docs = vectorstore.similarity_search_with_relevance_scores(query, k=2)
            
            if not docs:
                print("  No relevant documents found.")
            else:
                for i, (doc, score) in enumerate(docs):
                    print(f"  Result {i+1} - Relevance: {score:.4f}")
                    print(f"  Title: {doc.metadata.get('title', 'Unknown')}")
                    print(f"  Content snippet: {doc.page_content[:100]}...")
                    print()
        
        print("Database test completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error testing database: {e}")
        return False

if __name__ == "__main__":
    # Use the database path from command line or default
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./chroma_db"
    test_database(db_path)