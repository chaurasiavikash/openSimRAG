import os
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import glob

class VectorDatabase:
    def __init__(self, input_dir="./chunked_data", output_dir="./vector_db", 
                 model_name="all-MiniLM-L6-v2"):
        """
        Initialize the vector database creator.
        
        Args:
            input_dir (str): Directory containing chunked documents
            output_dir (str): Directory to save vector database
            model_name (str): Name of the sentence transformer model to use
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the sentence transformer model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=os.path.join(output_dir, "chroma_db"),
            anonymized_telemetry=False
        ))
        
        # Create collection
        try:
            self.collection = self.chroma_client.get_collection(name="opensim_docs")
            print("Collection 'opensim_docs' already exists, using existing collection.")
        except:
            self.collection = self.chroma_client.create_collection(
                name="opensim_docs",
                metadata={"description": "OpenSim documentation chunks"}
            )
            print("Created new collection 'opensim_docs'.")
    
    def generate_embeddings(self, chunks):
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks (list): List of chunk dictionaries
            
        Returns:
            list: List of chunk dictionaries with embeddings
        """
        # Extract text content from chunks
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
        return chunks
    
    def _prepare_metadata_for_chromadb(self, metadata):
        """
        Prepare metadata for ChromaDB by converting non-primitive types to strings.
        
        Args:
            metadata (dict): Original metadata
            
        Returns:
            dict: ChromaDB-compatible metadata
        """
        result = {}
        
        for key, value in metadata.items():
            # Convert lists to strings
            if isinstance(value, list):
                result[key] = "|".join(str(item) for item in value)
            # Convert dictionaries to strings
            elif isinstance(value, dict):
                result[key] = json.dumps(value)
            # Keep primitive types as is
            elif isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = value
            # Convert anything else to string
            else:
                result[key] = str(value)
        
        return result
    
    def add_to_chromadb(self, chunks):
        """
        Add chunks with embeddings to ChromaDB.
        
        Args:
            chunks (list): List of chunk dictionaries with embeddings
        """
        # Prepare data for ChromaDB
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        
        # Process metadata to ensure it's compatible with ChromaDB
        processed_metadatas = []
        for chunk in chunks:
            processed_metadata = self._prepare_metadata_for_chromadb(chunk["metadata"])
            processed_metadatas.append(processed_metadata)
        
        # Add to collection in batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
            try:
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=processed_metadatas[i:end_idx]
                )
                print(f"Added batch {i//batch_size + 1} to ChromaDB ({end_idx-i} documents)")
            except Exception as e:
                print(f"Error adding batch to ChromaDB: {e}")
                # Continue with next batch
                continue
        
        print(f"Added {len(ids)} documents to ChromaDB collection.")
    
    def save_embeddings_json(self, chunks, filename="embeddings.json"):
        """
        Save chunks with embeddings to JSON file.
        
        Args:
            chunks (list): List of chunk dictionaries with embeddings
            filename (str): Name of output file
        """
        output_file = os.path.join(self.output_dir, filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Saved embeddings to {output_file}")
    
    def process_chunks(self):
        """
        Process all chunks in the input directory.
        """
        # Check if all_chunks.json exists
        all_chunks_file = os.path.join(self.input_dir, "all_chunks.json")
        
        if os.path.exists(all_chunks_file):
            # Load all chunks from single file
            with open(all_chunks_file, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
            
            # Generate embeddings
            all_chunks_with_embeddings = self.generate_embeddings(all_chunks)
            
            # Save embeddings to file
            self.save_embeddings_json(all_chunks_with_embeddings)
            
            # Add to ChromaDB
            self.add_to_chromadb(all_chunks_with_embeddings)
        else:
            # Get all JSON files in the input directory
            json_files = glob.glob(os.path.join(self.input_dir, "chunks_*.json"))
            all_chunks_with_embeddings = []
            
            with tqdm(total=len(json_files), desc="Processing chunk files") as pbar:
                for json_file in json_files:
                    try:
                        # Read the chunks
                        with open(json_file, 'r', encoding='utf-8') as f:
                            chunks = json.load(f)
                        
                        # Generate embeddings
                        chunks_with_embeddings = self.generate_embeddings(chunks)
                        all_chunks_with_embeddings.extend(chunks_with_embeddings)
                        
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing {json_file}: {e}")
                        pbar.update(1)
            
            # Save all embeddings to a single file
            self.save_embeddings_json(all_chunks_with_embeddings)
            
            # Add all to ChromaDB
            self.add_to_chromadb(all_chunks_with_embeddings)
    
    def test_query(self, query, top_k=5):
        """
        Test vector database with a query.
        
        Args:
            query (str): Query text
            top_k (int): Number of results to return
        """
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        print(f"\nQuery: {query}")
        print(f"Top {top_k} results:")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0], 
            results["metadatas"][0],
            results["distances"][0]
        )):
            print(f"\n--- Result {i+1} (Distance: {distance:.4f}) ---")
            print(f"Title: {metadata.get('title', 'N/A')}")
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Section: {metadata.get('section', 'N/A')}")
            print(f"Content: {doc[:200]}...")


if __name__ == "__main__":
    # Create and run the vector database
    db = VectorDatabase()
    db.process_chunks()
    
    # Test with some sample queries
    sample_queries = [
        "How do I install OpenSim?",
        "What is forward dynamics in OpenSim?",
        "How to create a muscle model in OpenSim?",
        "OpenSim Python API example"
    ]
    
    for query in sample_queries:
        db.test_query(query)