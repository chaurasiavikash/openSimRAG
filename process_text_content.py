import os
import argparse
import logging
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

class TextContentProcessor:
    def __init__(self, 
                 content_dir="./opensim_content", 
                 existing_db_path="./chroma_db",
                 embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the text content processor to add content to the existing database.
        
        Args:
            content_dir (str): Directory containing the text content files
            existing_db_path (str): Path to the existing ChromaDB
            embedding_model (str): Name of the embedding model to use
        """
        self.content_dir = content_dir
        self.existing_db_path = existing_db_path
        self.embedding_model = embedding_model
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(content_dir, "processing_log.txt")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()
        
        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Create text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", "! ", "? "]
        )
    
    def process_text_file(self, file_path):
        """
        Process a single text file.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            list: List of document chunks
        """
        try:
            # Extract filename and source type
            filename = os.path.basename(file_path)
            source_type = "Unknown"
            
            if filename.startswith("forum_"):
                source_type = "Forum"
            elif filename.startswith("github_"):
                source_type = "GitHub"
            elif filename.startswith("doc_"):
                source_type = "Documentation"
            
            # Extract title from file
            title = filename.replace("forum_", "").replace("github_", "").replace("doc_", "").replace(".txt", "")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract URL if present in file
            url = "Unknown"
            if "URL: " in content:
                url_line = content.split("URL: ")[1].split("\n")[0]
                url = url_line.strip()
            
            # Create a single document
            document = Document(
                page_content=content,
                metadata={
                    "title": title,
                    "source": f"{source_type}: {filename}",
                    "url": url,
                    "content_type": source_type.lower()
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Add section metadata based on content
            for i, chunk in enumerate(chunks):
                # We'll add a simple section identifier
                chunk.metadata["section"] = f"Section {i+1}"
            
            self.logger.info(f"Processed {file_path} - Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def process_all_content(self):
        """
        Process all text files in the content directory.
        
        Returns:
            list: List of all document chunks
        """
        all_chunks = []
        
        # Get all text files
        text_files = [os.path.join(self.content_dir, f) for f in os.listdir(self.content_dir) 
                     if f.endswith('.txt') and os.path.isfile(os.path.join(self.content_dir, f))]
        
        self.logger.info(f"Found {len(text_files)} text files to process")
        
        # Process each file
        for file_path in tqdm(text_files, desc="Processing text files"):
            chunks = self.process_text_file(file_path)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def add_to_existing_database(self, chunks):
        """
        Add chunks to the existing vector database.
        
        Args:
            chunks (list): List of document chunks
        """
        self.logger.info(f"Adding {len(chunks)} chunks to existing database at {self.existing_db_path}")
        
        try:
            # Direct ChromaDB approach instead of relying on persist()
            client = chromadb.PersistentClient(path=self.existing_db_path)
            
            # Get the collection
            try:
                collection = client.get_collection(name="opensim_docs")
                self.logger.info(f"Found existing collection 'opensim_docs'")
            except Exception as e:
                self.logger.warning(f"Collection not found: {e}")
                self.logger.info(f"Creating new collection 'opensim_docs'")
                collection = client.create_collection(name="opensim_docs")
            
            # Process chunks for adding to ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                # Generate a unique ID for each chunk
                chunk_id = f"txt_{i}_{hash(chunk.page_content)}"
                ids.append(chunk_id)
                documents.append(chunk.page_content)
                
                # Process metadata for compatibility
                processed_metadata = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, list):
                        processed_metadata[key] = "|".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        processed_metadata[key] = str(value)
                    else:
                        processed_metadata[key] = value
                
                metadatas.append(processed_metadata)
            
            # Get embeddings for each document
            embeddings = []
            for doc in documents:
                embedding = self.embeddings.embed_query(doc)
                embeddings.append(embedding)
            
            # Add data to collection
            self.logger.info(f"Adding {len(documents)} documents to ChromaDB collection")
            
            # Add in batches to prevent overload
            batch_size = 20
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                batch_ids = ids[i:end_idx]
                batch_docs = documents[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_metadata = metadatas[i:end_idx]
                
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata
                )
                
                self.logger.info(f"Added batch {i//batch_size + 1}, documents {i} to {end_idx}")
            
            self.logger.info(f"Successfully added {len(documents)} documents to collection")
            return True
                
        except Exception as e:
            self.logger.error(f"Error adding to database: {e}")
            return False
    
    def run(self):
        """Run the complete processing pipeline"""
        # Step 1: Process all content files
        self.logger.info("Starting content processing...")
        chunks = self.process_all_content()
        
        if not chunks:
            self.logger.error("No chunks created. Check the text files.")
            return False
        
        # Step 2: Add to existing database
        self.logger.info("Adding to existing vector database...")
        success = self.add_to_existing_database(chunks)
        
        # Summary
        self.logger.info("=== Processing Summary ===")
        self.logger.info(f"Processed content from: {self.content_dir}")
        self.logger.info(f"Created {len(chunks)} chunks")
        self.logger.info(f"Added to database at: {self.existing_db_path}")
        self.logger.info(f"Status: {'Success' if success else 'Failed'}")
        
        return success

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Process OpenSim text content and add to existing database")
    
    parser.add_argument("--content_dir", type=str, default="./opensim_content",
                       help="Directory containing the text content")
    parser.add_argument("--db_path", type=str, default="./chroma_db",
                       help="Path to the existing database")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                       help="Name of the embedding model to use")
    
    args = parser.parse_args()
    
    # Create and run processor
    processor = TextContentProcessor(
        content_dir=args.content_dir,
        existing_db_path=args.db_path,
        embedding_model=args.embedding_model
    )
    
    processor.run()

if __name__ == "__main__":
    main()