import os
import json
import argparse
import logging
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class AcademicPaperProcessor:
    def __init__(self, 
                 papers_dir="./academic_papers", 
                 output_dir="./academic_db",
                 embedding_model="BAAI/bge-large-en-v1.5",
                 merge_with_existing=False,
                 existing_db_path="./chroma_db"):
        """
        Initialize the academic paper processor.
        
        Args:
            papers_dir (str): Directory containing downloaded papers
            output_dir (str): Directory to save the processed database
            embedding_model (str): Name of the embedding model to use
            merge_with_existing (bool): Whether to merge with an existing database
            existing_db_path (str): Path to the existing database if merging
        """
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.embedding_model = embedding_model
        self.merge_with_existing = merge_with_existing
        self.existing_db_path = existing_db_path
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "process_log.txt")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("PaperProcessor")
        
        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Create text splitter optimized for academic content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", "! ", "? "]
        )
    
    def process_pdf(self, pdf_path, metadata_path=None):
        """
        Process a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            metadata_path (str): Path to the metadata JSON file (optional)
            
        Returns:
            list: List of document chunks
        """
        try:
            # Load metadata if available
            metadata = {}
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                # Extract title from filename
                filename = os.path.basename(pdf_path)
                title = filename.replace('.pdf', '')
                metadata = {
                    'title': title,
                    'year': 'Unknown',
                    'author': 'Unknown',
                    'venue': 'Unknown'
                }
            
            # Extract paper title
            paper_title = metadata.get('title', os.path.basename(pdf_path).replace('.pdf', ''))
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Skip if no text extracted
            if not pages:
                self.logger.warning(f"No text extracted from {pdf_path}")
                return []
            
            # Add metadata to pages
            for page in pages:
                page.metadata.update({
                    "title": paper_title,
                    "source": f"Academic paper: {paper_title}",
                    "authors": metadata.get('author', 'Unknown'),
                    "year": metadata.get('year', 'Unknown'),
                    "venue": metadata.get('venue', 'Unknown'),
                    "content_type": "research_paper"
                })
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Add section labels based on content
            for i, chunk in enumerate(chunks):
                content = chunk.page_content.lower()
                
                # Try to identify the section
                if i == 0:
                    chunk.metadata['section'] = 'Abstract'
                elif 'method' in content or 'methodology' in content:
                    chunk.metadata['section'] = 'Methods'
                elif 'result' in content or 'discussion' in content:
                    chunk.metadata['section'] = 'Results and Discussion'
                elif 'conclusion' in content:
                    chunk.metadata['section'] = 'Conclusion'
                elif 'reference' in content or 'bibliography' in content:
                    chunk.metadata['section'] = 'References'
                else:
                    chunk.metadata['section'] = f'Section {i+1}'
            
            self.logger.info(f"Processed {pdf_path} - Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return []
    
    def process_all_papers(self):
        """
        Process all papers in the directory.
        
        Returns:
            list: List of all document chunks
        """
        all_chunks = []
        
        # Get all PDF files
        pdf_files = []
        for filename in os.listdir(self.papers_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.papers_dir, filename)
                metadata_path = pdf_path.replace('.pdf', '.json')
                if not os.path.exists(metadata_path):
                    metadata_path = None
                
                pdf_files.append((pdf_path, metadata_path))
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        for pdf_path, metadata_path in tqdm(pdf_files, desc="Processing PDFs"):
            chunks = self.process_pdf(pdf_path, metadata_path)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def create_vector_database(self, chunks):
        """
        Create a vector database from the chunks.
        
        Args:
            chunks (list): List of document chunks
        """
        # If merging with existing database
        if self.merge_with_existing and os.path.exists(self.existing_db_path):
            self.logger.info(f"Merging with existing database at {self.existing_db_path}")
            
            try:
                # Load existing database
                existing_db = Chroma(
                    persist_directory=self.existing_db_path,
                    embedding_function=self.embeddings,
                    collection_name="opensim_docs"
                )
                
                # Add new chunks
                self.logger.info(f"Adding {len(chunks)} chunks to existing database")
                existing_db.add_documents(chunks)
                
                self.logger.info("Successfully merged with existing database")
                return
                
            except Exception as e:
                self.logger.error(f"Error merging with existing database: {e}")
                self.logger.info("Falling back to creating a new database")
        
        # Create new database
        self.logger.info(f"Creating new vector database with {len(chunks)} chunks")
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.output_dir,
                collection_name="opensim_academic"
            )
            
            # Persist the database
            vectorstore.persist()
            
            self.logger.info(f"Vector database created successfully at {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating vector database: {e}")
    
    def run(self):
        """Run the complete paper processing pipeline"""
        # Step 1: Process all papers
        self.logger.info("Starting paper processing...")
        chunks = self.process_all_papers()
        
        if not chunks:
            self.logger.error("No chunks created. Check the PDF files.")
            return False
        
        # Step 2: Create vector database
        self.logger.info("Creating vector database...")
        self.create_vector_database(chunks)
        
        # Summary
        self.logger.info("=== Processing Summary ===")
        self.logger.info(f"Processed papers from: {self.papers_dir}")
        self.logger.info(f"Created {len(chunks)} chunks")
        self.logger.info(f"Database saved to: {self.output_dir}")
        
        return True

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Process academic papers for the OpenSim RAG system")
    
    parser.add_argument("--papers_dir", type=str, default="./academic_papers",
                       help="Directory containing downloaded papers")
    parser.add_argument("--output_dir", type=str, default="./academic_db",
                       help="Directory to save the processed database")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5",
                       help="Name of the embedding model to use")
    parser.add_argument("--merge", action="store_true",
                       help="Merge with existing database")
    parser.add_argument("--existing_db", type=str, default="./chroma_db",
                       help="Path to the existing database if merging")
    
    args = parser.parse_args()
    
    # Create and run processor
    processor = AcademicPaperProcessor(
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        merge_with_existing=args.merge,
        existing_db_path=args.existing_db
    )
    
    processor.run()

if __name__ == "__main__":
    main()