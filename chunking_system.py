import os
import json
import re
import glob
from tqdm import tqdm
import uuid

class DocumentChunker:
    def __init__(self, input_dir="./processed_data", output_dir="./chunked_data", 
                 chunk_size=500, chunk_overlap=100):
        """
        Initialize the document chunker.
        
        Args:
            input_dir (str): Directory containing processed documents
            output_dir (str): Directory to save chunked documents
            chunk_size (int): Target size of chunks in tokens (words)
            chunk_overlap (int): Overlap between chunks in tokens
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def split_text_into_chunks(self, text, heading="", document_metadata=None):
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to split
            heading (str): Section heading for context
            document_metadata (dict): Original document metadata
            
        Returns:
            list: List of chunk dictionaries
        """
        # Simple word-based tokenization (approximate)
        words = text.split()
        total_words = len(words)
        chunks = []
        
        if total_words <= self.chunk_size:
            # If text is shorter than chunk_size, return as single chunk
            chunk_content = text
            chunk_id = str(uuid.uuid4())
            
            chunks.append({
                "id": chunk_id,
                "document_id": document_metadata.get("id", ""),
                "content": chunk_content,
                "metadata": {
                    "source": document_metadata.get("url", ""),
                    "title": document_metadata.get("title", ""),
                    "section": heading,
                    "hierarchy": document_metadata.get("hierarchy", []),
                    "content_type": document_metadata.get("content_type", ""),
                    "position": 0,
                    "tags": [
                        document_metadata.get("content_type", ""),
                        document_metadata.get("category", ""),
                        document_metadata.get("difficulty", "")
                    ]
                }
            })
        else:
            # Create overlapping chunks
            start_idx = 0
            chunk_idx = 0
            
            while start_idx < total_words:
                # Calculate end index for current chunk
                end_idx = min(start_idx + self.chunk_size, total_words)
                
                # Get the chunk text
                chunk_content = " ".join(words[start_idx:end_idx])
                chunk_id = str(uuid.uuid4())
                
                chunks.append({
                    "id": chunk_id,
                    "document_id": document_metadata.get("id", ""),
                    "content": chunk_content,
                    "metadata": {
                        "source": document_metadata.get("url", ""),
                        "title": document_metadata.get("title", ""),
                        "section": heading,
                        "hierarchy": document_metadata.get("hierarchy", []),
                        "content_type": document_metadata.get("content_type", ""),
                        "position": chunk_idx,
                        "tags": [
                            document_metadata.get("content_type", ""),
                            document_metadata.get("category", ""),
                            document_metadata.get("difficulty", "")
                        ]
                    }
                })
                
                # Move to next chunk with overlap
                start_idx += (self.chunk_size - self.chunk_overlap)
                chunk_idx += 1
        
        return chunks
    
    def process_document(self, document):
        """
        Process a document into chunks.
        
        Args:
            document (dict): Document to process
            
        Returns:
            list: List of chunk dictionaries
        """
        chunks = []
        
        # Check if the document has sections
        if 'sections' in document and document['sections']:
            for section in document['sections']:
                section_heading = section.get('heading', '')
                section_content = section.get('content', '')
                
                if section_content:
                    # Process section content into chunks
                    section_chunks = self.split_text_into_chunks(
                        section_content, 
                        heading=section_heading, 
                        document_metadata=document
                    )
                    chunks.extend(section_chunks)
        else:
            # Process the whole document as one section
            whole_content = document.get('raw_text', '')
            if whole_content:
                content_chunks = self.split_text_into_chunks(
                    whole_content, 
                    heading="", 
                    document_metadata=document
                )
                chunks.extend(content_chunks)
        
        return chunks
    
    def chunk_special_content(self, document):
        """
        Create special chunks for code blocks, tables, etc.
        
        Args:
            document (dict): Document to process
            
        Returns:
            list: List of special chunk dictionaries
        """
        special_chunks = []
        
        # Process code blocks if present
        if 'code_blocks' in document and document['code_blocks']:
            for idx, code_block in enumerate(document['code_blocks']):
                chunk_id = str(uuid.uuid4())
                
                special_chunks.append({
                    "id": chunk_id,
                    "document_id": document.get("id", ""),
                    "content": code_block.get("content", ""),
                    "metadata": {
                        "source": document.get("url", ""),
                        "title": document.get("title", ""),
                        "section": "Code Example",
                        "hierarchy": document.get("hierarchy", []),
                        "content_type": "code_example",
                        "position": idx,
                        "language": code_block.get("language", "unknown"),
                        "tags": [
                            "code_example",
                            document.get("category", ""),
                            document.get("difficulty", "")
                        ]
                    }
                })
        
        return special_chunks
    
    def chunk_all_documents(self):
        """
        Process all documents in the input directory into chunks.
        """
        # Get all JSON files in the input directory
        json_files = glob.glob(os.path.join(self.input_dir, "*.json"))
        all_chunks = []
        
        with tqdm(total=len(json_files), desc="Chunking documents") as pbar:
            for json_file in json_files:
                try:
                    # Read the document
                    with open(json_file, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                    
                    # Process document into chunks
                    document_chunks = self.process_document(document)
                    
                    # Process special content
                    special_chunks = self.chunk_special_content(document)
                    
                    # Combine all chunks
                    combined_chunks = document_chunks + special_chunks
                    all_chunks.extend(combined_chunks)
                    
                    # Also save individual document chunks
                    output_file = os.path.join(self.output_dir, f"chunks_{os.path.basename(json_file)}")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(combined_chunks, f, ensure_ascii=False, indent=2)
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"Error chunking {json_file}: {e}")
                    pbar.update(1)
        
        # Save all chunks combined
        all_chunks_file = os.path.join(self.output_dir, "all_chunks.json")
        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Chunking complete. Created {len(all_chunks)} chunks from {len(json_files)} documents.")
        return all_chunks


if __name__ == "__main__":
    # Create and run the chunker
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunker.chunk_all_documents()