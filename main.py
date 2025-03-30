import os
import argparse
import time
from opensim_crawler import OpenSimCrawler
from content_processor import ContentProcessor
from chunking_system import DocumentChunker
from vector_database import VectorDatabase

def main():
    """
    Run the complete OpenSim documentation scraping pipeline.
    """
    parser = argparse.ArgumentParser(description="OpenSim Documentation Scraper")
    
    parser.add_argument("--max_pages", type=int, default=200, 
                       help="Maximum number of pages to crawl")
    parser.add_argument("--max_depth", type=int, default=3, 
                       help="Maximum depth for crawling")
    parser.add_argument("--chunk_size", type=int, default=500, 
                       help="Target size of chunks in tokens (words)")
    parser.add_argument("--chunk_overlap", type=int, default=100, 
                       help="Overlap between chunks in tokens")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", 
                       help="Name of the sentence transformer model")
    parser.add_argument("--skip_crawl", action="store_true", 
                       help="Skip the crawling step")
    parser.add_argument("--skip_process", action="store_true", 
                       help="Skip the content processing step")
    parser.add_argument("--skip_chunk", action="store_true", 
                       help="Skip the chunking step")
    parser.add_argument("--skip_vector", action="store_true", 
                       help="Skip the vector database creation step")
    
    args = parser.parse_args()
    
    # Define the output directories
    scraped_dir = "./scraped_data"
    processed_dir = "./processed_data"
    chunked_dir = "./chunked_data"
    vector_dir = "./vector_db"
    
    # Create directories if they don't exist
    for directory in [scraped_dir, processed_dir, chunked_dir, vector_dir]:
        os.makedirs(directory, exist_ok=True)
    
    total_start_time = time.time()
    
    # Step 1: Crawl the documentation
    if not args.skip_crawl:
        print("\n=== Step 1: Crawling OpenSim Documentation ===")
        start_time = time.time()
        
        # Define the starting URLs
        start_urls = [
            "https://simtk.org/projects/opensim",
            "https://simtk-confluence.stanford.edu/display/OpenSim/User%27s+Guide",
            "https://simtk-confluence.stanford.edu/display/OpenSim/Tutorials",
            "https://simtk.org/api_docs/opensim/api_docs/"
        ]
        
        # Create and run the crawler
        crawler = OpenSimCrawler(start_urls=start_urls, output_dir=scraped_dir, delay=2)
        crawler.crawl(max_pages=args.max_pages, max_depth=args.max_depth)
        
        elapsed = time.time() - start_time
        print(f"Crawling completed in {elapsed:.2f} seconds")
    else:
        print("\n=== Skipping Step 1: Crawling ===")
    
    # Step 2: Process the content
    if not args.skip_process:
        print("\n=== Step 2: Processing Content ===")
        start_time = time.time()
        
        processor = ContentProcessor(input_dir=scraped_dir, output_dir=processed_dir)
        processor.process_all_documents()
        
        elapsed = time.time() - start_time
        print(f"Content processing completed in {elapsed:.2f} seconds")
    else:
        print("\n=== Skipping Step 2: Content Processing ===")
    
    # Step 3: Chunk the documents
    if not args.skip_chunk:
        print("\n=== Step 3: Chunking Documents ===")
        start_time = time.time()
        
        chunker = DocumentChunker(
            input_dir=processed_dir, 
            output_dir=chunked_dir,
            chunk_size=args.chunk_size, 
            chunk_overlap=args.chunk_overlap
        )
        chunker.chunk_all_documents()
        
        elapsed = time.time() - start_time
        print(f"Chunking completed in {elapsed:.2f} seconds")
    else:
        print("\n=== Skipping Step 3: Chunking ===")
    
    # Step 4: Create the vector database
    if not args.skip_vector:
        print("\n=== Step 4: Creating Vector Database ===")
        start_time = time.time()
        
        db = VectorDatabase(
            input_dir=chunked_dir, 
            output_dir=vector_dir,
            model_name=args.model_name
        )
        db.process_chunks()
        
        # Test with some sample queries
        sample_queries = [
            "How do I install OpenSim?",
            "What is forward dynamics in OpenSim?",
            "How to create a muscle model in OpenSim?",
            "OpenSim Python API example"
        ]
        
        print("\n=== Testing Vector Database with Sample Queries ===")
        for query in sample_queries:
            db.test_query(query)
        
        elapsed = time.time() - start_time
        print(f"Vector database creation completed in {elapsed:.2f} seconds")
    else:
        print("\n=== Skipping Step 4: Vector Database Creation ===")
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal pipeline execution time: {total_elapsed:.2f} seconds")
    print("\nOpenSim documentation scraping completed successfully!")


if __name__ == "__main__":
    main()