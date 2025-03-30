from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# Directory containing your downloaded PDFs
pdf_dir = "./academic_papers"
# Output directory for the processed database
output_dir = "./academic_db"

# Create embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create text splitter for academic content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", "! ", "? "]
)

# Process each PDF
all_chunks = []
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        try:
            # Extract paper details from filename (you could use a naming convention)
            paper_title = filename.replace(".pdf", "")
            
            # Load and process PDF
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            pages = loader.load()
            
            # Extract and process text from PDF
            for page in pages:
                page.metadata["title"] = paper_title
                page.metadata["source"] = f"Academic paper: {paper_title}"
                page.metadata["content_type"] = "research_paper"
            
            # Split into chunks
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            
            print(f"Processed: {filename} - {len(chunks)} chunks")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Create or update database
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=output_dir,
    collection_name="opensim_academic"
)