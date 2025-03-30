import os
import argparse
import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline, BitsAndBytesConfig
from rag_utils import process_chroma_metadata, format_sources_for_display

class OpenSimRAG:
    def __init__(self, vector_db_path="./vector_db/chroma_db", 
                 embedding_model_name="all-MiniLM-L6-v2",
                 llm_model_name="HuggingFaceH4/zephyr-7b-beta"):
        """
        Initialize the OpenSim RAG system with Zephyr or other large models.
        
        Args:
            vector_db_path (str): Path to the ChromaDB database
            embedding_model_name (str): Name of the embedding model
            llm_model_name (str): Name of the LLM to use for response generation
        """
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        
        print(f"Initializing RAG system...")
        print(f"Loading embeddings model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Load the vector database
        print(f"Loading vector database from: {vector_db_path}")
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings,
            collection_name="opensim_docs"
        )
        
        # Create a retriever with increased document count
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Retrieve 10 documents
        )
        
        # Initialize the LLM
        print(f"Loading LLM: {llm_model_name}")
        self.llm = self._init_large_llm(llm_model_name)
        
        # Create the QA chain
        self.qa_chain = self._create_qa_chain()
        
        print("RAG system initialized successfully!")
    
    def _init_large_llm(self, model_name):
        """
        Initialize a large language model with quantization for memory efficiency.
        
        Args:
            model_name (str): Name of the model to use
            
        Returns:
            LLM: Initialized LLM
        """
        try:
            print(f"Loading model: {model_name}")
            print("This may take a few minutes for large models...")
            
            # Configure quantization for memory efficiency
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                # Load the tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Add special tokens handling for Zephyr model
                if "zephyr" in model_name.lower():
                    # Zephyr uses specific formatting
                    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '\n<|assistant|>' }}\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] + '\n' }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + '\n' }}\n{% endif %}\n{% endfor %}"
                
                # Load the model with quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
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
                    top_k=50,
                    repetition_penalty=1.2
                )
                
                # Wrap pipeline in LangChain
                llm = HuggingFacePipeline(pipeline=pipe)
                print("Model loaded successfully!")
                return llm
            except Exception as e:
                # If BitsAndBytes fails, try without quantization
                print(f"Quantization error: {e}")
                print("Trying without quantization...")
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95
                )
                
                llm = HuggingFacePipeline(pipeline=pipe)
                print("Model loaded successfully without quantization!")
                return llm
                
        except Exception as e:
            print(f"Error initializing large model: {e}")
            print("This might be due to insufficient GPU memory or CUDA issues.")
            print("Falling back to a smaller model...")
            
            fallback_model = "google/flan-t5-large"
            print(f"Loading fallback model: {fallback_model}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
                
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95
                )
                
                llm = HuggingFacePipeline(pipeline=pipe)
                return llm
            except Exception as e:
                print(f"Error loading fallback model: {e}")
                print("Falling back to the smallest model...")
                
                smallest_fallback = "google/flan-t5-small"
                tokenizer = AutoTokenizer.from_pretrained(smallest_fallback)
                model = AutoModelForSeq2SeqLM.from_pretrained(smallest_fallback)
                
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95
                )
                
                llm = HuggingFacePipeline(pipeline=pipe)
                return llm
    
    def _create_qa_chain(self):
        """
        Create the QA chain with a prompt template optimized for the model.
        
        Returns:
            RetrievalQA: Configured QA chain
        """
        # Check if we're using Zephyr
        if "zephyr" in self.llm_model_name.lower():
            # Define a prompt template optimized for Zephyr
            template = """<|system|>
You are an expert on OpenSim, a biomechanical simulation software used for modeling, simulating, and analyzing neuromusculoskeletal systems. Answer questions based only on the provided context.
<|user|>
I need you to answer the following question about OpenSim based on the documentation provided:

Question: {question}

Here are relevant sections from the OpenSim documentation:

{context}

Please provide a comprehensive, detailed answer using only the information provided. Include specific steps, commands, or file paths if they are mentioned in the documentation.
<|assistant|>
"""
        else:
            # Generic template for other models
            template = """
You are an expert on OpenSim, a biomechanical simulation software used for modeling, simulating, and analyzing neuromusculoskeletal systems.

Question: {question}

Here are relevant sections from the OpenSim documentation:

{context}

Provide a comprehensive, detailed answer using only the information provided. Include specific steps, commands, or file paths if they are mentioned in the documentation.
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        chain_type_kwargs = {"prompt": prompt}
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        
        return qa_chain
    
    def query(self, question):
        """
        Query the RAG system with a question.
        
        Args:
            question (str): Question about OpenSim
            
        Returns:
            dict: Response and source documents
        """
        print(f"\nQuerying: {question}")
        
        try:
            # Use the invoke method for newer LangChain versions
            result = self.qa_chain.invoke({"query": question})
            
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Clean up the response for Zephyr model
            if "zephyr" in self.llm_model_name.lower():
                # Remove any system or user prompts that might be in the output
                if "<|user|>" in answer:
                    answer = answer.split("<|user|>")[0].strip()
                if "<|system|>" in answer:
                    answer = answer.split("<|system|>")[0].strip()
                if "<|assistant|>" in answer:
                    answer = answer.split("<|assistant|>")[-1].strip()
            
            return {
                "question": question,
                "answer": answer,
                "source_docs": source_docs
            }
            
        except Exception as e:
            print(f"Error during query processing: {e}")
            # Provide a fallback response
            return {
                "question": question,
                "answer": f"I encountered an error while processing your question. This might be due to resource constraints or an issue with the model. Error details: {str(e)}",
                "source_docs": []
            }
    
    def interactive_mode(self):
        """
        Run the RAG system in interactive mode.
        """
        print(f"\n=== OpenSim RAG Interactive Mode ({self.llm_model_name}) ===")
        print("Ask questions about OpenSim (type 'exit' to quit)")
        
        while True:
            question = input("\nQuestion: ")
            
            if question.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode.")
                break
            
            try:
                result = self.query(question)
                
                print("\nAnswer:")
                print(result["answer"])
                
                # Process and display sources
                if result["source_docs"]:
                    print("\nSources:")
                    for i, doc in enumerate(result["source_docs"][:3]):
                        # Process metadata to handle string-encoded lists and dicts
                        metadata = process_chroma_metadata(doc.metadata)
                        print(f"{i+1}. {metadata.get('title', 'Unknown')} - {metadata.get('source', 'Unknown')}")
                        if "section" in metadata and metadata["section"]:
                            print(f"   Section: {metadata['section']}")
                        print(f"   Snippet: {doc.page_content[:100]}...")
                else:
                    print("\nNo source documents were retrieved.")
            except Exception as e:
                print(f"Error: {e}")
                print("Please try a different question or check the system setup.")


def main():
    """
    Run the OpenSim RAG system with large language model.
    """
    parser = argparse.ArgumentParser(description="OpenSim RAG System with Large Language Model")
    
    parser.add_argument("--vector_db_path", type=str, default="./vector_db/chroma_db", 
                       help="Path to the ChromaDB database")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", 
                       help="Name of the embedding model")
    parser.add_argument("--llm_model", type=str, default="HuggingFaceH4/zephyr-7b-beta", 
                       help="Name of the LLM to use for response generation")
    parser.add_argument("--question", type=str, 
                       help="Single question to answer (if not provided, interactive mode is used)")
    
    args = parser.parse_args()
    
    # Create the RAG system
    rag_system = OpenSimRAG(
        vector_db_path=args.vector_db_path,
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model
    )
    
    if args.question:
        # Answer a single question
        result = rag_system.query(args.question)
        
        print("\nAnswer:")
        print(result["answer"])
        
        print("\nSources:")
        for i, doc in enumerate(result["source_docs"][:3]):
            # Process metadata to handle string-encoded lists and dicts
            metadata = process_chroma_metadata(doc.metadata)
            print(f"{i+1}. {metadata.get('title', 'Unknown')} - {metadata.get('source', 'Unknown')}")
            if "section" in metadata and metadata["section"]:
                print(f"   Section: {metadata['section']}")
            print(f"   Snippet: {doc.page_content[:100]}...")
    else:
        # Run in interactive mode
        rag_system.interactive_mode()


if __name__ == "__main__":
    main()