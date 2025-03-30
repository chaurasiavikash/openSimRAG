import os
import argparse
import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from rag_utils import process_chroma_metadata, format_sources_for_display

class OpenSimOpenAIRAG:
    def __init__(self, vector_db_path="./vector_db/chroma_db", 
                 embedding_model_name="all-MiniLM-L6-v2",
                 openai_api_key=None,
                 model_name="gpt-3.5-turbo"):
        """
        Initialize the OpenSim RAG system with OpenAI integration.
        
        Args:
            vector_db_path (str): Path to the ChromaDB database
            embedding_model_name (str): Name of the embedding model to use for retrieval
            openai_api_key (str): OpenAI API key (if None, will use OPENAI_API_KEY env variable)
            model_name (str): OpenAI model name to use
        """
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model_name
        
        # Set API key from parameter or environment variable
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key must be provided as parameter or set as OPENAI_API_KEY environment variable")
        
        print(f"Initializing OpenAI RAG system...")
        print(f"Loading embeddings model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Load the vector database
        print(f"Loading vector database from: {vector_db_path}")
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings,
            collection_name="opensim_docs"
        )
        
        # Create a retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.7}
        )
        
        # Initialize the OpenAI LLM
        print(f"Initializing OpenAI model: {model_name}")
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            verbose=True
        )
        
        # Create conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the QA chain
        self.qa_chain = self._create_qa_chain()
        
        print("RAG system initialized successfully!")
    
    def _create_qa_chain(self):
        """
        Create the QA chain with the appropriate prompt template.
        
        Returns:
            RetrievalQA: Configured QA chain
        """
        # Define the prompt template
        template = """
        You are an expert assistant for OpenSim, a biomechanical simulation software widely used in research and clinical settings.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer or the context doesn't provide the information needed, say so - don't try to make up an answer.
        
        The current date is {current_date}.
        
        Retrieved context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer the question in a comprehensive, accurate, and helpful manner. If the context includes code examples or technical commands, present them in a clear and well-formatted way. If relevant, suggest additional resources or related topics the user might be interested in.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "chat_history", "current_date"]
        )
        
        # Create the chain
        chain_type_kwargs = {
            "prompt": prompt,
            "memory": self.memory
        }
        
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
        
        # Get current date for context
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Query the chain
        result = self.qa_chain({
            "query": question,
            "current_date": current_date
        })
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
        # Process source documents for better presentation
        sources = []
        for doc in source_docs:
            # Process metadata to handle string-encoded lists and dicts
            metadata = process_chroma_metadata(doc.metadata)
            source = {
                "title": metadata.get("title", "Unknown"),
                "url": metadata.get("source", "Unknown"),
                "section": metadata.get("section", ""),
                "content_type": metadata.get("content_type", ""),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    
    def interactive_mode(self):
        """
        Run the RAG system in interactive mode.
        """
        print("\n=== OpenSim OpenAI RAG Interactive Mode ===")
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
                
                print("\nSources:")
                for i, source in enumerate(result["sources"][:3]):
                    print(f"{i+1}. {source['title']}")
                    print(f"   URL: {source['url']}")
                    if source['section']:
                        print(f"   Section: {source['section']}")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """
    Run the OpenSim OpenAI RAG system.
    """
    parser = argparse.ArgumentParser(description="OpenSim OpenAI RAG System")
    
    parser.add_argument("--vector_db_path", type=str, default="./vector_db/chroma_db", 
                       help="Path to the ChromaDB database")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", 
                       help="Name of the embedding model")
    parser.add_argument("--openai_api_key", type=str, 
                       help="OpenAI API key (if not provided, will use OPENAI_API_KEY env variable)")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", 
                       help="OpenAI model name to use")
    parser.add_argument("--question", type=str, 
                       help="Single question to answer (if not provided, interactive mode is used)")
    
    args = parser.parse_args()
    
    # Create the RAG system
    rag_system = OpenSimOpenAIRAG(
        vector_db_path=args.vector_db_path,
        embedding_model_name=args.embedding_model,
        openai_api_key=args.openai_api_key,
        model_name=args.model_name
    )
    
    if args.question:
        # Answer a single question
        result = rag_system.query(args.question)
        
        print("\nAnswer:")
        print(result["answer"])
        
        print("\nSources:")
        for i, source in enumerate(result["sources"][:3]):
            print(f"{i+1}. {source['title']}")
            print(f"   URL: {source['url']}")
            if source['section']:
                print(f"   Section: {source['section']}")
    else:
        # Run in interactive mode
        rag_system.interactive_mode()


if __name__ == "__main__":
    main()