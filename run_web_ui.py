#!/usr/bin/env python
"""
Run the OpenSim RAG Web UI with commonly used settings.
"""

import os
import subprocess
import argparse

def main():
    """Run the web UI with the specified settings"""
    parser = argparse.ArgumentParser(description="Run the OpenSim RAG Web UI")
    
    parser.add_argument("--db_path", type=str, default="./chroma_db",
                       help="Path to the ChromaDB database")
    parser.add_argument("--model", type=str, default="HuggingFaceH4/zephyr-7b-beta",
                       help="Name of the language model to use")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to run the server on (use 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run the server on")
    
    args = parser.parse_args()
    
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    # Ensure the HTML template exists
    if not os.path.exists("templates/index.html"):
        print("Error: templates/index.html not found")
        print("Please make sure the template file is in the correct location")
        return 1
    
    # Check if the database exists
    if not os.path.exists(args.db_path):
        print(f"Warning: Database path {args.db_path} does not exist")
        print("You may need to create the database first with create_opensim_db.py")
        
        # Ask if the user wants to continue
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Start the web server
    print(f"Starting OpenSim RAG Web UI...")
    print(f"Database: {args.db_path}")
    print(f"Model: {args.model}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    
    # Build the command
    cmd = [
        "python", "app.py",
        "--db_path", args.db_path,
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    # Run the command
    try:
        subprocess.run(cmd)
        return 0
    except KeyboardInterrupt:
        print("\nServer stopped")
        return 0
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1

if __name__ == "__main__":
    exit(main())