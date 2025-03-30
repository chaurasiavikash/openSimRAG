#!/bin/bash

echo "===== OpenSim RAG System Setup and Run ====="
echo "This script will create a new OpenSim knowledge base and run the RAG system."

# Create the OpenSim vector database
echo ""
echo "Step 1: Creating OpenSim knowledge base..."
python create_opensim_db.py

# Test that the database was created correctly
echo ""
echo "Step 2: Testing the database..."
python test_database.py

# Run the enhanced RAG system
echo ""
echo "Step 3: Starting the RAG system..."
python enhanced_rag.py --vector_db_path ./chroma_db --debug

echo "===== Setup and run complete ====="