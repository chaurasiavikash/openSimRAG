# 🧠 OpenSim RAG System

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/chaurasiavikash/openSimRAG.git)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7+-yellow.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-orange.svg)](https://huggingface.co/)

A Retrieval Augmented Generation (RAG) system for OpenSim biomechanical simulation software. This system enables natural language querying of OpenSim documentation, providing accurate, contextual answers about OpenSim functionality, implementation, and usage.

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Usage](#-usage)
- [System Architecture](#-system-architecture)
- [Enhancing the Knowledge Base](#-enhancing-the-knowledge-base)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## 🔍 Overview

The OpenSim RAG system combines the power of modern language models with a specialized knowledge base of OpenSim documentation and content. It allows users to:

- Ask natural language questions about OpenSim
- Receive accurate, context-aware answers
- Get information from official documentation, GitHub code, and other sources
- Access embedded knowledge without requiring internet searches

This system is particularly useful for researchers, students, and practitioners working with OpenSim who need quick access to specific information about the software's capabilities, APIs, and implementation details.

## 🚀 Installation

### Prerequisites

- Python 3.7 or later
- pip (Python package installer)
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/chaurasiavikash/openSimRAG.git
cd openSimRAG
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Creating the Knowledge Base

Before using the RAG system, you need to create a knowledge base with OpenSim documentation:

```bash
python create_opensim_db.py
```

This will create a vector database in the `./chroma_db` directory with essential OpenSim information.

### Running the RAG System

To start the interactive RAG system:

```bash
python fixed_rag.py --db_path ./chroma_db
```

Once running, you can ask questions about OpenSim:

```
Question: What is OpenSim used for?
Question: How do I install OpenSim on Windows?
Question: What are markers in OpenSim?
Question: How does forward dynamics work in OpenSim?
```

### Enhancing with Additional Content

To enhance the knowledge base with content from OpenSim GitHub repositories and documentation:

```bash
# Step 1: Scrape additional OpenSim content
python scrape_opensim_content.py

# Step 2: Process and add to the knowledge base
python process_text_content_fixed.py --content_dir ./opensim_content --db_path ./chroma_db
```

## 🏗️ System Architecture

The OpenSim RAG system consists of several components:

1. **Vector Database** (ChromaDB)
   - Stores text chunks and their vector embeddings
   - Enables semantic search for relevant information

2. **Embedding Model** (all-MiniLM-L6-v2)
   - Converts text to vector embeddings
   - Used for similarity search in the vector database

3. **Language Model** (Zephyr-7B)
   - Generates natural language responses
   - Integrates retrieved context with user queries

4. **Retrieval System**
   - Finds relevant documents based on question semantics
   - Provides context for the language model

5. **Content Processing Pipeline**
   - Chunks documents into manageable segments
   - Handles metadata for source tracking
   - Prepares text for embedding and storage

## 📚 Enhancing the Knowledge Base

The system's performance depends on the quality and breadth of its knowledge base. Here are ways to enhance it:

### Adding Documentation

1. **Scrape Additional Content**:
   ```bash
   python scrape_opensim_content.py --output_dir ./more_content
   ```

2. **Manual Additions**:
   - Add text files with useful OpenSim information to `./opensim_content`
   - Name files descriptively (e.g., `doc_installation_guide.txt`)
   - Include content with detailed technical information

3. **Process New Content**:
   ```bash
   python process_text_content_fixed.py --content_dir ./your_content_dir --db_path ./chroma_db
   ```

### Improving Embeddings

For better retrieval quality, you can use a more powerful embedding model:

```bash
python create_opensim_db.py --embedding_model "BAAI/bge-large-en-v1.5"
```

## ⚠️ Troubleshooting

### Common Issues

1. **No relevant documents retrieved**:
   - Ensure your database contains appropriate content
   - Try rephrasing your question
   - Add more domain-specific content to the knowledge base

2. **Memory issues with large models**:
   - Use 4-bit quantization (enabled by default)
   - Try a smaller language model like `google/flan-t5-large`
   - Reduce the chunk size in processing scripts

3. **ChromaDB errors**:
   - Make sure the database path exists
   - Check for compatible versions of langchain and chromadb
   - Try recreating the database if it becomes corrupted

### Database Verification

To verify your database is working correctly:

```bash
python fixed_rag.py --db_path ./chroma_db --question "What is OpenSim?"
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The OpenSim team for their excellent documentation and open-source software
- Hugging Face for providing open-source language and embedding models
- The LangChain community for creating tools that enable building RAG systems

---

📧 For issues, suggestions, or contributions, please [open an issue](https://github.com/chaurasiavikash/openSimRAG/issues) on the GitHub repository.