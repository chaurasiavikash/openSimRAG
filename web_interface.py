from flask import Flask, render_template, request, jsonify
import os
import argparse
import time
from datetime import datetime
import markdown
import json

# Import the OpenAI RAG implementation
from opensim_openai_rag import OpenSimOpenAIRAG

app = Flask(__name__)

# Global variable to hold the RAG system
rag_system = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle the query request"""
    question = request.json.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Query the RAG system
        start_time = time.time()
        result = rag_system.query(question)
        elapsed_time = time.time() - start_time
        
        # Convert the answer to HTML format if it contains markdown
        answer_html = markdown.markdown(result['answer'])
        
        # Prepare the response
        response = {
            'answer': answer_html,
            'sources': result['sources'],
            'elapsed_time': f"{elapsed_time:.2f} seconds"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_app(vector_db_path, embedding_model, openai_api_key=None, model_name="gpt-3.5-turbo"):
    """Create and configure the Flask app with RAG system"""
    global rag_system
    
    # Initialize the RAG system
    rag_system = OpenSimOpenAIRAG(
        vector_db_path=vector_db_path,
        embedding_model_name=embedding_model,
        openai_api_key=openai_api_key,
        model_name=model_name
    )
    
    # Ensure the templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenSim Documentation Assistant</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #d1ecf1;
            color: #0c5460;
            margin-left: 20%;
        }
        .assistant-message {
            background-color: #e2e3e5;
            color: #383d41;
            margin-right: 20%;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 10px 0;
        }
        .source-list {
            font-size: 0.85em;
            color: #666;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        code {
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">OpenSim Documentation Assistant</h1>
        <p class="lead mb-4">Ask questions about OpenSim biomechanical simulation software!</p>
        
        <div class="chat-container" id="chatContainer">
            <div class="message assistant-message">
                <p>Hello! I'm your OpenSim documentation assistant. How can I help you today?</p>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Thinking...</p>
        </div>
        
        <form id="questionForm" class="mb-4">
            <div class="input-group">
                <input type="text" id="questionInput" class="form-control" placeholder="Ask a question about OpenSim..." required>
                <button type="submit" class="btn btn-primary">Ask</button>
            </div>
        </form>
        
        <div class="alert alert-info" role="alert">
            <h4 class="alert-heading">Example Questions</h4>
            <ul>
                <li>How do I install OpenSim?</li>
                <li>What is forward dynamics in OpenSim?</li>
                <li>How do I create a muscle model?</li>
                <li>Can you explain inverse kinematics?</li>
            </ul>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const questionForm = document.getElementById('questionForm');
            const questionInput = document.getElementById('questionInput');
            const chatContainer = document.getElementById('chatContainer');
            const loading = document.getElementById('loading');
            
            questionForm.addEventListener('submit', function(event) {
                event.preventDefault();
                
                const question = questionInput.value.trim();
                if (!question) return;
                
                // Add user message
                addMessage(question, 'user');
                
                // Clear input
                questionInput.value = '';
                
                // Show loading indicator
                loading.style.display = 'block';
                
                // Send request to server
                fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        question: question
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    loading.style.display = 'none';
                    
                    if (data.error) {
                        addMessage('Error: ' + data.error, 'assistant');
                    } else {
                        // Format response with sources
                        let message = data.answer;
                        
                        if (data.sources && data.sources.length > 0) {
                            message += '<div class="source-list"><strong>Sources:</strong><ol>';
                            data.sources.slice(0, 3).forEach(source => {
                                message += `<li>${source.title} - <a href="${source.url}" target="_blank">Link</a>`;
                                if (source.section) {
                                    message += ` (${source.section})`;
                                }
                                message += '</li>';
                            });
                            message += '</ol></div>';
                        }
                        
                        addMessage(message, 'assistant');
                    }
                })
                .catch(error => {
                    // Hide loading indicator
                    loading.style.display = 'none';
                    
                    // Show error message
                    addMessage('Error connecting to server: ' + error, 'assistant');
                });
            });
            
            function addMessage(content, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.innerHTML = `<p>${content}</p>`;
                
                chatContainer.appendChild(messageDiv);
                
                // Scroll to bottom
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });
    </script>
</body>
</html>
        """)
    
    return app

def main():
    """Run the Flask app"""
    parser = argparse.ArgumentParser(description="OpenSim RAG Web Interface")
    
    parser.add_argument("--vector_db_path", type=str, default="./vector_db/chroma_db", 
                       help="Path to the ChromaDB database")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", 
                       help="Name of the embedding model")
    parser.add_argument("--openai_api_key", type=str, 
                       help="OpenAI API key (if not provided, will use OPENAI_API_KEY env variable)")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", 
                       help="OpenAI model name to use")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                       help="Host to run the Flask app on")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Port to run the Flask app on")
    
    args = parser.parse_args()
    
    # Create the app
    app = create_app(
        vector_db_path=args.vector_db_path,
        embedding_model=args.embedding_model,
        openai_api_key=args.openai_api_key,
        model_name=args.model_name
    )
    
    # Run the app
    print(f"Starting web interface at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()