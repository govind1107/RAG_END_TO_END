from flask import Flask,request,jsonify
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.data_preprocessing import load_and_preprocess_pdfs
from modules.embedding_generation import generate_embeddings
from modules.vector_store import store_embeddings_in_faiss
from modules.retrieval import retrieve_relevent_documents
from modules.response_generation import generate_response
from modules.logger import app_logger



from modules.custom_exceptions import (
    EmbeddingGenerationError,VectorStoreError,RetrievalError,DataProcessingError,ResponseGenerationError
)


app = Flask(__name__)

documents = []
embeddings = []
vector_store = None

def initialize_vector_store():
    try:
        folder_path = r"C:\Users\HP\Desktop\ML\rag_on_large_doc\data"

        app_logger.info(f"Loading and preprocessing the pdfs")

        documents = load_and_preprocess_pdfs(folder_path)
        app_logger.info(f"Embedding generation")

        embeddings = generate_embeddings(documents)
        app_logger.info(f"Embedding storage")

        vector_store = store_embeddings_in_faiss(embeddings,documents)

        app_logger.info(f"DONE")

        return vector_store


    except (EmbeddingGenerationError,DataProcessingError,VectorStoreError,ResponseGenerationError,RetrievalError) as e:
        app_logger.critical(f"Critical Error during init : {str(e)}")
        raise
# Initialize vector store globally
vector_store = initialize_vector_store()

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        
        if not user_query:
            app_logger.warning("No query provided in the request.")
            return jsonify({"error": "Query parameter is missing."}), 400
        
        app_logger.info(f"Received query: {user_query}")
        
        # Retrieve relevant documents
        relevant_docs = retrieve_relevent_documents(vector_store, user_query)
        
        if not relevant_docs:
            app_logger.info("No relevant documents found. Generating response without context.")
            response = generate_response(user_query, [])
        else:
            # Generate response using context
            response = generate_response(user_query, relevant_docs)
        
        return jsonify({"response": response}), 200
    
    except (RetrievalError, ResponseGenerationError) as e:
        app_logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app_logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred."}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)