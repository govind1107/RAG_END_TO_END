from langchain_openai import OpenAIEmbeddings
from configs.config import OPENAI_API_KEY,TOP_K_RESULTS
from modules.logger import app_logger
from modules.custom_exceptions import RetrievalError

import numpy as np

def retrieve_relevent_documents(vector_store,query):
    try:
        app_logger.info("Retrieving relevent documents")


        if vector_store is None or vector_store.index is None:
            app_logger.error("Vector store or index is None, cannot perform similarity search.")
            raise RetrievalError("Vector store or index is None, cannot perform similarity search.")

        if not query:
            raise RetrievalError("No Query")
        
        embedding_model = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

        query_embedding = embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding)

        # app_logger.info(f"Embedded {query_embedding} relevent documents")


        # results = vector_store.similarity_search_by_vector(query_embedding,k=TOP_K_RESULTS)

        if vector_store is None:
            app_logger.info("Vector store is None")

                # Perform similarity search
        distances, indices = vector_store.index.search(query_embedding.reshape(1, -1), TOP_K_RESULTS)

        # Retrieve the documents based on indices
        results = [vector_store.docstore.search(vector_store.index_to_docstore_id[idx]) for idx in indices[0]]


        app_logger.info(f"Retrived {len(results)} relevent documents")

        return results

    except Exception as e:

        app_logger.error(f"Error in document retrieval {str(e)}")
        raise RetrievalError("Failed to retrive relevent documents ") from e