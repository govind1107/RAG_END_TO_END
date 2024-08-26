from langchain.vectorstores import FAISS
import faiss
from modules.logger import app_logger
from modules.custom_exceptions import VectorStoreError

from langchain.docstore import InMemoryDocstore
# from langchain.docstore.document import Document
from uuid import uuid4

from langchain_core.documents import Document
import os
import numpy as np

def store_embeddings_in_faiss(embeddings,docs):
    try:
        app_logger.info("Storing embeddings in FAISS vector store")

        # Check that embeddings are not empty and have consistent dimensions
        if isinstance(embeddings, list) and len(embeddings) > 0:
            embedding_dim = len(embeddings[0])
        else:
            app_logger.error("Embeddings list is empty or not correctly formatted.")
            raise ValueError("Embeddings list is empty or not correctly formatted.")

        # Create a new FAISS index
        f_index = faiss.IndexFlatL2(embedding_dim)


        # Ensure the FAISS index is correctly created
        if f_index.is_trained:
            app_logger.info(f"FAISS index initialized with dimensionality: {embedding_dim}")
        else:
            app_logger.error("FAISS index is not trained.")
            raise ValueError("FAISS index is not properly initialized.")
        


        # Convert embeddings to a NumPy array and add to FAISS index
        embeddings_array = np.array(embeddings)
        f_index.add(embeddings_array)

        # # Create index-to-docstore mapping and docstore
        index_to_docstore_id = {i: str(i) for i in range(len(docs))}
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

        # Create the FAISS vector store

        print(len(docs))


        uuids = [str(uuid4()) for _ in range(len(docs))]

        vector_store = FAISS(index = f_index,embedding_function= embeddings,docstore = docstore, index_to_docstore_id=index_to_docstore_id)

        os.makedirs("db",exist_ok=True)

        faiss.write_index(f_index,"db/vector_db.faiss")

        app_logger.info("Stored embeddings in FAISS vector store successfully")

        if vector_store is None  or vector_store.index is None:
            app_logger.error("Vector store initialization failed, returning None")

        return vector_store




    except Exception as e:
        app_logger.info(f"Error in vector store operations : {str(e)}")
        raise VectorStoreError("Failed to store embedding to FAISS") from e
    


def update_faiss_with_new_embeddings(vector_store,new_embeddings,new_docs):
    try:
        app_logger.info("Storing embeddings in FAISS vector store")



        if not new_embeddings or not new_docs:
            raise VectorStoreError("new Embeddings or new documents are missing for store")
        

        vector_store.add_embeddings(new_embeddings,new_docs)

        faiss.write_index(vector_store.index,"db/vector_db.faiss")


        app_logger.info("Stored udated embeddings in FAISS vector store successfully")

        return vector_store




    except Exception as e:
        app_logger.info(f"Error in vector store operations : {str(e)}")
        raise VectorStoreError("Failed to update embedding to FAISS") from e
