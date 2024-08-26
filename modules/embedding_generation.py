from langchain_openai.embeddings import OpenAIEmbeddings
from configs.config import OPENAI_API_KEY
from modules.logger import app_logger
from modules.custom_exceptions import EmbeddingGenerationError


def generate_embeddings(docs):
    try:

        app_logger.info("Starting embeddings generation")

        embedding_model = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
        embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])



    # Ensure all embeddings have the same shape
        embedding_shape = set([len(e) for e in embeddings])
        if len(embedding_shape) > 1:
            app_logger.error("Inconsistent embedding dimensions found.")
            raise ValueError("Embeddings have inconsistent dimensions.")


        app_logger.info(f"Generated embeddings for {len(docs)} documents")

        return embeddings
    
    except Exception as e:

        app_logger.error(f"Error in embedding generation : {str(e)}")

        raise EmbeddingGenerationError("Failed to generate embeddings") from e
