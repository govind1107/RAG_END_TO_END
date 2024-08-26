# modules/custom_exceptions.py

class DataProcessingError(Exception):
    """Raised when an error occurs during data preprocessing."""
    pass

class EmbeddingGenerationError(Exception):
    """Raised when an error occurs during embedding generation."""
    pass

class VectorStoreError(Exception):
    """Raised when an error occurs during vector store operations."""
    pass

class RetrievalError(Exception):
    """Raised when an error occurs during document retrieval."""
    pass

class ResponseGenerationError(Exception):
    """Raised when an error occurs during response generation."""
    pass
