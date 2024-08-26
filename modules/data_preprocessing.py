import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configs.config import CHUNK_SIZE,CHUNK_OVERLAP
from modules.logger import app_logger
from modules.custom_exceptions import DataProcessingError


def load_and_preprocess_pdfs(folder_path):
    documents = []

    try:

        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(folder_path,filename)
                app_logger.info(f"Processing PDF : {filename}")


                loader = PyMuPDFLoader(pdf_path)

                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE ,chunk_overlap = CHUNK_OVERLAP)
        split_docs = text_splitter.split_documents(documents)

        

        app_logger.info(f"Sucessfully processed {len(split_docs)} documents chunks")

        return split_docs



    except Exception as e:
        app_logger.error(f"Error in data processing: {str(e)}")
        raise DataProcessingError(f"Failed to preprocess PDFs in folder {folder_path}")  from e


