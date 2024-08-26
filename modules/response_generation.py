from langchain.llms import OpenAI
from configs.config import OPENAI_API_KEY
from modules.logger import app_logger
from modules.custom_exceptions import ResponseGenerationError


def generate_response(context,question):
    try:
        app_logger.info("Generating resonse based on context and question")
        if not context or not question:
            raise ResponseGenerationError("Context or question is missing for response generation")
        
        prompt = (
            "USe the following context to answer the question . \n\n"
            f"Context : \n\n{context}"
            f"Question: \n\n {question}"
            "Answer : "
        )

        llm = OpenAI(temperature=0.7,openai_api_key = OPENAI_API_KEY)

        response = llm(prompt)

        app_logger.info("Generated resonse based on context and question")

        return response





    except Exception as e:
        app_logger.error(f"Error in response generation : {str(e)}")
        raise ResponseGenerationError("Failed to generate response") from e