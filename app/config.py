import os
import logging
from tabnanny import verbose

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IntelliScout")


class Config:
    # Load Groq API key from .env
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    # Default model configuration
    MODEL_ID = "groq/gemma2-9b-it"
    TEMPERATURE = 0.1
    
    # Configuration for ScrapeGraphAI
    GRAPH_CONFIG = {
        "llm": {
            "model": MODEL_ID,
            "api_key": GROQ_API_KEY,
            "temperature": TEMPERATURE
        },
        "headless": False,
        "verbose": True,
    }

    @classmethod
    def update_model(cls, model_id, temperature=None):
        """Update the model configuration.
        
        Args:
            model_id (str): The ID of the model to use
            temperature (float, optional): The temperature value to use
        """
        logger.info(f"Updating model configuration to: {model_id}")
        cls.MODEL_ID = f"groq/{model_id}" if not model_id.startswith("groq/") else model_id
        if temperature is not None:
            cls.TEMPERATURE = temperature
        cls.GRAPH_CONFIG["llm"]["model"] = cls.MODEL_ID
        cls.GRAPH_CONFIG["llm"]["temperature"] = cls.TEMPERATURE
        logger.info(f"Model updated successfully to: {cls.MODEL_ID} with temperature: {cls.TEMPERATURE}")