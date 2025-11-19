"""Google Gemini model implementation."""
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI

from app.base import BaseModel, ModelBackend
from app.config import get_config


class GeminiModel(BaseModel):
    """Google Gemini model wrapper using LangChain."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "gemini-pro")
        
        app_config = get_config()
        api_key = app_config.google_api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000),
            google_api_key=api_key
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Google Gemini."""
        response = self.llm.invoke(prompt)
        return response.content
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name
    
    def get_backend(self) -> ModelBackend:
        """Get backend type."""
        return ModelBackend.GEMINI
