"""Groq model implementation."""
from typing import Any, Dict
from langchain_groq import ChatGroq

from app.base import BaseModel, ModelBackend
from app.config import get_config


class GroqModel(BaseModel):
    """Groq model wrapper using LangChain."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "mixtral-8x7b-32768")
        
        app_config = get_config()
        api_key = app_config.groq_api_key
        
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000),
            api_key=api_key
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Groq."""
        response = self.llm.invoke(prompt)
        return response.content
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name
    
    def get_backend(self) -> ModelBackend:
        """Get backend type."""
        return ModelBackend.GROQ
