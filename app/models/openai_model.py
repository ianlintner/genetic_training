"""OpenAI model implementation."""
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI

from app.base import BaseModel, ModelBackend
from app.config import get_config


class OpenAIModel(BaseModel):
    """OpenAI model wrapper using LangChain."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "gpt-4o-mini")
        
        app_config = get_config()
        api_key = app_config.openai_api_key
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000),
            api_key=api_key
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        response = self.llm.invoke(prompt)
        return response.content
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name
    
    def get_backend(self) -> ModelBackend:
        """Get backend type."""
        return ModelBackend.OPENAI
