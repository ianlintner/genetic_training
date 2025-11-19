"""Anthropic model implementation."""
from typing import Any, Dict
from langchain_anthropic import ChatAnthropic

from app.base import BaseModel, ModelBackend
from app.config import get_config


class AnthropicModel(BaseModel):
    """Anthropic Claude model wrapper using LangChain."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "claude-3-haiku-20240307")
        
        app_config = get_config()
        api_key = app_config.anthropic_api_key
        
        self.llm = ChatAnthropic(
            model=self.model_name,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000),
            api_key=api_key
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic."""
        response = self.llm.invoke(prompt)
        return response.content
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name
    
    def get_backend(self) -> ModelBackend:
        """Get backend type."""
        return ModelBackend.ANTHROPIC
