"""CTransformers model implementation."""
from typing import Any, Dict
from ctransformers import AutoModelForCausalLM

from app.base import BaseModel, ModelBackend


class CTransformersModel(BaseModel):
    """CTransformers model wrapper for GGML models."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.model_path = model_path
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=config.get("model_type", "mistral"),
            context_length=config.get("context_length", 2048),
            gpu_layers=config.get("gpu_layers", 0)
        )
        
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using CTransformers."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        output = self.llm(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        return output
    
    def get_model_name(self) -> str:
        """Get model name."""
        return f"ctransformers:{self.model_path}"
    
    def get_backend(self) -> ModelBackend:
        """Get backend type."""
        return ModelBackend.CTRANSFORMERS
