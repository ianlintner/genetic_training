"""llama.cpp model implementation."""
from typing import Any, Dict
from llama_cpp import Llama

from app.base import BaseModel, ModelBackend


class LlamaCppModel(BaseModel):
    """llama.cpp model wrapper for GGUF models."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.model_path = model_path
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=config.get("n_ctx", 2048),
            n_gpu_layers=config.get("n_gpu_layers", 0),
            verbose=False
        )
        
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using llama.cpp."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=kwargs.get("stop", [])
        )
        
        return output["choices"][0]["text"]
    
    def get_model_name(self) -> str:
        """Get model name."""
        return f"llama_cpp:{self.model_path}"
    
    def get_backend(self) -> ModelBackend:
        """Get backend type."""
        return ModelBackend.LLAMA_CPP
