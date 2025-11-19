"""
Model registry for loading and managing LLM models from multiple backends.
"""
from typing import Dict, Optional, Any
from loguru import logger

from app.base import BaseModel, ModelBackend
from app.config import get_config


class ModelRegistry:
    """Central registry for managing multiple LLM models."""
    
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self._config = get_config()
    
    def register_model(self, name: str, model: BaseModel):
        """Register a model instance."""
        self._models[name] = model
        logger.info(f"Registered model: {name} ({model.get_backend()})")
    
    def get_model(self, name: Optional[str] = None) -> BaseModel:
        """
        Get a model by name or the default selected model.
        
        Args:
            name: Model name (optional, uses selected_model if None)
            
        Returns:
            BaseModel instance
        """
        if name is None:
            name = self._config.models.get("selected_model", "gpt-4o-mini")
        
        if name not in self._models:
            # Try to load the model
            self._load_model(name)
        
        return self._models.get(name)
    
    def _load_model(self, name: str):
        """Load a model based on configuration."""
        registry = self._config.models.get("registry", {})
        
        # Try to find model in registry
        for backend, backend_config in registry.items():
            if isinstance(backend_config, dict) and backend_config.get("model_name") == name:
                self._load_model_by_backend(backend, backend_config)
                return
        
        # Check if it's a backend name
        if name in registry:
            self._load_model_by_backend(name, registry[name])
            return
        
        logger.warning(f"Model {name} not found in registry")
    
    def _load_model_by_backend(self, backend: str, config: Dict[str, Any]):
        """Load model from a specific backend."""
        try:
            if backend == "openai":
                from app.models.openai_model import OpenAIModel
                model = OpenAIModel(config)
                self.register_model(config["model_name"], model)
            
            elif backend == "anthropic":
                from app.models.anthropic_model import AnthropicModel
                model = AnthropicModel(config)
                self.register_model(config["model_name"], model)
            
            elif backend == "groq":
                from app.models.groq_model import GroqModel
                model = GroqModel(config)
                self.register_model(config["model_name"], model)
            
            elif backend == "gemini":
                from app.models.gemini_model import GeminiModel
                model = GeminiModel(config)
                self.register_model(config["model_name"], model)
            
            elif backend == "local_transformers":
                from app.models.transformers_model import TransformersModel
                model = TransformersModel(config)
                self.register_model(config["model_name"], model)
            
            elif backend == "llama_cpp":
                from app.models.llama_cpp_model import LlamaCppModel
                model_path = self._config.models.get("local_model_path")
                model = LlamaCppModel(model_path, config)
                self.register_model("llama_cpp", model)
            
            elif backend == "ctransformers":
                from app.models.ctransformers_model import CTransformersModel
                model_path = self._config.models.get("local_model_path")
                model = CTransformersModel(model_path, config)
                self.register_model("ctransformers", model)
            
        except Exception as e:
            logger.error(f"Failed to load {backend} model: {e}")
            raise
    
    def list_models(self) -> list:
        """List all registered models."""
        return list(self._models.keys())


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
