"""Local Transformers model implementation."""
from typing import Any, Dict

from app.base import BaseModel, ModelBackend

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TransformersModel(BaseModel):
    """Local Transformers model wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers and PyTorch are not installed. "
                "Install with: pip install transformers torch"
            )
        
        self.config = config
        self.model_name = config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2")
        
        device = config.get("device", "auto")
        torch_dtype_str = config.get("torch_dtype", "auto")
        
        # Convert string to torch dtype
        if torch_dtype_str == "auto":
            torch_dtype = "auto"
        elif torch_dtype_str == "float16":
            torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype=torch_dtype
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=config.get("max_tokens", 1000),
            temperature=config.get("temperature", 0.7)
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using local Transformers."""
        result = self.pipeline(prompt, **kwargs)
        return result[0]["generated_text"]
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name
    
    def get_backend(self) -> ModelBackend:
        """Get backend type."""
        return ModelBackend.LOCAL_TRANSFORMERS
