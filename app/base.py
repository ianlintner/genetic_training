"""
Base interfaces and abstract classes for the LLM Validator system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ModelBackend(str, Enum):
    """Supported model backends."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    GEMINI = "gemini"
    LOCAL_TRANSFORMERS = "local_transformers"
    LLAMA_CPP = "llama_cpp"
    CTRANSFORMERS = "ctransformers"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    score: float
    heuristic_scores: Dict[str, float]
    judge_score: Optional[float]
    details: Dict[str, Any]
    passed: bool
    timestamp: datetime
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class GenerationResult:
    """Result of a generation operation."""
    text: str
    model: str
    prompt: str
    validation: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModel(ABC):
    """Base interface for all LLM models."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model identifier."""
        pass
    
    @abstractmethod
    def get_backend(self) -> ModelBackend:
        """Get the backend type."""
        pass


class BaseHeuristic(ABC):
    """Base interface for heuristic validators."""
    
    def __init__(self, weight: float = 1.0, enabled: bool = True):
        self.weight = weight
        self.enabled = enabled
    
    @abstractmethod
    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Validate text using this heuristic.
        
        Args:
            text: Text to validate
            context: Optional context (e.g., prompt, reference text)
            
        Returns:
            Score between 0.0 (bad) and 1.0 (good)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the heuristic name."""
        pass
    
    def get_details(self) -> Dict[str, Any]:
        """Get additional details about the validation."""
        return {}


class BaseValidator(ABC):
    """Base interface for validators."""
    
    @abstractmethod
    def validate(
        self,
        text: str,
        prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate generated text.
        
        Args:
            text: Generated text to validate
            prompt: Original prompt (optional)
            context: Additional context (optional)
            
        Returns:
            ValidationResult with scores and details
        """
        pass


class BasePipeline(ABC):
    """Base interface for pipelines."""
    
    @abstractmethod
    def run(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Run the pipeline.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult with validation
        """
        pass


class BaseStore(ABC):
    """Base interface for storage backends."""
    
    @abstractmethod
    def save(self, record: Dict[str, Any]) -> str:
        """
        Save a record.
        
        Args:
            record: Record to save
            
        Returns:
            Record ID
        """
        pass
    
    @abstractmethod
    def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query records.
        
        Args:
            filters: Query filters
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            List of records
        """
        pass
    
    @abstractmethod
    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a record by ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            Record or None if not found
        """
        pass
    
    @abstractmethod
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a record.
        
        Args:
            record_id: Record ID
            updates: Fields to update
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """
        Delete a record.
        
        Args:
            record_id: Record ID
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the storage connection."""
        pass
