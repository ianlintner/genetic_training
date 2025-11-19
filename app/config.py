"""
Configuration management for the LLM Validator system.
"""
import os
from typing import Any, Dict, Optional
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    n_ctx: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    context_length: Optional[int] = None
    gpu_layers: Optional[int] = None
    model_type: Optional[str] = None


class HeuristicConfig(BaseModel):
    """Configuration for a heuristic validator."""
    enabled: bool = True
    weight: float = 0.2
    threshold: Optional[float] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    max_ngram_repeat: Optional[int] = None


class ScoringConfig(BaseModel):
    """Scoring configuration."""
    weights: Dict[str, float] = Field(default_factory=lambda: {"heuristics": 0.4, "judge": 0.6})
    heuristics: Dict[str, HeuristicConfig] = Field(default_factory=dict)
    judge: Dict[str, Any] = Field(default_factory=dict)


class ValidationConfig(BaseModel):
    """Validation configuration."""
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    adaptive_learning: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    max_retries: int = 3
    timeout: int = 60
    max_revisions: Optional[int] = None
    quality_threshold: Optional[float] = None


class TrainingConfig(BaseModel):
    """Training configuration."""
    datasets: list = Field(default_factory=list)
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2


class EvolutionConfig(BaseModel):
    """Evolution/genetic algorithm configuration."""
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elitism: float = 0.1
    mutation_ranges: Dict[str, float] = Field(default_factory=dict)
    fitness: Dict[str, Any] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    """Storage backend configuration."""
    backend: str = "json"
    json: Dict[str, str] = Field(default_factory=lambda: {"path": "data/results.json"})
    sqlite: Dict[str, str] = Field(default_factory=lambda: {"path": "data/results.db"})
    chromadb: Dict[str, str] = Field(default_factory=dict)
    postgresql: Dict[str, Any] = Field(default_factory=dict)


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    cors_origins: list = Field(default_factory=list)


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    # Model settings
    models: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation settings
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    
    # Pipeline settings
    pipelines: Dict[str, PipelineConfig] = Field(default_factory=dict)
    
    # Training settings
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    # Evolution settings
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    
    # Storage settings
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # API settings
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Logging settings
    logging: Dict[str, Any] = Field(default_factory=dict)
    
    # API Keys from environment
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file and environment variables.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        AppConfig instance with loaded configuration
    """
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {}
    
    return AppConfig(**config_data)


# Global config instance
config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = load_config()
    return config


def reload_config(config_path: Optional[str] = None) -> AppConfig:
    """Reload configuration from file."""
    global config
    config = load_config(config_path)
    return config
