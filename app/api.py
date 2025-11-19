"""
FastAPI application with validation, training, and evolution endpoints.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from loguru import logger
import sys
from pathlib import Path

from app.config import get_config
from app.models import get_registry
from app.validators import HybridValidator
from app.pipelines import (
    GenerateValidatePipeline,
    ValidateOnlyPipeline,
    ReviseLoopPipeline
)
from app.training import TrainingPipeline
from app.evolution import GeneticOptimizer
from app.storage import create_store

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("logs/app.log", rotation="500 MB", level="DEBUG")

# Initialize app
app = FastAPI(
    title="Multi-LLM Validator API",
    description="Production-grade LLM validation system with multiple backends and genetic optimization",
    version="1.0.0"
)

# Load config
config = get_config()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
validator = HybridValidator()
store = create_store(config.storage.backend)

# Pydantic models for API
class GenerateRequest(BaseModel):
    """Request to generate and validate text."""
    prompt: str = Field(..., description="Input prompt for generation")
    model: Optional[str] = Field(None, description="Model to use (optional)")
    temperature: Optional[float] = Field(0.7, description="Generation temperature")
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens to generate")

class ValidateRequest(BaseModel):
    """Request to validate text."""
    text: str = Field(..., description="Text to validate")
    prompt: Optional[str] = Field(None, description="Original prompt (optional)")
    threshold: Optional[float] = Field(0.7, description="Quality threshold")

class ReviseRequest(BaseModel):
    """Request to generate with revision loop."""
    prompt: str = Field(..., description="Input prompt")
    model: Optional[str] = Field(None, description="Model to use")
    max_revisions: Optional[int] = Field(3, description="Maximum revision attempts")
    quality_threshold: Optional[float] = Field(0.8, description="Quality threshold")

class TrainingRequest(BaseModel):
    """Request to start training."""
    datasets: Optional[List[str]] = Field(None, description="Datasets to use")
    epochs: Optional[int] = Field(10, description="Training epochs")
    background: Optional[bool] = Field(True, description="Run in background")

class EvolutionRequest(BaseModel):
    """Request to start evolution."""
    population_size: Optional[int] = Field(20, description="Population size")
    generations: Optional[int] = Field(50, description="Number of generations")
    background: Optional[bool] = Field(True, description="Run in background")


# Health check
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Multi-LLM Validator",
        "version": "1.0.0"
    }


# Validation endpoints
@app.post("/validate/single", tags=["Validation"])
async def validate_single(request: ValidateRequest):
    """
    Validate a single text without generation.
    
    Args:
        request: Validation request
        
    Returns:
        Validation result
    """
    try:
        logger.info(f"Validating text (length: {len(request.text)})")
        
        pipeline = ValidateOnlyPipeline(validator)
        result = pipeline.run(
            prompt=request.prompt or "",
            text=request.text,
            threshold=request.threshold
        )
        
        # Store result
        store.save({
            "type": "validation",
            "text": result.text,
            "prompt": result.prompt,
            "score": result.validation.score,
            "passed": result.validation.passed,
            "details": result.validation.details
        })
        
        return {
            "score": result.validation.score,
            "passed": result.validation.passed,
            "heuristic_scores": result.validation.heuristic_scores,
            "judge_score": result.validation.judge_score,
            "details": result.validation.details,
            "timestamp": result.validation.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/generate", tags=["Validation"])
async def validate_generate(request: GenerateRequest):
    """
    Generate text and validate it.
    
    Args:
        request: Generation request
        
    Returns:
        Generation and validation result
    """
    try:
        logger.info(f"Generating and validating: {request.prompt[:50]}...")
        
        pipeline = GenerateValidatePipeline(request.model, validator)
        result = pipeline.run(
            request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Store result
        store.save({
            "type": "generation",
            "model": result.model,
            "prompt": result.prompt,
            "text": result.text,
            "score": result.validation.score,
            "passed": result.validation.passed,
            "details": result.validation.details
        })
        
        return {
            "text": result.text,
            "model": result.model,
            "validation": {
                "score": result.validation.score,
                "passed": result.validation.passed,
                "heuristic_scores": result.validation.heuristic_scores,
                "judge_score": result.validation.judge_score,
                "details": result.validation.details
            }
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/revise", tags=["Validation"])
async def validate_revise(request: ReviseRequest):
    """
    Generate text with iterative revision until quality threshold is met.
    
    Args:
        request: Revision request
        
    Returns:
        Final generation and validation result
    """
    try:
        logger.info(f"Starting revision loop: {request.prompt[:50]}...")
        
        pipeline = ReviseLoopPipeline(
            model_name=request.model,
            validator=validator,
            max_revisions=request.max_revisions,
            quality_threshold=request.quality_threshold
        )
        
        result = pipeline.run(request.prompt)
        
        # Store result
        store.save({
            "type": "revision",
            "model": result.model,
            "prompt": result.prompt,
            "text": result.text,
            "score": result.validation.score,
            "passed": result.validation.passed,
            "metadata": result.metadata
        })
        
        return {
            "text": result.text,
            "model": result.model,
            "validation": {
                "score": result.validation.score,
                "passed": result.validation.passed,
                "heuristic_scores": result.validation.heuristic_scores,
                "judge_score": result.validation.judge_score
            },
            "revision_history": result.metadata.get("revision_history", [])
        }
        
    except Exception as e:
        logger.error(f"Revision loop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoints
@app.post("/train/run", tags=["Training"])
async def train_run(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Run training pipeline to optimize validator weights.
    
    Args:
        request: Training request
        background_tasks: FastAPI background tasks
        
    Returns:
        Training job info or results
    """
    try:
        pipeline = TrainingPipeline()
        
        if request.background:
            # Run in background
            job_id = f"train_{hash(str(request.dict()))}"
            background_tasks.add_task(pipeline.run_training)
            
            return {
                "status": "started",
                "job_id": job_id,
                "message": "Training started in background"
            }
        else:
            # Run synchronously
            logger.info("Starting training pipeline...")
            result = pipeline.run_training()
            
            return {
                "status": "completed",
                "best_weights": result["best_weights"],
                "best_score": result["best_score"],
                "train_metrics": result["train_metrics"],
                "val_metrics": result["val_metrics"]
            }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Evolution endpoints
@app.post("/evolve/run", tags=["Evolution"])
async def evolve_run(request: EvolutionRequest, background_tasks: BackgroundTasks):
    """
    Run genetic algorithm evolution to optimize validator configuration.
    
    Args:
        request: Evolution request
        background_tasks: FastAPI background tasks
        
    Returns:
        Evolution job info or results
    """
    try:
        optimizer = GeneticOptimizer()
        
        if request.population_size:
            optimizer.population_size = request.population_size
        if request.generations:
            optimizer.generations = request.generations
        
        if request.background:
            # Run in background
            job_id = f"evolve_{hash(str(request.dict()))}"
            background_tasks.add_task(optimizer.evolve)
            
            return {
                "status": "started",
                "job_id": job_id,
                "message": "Evolution started in background",
                "population_size": optimizer.population_size,
                "generations": optimizer.generations
            }
        else:
            # Run synchronously
            logger.info("Starting genetic evolution...")
            result = optimizer.evolve()
            
            return {
                "status": "completed",
                "best_configuration": result["best_configuration"],
                "best_fitness": result["best_fitness"],
                "summary": {
                    "generations": optimizer.generations,
                    "population_size": optimizer.population_size
                }
            }
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management
@app.get("/models/list", tags=["Models"])
async def list_models():
    """List available models."""
    registry = get_registry()
    return {
        "models": registry.list_models(),
        "selected": config.models.get("selected_model")
    }


# Configuration
@app.get("/config", tags=["Configuration"])
async def get_config_endpoint():
    """Get current configuration."""
    return {
        "validation": {
            "weights": validator.weights,
            "adaptive_learning": config.validation.adaptive_learning
        },
        "storage": {
            "backend": config.storage.backend
        },
        "models": {
            "selected": config.models.get("selected_model")
        }
    }


# Statistics
@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """Get validation statistics."""
    try:
        records = store.query(limit=1000)
        
        if not records:
            return {"message": "No validation data available"}
        
        total = len(records)
        passed = sum(1 for r in records if r.get("passed", False))
        scores = [r.get("score", 0) for r in records if "score" in r]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "average_score": avg_score
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.api.log_level
    )
