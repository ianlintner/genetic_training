#!/usr/bin/env python3
"""
Main entry point for running the LLM validator application.
"""
import sys
import argparse
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_config, reload_config
from loguru import logger


def run_api():
    """Run the FastAPI server."""
    import uvicorn
    from app.api import app
    
    config = get_config()
    
    logger.info(f"Starting API server on {config.api.host}:{config.api.port}")
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.api.log_level
    )


def run_training():
    """Run training pipeline."""
    from app.training import TrainingPipeline
    
    logger.info("Starting training pipeline...")
    pipeline = TrainingPipeline()
    result = pipeline.run_training()
    
    logger.info("Training complete!")
    logger.info(f"Best weights: {result['best_weights']}")
    logger.info(f"Best score: {result['best_score']:.4f}")


def run_evolution():
    """Run genetic evolution."""
    from app.evolution import GeneticOptimizer
    
    logger.info("Starting genetic evolution...")
    optimizer = GeneticOptimizer()
    result = optimizer.evolve()
    
    logger.info("Evolution complete!")
    logger.info(f"Best fitness: {result['best_fitness']:.4f}")
    logger.info(f"Best configuration: {result['best_configuration']}")


def validate_text(text: str, prompt: str = None):
    """Validate a single text."""
    from app.validators import HybridValidator
    
    validator = HybridValidator()
    result = validator.validate(text, prompt=prompt)
    
    logger.info(f"Validation score: {result.score:.3f}")
    logger.info(f"Passed: {result.passed}")
    logger.info(f"Heuristic scores: {result.heuristic_scores}")
    logger.info(f"Judge score: {result.judge_score}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Multi-LLM Validator CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Run API server")
    api_parser.add_argument("--config", help="Path to config file")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    
    # Evolution command
    evolve_parser = subparsers.add_parser("evolve", help="Run genetic evolution")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate text")
    validate_parser.add_argument("text", help="Text to validate")
    validate_parser.add_argument("--prompt", help="Original prompt")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load config if specified
    if hasattr(args, "config") and args.config:
        reload_config(args.config)
    
    # Execute command
    if args.command == "api":
        run_api()
    elif args.command == "train":
        run_training()
    elif args.command == "evolve":
        run_evolution()
    elif args.command == "validate":
        validate_text(args.text, args.prompt)


if __name__ == "__main__":
    main()
