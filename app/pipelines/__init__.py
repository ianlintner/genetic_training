"""
Pipelines for generation, validation, and revision.
"""
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger

from app.base import BasePipeline, GenerationResult
from app.models import get_registry
from app.validators import HybridValidator
from app.config import get_config
from app.prompts import REVISION_PROMPT_TEMPLATE


class GenerateValidatePipeline(BasePipeline):
    """Pipeline that generates text and validates it."""
    
    def __init__(self, model_name: Optional[str] = None, validator: Optional[HybridValidator] = None):
        """
        Initialize pipeline.
        
        Args:
            model_name: Name of model to use for generation
            validator: Validator instance (creates default if None)
        """
        self.model_name = model_name
        self.validator = validator or HybridValidator()
        self.config = get_config().pipelines.get("generate_validate", {})
    
    def run(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate and validate text.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with validation
        """
        try:
            # Get model
            registry = get_registry()
            model = registry.get_model(self.model_name)
            
            # Generate text
            logger.info(f"Generating with model: {model.get_model_name()}")
            text = model.generate(prompt, **kwargs)
            
            # Validate
            logger.info("Validating generated text")
            validation = self.validator.validate(text, prompt=prompt)
            
            return GenerationResult(
                text=text,
                model=model.get_model_name(),
                prompt=prompt,
                validation=validation,
                metadata={
                    "pipeline": "generate_validate",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


class ValidateOnlyPipeline(BasePipeline):
    """Pipeline that only validates existing text."""
    
    def __init__(self, validator: Optional[HybridValidator] = None):
        """
        Initialize pipeline.
        
        Args:
            validator: Validator instance (creates default if None)
        """
        self.validator = validator or HybridValidator()
    
    def run(self, prompt: str, text: str = None, **kwargs) -> GenerationResult:
        """
        Validate existing text.
        
        Args:
            prompt: Original prompt
            text: Text to validate (required)
            **kwargs: Additional context
            
        Returns:
            GenerationResult with validation only
        """
        if text is None:
            raise ValueError("text parameter is required for validate-only pipeline")
        
        try:
            # Validate
            logger.info("Validating text")
            validation = self.validator.validate(text, prompt=prompt, context=kwargs)
            
            return GenerationResult(
                text=text,
                model="external",
                prompt=prompt,
                validation=validation,
                metadata={
                    "pipeline": "validate_only",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise


class ReviseLoopPipeline(BasePipeline):
    """Pipeline that iteratively revises text until quality threshold is met."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        validator: Optional[HybridValidator] = None,
        max_revisions: int = 3,
        quality_threshold: float = 0.8
    ):
        """
        Initialize pipeline.
        
        Args:
            model_name: Name of model to use
            validator: Validator instance
            max_revisions: Maximum number of revision attempts
            quality_threshold: Quality score threshold
        """
        self.model_name = model_name
        self.validator = validator or HybridValidator()
        
        config = get_config().pipelines.get("revise_loop", {})
        self.max_revisions = max_revisions or config.get("max_revisions", 3)
        self.quality_threshold = quality_threshold or config.get("quality_threshold", 0.8)
    
    def run(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate and iteratively revise text.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult with best version
        """
        registry = get_registry()
        model = registry.get_model(self.model_name)
        
        revision_history = []
        best_result = None
        best_score = 0.0
        
        # Initial generation
        logger.info("Initial generation")
        current_text = model.generate(prompt, **kwargs)
        
        for revision in range(self.max_revisions + 1):
            # Validate current text
            validation = self.validator.validate(current_text, prompt=prompt)
            
            result = GenerationResult(
                text=current_text,
                model=model.get_model_name(),
                prompt=prompt,
                validation=validation,
                metadata={
                    "pipeline": "revise_loop",
                    "revision": revision,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            revision_history.append(result)
            
            # Track best result
            if validation.score > best_score:
                best_score = validation.score
                best_result = result
            
            logger.info(f"Revision {revision}: score={validation.score:.3f}")
            
            # Check if threshold met
            if validation.score >= self.quality_threshold:
                logger.info(f"Quality threshold met at revision {revision}")
                break
            
            # Don't revise if this was the last attempt
            if revision >= self.max_revisions:
                break
            
            # Generate revision prompt
            revision_prompt = self._create_revision_prompt(
                prompt,
                current_text,
                validation
            )
            
            # Generate revised text
            logger.info(f"Generating revision {revision + 1}")
            current_text = model.generate(revision_prompt, **kwargs)
        
        # Add revision history to best result
        best_result.metadata["revision_history"] = [
            {
                "revision": r.metadata["revision"],
                "score": r.validation.score,
                "passed": r.validation.passed
            }
            for r in revision_history
        ]
        
        return best_result
    
    def _create_revision_prompt(self, original_prompt: str, text: str, validation) -> str:
        """Create a prompt for revision based on validation results."""
        issues = []
        
        # Collect issues from heuristics
        for name, score in validation.heuristic_scores.items():
            if score < 0.6:
                issues.append(f"{name} (score: {score:.2f})")
        
        # Get judge feedback
        judge_result = validation.details.get("judge_result", {})
        reasoning = judge_result.get("reasoning", "")
        
        revision_prompt = REVISION_PROMPT_TEMPLATE.format(
            original_prompt=original_prompt,
            text=text,
            issues=', '.join(issues) if issues else 'General quality improvement needed',
            reasoning=reasoning
        )
        
        return revision_prompt
