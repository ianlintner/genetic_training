"""
Main validation engine combining heuristics and LLM judge.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from app.base import BaseValidator, BaseHeuristic, ValidationResult
from app.config import get_config
from app.validators.heuristics import (
    TokenLengthHeuristic,
    RepetitionHeuristic,
    StructureHeuristic,
    HarmfulContentHeuristic,
    ContradictionHeuristic,
    HallucinationHeuristic
)
from app.validators.judge import LLMJudge
from app.validators.adaptive import AdaptiveLearner


class HybridValidator(BaseValidator):
    """Hybrid validator combining heuristics and LLM judge."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validator with configuration."""
        if config is None:
            app_config = get_config()
            config = app_config.validation.scoring
        
        self.config = config
        self.weights = config.weights
        
        # Initialize heuristics
        self.heuristics = self._init_heuristics(config.heuristics)
        
        # Initialize judge
        judge_config = config.judge
        self.judge = LLMJudge(
            model_name=judge_config.get("model"),
            temperature=judge_config.get("temperature", 0.3)
        )
        
        # Initialize adaptive learner
        app_config = get_config()
        adaptive_config = app_config.validation.adaptive_learning
        if adaptive_config.get("enabled", True):
            self.adaptive_learner = AdaptiveLearner(adaptive_config)
        else:
            self.adaptive_learner = None
    
    def _init_heuristics(self, config: Dict[str, Any]) -> List[BaseHeuristic]:
        """Initialize heuristic validators from config."""
        heuristics = []
        
        heuristic_classes = {
            "token_length": TokenLengthHeuristic,
            "repetition": RepetitionHeuristic,
            "structure": StructureHeuristic,
            "harmful_content": HarmfulContentHeuristic,
            "contradiction": ContradictionHeuristic,
            "hallucination": HallucinationHeuristic
        }
        
        for name, heuristic_class in heuristic_classes.items():
            if name in config:
                h_config = config[name]
                if isinstance(h_config, dict) and h_config.get("enabled", True):
                    try:
                        heuristic = heuristic_class(**h_config)
                        heuristics.append(heuristic)
                        logger.debug(f"Initialized heuristic: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {name}: {e}")
        
        return heuristics
    
    def validate(
        self,
        text: str,
        prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate text using hybrid approach.
        
        Args:
            text: Text to validate
            prompt: Original prompt
            context: Additional context
            
        Returns:
            ValidationResult with combined score
        """
        heuristic_scores = {}
        heuristic_details = {}
        
        # Run all heuristics
        for heuristic in self.heuristics:
            if heuristic.enabled:
                try:
                    score = heuristic.validate(text, context)
                    name = heuristic.get_name()
                    heuristic_scores[name] = score
                    heuristic_details[name] = heuristic.get_details()
                except Exception as e:
                    logger.error(f"Heuristic {heuristic.get_name()} failed: {e}")
        
        # Calculate weighted heuristic score
        total_weight = sum(h.weight for h in self.heuristics if h.enabled)
        if total_weight > 0:
            heuristic_score = sum(
                heuristic_scores.get(h.get_name(), 0.5) * h.weight
                for h in self.heuristics if h.enabled
            ) / total_weight
        else:
            heuristic_score = 0.5
        
        # Run LLM judge
        judge_result = None
        judge_score = None
        try:
            judge_result = self.judge.judge(text, prompt, context)
            judge_score = judge_result.get("overall", 0.5)
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            judge_score = 0.5
            judge_result = {"error": str(e)}
        
        # Combine scores
        final_score = (
            heuristic_score * self.weights["heuristics"] +
            judge_score * self.weights["judge"]
        )
        
        # Determine if validation passed
        threshold = context.get("threshold", 0.7) if context else 0.7
        passed = final_score >= threshold
        
        # Create result
        result = ValidationResult(
            score=final_score,
            heuristic_scores=heuristic_scores,
            judge_score=judge_score,
            details={
                "heuristic_details": heuristic_details,
                "judge_result": judge_result,
                "weights": self.weights,
                "threshold": threshold
            },
            passed=passed,
            timestamp=datetime.now()
        )
        
        # Update adaptive learner
        if self.adaptive_learner:
            try:
                self.adaptive_learner.update(result, text, prompt)
            except Exception as e:
                logger.warning(f"Adaptive learning update failed: {e}")
        
        return result
