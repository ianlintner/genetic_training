"""
Adaptive learning system for validator optimization.
"""
from typing import Dict, Any, List, Optional
from collections import deque
import json
from pathlib import Path
from loguru import logger

from app.base import ValidationResult


class AdaptiveLearner:
    """Learns from validation history to optimize scoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adaptive learner.
        
        Args:
            config: Adaptive learning configuration
        """
        self.config = config
        self.history_size = config.get("history_size", 1000)
        self.update_interval = config.get("update_interval", 100)
        self.learning_rate = config.get("learning_rate", 0.01)
        
        self.history: deque = deque(maxlen=self.history_size)
        self.update_count = 0
        
        # Statistics
        self.score_stats = {
            "mean": 0.5,
            "std": 0.2,
            "min": 0.0,
            "max": 1.0
        }
    
    def update(self, result: ValidationResult, text: str, prompt: Optional[str] = None):
        """
        Update learner with new validation result.
        
        Args:
            result: Validation result
            text: Validated text
            prompt: Original prompt
        """
        # Add to history
        self.history.append({
            "score": result.score,
            "heuristic_scores": result.heuristic_scores,
            "judge_score": result.judge_score,
            "passed": result.passed,
            "timestamp": result.timestamp.isoformat(),
            "text_length": len(text)
        })
        
        self.update_count += 1
        
        # Periodically update statistics
        if self.update_count % self.update_interval == 0:
            self._update_statistics()
    
    def _update_statistics(self):
        """Update score statistics from history."""
        if not self.history:
            return
        
        scores = [h["score"] for h in self.history]
        
        # Calculate statistics
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        self.score_stats = {
            "mean": mean_score,
            "std": std_dev,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores)
        }
        
        logger.info(f"Updated statistics: mean={mean_score:.3f}, std={std_dev:.3f}")
    
    def get_adjusted_threshold(self, base_threshold: float = 0.7) -> float:
        """
        Get adjusted threshold based on learned distribution.
        
        Args:
            base_threshold: Base threshold value
            
        Returns:
            Adjusted threshold
        """
        if not self.history:
            return base_threshold
        
        # Adjust threshold based on mean score
        mean = self.score_stats["mean"]
        
        # If mean is significantly different from 0.5, adjust threshold
        adjustment = (mean - 0.5) * self.learning_rate
        adjusted = base_threshold + adjustment
        
        # Clamp to reasonable range
        return max(0.3, min(0.9, adjusted))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "score_stats": self.score_stats,
            "history_size": len(self.history),
            "update_count": self.update_count
        }
    
    def save(self, path: str):
        """Save learner state to file."""
        state = {
            "config": self.config,
            "score_stats": self.score_stats,
            "update_count": self.update_count,
            "history": list(self.history)
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved adaptive learner state to {path}")
    
    def load(self, path: str):
        """Load learner state from file."""
        try:
            with open(path, "r") as f:
                state = json.load(f)
            
            self.score_stats = state.get("score_stats", self.score_stats)
            self.update_count = state.get("update_count", 0)
            
            history_data = state.get("history", [])
            self.history = deque(history_data, maxlen=self.history_size)
            
            logger.info(f"Loaded adaptive learner state from {path}")
        except FileNotFoundError:
            logger.warning(f"No saved state found at {path}")
        except Exception as e:
            logger.error(f"Failed to load adaptive learner state: {e}")
