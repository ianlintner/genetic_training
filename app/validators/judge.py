"""
LLM-based judge for validation.
"""
from typing import Dict, Any, Optional
import json
from loguru import logger

from app.base import BaseModel
from app.models import get_registry


class LLMJudge:
    """Uses an LLM to judge response quality."""
    
    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of AI-generated text.

Evaluate the following response based on these criteria:
1. Correctness: Is the information accurate and appropriate?
2. Coherence: Is the text well-structured and logical?
3. Hallucination: Does it contain false or fabricated information?
4. Completeness: Does it fully address the prompt?

Prompt: {prompt}

Response: {response}

Provide your evaluation as a JSON object with scores from 0.0 to 1.0 for each criterion:
{{
  "correctness": 0.0-1.0,
  "coherence": 0.0-1.0,
  "hallucination_free": 0.0-1.0,
  "completeness": 0.0-1.0,
  "overall": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Only respond with the JSON object, no other text."""
    
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.3):
        """
        Initialize LLM judge.
        
        Args:
            model_name: Name of model to use for judging
            temperature: Temperature for generation
        """
        self.model_name = model_name
        self.temperature = temperature
        self._model: Optional[BaseModel] = None
    
    def _get_model(self) -> BaseModel:
        """Get or initialize the judge model."""
        if self._model is None:
            registry = get_registry()
            self._model = registry.get_model(self.model_name)
        return self._model
    
    def judge(self, text: str, prompt: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Judge the quality of generated text.
        
        Args:
            text: Generated text to judge
            prompt: Original prompt (optional)
            context: Additional context (optional)
            
        Returns:
            Dictionary with scores and reasoning
        """
        try:
            model = self._get_model()
            
            judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
                prompt=prompt or "N/A",
                response=text
            )
            
            response = model.generate(judge_prompt, temperature=self.temperature)
            
            # Try to parse JSON from response
            # Look for JSON object in the response
            json_match = response
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                json_match = response[start:end]
            
            result = json.loads(json_match)
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse judge response: {e}")
            # Return default scores
            return {
                "correctness": 0.5,
                "coherence": 0.5,
                "hallucination_free": 0.5,
                "completeness": 0.5,
                "overall": 0.5,
                "reasoning": "Failed to parse judge response",
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {
                "correctness": 0.5,
                "coherence": 0.5,
                "hallucination_free": 0.5,
                "completeness": 0.5,
                "overall": 0.5,
                "reasoning": "Judge evaluation error",
                "error": str(e)
            }
