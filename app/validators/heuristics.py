"""
Heuristic validators for text quality assessment.
"""
from typing import Any, Dict, Optional
import re
from collections import Counter

from app.base import BaseHeuristic


class TokenLengthHeuristic(BaseHeuristic):
    """Validate token length is within acceptable range."""
    
    def __init__(self, weight: float = 0.15, min_tokens: int = 10, max_tokens: int = 2000, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self._details = {}
    
    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Validate token length."""
        # Simple whitespace tokenization
        tokens = text.split()
        token_count = len(tokens)
        self._details = {"token_count": token_count}
        
        if token_count < self.min_tokens:
            return max(0.0, token_count / self.min_tokens)
        elif token_count > self.max_tokens:
            return max(0.0, 1.0 - (token_count - self.max_tokens) / self.max_tokens)
        else:
            return 1.0
    
    def get_name(self) -> str:
        return "token_length"
    
    def get_details(self) -> Dict[str, Any]:
        return self._details


class RepetitionHeuristic(BaseHeuristic):
    """Detect excessive repetition in text."""
    
    def __init__(self, weight: float = 0.15, max_ngram_repeat: int = 5, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.max_ngram_repeat = max_ngram_repeat
        self._details = {}
    
    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Check for repetitive patterns."""
        words = text.lower().split()
        
        if len(words) < 3:
            return 1.0
        
        # Check 3-grams
        ngrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        ngram_counts = Counter(ngrams)
        
        max_repeats = max(ngram_counts.values()) if ngram_counts else 0
        self._details = {"max_repeats": max_repeats}
        
        if max_repeats <= 1:
            return 1.0
        elif max_repeats >= self.max_ngram_repeat:
            return 0.0
        else:
            return 1.0 - (max_repeats - 1) / (self.max_ngram_repeat - 1)
    
    def get_name(self) -> str:
        return "repetition"
    
    def get_details(self) -> Dict[str, Any]:
        return self._details


class StructureHeuristic(BaseHeuristic):
    """Validate text structure and formatting."""
    
    def __init__(self, weight: float = 0.10, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self._details = {}
    
    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Check text structure."""
        score = 1.0
        issues = []
        
        # Check for complete sentences
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = [s for s in sentences if s.strip()]
        
        if len(complete_sentences) == 0:
            score *= 0.5
            issues.append("no_complete_sentences")
        
        # Check for balanced parentheses/brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for char in text:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack[-1]] != char:
                    score *= 0.8
                    issues.append("unbalanced_brackets")
                    break
                stack.pop()
        
        if stack:
            score *= 0.8
            issues.append("unclosed_brackets")
        
        # Check for reasonable capitalization
        if text and text[0].islower():
            score *= 0.9
            issues.append("no_initial_capital")
        
        self._details = {"issues": issues, "sentence_count": len(complete_sentences)}
        return max(0.0, score)
    
    def get_name(self) -> str:
        return "structure"
    
    def get_details(self) -> Dict[str, Any]:
        return self._details


class HarmfulContentHeuristic(BaseHeuristic):
    """Detect potentially harmful content."""
    
    def __init__(self, weight: float = 0.25, threshold: float = 0.5, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.threshold = threshold
        self._details = {}
        
        # Initialize detoxify model lazily
        self._model = None
    
    def _get_model(self):
        """Lazy load detoxify model."""
        if self._model is None:
            try:
                from detoxify import Detoxify
                self._model = Detoxify('original')
            except Exception as e:
                # Fallback to simple keyword matching
                self._model = "fallback"
        return self._model
    
    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Check for harmful content."""
        model = self._get_model()
        
        if model == "fallback":
            # Simple keyword-based fallback
            harmful_keywords = ['hate', 'violence', 'explicit', 'illegal']
            text_lower = text.lower()
            found = [kw for kw in harmful_keywords if kw in text_lower]
            score = 1.0 - (len(found) * 0.2)
            self._details = {"method": "keyword", "found": found}
        else:
            # Use detoxify model
            results = model.predict(text)
            max_toxicity = max(results.values())
            score = 1.0 - max_toxicity
            self._details = {"method": "detoxify", "scores": results}
        
        return max(0.0, score)
    
    def get_name(self) -> str:
        return "harmful_content"
    
    def get_details(self) -> Dict[str, Any]:
        return self._details


class ContradictionHeuristic(BaseHeuristic):
    """Detect internal contradictions in text."""
    
    def __init__(self, weight: float = 0.20, threshold: float = 0.7, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.threshold = threshold
        self._details = {}
    
    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Check for contradictions."""
        # Simple heuristic: look for negation patterns and opposing statements
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # Look for obvious contradictions
        contradiction_patterns = [
            (r'\bnot\b', r'\bis\b'),
            (r'\bno\b', r'\byes\b'),
            (r'\btrue\b', r'\bfalse\b'),
            (r'\bcorrect\b', r'\bincorrect\b'),
        ]
        
        contradictions = 0
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i+1:]:
                for pattern1, pattern2 in contradiction_patterns:
                    if re.search(pattern1, sent1, re.I) and re.search(pattern2, sent2, re.I):
                        contradictions += 1
        
        self._details = {"contradictions": contradictions, "sentence_count": len(sentences)}
        
        if contradictions == 0:
            return 1.0
        else:
            return max(0.0, 1.0 - (contradictions * 0.3))
    
    def get_name(self) -> str:
        return "contradiction"
    
    def get_details(self) -> Dict[str, Any]:
        return self._details


class HallucinationHeuristic(BaseHeuristic):
    """Detect potential hallucinations."""
    
    def __init__(self, weight: float = 0.15, threshold: float = 0.6, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.threshold = threshold
        self._details = {}
    
    def validate(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Check for hallucination patterns."""
        score = 1.0
        issues = []
        
        # Check for hedging language (often indicates uncertainty/potential hallucination)
        hedging_patterns = [
            r'\bmaybe\b', r'\bperhaps\b', r'\bmight\b', r'\bcould\b',
            r'\bpossibly\b', r'\bprobably\b', r'\bseemingly\b'
        ]
        
        hedging_count = sum(1 for pattern in hedging_patterns if re.search(pattern, text, re.I))
        
        # Check for overly specific but unverifiable claims
        specific_patterns = [
            r'\d+\.\d+%',  # Exact percentages
            r'\bon [A-Z][a-z]+ \d+, \d{4}\b',  # Specific dates
        ]
        
        specific_count = sum(1 for pattern in specific_patterns if re.search(pattern, text))
        
        # Penalize excessive hedging or specificity
        if hedging_count > 3:
            score *= 0.8
            issues.append(f"excessive_hedging:{hedging_count}")
        
        if specific_count > 5:
            score *= 0.9
            issues.append(f"excessive_specificity:{specific_count}")
        
        self._details = {
            "hedging_count": hedging_count,
            "specific_claims": specific_count,
            "issues": issues
        }
        
        return max(0.0, score)
    
    def get_name(self) -> str:
        return "hallucination"
    
    def get_details(self) -> Dict[str, Any]:
        return self._details
