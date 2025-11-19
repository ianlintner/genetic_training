"""
Unit tests for heuristic validators.
"""
import pytest
from app.validators.heuristics import (
    TokenLengthHeuristic,
    RepetitionHeuristic,
    StructureHeuristic,
    ContradictionHeuristic,
    HallucinationHeuristic
)


class TestTokenLengthHeuristic:
    """Test token length validation."""
    
    def test_valid_length(self):
        """Test text with valid length."""
        heuristic = TokenLengthHeuristic(min_tokens=5, max_tokens=100)
        text = "This is a reasonably sized text that should pass."
        score = heuristic.validate(text)
        assert score == 1.0
    
    def test_too_short(self):
        """Test text that's too short."""
        heuristic = TokenLengthHeuristic(min_tokens=10, max_tokens=100)
        text = "Too short"
        score = heuristic.validate(text)
        assert 0 <= score < 1.0
    
    def test_too_long(self):
        """Test text that's too long."""
        heuristic = TokenLengthHeuristic(min_tokens=5, max_tokens=10)
        text = " ".join(["word"] * 20)
        score = heuristic.validate(text)
        assert 0 <= score < 1.0
    
    def test_details(self):
        """Test that details are provided."""
        heuristic = TokenLengthHeuristic()
        heuristic.validate("Test text here")
        details = heuristic.get_details()
        assert "token_count" in details
        assert details["token_count"] == 3


class TestRepetitionHeuristic:
    """Test repetition detection."""
    
    def test_no_repetition(self):
        """Test text without repetition."""
        heuristic = RepetitionHeuristic()
        text = "This is unique text without any repetition patterns."
        score = heuristic.validate(text)
        assert score == 1.0
    
    def test_with_repetition(self):
        """Test text with repetition."""
        heuristic = RepetitionHeuristic(max_ngram_repeat=2)
        text = "Hello world. Hello world. Hello world. Hello world."
        score = heuristic.validate(text)
        assert score < 1.0
    
    def test_short_text(self):
        """Test very short text."""
        heuristic = RepetitionHeuristic()
        text = "Hi"
        score = heuristic.validate(text)
        assert score == 1.0


class TestStructureHeuristic:
    """Test structure validation."""
    
    def test_good_structure(self):
        """Test well-structured text."""
        heuristic = StructureHeuristic()
        text = "This is a well-structured sentence with proper capitalization."
        score = heuristic.validate(text)
        assert score == 1.0
    
    def test_no_initial_capital(self):
        """Test text without initial capital."""
        heuristic = StructureHeuristic()
        text = "this starts with lowercase."
        score = heuristic.validate(text)
        assert score < 1.0
    
    def test_unbalanced_brackets(self):
        """Test text with unbalanced brackets."""
        heuristic = StructureHeuristic()
        text = "This has (unbalanced brackets."
        score = heuristic.validate(text)
        assert score < 1.0
    
    def test_details(self):
        """Test that details are provided."""
        heuristic = StructureHeuristic()
        heuristic.validate("Test.")
        details = heuristic.get_details()
        assert "issues" in details
        assert "sentence_count" in details


class TestContradictionHeuristic:
    """Test contradiction detection."""
    
    def test_no_contradiction(self):
        """Test text without contradictions."""
        heuristic = ContradictionHeuristic()
        text = "The sky is blue. The grass is green."
        score = heuristic.validate(text)
        assert score == 1.0
    
    def test_single_sentence(self):
        """Test single sentence (no contradiction possible)."""
        heuristic = ContradictionHeuristic()
        text = "This is a single sentence."
        score = heuristic.validate(text)
        assert score == 1.0


class TestHallucinationHeuristic:
    """Test hallucination detection."""
    
    def test_normal_text(self):
        """Test normal text without hallucination patterns."""
        heuristic = HallucinationHeuristic()
        text = "This is a straightforward statement about facts."
        score = heuristic.validate(text)
        assert 0.5 <= score <= 1.0
    
    def test_excessive_hedging(self):
        """Test text with excessive hedging."""
        heuristic = HallucinationHeuristic()
        text = "Maybe perhaps possibly it might could probably be true."
        score = heuristic.validate(text)
        assert score < 1.0
    
    def test_details(self):
        """Test that details are provided."""
        heuristic = HallucinationHeuristic()
        heuristic.validate("Test text")
        details = heuristic.get_details()
        assert "hedging_count" in details
        assert "specific_claims" in details
