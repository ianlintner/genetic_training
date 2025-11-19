"""
Unit tests for configuration management.
"""
import pytest
from app.config import load_config


class TestConfig:
    """Test configuration loading."""
    
    def test_default_config(self):
        """Test loading default configuration."""
        config = load_config("configs/config.yaml")
        assert config is not None
        assert hasattr(config, "models")
        assert hasattr(config, "validation")
        assert hasattr(config, "storage")
        assert hasattr(config, "api")
