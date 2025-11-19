"""
Unit tests for storage backends.
"""
import pytest
import os
import tempfile

from app.storage import JSONStore, SQLiteStore


class TestJSONStore:
    """Test JSON storage backend."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_save_and_retrieve(self, temp_file):
        """Test saving and retrieving a record."""
        store = JSONStore(temp_file)
        record = {"text": "Test", "score": 0.85}
        record_id = store.save(record)
        assert record_id is not None
        retrieved = store.get_by_id(record_id)
        assert retrieved is not None
        assert retrieved["text"] == "Test"
        assert retrieved["score"] == 0.85
        store.close()
