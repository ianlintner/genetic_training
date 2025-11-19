"""
Storage backends for validation results and training data.
"""
import json
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import uuid
from loguru import logger

from app.base import BaseStore


class JSONStore(BaseStore):
    """JSON file-based storage."""
    
    def __init__(self, path: str = "data/results.json"):
        """Initialize JSON store."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load {self.path}, starting fresh")
                return {}
        return {}
    
    def _save_to_disk(self):
        """Save data to JSON file."""
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
    
    def save(self, record: Dict[str, Any]) -> str:
        """Save a record."""
        record_id = record.get("id", str(uuid.uuid4()))
        record["id"] = record_id
        record["created_at"] = datetime.now().isoformat()
        
        self._data[record_id] = record
        self._save_to_disk()
        
        return record_id
    
    def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query records."""
        results = list(self._data.values())
        
        # Apply filters
        if filters:
            results = [
                r for r in results
                if all(r.get(k) == v for k, v in filters.items())
            ]
        
        # Apply pagination
        return results[offset:offset + limit]
    
    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        return self._data.get(record_id)
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """Update a record."""
        if record_id in self._data:
            self._data[record_id].update(updates)
            self._data[record_id]["updated_at"] = datetime.now().isoformat()
            self._save_to_disk()
            return True
        return False
    
    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        if record_id in self._data:
            del self._data[record_id]
            self._save_to_disk()
            return True
        return False
    
    def close(self):
        """Close store."""
        pass


class SQLiteStore(BaseStore):
    """SQLite database storage."""
    
    def __init__(self, path: str = "data/results.db"):
        """Initialize SQLite store."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """)
        self.conn.commit()
    
    def save(self, record: Dict[str, Any]) -> str:
        """Save a record."""
        record_id = record.get("id", str(uuid.uuid4()))
        record["id"] = record_id
        
        created_at = datetime.now().isoformat()
        data_json = json.dumps(record, default=str)
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO records (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (record_id, data_json, created_at, created_at)
        )
        self.conn.commit()
        
        return record_id
    
    def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query records."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT data FROM records ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        
        results = []
        for row in cursor.fetchall():
            record = json.loads(row["data"])
            
            # Apply filters
            if filters is None or all(record.get(k) == v for k, v in filters.items()):
                results.append(record)
        
        return results
    
    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        
        if row:
            return json.loads(row["data"])
        return None
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """Update a record."""
        record = self.get_by_id(record_id)
        if record:
            record.update(updates)
            record["updated_at"] = datetime.now().isoformat()
            
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE records SET data = ?, updated_at = ? WHERE id = ?",
                (json.dumps(record, default=str), record["updated_at"], record_id)
            )
            self.conn.commit()
            return True
        return False
    
    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM records WHERE id = ?", (record_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class ChromaDBStore(BaseStore):
    """ChromaDB vector store."""
    
    def __init__(self, path: str = "data/chromadb", collection_name: str = "validation_results"):
        """Initialize ChromaDB store."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.Client(Settings(
                persist_directory=path,
                anonymized_telemetry=False
            ))
            self.collection = self.client.get_or_create_collection(collection_name)
        except ImportError:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")
    
    def save(self, record: Dict[str, Any]) -> str:
        """Save a record."""
        record_id = record.get("id", str(uuid.uuid4()))
        record["id"] = record_id
        
        # Extract text for embedding
        text = record.get("text", json.dumps(record))
        
        self.collection.add(
            ids=[record_id],
            documents=[text],
            metadatas=[self._flatten_dict(record)]
        )
        
        return record_id
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary for ChromaDB metadata."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            elif isinstance(v, (str, int, float, bool)):
                items.append((new_key, v))
            else:
                items.append((new_key, str(v)))
        return dict(items)
    
    def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query records."""
        results = self.collection.get(limit=limit, offset=offset)
        
        records = []
        for i, metadata in enumerate(results["metadatas"]):
            record = metadata.copy()
            record["text"] = results["documents"][i]
            records.append(record)
        
        return records
    
    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        results = self.collection.get(ids=[record_id])
        
        if results["ids"]:
            record = results["metadatas"][0].copy()
            record["text"] = results["documents"][0]
            return record
        return None
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """Update a record."""
        record = self.get_by_id(record_id)
        if record:
            record.update(updates)
            text = record.get("text", json.dumps(record))
            
            self.collection.update(
                ids=[record_id],
                documents=[text],
                metadatas=[self._flatten_dict(record)]
            )
            return True
        return False
    
    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        try:
            self.collection.delete(ids=[record_id])
            return True
        except Exception:
            return False
    
    def close(self):
        """Close store."""
        pass


def create_store(backend: str = "json", config: Optional[Dict[str, Any]] = None) -> BaseStore:
    """
    Factory function to create appropriate store.
    
    Args:
        backend: Storage backend type
        config: Backend-specific configuration
        
    Returns:
        BaseStore instance
    """
    if config is None:
        from app.config import get_config
        app_config = get_config()
        # Get the config using both the original and aliased field names
        if backend == "json":
            config = app_config.storage.json_config
        elif backend == "sqlite":
            config = app_config.storage.sqlite_config
        elif backend == "chromadb":
            config = app_config.storage.chromadb_config
        elif backend == "postgresql":
            config = app_config.storage.postgresql_config
        else:
            config = {}
    
    if backend == "json":
        path = config.get("path", "data/results.json")
        return JSONStore(path)
    
    elif backend == "sqlite":
        path = config.get("path", "data/results.db")
        return SQLiteStore(path)
    
    elif backend == "chromadb":
        path = config.get("path", "data/chromadb")
        collection_name = config.get("collection_name", "validation_results")
        return ChromaDBStore(path, collection_name)
    
    else:
        raise ValueError(f"Unknown storage backend: {backend}")
