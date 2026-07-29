import sqlite3
import json
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

class BaseStorage(ABC):
    """Abstract base class for memory storage."""
    
    @abstractmethod
    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Saves a single message to the session."""
        pass
        
    @abstractmethod
    def save_system_message(self, session_id: str, content: str) -> None:
        """Saves or updates the system message for a session (upsert)."""
        pass
        
    @abstractmethod
    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves all messages for a given session."""
        pass
        
    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Clears all messages for a session."""
        pass
        
    def close(self):
        """Clean up resources if needed."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class InMemoryStorage(BaseStorage):
    """Volatile storage that keeps messages in memory. Useful for testing."""
    
    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._system_store: Dict[str, str] = {}
        self._lock = threading.Lock()
        
    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append({"role": role, "content": content})
        
    def save_system_message(self, session_id: str, content: str) -> None:
        with self._lock:
            self._system_store[session_id] = content
        
    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            msgs = []
            if session_id in self._system_store:
                msgs.append({"role": "system", "content": self._system_store[session_id]})
            msgs.extend(self._store.get(session_id, []).copy())
            return msgs
        
    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
            if session_id in self._system_store:
                del self._system_store[session_id]

def get_default_db_path() -> str:
    home = Path.home() / ".ailocalmemory"
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "ailocalmemory.db")

class SQLiteStorage(BaseStorage):
    """Persistent storage using SQLite database."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_default_db_path()
        self._lock = threading.Lock()
        # Connection pool/cache (one connection per instance)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        
    def _init_db(self):
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self._conn.commit()
            
    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            self._conn.commit()
        
    def save_system_message(self, session_id: str, content: str) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            # Upsert logic: Delete old system message, then insert new one
            cursor.execute(
                "DELETE FROM messages WHERE session_id = ? AND role = 'system'",
                (session_id,)
            )
            cursor.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "system", content)
            )
            self._conn.commit()
            
    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            cursor = self._conn.cursor()
            # We need system message first, then the rest ordered by ID
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY CASE WHEN role = 'system' THEN 0 ELSE 1 END, id ASC",
                (session_id,)
            )
            rows = cursor.fetchall()
            return [{"role": row[0], "content": row[1]} for row in rows]
            
    def clear_session(self, session_id: str) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.commit()

    def close(self):
        with self._lock:
            if hasattr(self, '_conn') and self._conn:
                self._conn.close()
                self._conn = None
