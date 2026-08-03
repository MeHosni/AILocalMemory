import threading
import uuid
import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

def get_default_chroma_path() -> str:
    # Memaksa penyimpanan database ke dalam folder proyek lokal
    project_root = Path(r"C:\Users\MSI Laptop\Documents\phyton\library")
    db_dir = project_root / "ai_data" / "chroma_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir)

class VectorDatabase:
    """
    Long-Term Memory Storage using ChromaDB.
    Automatically vectorizes messages and performs Semantic Search.
    """
    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.Lock()
        
        if not CHROMADB_AVAILABLE:
            print("[VectorDatabase] ChromaDB is not installed. Long-term memory is disabled.")
            self.collection = None
            return
            
        self.db_path = db_path or get_default_chroma_path()
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # We use the default ONNX MiniLM model so it works instantly without Ollama dependencies
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection_name = "ailocal_longterm_memory"
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, 
                embedding_function=self.ef
            )
        except Exception as e:
            print(f"[VectorDatabase] Error initializing collection: {e}")
            self.collection = None
            
        self._lock = threading.Lock()

    def upsert_memory(self, session_id: str, role: str, content: str):
        """Vectorizes and stores a message into the long-term memory."""
        if not self.collection:
            return
            
        with self._lock:
            try:
                msg_id = str(uuid.uuid4())
                self.collection.add(
                    documents=[f"[{role.capitalize()}]: {content}"],
                    metadatas=[{"session_id": session_id, "role": role}],
                    ids=[msg_id]
                )
            except Exception as e:
                print(f"[VectorDatabase] Failed to save memory: {e}")

    def recall_memories(self, session_id: str, query: str, n_results: int = 3) -> List[str]:
        """Searches for past memories semantically similar to the query."""
        if not self.collection:
            return []
            
        with self._lock:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where={"session_id": session_id}
                )
                
                if results and results['documents'] and results['documents'][0]:
                    memories = []
                    docs = results['documents'][0]
                    metas = results['metadatas'][0]
                    
                    for i in range(len(docs)):
                        # Format "Role: content" sudah dimasukkan sejak awal di dokumen
                        memories.append(docs[i])
                    return memories
            except Exception as e:
                print(f"[VectorDatabase] Failed to recall memory: {e}")
                
        return []
