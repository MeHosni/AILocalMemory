from typing import List, Dict, Optional
from .storage import BaseStorage, InMemoryStorage, SQLiteStorage
from .optimizer import BaseOptimizer, TokenLimitOptimizer
from .vector import VectorDatabase

class ChatSession:
    """
    Manages the conversation history for a specific session/user.
    Acts as the primary interface for the LocalMemory library.
    """
    
    def __init__(
        self, 
        session_id: str, 
        storage: str = "memory",
        storage_kwargs: Optional[Dict] = None,
        optimizer: Optional[BaseOptimizer] = None,
        enable_rag: bool = True
    ):
        self.session_id = session_id
        
        # Setup Vector Storage (Long-Term Memory)
        self.enable_rag = enable_rag
        if self.enable_rag:
            self.vector_db = VectorDatabase()
        
        # Setup storage
        storage_kwargs = storage_kwargs or {}
        if storage == "memory":
            self.storage = InMemoryStorage()
        elif storage == "sqlite":
            self.storage = SQLiteStorage(**storage_kwargs)
        elif isinstance(storage, BaseStorage):
            self.storage = storage
        else:
            raise ValueError(f"Unknown storage type: {storage}")
            
        # Setup optimizer (default to TokenLimitOptimizer to prevent crashes)
        self.optimizer = optimizer or TokenLimitOptimizer(max_tokens=8192)

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the session."""
        self.storage.save_message(self.session_id, role, content)
        
        if self.enable_rag and role in ["user", "assistant"]:
            self.vector_db.upsert_memory(self.session_id, role, content)
        
    def add_user_message(self, content: str) -> None:
        """Helper to add a user message."""
        self.add_message("user", content)
        
    def add_assistant_message(self, content: str) -> None:
        """Helper to add an assistant message."""
        self.add_message("assistant", content)
        
    def add_system_message(self, content: str) -> None:
        """
        Helper to add a system message. Uses upsert logic to ensure
        only one system message exists per session.
        """
        self.storage.save_system_message(self.session_id, content)

    def get_full_history(self) -> List[Dict[str, str]]:
        """Returns the full unoptimized history."""
        return self.storage.get_messages(self.session_id)
        
    def get_context(self) -> List[Dict[str, str]]:
        """
        Returns the optimized (truncated) history ready to be sent to the LLM.
        """
        messages = self.get_full_history()
        
        if self.enable_rag and messages:
            # Find the last user message to use as the semantic query
            last_user_msg = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), None)
            if last_user_msg:
                memories = self.vector_db.recall_memories(self.session_id, last_user_msg, n_results=3)
                if memories:
                    memory_text = "\n".join([f"- {m}" for m in memories])
                    rag_prompt = f"\n\n[LONG-TERM MEMORY RECALL]\nHere are relevant past memories from this user:\n{memory_text}\nUse this context to inform your response if relevant."
                    
                    # Inject into the system message (which is always preserved by the optimizer)
                    if messages[0].get("role") == "system":
                        messages[0]["content"] += rag_prompt
                        
        return self.optimizer.optimize(messages)
        
    def clear(self) -> None:
        """Clears all messages from the session."""
        self.storage.clear_session(self.session_id)
        
    def close(self) -> None:
        """Closes underlying storage resources (e.g. database connections)."""
        if hasattr(self.storage, "close"):
            self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
