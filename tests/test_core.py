import pytest
import os
from ailocalmemory.core.session import ChatSession
from ailocalmemory.core.storage import InMemoryStorage, SQLiteStorage
from ailocalmemory.core.optimizer import SlidingWindowOptimizer

def test_in_memory_storage():
    session = ChatSession(session_id="test_mem", storage="memory")
    session.add_user_message("Hello")
    session.add_assistant_message("Hi there")
    
    history = session.get_full_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    
    session.clear()
    assert len(session.get_full_history()) == 0

def test_sqlite_storage(tmp_path):
    db_path = tmp_path / "test.db"
    
    # Session 1 writes data
    session1 = ChatSession(
        session_id="test_db", 
        storage="sqlite", 
        storage_kwargs={"db_path": str(db_path)}
    )
    session1.add_user_message("Message 1")
    session1.add_assistant_message("Response 1")
    
    # Session 2 (new instance) reads data
    session2 = ChatSession(
        session_id="test_db", 
        storage="sqlite", 
        storage_kwargs={"db_path": str(db_path)}
    )
    history = session2.get_full_history()
    
    assert len(history) == 2
    assert history[0]["content"] == "Message 1"
    
    session2.clear()
    assert len(session2.get_full_history()) == 0

def test_sliding_window_optimizer():
    optimizer = SlidingWindowOptimizer(k=2)
    session = ChatSession(
        session_id="test_opt", 
        storage="memory", 
        optimizer=optimizer
    )
    
    session.add_user_message("M1")
    session.add_assistant_message("M2")
    session.add_user_message("M3")
    session.add_assistant_message("M4")
    
    # Full history is 4
    assert len(session.get_full_history()) == 4
    
    # Context should only be the last 2 messages (M3 and M4)
    context = session.get_context()
    assert len(context) == 2
    assert context[0]["content"] == "M3"
    assert context[1]["content"] == "M4"
