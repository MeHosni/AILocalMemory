import pytest
import httpx
from ailocalmemory.core.session import ChatSession
from ailocalmemory.adapters.ollama_adapter import OllamaAdapter
from ailocalmemory.adapters.openai_adapter import OpenAICompatibleAdapter
import pytest_asyncio
import asyncio

@pytest.fixture
def session():
    # Use memory storage for fast, isolated tests
    return ChatSession(session_id="test_session", storage="memory")

def test_ollama_send_sync(session, httpx_mock):
    # Mock the Ollama API response
    httpx_mock.add_response(
        json={"message": {"content": "Hello from mocked Ollama!"}}
    )
    
    adapter = OllamaAdapter("llama3", session)
    response = adapter.send("Hello")
    
    assert response == "Hello from mocked Ollama!"
    
    # Check if memory was updated
    history = session.get_full_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hello from mocked Ollama!"

@pytest.mark.asyncio
async def test_ollama_send_async(session, httpx_mock):
    httpx_mock.add_response(
        json={"message": {"content": "Async Hello from mocked Ollama!"}}
    )
    
    adapter = OllamaAdapter("llama3", session)
    response = await adapter.send_async("Hello async")
    
    assert response == "Async Hello from mocked Ollama!"
    
    history = session.get_full_history()
    assert history[1]["content"] == "Async Hello from mocked Ollama!"

def test_openai_send_sync(session, httpx_mock):
    # Mock the OpenAI API response
    httpx_mock.add_response(
        json={
            "choices": [
                {"message": {"content": "Hello from mocked OpenAI!"}}
            ]
        }
    )
    
    adapter = OpenAICompatibleAdapter("local-model", session)
    response = adapter.send("Hello OpenAI")
    
    assert response == "Hello from mocked OpenAI!"
    
    history = session.get_full_history()
    assert len(history) == 2
    assert history[1]["content"] == "Hello from mocked OpenAI!"

@pytest.mark.asyncio
async def test_openai_send_async(session, httpx_mock):
    httpx_mock.add_response(
        json={
            "choices": [
                {"message": {"content": "Async Hello from mocked OpenAI!"}}
            ]
        }
    )
    
    adapter = OpenAICompatibleAdapter("local-model", session)
    response = await adapter.send_async("Hello async OpenAI")
    
    assert response == "Async Hello from mocked OpenAI!"
    
    history = session.get_full_history()
    assert history[1]["content"] == "Async Hello from mocked OpenAI!"
