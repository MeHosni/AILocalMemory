from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Generator, AsyncGenerator, Union
from ..core.session import ChatSession

class BaseAdapter(ABC):
    """
    Abstract base class for all AI adapters.
    An adapter wraps a ChatSession and a specific AI API (like Ollama or OpenAI).
    """
    
    def __init__(self, memory_session: ChatSession):
        self.memory = memory_session
        
    @abstractmethod
    def send(self, message: str, stream: bool = False, save_history: bool = True, **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Sends a message to the AI.
        If stream=False, returns the full string response.
        If stream=True, returns a generator that yields string chunks, and saves to memory at the end.
        If save_history=False, the message and response will not be logged in the database.
        """
        pass
        
    @abstractmethod
    async def send_async(self, message: str, stream: bool = False, save_history: bool = True, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Asynchronous version of send().
        If stream=False, returns the full string response.
        If stream=True, returns an async generator yielding string chunks.
        If save_history=False, the message and response will not be logged in the database.
        """
        pass
