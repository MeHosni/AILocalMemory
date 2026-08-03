import httpx
import json
import asyncio
from typing import Generator, AsyncGenerator, Union, Any
from .base import BaseAdapter
from ..core.session import ChatSession

class OllamaAdapter(BaseAdapter):
    """
    Adapter for connecting to an Ollama instance directly via its REST API.
    """
    
    def __init__(
        self, 
        model: str, 
        memory_session: ChatSession, 
        base_url: str = "http://localhost:11434"
    ):
        super().__init__(memory_session)
        self.model = model
        self.base_url = base_url.rstrip("/")
        
    def _prepare_request(self, message: str, stream: bool, save_history: bool = True, bypass_memory: bool = False, **kwargs):
        history_message = kwargs.pop("history_message", message)
        is_system_trigger = kwargs.pop("is_system_trigger", False)
        
        if save_history and not is_system_trigger:
            self.memory.add_user_message(history_message)
            
        if bypass_memory:
            messages = [{"role": "user", "content": message}]
        else:
            messages = self.memory.get_context()
            if save_history and not is_system_trigger:
                # Replace the content of the last user message in the context (which was just added)
                # with the augmented RAG message so the LLM sees the search results.
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] = message
            else:
                messages.append({"role": "user", "content": message})
            
        # Parse multimodal <IMAGE:...> tags from messages without breaking SQLite
        import re
        for msg in messages:
            images = re.findall(r'<IMAGE:(.*?)>', msg.get("content", ""))
            if images:
                msg["images"] = images
                msg["content"] = re.sub(r'<IMAGE:.*?>', '', msg["content"])
                
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        return url, payload
        
    def send(self, message: str, stream: bool = False, save_history: bool = True, **kwargs) -> Union[str, Generator[str, None, None]]:
        is_system_trigger = kwargs.get("is_system_trigger", False)
        url, payload = self._prepare_request(message, stream, save_history=save_history, **kwargs)
        
        try:
            if not stream:
                with httpx.Client() as client:
                    response = client.post(url, json=payload, timeout=300.0)
                    response.raise_for_status()
                    data = response.json()
                    assistant_content = data.get("message", {}).get("content", "")
                    if assistant_content and save_history and not is_system_trigger:
                        self.memory.add_assistant_message(assistant_content)
                    return assistant_content
            else:
                def response_generator():
                    full_content = []
                    try:
                        with httpx.Client() as client:
                            with client.stream("POST", url, json=payload, timeout=300.0) as response:
                                response.raise_for_status()
                                for line in response.iter_lines():
                                    if line:
                                        try:
                                            chunk = json.loads(line)
                                            content = chunk.get("message", {}).get("content", "")
                                            if content:
                                                full_content.append(content)
                                                yield content
                                        except json.JSONDecodeError:
                                            continue
                    finally:
                        if full_content and save_history and not is_system_trigger:
                            self.memory.add_assistant_message("".join(full_content))
                
                return response_generator()
                
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to Ollama at {self.base_url}: {str(e)}"
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg
            
    async def send_async(self, message: str, stream: bool = False, save_history: bool = True, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        is_system_trigger = kwargs.get("is_system_trigger", False)
        url, payload = await asyncio.to_thread(self._prepare_request, message, stream, save_history=save_history, **kwargs)
        
        try:
            if not stream:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload, timeout=300.0)
                    response.raise_for_status()
                    data = response.json()
                    assistant_content = data.get("message", {}).get("content", "")
                    if assistant_content and save_history and not is_system_trigger:
                        await asyncio.to_thread(self.memory.add_assistant_message, assistant_content)
                    return assistant_content
            else:
                async def response_generator():
                    full_content = []
                    try:
                        async with httpx.AsyncClient() as client:
                            async with client.stream("POST", url, json=payload, timeout=300.0) as response:
                                response.raise_for_status()
                                async for line in response.aiter_lines():
                                    if line:
                                        try:
                                            chunk = json.loads(line)
                                            content = chunk.get("message", {}).get("content", "")
                                            if content:
                                                full_content.append(content)
                                                yield content
                                        except json.JSONDecodeError:
                                            continue
                    except httpx.RequestError as e:
                        yield f"Failed to connect to Ollama at {self.base_url}: {str(e)}"
                    finally:
                        if full_content and save_history and not is_system_trigger:
                            await asyncio.to_thread(self.memory.add_assistant_message, "".join(full_content))
                        
                return response_generator()
                
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to Ollama at {self.base_url}: {str(e)}"
            if stream:
                async def error_gen(): yield error_msg
                return error_gen()
            return error_msg
