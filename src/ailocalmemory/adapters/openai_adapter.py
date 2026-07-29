import httpx
import json
import asyncio
from typing import Generator, AsyncGenerator, Union, Any
from .base import BaseAdapter
from ..core.session import ChatSession

class OpenAICompatibleAdapter(BaseAdapter):
    """
    Adapter for connecting to OpenAI-compatible REST endpoints 
    (e.g., LM Studio, vLLM, text-generation-webui).
    """
    
    def __init__(
        self, 
        model: str, 
        memory_session: ChatSession, 
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed"
    ):
        super().__init__(memory_session)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        
    def _prepare_request(self, message: str, stream: bool, **kwargs):
        self.memory.add_user_message(message)
        messages = self.memory.get_context()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        return url, headers, payload
        
    def send(self, message: str, stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        url, headers, payload = self._prepare_request(message, stream, **kwargs)
        
        try:
            if not stream:
                with httpx.Client() as client:
                    response = client.post(url, json=payload, headers=headers, timeout=60.0)
                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return ""
                    assistant_content = choices[0].get("message", {}).get("content", "")
                    if assistant_content:
                        self.memory.add_assistant_message(assistant_content)
                    return assistant_content
            else:
                def response_generator():
                    full_content = []
                    try:
                        with httpx.Client() as client:
                            with client.stream("POST", url, json=payload, headers=headers, timeout=60.0) as response:
                                response.raise_for_status()
                                for line in response.iter_lines():
                                    if line:
                                        if line.startswith("data: ") and line != "data: [DONE]":
                                            try:
                                                chunk = json.loads(line[6:])
                                                choices = chunk.get("choices", [])
                                                if choices:
                                                    delta = choices[0].get("delta", {}).get("content", "")
                                                    if delta:
                                                        full_content.append(delta)
                                                        yield delta
                                            except json.JSONDecodeError:
                                                continue
                    finally:
                        if full_content:
                            self.memory.add_assistant_message("".join(full_content))
                
                return response_generator()
                
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to API at {self.base_url}: {str(e)}"
            if stream:
                def error_gen(): yield error_msg
                return error_gen()
            return error_msg
            
    async def send_async(self, message: str, stream: bool = False, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        url, headers, payload = await asyncio.to_thread(self._prepare_request, message, stream, **kwargs)
        
        try:
            if not stream:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload, headers=headers, timeout=60.0)
                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return ""
                    assistant_content = choices[0].get("message", {}).get("content", "")
                    if assistant_content:
                        await asyncio.to_thread(self.memory.add_assistant_message, assistant_content)
                    return assistant_content
            else:
                async def response_generator():
                    full_content = []
                    try:
                        async with httpx.AsyncClient() as client:
                            async with client.stream("POST", url, json=payload, headers=headers, timeout=60.0) as response:
                                response.raise_for_status()
                                async for line in response.aiter_lines():
                                    if line:
                                        if line.startswith("data: ") and line != "data: [DONE]":
                                            try:
                                                chunk = json.loads(line[6:])
                                                choices = chunk.get("choices", [])
                                                if choices:
                                                    delta = choices[0].get("delta", {}).get("content", "")
                                                    if delta:
                                                        full_content.append(delta)
                                                        yield delta
                                            except json.JSONDecodeError:
                                                continue
                    finally:
                        if full_content:
                            await asyncio.to_thread(self.memory.add_assistant_message, "".join(full_content))
                        
                return response_generator()
                
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to API at {self.base_url}: {str(e)}"
            if stream:
                async def error_gen(): yield error_msg
                return error_gen()
            return error_msg
