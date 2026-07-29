# AILocalMemory 🧠

*Enterprise-Grade Agnostic Memory Management Library for Local AI.*

[![PyPI version](https://badge.fury.io/py/ailocalmemory.svg)](https://badge.fury.io/py/ailocalmemory)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AILocalMemory is a robust, lightweight Python library designed to handle chat history, state persistence, and context window optimization for any local AI endpoint (Ollama, LM Studio, vLLM, Llama.cpp, etc.). 

Since most local AI instances are stateless, AILocalMemory handles the tedious parts of conversational AI: storing past messages, truncating them when they get too long (so your AI doesn't crash from OOM), and abstracting the API calls.

## Features ✨
- **Plug-and-Play Adapters:** Built-in adapters for Ollama and OpenAI-compatible endpoints.
- **Async & Streaming Support:** Fully supports asynchronous operations (`asyncio`) and token streaming for responsive UIs.
- **Persistent Memory:** Keep conversations across reboots using SQLite (Thread-safe).
- **Smart Context Optimization:** Automatically prunes old messages to fit your model's maximum token limit.

## Installation 📦

You can install AILocalMemory directly from PyPI:

```bash
pip install ailocalmemory
```

## Quick Start (Using Adapters) 🚀

The easiest way to use the library is via built-in adapters. 

```python
from ailocalmemory import ChatSession, OllamaAdapter

# 1. Initialize a session (keeps messages in memory by default)
session = ChatSession(session_id="user_1")

# 2. Wrap it with an adapter (e.g., Ollama)
chat = OllamaAdapter(model="llama3", memory_session=session)

# 3. Chat! The adapter automatically handles context saving and retrieving.
response = chat.send("Hello, my name is Alice and my favorite color is blue.")
print("AI:", response)

response = chat.send("What is my name and favorite color?")
print("AI:", response) # It remembers!
```

## Advanced: Streaming & Async ⚡

AILocalMemory is built for modern applications like FastAPI or Discord bots.

### Streaming
```python
# Stream the response token by token
response_stream = chat.send("Write a long story", stream=True)
for chunk in response_stream:
    print(chunk, end="", flush=True)
```

### Async / Await
```python
import asyncio
from ailocalmemory import ChatSession, OllamaAdapter

async def main():
    session = ChatSession(session_id="user_2", storage="sqlite")
    chat = OllamaAdapter(model="llama3", memory_session=session)
    
    # Async Streaming
    stream = await chat.send_async("Tell me a joke", stream=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Storage Options 💾

- `memory` (Default): Volatile, lost when script ends.
- `sqlite`: Persistent, thread-safe, saves to a local `~/.ailocalmemory/ailocalmemory.db` file automatically.

## Optimizers ⚙️

By default, the session uses `TokenLimitOptimizer(max_tokens=8192)` to ensure your context doesn't explode. You can also use `SlidingWindowOptimizer` to just keep the last $K$ messages.

```python
from ailocalmemory import ChatSession, SlidingWindowOptimizer

# Keeps the last 10 messages and ensures it stays under 8192 tokens
opt = SlidingWindowOptimizer(k=10, max_tokens=8192)
session = ChatSession(session_id="user", optimizer=opt)
```
