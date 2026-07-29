# AILocalMemory 🧠

*Agnostic Memory Management Library for Local AI.*

AILocalMemory is a lightweight Python library designed to handle chat history, state persistence, and context window optimization for any local AI endpoint (Ollama, LM Studio, vLLM, Llama.cpp, etc.). 

Since most local AI instances are stateless, AILocalMemory handles the tedious parts of conversational AI: storing past messages, truncating them when they get too long (so your AI doesn't crash from OOM), and abstracting the API calls.

## Installation

```bash
pip install -e .
```

## Quick Start (Using Adapters)

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

## Pure Agnostic Usage

If you prefer calling the AI endpoint yourself, you can use `AILocalMemory` purely as a state manager:

```python
from ailocalmemory import ChatSession

session = ChatSession(session_id="user_1", storage="sqlite")
session.add_user_message("What is the capital of France?")

# Get the optimized context (truncated to fit model limits)
context = session.get_context()

# ... send `context` to your AI engine of choice ...
response_text = "Paris"

session.add_assistant_message(response_text)
```

## Storage Options

- `memory` (Default): Volatile, lost when script ends.
- `sqlite`: Persistent, saves to a local `ailocalmemory.db` file automatically.

## Optimizers

By default, the session uses `TokenLimitOptimizer(max_tokens=2048)` to ensure your context doesn't explode. You can also use `SlidingWindowOptimizer` to just keep the last $K$ messages.
