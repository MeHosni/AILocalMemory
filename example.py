import httpx
from ailocalmemory import ChatSession, OllamaAdapter

def test_ollama_is_running():
    try:
        httpx.get("http://localhost:11434", timeout=2.0)
        return True
    except httpx.RequestError:
        return False

def main():
    print("Welcome to AILocalMemory v0.3.0 Example!")
    print("We will create a persistent SQLite session and talk to Ollama.")
    
    if not test_ollama_is_running():
        print("WARNING: Ollama is not running on http://localhost:11434.")
        print("Please start Ollama to see the adapter in action.")
        return
        
    # 1. Initialize persistent session using context manager!
    with ChatSession(session_id="example_user", storage="sqlite") as session:
        # 2. Check if there are past messages
        history = session.get_full_history()
        if history:
            print(f"Loaded {len(history)} past messages from SQLite database!")
        else:
            print("No past history found. Starting a fresh conversation.")
            # We can upsert system prompt
            session.add_system_message("You are a helpful AI assistant.")
            
        # 3. Setup adapter
        chat = OllamaAdapter(model="llama3", memory_session=session)
        
        # 4. Interactive chat loop
        print("\nStart chatting! (type 'quit' to exit, 'clear' to erase memory)")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'clear':
                session.clear()
                print("Memory cleared!")
                continue
                
            print("AI: ", end="", flush=True)
            try:
                # Enable streaming!
                response_stream = chat.send(user_input, stream=True)
                
                if isinstance(response_stream, str):
                    print(response_stream)
                else:
                    for chunk in response_stream:
                        print(chunk, end="", flush=True)
                    print()
            except Exception as e:
                print(f"\nError communicating with AI: {e}")

if __name__ == "__main__":
    main()
