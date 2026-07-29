from abc import ABC, abstractmethod
from typing import List, Dict

class BaseOptimizer(ABC):
    """Abstract base class for context optimizers."""
    
    @abstractmethod
    def optimize(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Takes a full list of messages and returns an optimized (truncated) list."""
        pass

class TokenLimitOptimizer(BaseOptimizer):
    """
    A heuristic optimizer that removes oldest messages until the 
    estimated token count is below the limit.
    """
    
    def __init__(self, max_tokens: int = 8192, chars_per_token: float = 4.0):
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        
    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token)
        
    def optimize(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return []
            
        has_system_prompt = messages[0].get("role") == "system"
        if has_system_prompt:
            system_msg = [messages[0]]
            chat_history = messages[1:]
            current_tokens = self._estimate_tokens(messages[0].get("content", ""))
        else:
            system_msg = []
            chat_history = messages
            current_tokens = 0
            
        # Add messages from newest to oldest until limit is reached
        kept_messages = []
        is_first = True
        
        for msg in reversed(chat_history):
            msg_tokens = self._estimate_tokens(msg.get("content", ""))
            
            # If it's the newest message (first in reversed list), keep it regardless of limit
            if is_first:
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens + 10
                is_first = False
                continue
                
            # For older messages, only keep them if they fit
            if current_tokens + msg_tokens + 10 <= self.max_tokens:
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens + 10
            else:
                break
                
        return system_msg + kept_messages

class SlidingWindowOptimizer(TokenLimitOptimizer):
    """
    Keeps only the last K messages, AND ensures it doesn't exceed the token limit.
    Inherits from TokenLimitOptimizer to prevent OOM on very long single messages.
    """
    
    def __init__(self, k: int = 10, max_tokens: int = 8192):
        super().__init__(max_tokens=max_tokens)
        self.k = k
        
    def optimize(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return []
            
        has_system_prompt = messages[0].get("role") == "system"
        
        if has_system_prompt:
            system_msg = [messages[0]]
            chat_history = messages[1:]
        else:
            system_msg = []
            chat_history = messages
            
        # Keep only the last K messages from chat history
        optimized_history = chat_history[-self.k:] if len(chat_history) > self.k else chat_history
        
        # Then pass through token limit optimizer to be absolutely safe
        return super().optimize(system_msg + optimized_history)
