"""Port: text generation from prompts (chat / completion)."""

from abc import ABC, abstractmethod


class LlmPort(ABC):
    """Abstraction over a language model for grounded answers."""

    @abstractmethod
    def generate(self, *, system: str, user: str) -> str:
        """Return model text for the given system and user messages."""
