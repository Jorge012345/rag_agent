"""Port: text generation from prompts (chat / completion)."""

from abc import ABC, abstractmethod
from typing import Any


class LlmPort(ABC):
    """Abstraction over a language model for grounded answers."""

    @abstractmethod
    def generate(self, *, system: str, user: str) -> str:
        """Return model text for the given system and user messages."""

    @abstractmethod
    def generate_with_template(self, template_name: str, **variables: Any) -> str:
        """Generate response using a named prompt template with variables."""
