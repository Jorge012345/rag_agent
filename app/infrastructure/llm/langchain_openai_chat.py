"""OpenAI chat completions via LangChain (infrastructure)."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.domain.ports.llm_port import LlmPort
from app.infrastructure.settings import SETTINGS
from app.infrastructure.llm.prompts import PROMPT_TEMPLATES


class LangChainOpenAIChatAdapter(LlmPort):
    """Generates natural-language replies using ChatOpenAI."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            api_key=SecretStr(SETTINGS.OPENAI.OPENAI_API_KEY),
            model=SETTINGS.OPENAI.MODEL_NAME,
            temperature=SETTINGS.OPENAI.TEMPERATURE,
        )
        # Pre-build LCEL chains for each template (prompt | llm | output_parser)
        self._chains = {
            name: template | self._llm | StrOutputParser()
            for name, template in PROMPT_TEMPLATES.items()
        }

    def generate(self, *, system: str, user: str) -> str:
        """Generate response from system and user strings (legacy method)."""
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        out = self._llm.invoke(messages)
        text = out.content
        return text if isinstance(text, str) else str(text)

    def generate_with_template(self, template_name: str, **variables: Any) -> str:
        """Generate response using LCEL chain with named prompt template."""
        if template_name not in self._chains:
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available templates: {list(self._chains.keys())}"
            )

        chain = self._chains[template_name]
        result = chain.invoke(variables)
        return result
