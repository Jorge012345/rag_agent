"""OpenAI chat completions via LangChain (infrastructure)."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.domain.ports.llm_port import LlmPort
from app.infrastructure.settings import SETTINGS


class LangChainOpenAIChatAdapter(LlmPort):
    """Generates natural-language replies using ChatOpenAI."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            api_key=SecretStr(SETTINGS.OPENAI.OPENAI_API_KEY),
            model=SETTINGS.OPENAI.MODEL_NAME,
            temperature=SETTINGS.OPENAI.TEMPERATURE,
        )

    def generate(self, *, system: str, user: str) -> str:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        out = self._llm.invoke(messages)
        text = out.content
        return text if isinstance(text, str) else str(text)
