"""LLM prompt templates using LangChain."""

from langchain_core.prompts import ChatPromptTemplate

# RAG System Message
RAG_SYSTEM_MESSAGE = """You are a helpful assistant answering questions about documents the user has ingested.
Use ONLY the provided context excerpts to support your answer. If the context does not contain enough information, say so clearly instead of guessing.
Write in clear, natural prose. Match the language of the user's question when possible."""

# RAG Human Message
RAG_HUMAN_MESSAGE = """Context excerpts:
---
{context}
---

Question: {question}

Answer based on the context above:"""

# RAG Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_MESSAGE),
    ("human", RAG_HUMAN_MESSAGE),
])

# Registry of all available templates
PROMPT_TEMPLATES = {
    "rag": RAG_PROMPT,
}
