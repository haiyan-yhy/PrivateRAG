"""LLM + Embedding factory — swap providers via config.LLM_PROVIDER."""

from __future__ import annotations

import logging
from typing import Tuple

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

import config

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
你是一个基于知识库回答问题的 AI 助手。

请严格基于下面提供的上下文回答问题。
如果上下文中没有答案，请明确回答：
"我无法从当前知识库中找到答案。"

【上下文】
{context}

【用户问题】
{query}

【回答要求】
1. 只基于上下文回答
2. 不允许编造信息
3. 尽量简洁清晰
4. 最后附上引用来源（文件名）
"""


def _build_openai() -> Tuple[LLM, BaseEmbedding]:
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")

    llm = OpenAI(model=config.OPENAI_LLM_MODEL, api_key=config.OPENAI_API_KEY)
    embed = OpenAIEmbedding(
        model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY
    )
    return llm, embed


def _build_anthropic() -> Tuple[LLM, BaseEmbedding]:
    from llama_index.llms.anthropic import Anthropic
    from llama_index.embeddings.openai import OpenAIEmbedding

    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "Anthropic provider requires OPENAI_API_KEY for embeddings "
            "(Claude does not expose an embedding API). "
            "Alternatively, set LLM_PROVIDER=ollama for a fully local setup."
        )

    llm = Anthropic(
        model=config.ANTHROPIC_LLM_MODEL, api_key=config.ANTHROPIC_API_KEY
    )
    embed = OpenAIEmbedding(
        model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY
    )
    return llm, embed


def _build_ollama() -> Tuple[LLM, BaseEmbedding]:
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding

    llm = Ollama(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=120.0,
    )
    embed = OllamaEmbedding(
        model_name=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    return llm, embed


def _build_deepseek() -> Tuple[LLM, BaseEmbedding]:
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    if not config.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is not set.")

    # DeepSeek API is OpenAI-compatible
    llm = OpenAILike(
        model=config.DEEPSEEK_LLM_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        api_base=config.DEEPSEEK_BASE_URL,
        is_chat_model=True,
        context_window=65536,
    )
    # DeepSeek has no embedding API — use a local HuggingFace model instead
    embed = HuggingFaceEmbedding(model_name=config.DEEPSEEK_EMBED_MODEL)
    return llm, embed


def get_llm_and_embedding() -> Tuple[LLM, BaseEmbedding]:
    provider = config.LLM_PROVIDER.lower()
    logger.info("Loading provider: %s", provider)

    if provider == "openai":
        return _build_openai()
    if provider == "anthropic":
        return _build_anthropic()
    if provider == "ollama":
        return _build_ollama()
    if provider == "deepseek":
        return _build_deepseek()

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Choose: openai | anthropic | ollama | deepseek"
    )


def build_prompt(query: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, query=query)


def build_chat_messages(query: str, context: str, history: list[dict]):
    from llama_index.core.llms import ChatMessage, MessageRole

    system_content = (
        "你是一个基于知识库回答问题的 AI 助手。\n"
        "请严格基于下面提供的上下文回答问题。\n"
        "如果上下文中没有答案，请明确回答："我无法从当前知识库中找到答案。"\n\n"
        f"【上下文】\n{context}\n\n"
        "【回答要求】\n"
        "1. 只基于上下文回答\n"
        "2. 不允许编造信息\n"
        "3. 尽量简洁清晰\n"
        "4. 最后附上引用来源（文件名）"
    )
    messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_content)]
    for msg in history:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=msg["content"]))
    messages.append(ChatMessage(role=MessageRole.USER, content=query))
    return messages
